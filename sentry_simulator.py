import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from sentry_mpc_detailed import (
    MPCConfig,
    POSTURE_ATTACK,
    POSTURE_DEFENSE,
    POSTURE_MOVE,
    SentryMPC,
    SentryObservation,
    SentryState,
)

try:
    import matplotlib

    HAVE_MPL = False
    MPL_BACKEND = None
    plt = None
    FuncAnimation = None
    Rectangle = None

    for backend in ("TkAgg", "QtAgg", "Agg"):
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from matplotlib.patches import Rectangle

            HAVE_MPL = True
            MPL_BACKEND = backend
            break
        except Exception:
            continue
except Exception:
    plt = None
    FuncAnimation = None
    Rectangle = None
    HAVE_MPL = False
    MPL_BACKEND = None

ANIM = None


POSTURE_LABELS = {
    POSTURE_ATTACK: "进攻",
    POSTURE_DEFENSE: "防御",
    POSTURE_MOVE: "移动",
}


@dataclass
class SentryParams:
    max_hp: float = 400.0
    heat_limit: float = 260.0
    base_cooling: float = 30.0
    buffer_max: float = 60.0
    power_limit: float = 100.0
    switch_cd: float = 5.0
    degrade_limit: float = 180.0
    heat_per_shot: float = 10.0
    heat_per_sec_lock: float = 100.0
    enemy_max_hp: float = 400.0


@dataclass
class SimConfig:
    dt: float = 0.1
    total_time: float = 0.0
    seed: int = 0
    field_w: float = 28.0
    field_h: float = 15.0
    sensor_range: float = 18.0
    lock_range: float = 8.0
    enemy_speed: float = 1.6
    enemy_turn_period: float = 8.0
    sentry_speed: float = 2.2
    waypoint_radius: float = 0.7
    history_len: int = 2000


@dataclass
class BuffState:
    defense: float = 0.0
    vulnerability: float = 0.0
    cooling_bonus: float = 0.0
    heal_rate: float = 0.0


@dataclass
class SentrySimState:
    x: float
    y: float
    posture: int = POSTURE_MOVE
    hp: float = 400.0
    heat: float = 0.0
    buffer_energy: float = 60.0
    time_in_posture: float = 0.0
    switch_cd: float = 0.0
    weapon_lock: str = "none"
    chassis_disabled: float = 0.0
    time_since_combat: float = 0.0


@dataclass
class EnemyState:
    x: float
    y: float
    vx: float
    vy: float
    hp: float


@dataclass
class Zone:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str


def posture_effects(posture: int, degraded: bool) -> Tuple[float, float, float, float]:
    if posture == POSTURE_ATTACK:
        cooling_mult = 2.0 if degraded else 3.0
        defense = 0.0
        vulnerability = 0.25
        power_mult = 0.5
    elif posture == POSTURE_DEFENSE:
        cooling_mult = 1.0 / 3.0
        defense = 0.25 if degraded else 0.5
        vulnerability = 0.0
        power_mult = 0.5
    else:
        cooling_mult = 1.0 / 3.0
        defense = 0.0
        vulnerability = 0.25
        power_mult = 1.2 if degraded else 1.5
    return cooling_mult, defense, vulnerability, power_mult


def enemy_dps(distance: float, visible: bool) -> float:
    if not visible:
        return 0.0
    if distance < 3.0:
        return 200.0
    if distance < 8.0:
        return 50.0
    return 10.0


def heat_demand(visible: bool, aim_locked: bool) -> float:
    if aim_locked:
        return 300.0
    if visible:
        return 50.0
    return 0.0


def enemy_hit_rate(distance: float, visible: bool, aim_locked: bool) -> float:
    if not visible:
        return 0.0
    if aim_locked:
        return 0.35
    if distance < 12.0:
        return 0.15
    return 0.05


def pick_waypoint(rng: np.random.Generator, cfg: SimConfig) -> Tuple[float, float]:
    return rng.uniform(1.0, cfg.field_w - 1.0), rng.uniform(1.0, cfg.field_h - 1.0)


class SentrySimulator:
    def __init__(
        self,
        sim_cfg: SimConfig,
        params: SentryParams,
        auto_reset_on_death: bool = True,
        auto_respawn_enemy: bool = True,
    ) -> None:
        self.cfg = sim_cfg
        self.params = params
        self.auto_reset_on_death = auto_reset_on_death
        self.auto_respawn_enemy = auto_respawn_enemy
        self.rng = np.random.default_rng(sim_cfg.seed)
        self.zones = self._build_zones()
        self.reset()

    def _build_zones(self) -> List[Zone]:
        return [
            Zone("supply", 1.0, 1.0, 4.5, 4.0, "supply"),
            Zone("high_ground", 11.0, 5.5, 17.0, 9.5, "high_ground"),
        ]

    def reset(self) -> None:
        self.time = 0.0
        self.enemy_respawns = 0
        self.sentry_respawns = 0
        self.posture_time_attack = 0.0
        self.posture_time_defense = 0.0
        self.posture_time_move = 0.0
        self.state = SentrySimState(
            x=2.0,
            y=self.cfg.field_h * 0.5,
            posture=POSTURE_MOVE,
            hp=self.params.max_hp,
            heat=0.0,
            buffer_energy=self.params.buffer_max,
        )
        ex, ey = self.cfg.field_w - 4.0, self.cfg.field_h * 0.5
        angle = self.rng.uniform(0, 2 * math.pi)
        self.enemy_turn_time = self.rng.uniform(2.0, self.cfg.enemy_turn_period)
        self.enemy = EnemyState(
            x=ex,
            y=ey,
            vx=self.cfg.enemy_speed * math.cos(angle),
            vy=self.cfg.enemy_speed * math.sin(angle),
            hp=self.params.enemy_max_hp,
        )
        self.target = pick_waypoint(self.rng, self.cfg)
        self.logs: Dict[str, List[float]] = {k: [] for k in self._log_keys()}
        self.reason_log: List[str] = []

    def _log_keys(self) -> List[str]:
        return [
            "time",
            "x",
            "y",
            "enemy_x",
            "enemy_y",
            "enemy_hp",
            "target_x",
            "target_y",
            "hp",
            "heat",
            "buffer",
            "posture",
            "time_attack",
            "time_defense",
            "time_move",
            "time_in_posture",
            "switch_cd",
            "weapon_lock",
            "chassis_disabled",
            "enemy_distance",
            "target_distance",
            "enemy_visible",
            "aim_locked",
            "incoming_dps",
            "damage",
            "heat_demand",
            "actual_fire_heat",
            "cooling_rate",
            "power_demand",
            "power_limit",
            "out_of_combat",
            "requested_action",
            "cooldown_blocked",
            "illegal_switch",
            "action",
            "cost",
        ]

    def _in_zone(self, x: float, y: float, zone: Zone) -> bool:
        return zone.x0 <= x <= zone.x1 and zone.y0 <= y <= zone.y1

    def _collect_buffs(self) -> BuffState:
        buff = BuffState()
        in_supply = any(
            z.kind == "supply" and self._in_zone(self.state.x, self.state.y, z)
            for z in self.zones
        )
        in_high = any(
            z.kind == "high_ground" and self._in_zone(self.state.x, self.state.y, z)
            for z in self.zones
        )
        if in_high:
            buff.defense = max(buff.defense, 0.25)
        if in_supply:
            if self.time >= 240.0 and self.state.time_since_combat >= 6.0:
                buff.heal_rate = max(buff.heal_rate, 0.25 * self.params.max_hp)
            else:
                buff.heal_rate = max(buff.heal_rate, 0.10 * self.params.max_hp)
        return buff

    def _update_enemy(self) -> None:
        if self.time >= self.enemy_turn_time:
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.cfg.enemy_speed * self.rng.uniform(0.8, 1.2)
            self.enemy.vx = speed * math.cos(angle)
            self.enemy.vy = speed * math.sin(angle)
            self.enemy_turn_time = self.time + self.cfg.enemy_turn_period * self.rng.uniform(0.7, 1.3)
        self.enemy.x += self.enemy.vx * self.cfg.dt
        self.enemy.y += self.enemy.vy * self.cfg.dt
        if self.enemy.x <= 0.5 or self.enemy.x >= self.cfg.field_w - 0.5:
            self.enemy.vx *= -1.0
        if self.enemy.y <= 0.5 or self.enemy.y >= self.cfg.field_h - 0.5:
            self.enemy.vy *= -1.0

    def _respawn_enemy(self) -> None:
        ex, ey = pick_waypoint(self.rng, self.cfg)
        angle = self.rng.uniform(0, 2 * math.pi)
        speed = self.cfg.enemy_speed * self.rng.uniform(0.8, 1.2)
        self.enemy = EnemyState(
            x=ex,
            y=ey,
            vx=speed * math.cos(angle),
            vy=speed * math.sin(angle),
            hp=self.params.enemy_max_hp,
        )
        self.enemy_respawns += 1


    def _update_target(self) -> None:
        dx = self.target[0] - self.state.x
        dy = self.target[1] - self.state.y
        if math.hypot(dx, dy) <= self.cfg.waypoint_radius:
            self.target = pick_waypoint(self.rng, self.cfg)

    def _move_sentry(self, speed: float) -> None:
        dx = self.target[0] - self.state.x
        dy = self.target[1] - self.state.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        ux, uy = dx / dist, dy / dist
        self.state.x = np.clip(
            self.state.x + ux * speed * self.cfg.dt,
            0.0,
            self.cfg.field_w,
        )
        self.state.y = np.clip(
            self.state.y + uy * speed * self.cfg.dt,
            0.0,
            self.cfg.field_h,
        )

    def _explain_action(self, action: int, obs: SentryObservation) -> str:
        reasons = []
        if self.state.switch_cd > 0:
            reasons.append("切换冷却中，保持当前姿态")

        incoming = enemy_dps(obs.enemy_distance, obs.enemy_visible)
        if action == POSTURE_ATTACK:
            if obs.is_aim_locked:
                if self.state.heat > 0.8 * self.params.heat_limit:
                    reasons.append("热量偏高，选择进攻姿态散热并输出")
                else:
                    reasons.append("目标锁定，优先输出")
            elif obs.enemy_visible:
                reasons.append("敌人可见，保持压制输出")
        elif action == POSTURE_DEFENSE:
            if self.state.hp < 0.3 * self.params.max_hp:
                reasons.append("血量偏低，优先防御")
            if incoming > 60.0 or (obs.enemy_visible and obs.enemy_distance < 8.0):
                reasons.append("威胁较高，降低受伤")
            if not reasons:
                reasons.append("保守生存策略")
        elif action == POSTURE_MOVE:
            if obs.target_distance > 20.0:
                reasons.append("目标距离较远，机动赶路")
            if obs.enemy_visible and not obs.is_aim_locked:
                reasons.append("敌方可见但未锁定，拉扯位移")

        if self.state.time_in_posture > self.params.degrade_limit:
            reasons.append("姿态弱化风险")
        return "；".join(reasons) if reasons else "综合最优"

    def step(self, mpc: SentryMPC) -> Dict[str, float]:
        self._update_enemy()
        self._update_target()

        dx = self.enemy.x - self.state.x
        dy = self.enemy.y - self.state.y
        enemy_distance = math.hypot(dx, dy)
        enemy_visible = enemy_distance <= self.cfg.sensor_range
        aim_locked = enemy_visible and enemy_distance <= self.cfg.lock_range
        target_distance = math.hypot(
            self.target[0] - self.state.x,
            self.target[1] - self.state.y,
        )

        obs = SentryObservation(
            enemy_visible=enemy_visible,
            enemy_distance=enemy_distance,
            is_aim_locked=aim_locked,
            target_distance=target_distance,
            force_posture=0,
        )
        mpc_state = SentryState(
            self.state.posture,
            self.state.hp,
            self.state.heat,
            self.state.buffer_energy,
            self.state.time_in_posture,
            self.state.switch_cd,
        )

        pre_cd = self.state.switch_cd
        pre_posture = self.state.posture
        requested_action, cost = mpc.solve(mpc_state, obs)
        cooldown_blocked = 1.0 if pre_cd > 0.0 and requested_action != pre_posture else 0.0
        action = requested_action
        if pre_cd > 0.0:
            action = pre_posture
        illegal_switch = 1.0 if pre_cd > 0.0 and action != pre_posture else 0.0

        if action != self.state.posture and self.state.switch_cd <= 0.0:
            self.state.posture = action
            self.state.time_in_posture = 0.0
            self.state.switch_cd = self.params.switch_cd
        else:
            self.state.time_in_posture += self.cfg.dt
            self.state.switch_cd = max(0.0, self.state.switch_cd - self.cfg.dt)

        degraded = self.state.time_in_posture > self.params.degrade_limit
        cooling_mult, posture_def, posture_vuln, power_mult = posture_effects(
            self.state.posture,
            degraded,
        )

        buff = self._collect_buffs()
        defense = max(buff.defense, posture_def)
        vulnerability = max(buff.vulnerability, posture_vuln)
        damage_mult = max(0.0, 1.0 - defense + vulnerability)

        incoming = enemy_dps(enemy_distance, enemy_visible)
        damage = incoming * damage_mult * self.cfg.dt
        self.state.hp = max(0.0, self.state.hp - damage)

        if damage > 0.0:
            self.state.time_since_combat = 0.0
        else:
            self.state.time_since_combat += self.cfg.dt

        if buff.heal_rate > 0.0:
            self.state.hp = min(
                self.params.max_hp,
                self.state.hp + buff.heal_rate * self.cfg.dt,
            )

        if self.state.hp <= 0.0:
            if self.auto_reset_on_death:
                self.reset()
                return {
                    "requested_action": -1.0,
                    "action": -1.0,
                    "cooldown_blocked": 0.0,
                    "illegal_switch": 0.0,
                    "pre_cd": float(pre_cd),
                }
            self.state.hp = 0.0

        demand = heat_demand(enemy_visible, aim_locked)
        cooling_rate = (self.params.base_cooling + buff.cooling_bonus) * cooling_mult
        max_fire_possible = self.params.heat_limit - (
            self.state.heat - cooling_rate * self.cfg.dt
        )
        actual_fire_heat = min(demand * self.cfg.dt, max_fire_possible)
        actual_fire_heat = max(0.0, actual_fire_heat)

        if self.state.weapon_lock != "none":
            actual_fire_heat = 0.0
        self.state.heat = max(
            0.0,
            self.state.heat + actual_fire_heat - cooling_rate * self.cfg.dt,
        )

        if self.state.weapon_lock != "perma":
            if self.state.heat > self.params.heat_limit + self.params.heat_per_sec_lock:
                self.state.weapon_lock = "perma"
            elif self.state.heat > self.params.heat_limit:
                self.state.weapon_lock = "overheat"
            elif self.state.weapon_lock == "overheat" and self.state.heat <= 0.0:
                self.state.weapon_lock = "none"

        if actual_fire_heat > 0.0:
            self.state.time_since_combat = 0.0

        if actual_fire_heat > 0.0:
            shots = actual_fire_heat / self.params.heat_per_shot
            hit_rate = enemy_hit_rate(enemy_distance, enemy_visible, aim_locked)
            damage_per_shot = 20.0
            enemy_damage = shots * damage_per_shot * hit_rate
            self.enemy.hp = max(0.0, self.enemy.hp - enemy_damage)

        mobility = np.clip(target_distance / 20.0, 0.0, 1.0)
        power_limit = self.params.power_limit * max(1.0, power_mult)
        power_demand = 80.0 + 60.0 * mobility
        if self.state.chassis_disabled > 0.0:
            power_demand = 0.0
            self.state.chassis_disabled = max(0.0, self.state.chassis_disabled - self.cfg.dt)
        else:
            delta = (power_demand - power_limit) * self.cfg.dt
            self.state.buffer_energy = np.clip(
                self.state.buffer_energy - delta,
                0.0,
                self.params.buffer_max,
            )
            if self.state.buffer_energy <= 0.0 and power_demand > power_limit:
                self.state.chassis_disabled = 5.0
                self.state.buffer_energy = 0.0

        speed = 0.0
        if self.state.chassis_disabled <= 0.0:
            speed = self.cfg.sentry_speed * (0.7 + 0.3 * mobility) * max(0.6, power_mult)
        self._move_sentry(speed)

        out_of_combat = self.state.time_since_combat >= 6.0
        reason = self._explain_action(action, obs)

        if self.state.posture == POSTURE_ATTACK:
            self.posture_time_attack += self.cfg.dt
        elif self.state.posture == POSTURE_DEFENSE:
            self.posture_time_defense += self.cfg.dt
        else:
            self.posture_time_move += self.cfg.dt

        self._log(
            time=self.time,
            x=self.state.x,
            y=self.state.y,
            enemy_x=self.enemy.x,
            enemy_y=self.enemy.y,
            enemy_hp=self.enemy.hp,
            target_x=self.target[0],
            target_y=self.target[1],
            hp=self.state.hp,
            heat=self.state.heat,
            buffer=self.state.buffer_energy,
            posture=self.state.posture,
            time_attack=self.posture_time_attack,
            time_defense=self.posture_time_defense,
            time_move=self.posture_time_move,
            time_in_posture=self.state.time_in_posture,
            switch_cd=self.state.switch_cd,
            weapon_lock=0.0 if self.state.weapon_lock == "none" else 1.0,
            chassis_disabled=self.state.chassis_disabled,
            enemy_distance=enemy_distance,
            target_distance=target_distance,
            enemy_visible=1.0 if enemy_visible else 0.0,
            aim_locked=1.0 if aim_locked else 0.0,
            incoming_dps=incoming,
            damage=damage,
            heat_demand=demand,
            actual_fire_heat=actual_fire_heat,
            cooling_rate=cooling_rate,
            power_demand=power_demand,
            power_limit=power_limit,
            out_of_combat=1.0 if out_of_combat else 0.0,
            requested_action=requested_action,
            cooldown_blocked=cooldown_blocked,
            illegal_switch=illegal_switch,
            action=action,
            cost=cost,
        )
        self.reason_log.append(reason)

        if self.cfg.history_len > 0 and len(self.logs["time"]) > self.cfg.history_len:
            for key in self.logs:
                self.logs[key].pop(0)
            self.reason_log.pop(0)

        self.time += self.cfg.dt
        self.latest_reason = reason

        if self.enemy.hp <= 0.0 and self.auto_respawn_enemy:
            self._respawn_enemy()

        return {
            "requested_action": float(requested_action),
            "action": float(action),
            "cooldown_blocked": cooldown_blocked,
            "illegal_switch": illegal_switch,
            "pre_cd": float(pre_cd),
        }

    def _log(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.logs[key].append(float(value))


def run_sim(
    sim_cfg: SimConfig,
    params: SentryParams,
    total_time: float,
) -> Tuple[Dict[str, List[float]], List[str]]:
    mpc_cfg = MPCConfig(
        dt=sim_cfg.dt,
        horizon=10,
        max_hp=params.max_hp,
        max_heat=params.heat_limit,
        base_cooling=params.base_cooling,
        max_buffer=params.buffer_max,
        switch_cd_limit=params.switch_cd,
        degrade_time_limit=params.degrade_limit,
    )
    mpc = SentryMPC(mpc_cfg)
    sim = SentrySimulator(sim_cfg, params)

    steps = int(total_time / sim_cfg.dt)
    for _ in range(steps):
        sim.step(mpc)
    return sim.logs, sim.reason_log


def summarize(logs: Dict[str, List[float]]) -> str:
    if not logs["time"]:
        return "no data"
    hp_end = logs["hp"][-1]
    time_end = logs["time"][-1]
    dt = 0.0
    if len(logs["time"]) > 1:
        dt = logs["time"][1] - logs["time"][0]
    locked_time = sum(1 for v in logs["weapon_lock"] if v > 0.5) * dt
    power_cut = sum(1 for v in logs["chassis_disabled"] if v > 0.0) * dt
    switches = 0
    for i in range(1, len(logs["posture"])):
        if logs["posture"][i] != logs["posture"][i - 1]:
            switches += 1
    return (
        f"time={time_end:.1f}s, hp_end={hp_end:.1f}, locked_time={locked_time:.1f}s, "
        f"power_cut={power_cut:.1f}s, posture_switches={switches}"
    )


def run_duel_round(
    sim_cfg: SimConfig,
    params: SentryParams,
    seed: int,
) -> Dict[str, float]:
    cfg = SimConfig(
        dt=sim_cfg.dt,
        total_time=0.0,
        seed=seed,
        field_w=sim_cfg.field_w,
        field_h=sim_cfg.field_h,
        sensor_range=sim_cfg.sensor_range,
        lock_range=sim_cfg.lock_range,
        enemy_speed=sim_cfg.enemy_speed,
        enemy_turn_period=sim_cfg.enemy_turn_period,
        sentry_speed=sim_cfg.sentry_speed,
        waypoint_radius=sim_cfg.waypoint_radius,
        history_len=0,
    )
    mpc_cfg = MPCConfig(
        dt=cfg.dt,
        horizon=10,
        max_hp=params.max_hp,
        max_heat=params.heat_limit,
        base_cooling=params.base_cooling,
        max_buffer=params.buffer_max,
        switch_cd_limit=params.switch_cd,
        degrade_time_limit=params.degrade_limit,
    )
    mpc = SentryMPC(mpc_cfg)
    sim = SentrySimulator(
        cfg,
        params,
        auto_reset_on_death=False,
        auto_respawn_enemy=False,
    )

    prev_posture = sim.state.posture
    switch_count = 0
    lock_time = 0.0
    power_cut_time = 0.0
    heat_sum = 0.0
    buffer_sum = 0.0
    steps = 0
    cooldown_blocked_count = 0
    illegal_switch_count = 0

    while True:
        info = sim.step(mpc)
        steps += 1
        heat_sum += sim.state.heat
        buffer_sum += sim.state.buffer_energy
        cooldown_blocked_count += int(info.get("cooldown_blocked", 0.0) > 0.0)
        illegal_switch_count += int(info.get("illegal_switch", 0.0) > 0.0)
        if sim.state.weapon_lock != "none":
            lock_time += cfg.dt
        if sim.state.chassis_disabled > 0.0:
            power_cut_time += cfg.dt
        if sim.state.posture != prev_posture:
            switch_count += 1
            prev_posture = sim.state.posture

        if sim.enemy.hp <= 0.0 or sim.state.hp <= 0.0:
            break

    duration = sim.time
    outcome = 1.0 if sim.enemy.hp <= 0.0 else 0.0
    avg_heat = heat_sum / max(1, steps)
    avg_buffer = buffer_sum / max(1, steps)
    attack_time = sim.posture_time_attack
    defense_time = sim.posture_time_defense
    move_time = sim.posture_time_move

    return {
        "seed": float(seed),
        "duration": duration,
        "kill": outcome,
        "death": 1.0 - outcome,
        "switches": float(switch_count),
        "attack_time": attack_time,
        "defense_time": defense_time,
        "move_time": move_time,
        "lock_time": lock_time,
        "power_cut_time": power_cut_time,
        "avg_heat": avg_heat,
        "avg_buffer": avg_buffer,
        "cooldown_blocked_count": float(cooldown_blocked_count),
        "illegal_switch_count": float(illegal_switch_count),
    }


def summarize_duels(results: List[Dict[str, float]]) -> str:
    if not results:
        return "no duel results"
    kills = sum(r["kill"] for r in results)
    deaths = sum(r["death"] for r in results)
    kd = kills / deaths if deaths > 0 else float("inf")
    durations = [r["duration"] for r in results]
    kill_times = [r["duration"] for r in results if r["kill"] > 0.5]
    death_times = [r["duration"] for r in results if r["death"] > 0.5]
    switches = [r["switches"] for r in results]
    attack_ratio = [r["attack_time"] / max(r["duration"], 1e-6) for r in results]
    defense_ratio = [r["defense_time"] / max(r["duration"], 1e-6) for r in results]
    move_ratio = [r["move_time"] / max(r["duration"], 1e-6) for r in results]
    lock_times = [r["lock_time"] for r in results]
    power_cut_times = [r["power_cut_time"] for r in results]
    avg_heat = [r["avg_heat"] for r in results]
    cooldown_blocked = [r.get("cooldown_blocked_count", 0.0) for r in results]
    illegal_switch = [r.get("illegal_switch_count", 0.0) for r in results]

    def mean(values: List[float]) -> float:
        return sum(values) / max(1, len(values))

    return "\n".join(
        [
            f"rounds={len(results)}, kills={int(kills)}, deaths={int(deaths)}, kd={kd:.2f}",
            f"avg_duration={mean(durations):.1f}s, avg_kill_time={mean(kill_times):.1f}s, avg_death_time={mean(death_times):.1f}s",
            f"avg_switches={mean(switches):.2f}, switches_per_min={mean(switches) / max(mean(durations), 1e-6) * 60.0:.2f}",
            f"avg_posture_ratio=进攻{mean(attack_ratio):.2f}, 防御{mean(defense_ratio):.2f}, 移动{mean(move_ratio):.2f}",
            f"avg_lock_time={mean(lock_times):.1f}s, avg_power_cut_time={mean(power_cut_times):.1f}s, avg_heat={mean(avg_heat):.1f}",
            f"cooldown_blocked_total={int(sum(cooldown_blocked))}, illegal_switch_total={int(sum(illegal_switch))}",
        ]
    )


def write_duel_csv(path: str, results: List[Dict[str, float]]) -> None:
    import csv

    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in results:
            writer.writerow([row[k] for k in keys])


def plot_duel_results(path: str, results: List[Dict[str, float]]) -> None:
    if not HAVE_MPL:
        print("matplotlib not available; skip plotting")
        return
    if not results:
        return
    assert plt is not None

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    durations = np.array([r["duration"] for r in results])
    kills = np.array([r["kill"] for r in results])
    switches = np.array([r["switches"] for r in results])
    lock_times = np.array([r["lock_time"] for r in results])
    power_cut_times = np.array([r["power_cut_time"] for r in results])
    avg_heat = np.array([r["avg_heat"] for r in results])
    attack_ratio = np.array([r["attack_time"] / max(r["duration"], 1e-6) for r in results])
    defense_ratio = np.array([r["defense_time"] / max(r["duration"], 1e-6) for r in results])
    move_ratio = np.array([r["move_time"] / max(r["duration"], 1e-6) for r in results])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("姿态系统对抗统计（1000轮）")

    ax = axes[0, 0]
    kill_count = int(kills.sum())
    death_count = len(results) - kill_count
    ax.bar(["击杀", "被击杀"], [kill_count, death_count], color=["#4caf50", "#f44336"])
    ax.set_title("胜负统计")
    ax.set_ylabel("次数")

    ax = axes[0, 1]
    ax.hist(durations, bins=20, color="#2196f3", alpha=0.7)
    ax.set_title("轮次时长分布")
    ax.set_xlabel("秒")
    ax.set_ylabel("次数")

    ax = axes[1, 0]
    ax.hist(switches, bins=15, color="#9c27b0", alpha=0.7)
    ax.set_title("切姿态次数分布")
    ax.set_xlabel("次数")
    ax.set_ylabel("轮次")

    ax = axes[1, 1]
    ax.boxplot(
        [attack_ratio, defense_ratio, move_ratio],
        tick_labels=["进攻占比", "防御占比", "移动占比"],
    )
    ax.set_title("姿态占比分布")

    ax = axes[2, 0]
    ax.hist(avg_heat, bins=20, color="#ff9800", alpha=0.7)
    ax.set_title("平均热量分布")
    ax.set_xlabel("热量")
    ax.set_ylabel("轮次")

    ax = axes[2, 1]
    ax.hist(power_cut_times, bins=20, color="#607d8b", alpha=0.7, label="断电时长")
    ax.hist(lock_times, bins=20, color="#795548", alpha=0.6, label="锁枪时长")
    ax.set_title("断电/锁枪时长分布")
    ax.set_xlabel("秒")
    ax.set_ylabel("轮次")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(path: str, logs: Dict[str, List[float]], reasons: List[str]) -> None:
    import csv

    keys = list(logs.keys())
    rows = zip(*(logs[k] for k in keys), reasons)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys + ["reason"])
        writer.writerows(rows)


def visualize(logs: Dict[str, List[float]], reasons: List[str], cfg: SimConfig) -> None:
    if not HAVE_MPL:
        print("matplotlib not available; run with --no-viz or install matplotlib")
        print(f"python executable: {sys.executable}")
        print(f"python version: {sys.version.split()[0]}")
        print(f"install: \"{sys.executable}\" -m pip install matplotlib")
        return
    assert plt is not None
    assert FuncAnimation is not None
    assert Rectangle is not None

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    times = np.array(logs["time"])
    x = np.array(logs["x"])
    y = np.array(logs["y"])
    ex = np.array(logs["enemy_x"])
    ey = np.array(logs["enemy_y"])
    tx = np.array(logs["target_x"])
    ty = np.array(logs["target_y"])
    hp = np.array(logs["hp"])
    enemy_hp = np.array(logs["enemy_hp"])
    heat = np.array(logs["heat"])
    buffer = np.array(logs["buffer"])
    posture = np.array(logs["posture"])
    power_demand = np.array(logs["power_demand"])
    power_limit = np.array(logs["power_limit"])
    time_attack = np.array(logs["time_attack"])
    time_defense = np.array(logs["time_defense"])
    time_move = np.array(logs["time_move"])

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_mid = fig.add_subplot(gs[1, 1])
    ax_bottom = fig.add_subplot(gs[2, 1])

    ax_map.set_xlim(0, cfg.field_w)
    ax_map.set_ylim(0, cfg.field_h)
    ax_map.set_aspect("equal")
    ax_map.set_title("战场")
    ax_map.plot([0, cfg.field_w, cfg.field_w, 0, 0], [0, 0, cfg.field_h, cfg.field_h, 0], "k-")

    supply = Rectangle((1.0, 1.0), 3.5, 3.0, color="#d8f2d8", alpha=0.5)
    high = Rectangle((11.0, 5.5), 6.0, 4.0, color="#d8e8f8", alpha=0.5)
    ax_map.add_patch(supply)
    ax_map.add_patch(high)

    sentry_dot, = ax_map.plot([], [], "bo", label="哨兵")
    enemy_dot, = ax_map.plot([], [], "ro", label="敌方")
    target_dot, = ax_map.plot([], [], "gx", label="目标")
    ax_map.legend(loc="upper right")

    hp_line, = ax_top.plot([], [], label="己方血量")
    enemy_hp_line, = ax_top.plot([], [], label="敌方血量")
    heat_line, = ax_top.plot([], [], label="热量")
    ax_top.set_xlim(times[0], times[-1])
    ax_top.set_ylim(0, max(hp.max(), heat.max(), enemy_hp.max()) * 1.1)
    ax_top.legend(loc="upper right")
    ax_top.set_title("血量 / 热量")

    buffer_line, = ax_mid.plot([], [], label="缓冲能量")
    power_d_line, = ax_mid.plot([], [], label="功率需求")
    power_l_line, = ax_mid.plot([], [], label="功率上限")
    ax_mid.set_xlim(times[0], times[-1])
    ax_mid.set_ylim(0, max(power_demand.max(), power_limit.max(), buffer.max()) * 1.1)
    ax_mid.legend(loc="upper right")
    ax_mid.set_title("缓冲能量 / 功率")

    attack_line, = ax_bottom.plot([], [], label="进攻累计时长")
    defense_line, = ax_bottom.plot([], [], label="防御累计时长")
    move_line, = ax_bottom.plot([], [], label="移动累计时长")
    ax_bottom.set_xlim(times[0], times[-1])
    ax_bottom.set_ylim(0, max(time_attack.max(), time_defense.max(), time_move.max()) * 1.1 + 1.0)
    ax_bottom.legend(loc="upper right")
    ax_bottom.set_title("姿态累计时长")

    reason_text = ax_map.text(
        0.02,
        0.98,
        "",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    stride = max(1, int(0.2 / cfg.dt))

    def update(i: int) -> List:
        idx = min(i * stride, len(times) - 1)
        sentry_dot.set_data([x[idx]], [y[idx]])
        enemy_dot.set_data([ex[idx]], [ey[idx]])
        target_dot.set_data([tx[idx]], [ty[idx]])
        hp_line.set_data(times[: idx + 1], hp[: idx + 1])
        enemy_hp_line.set_data(times[: idx + 1], enemy_hp[: idx + 1])
        heat_line.set_data(times[: idx + 1], heat[: idx + 1])
        buffer_line.set_data(times[: idx + 1], buffer[: idx + 1])
        power_d_line.set_data(times[: idx + 1], power_demand[: idx + 1])
        power_l_line.set_data(times[: idx + 1], power_limit[: idx + 1])
        attack_line.set_data(times[: idx + 1], time_attack[: idx + 1])
        defense_line.set_data(times[: idx + 1], time_defense[: idx + 1])
        move_line.set_data(times[: idx + 1], time_move[: idx + 1])
        posture_label = POSTURE_LABELS.get(int(posture[idx]), "未知")
        reason = reasons[idx] if idx < len(reasons) else ""
        reason_text.set_text(
            f"姿态: {posture_label}\n原因: {reason}"
        )
        return [
            sentry_dot,
            enemy_dot,
            target_dot,
            hp_line,
            enemy_hp_line,
            heat_line,
            buffer_line,
            power_d_line,
            power_l_line,
            attack_line,
            defense_line,
            move_line,
            reason_text,
        ]

    frames = int(math.ceil(len(times) / stride))
    global ANIM
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    ANIM = anim
    plt.tight_layout()
    plt.show()


def visualize_live(sim_cfg: SimConfig, params: SentryParams) -> None:
    if not HAVE_MPL:
        print("matplotlib not available; run with --no-viz or install matplotlib")
        print(f"python executable: {sys.executable}")
        print(f"python version: {sys.version.split()[0]}")
        print(f"install: \"{sys.executable}\" -m pip install matplotlib")
        return

    assert plt is not None
    assert FuncAnimation is not None
    assert Rectangle is not None

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    mpc_cfg = MPCConfig(
        dt=sim_cfg.dt,
        horizon=10,
        max_hp=params.max_hp,
        max_heat=params.heat_limit,
        base_cooling=params.base_cooling,
        max_buffer=params.buffer_max,
        switch_cd_limit=params.switch_cd,
        degrade_time_limit=params.degrade_limit,
    )
    mpc = SentryMPC(mpc_cfg)
    sim = SentrySimulator(sim_cfg, params)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_mid = fig.add_subplot(gs[1, 1])
    ax_bottom = fig.add_subplot(gs[2, 1])

    ax_map.set_xlim(0, sim_cfg.field_w)
    ax_map.set_ylim(0, sim_cfg.field_h)
    ax_map.set_aspect("equal")
    ax_map.set_title("战场")
    ax_map.plot(
        [0, sim_cfg.field_w, sim_cfg.field_w, 0, 0],
        [0, 0, sim_cfg.field_h, sim_cfg.field_h, 0],
        "k-",
    )

    supply = Rectangle((1.0, 1.0), 3.5, 3.0, color="#d8f2d8", alpha=0.5)
    high = Rectangle((11.0, 5.5), 6.0, 4.0, color="#d8e8f8", alpha=0.5)
    ax_map.add_patch(supply)
    ax_map.add_patch(high)

    sentry_dot, = ax_map.plot([], [], "bo", label="哨兵")
    enemy_dot, = ax_map.plot([], [], "ro", label="敌方")
    target_dot, = ax_map.plot([], [], "gx", label="目标")
    ax_map.legend(loc="upper right")

    hp_line, = ax_top.plot([], [], label="己方血量")
    enemy_hp_line, = ax_top.plot([], [], label="敌方血量")
    heat_line, = ax_top.plot([], [], label="热量")
    ax_top.set_title("血量 / 热量")
    ax_top.legend(loc="upper right")

    buffer_line, = ax_mid.plot([], [], label="缓冲能量")
    power_d_line, = ax_mid.plot([], [], label="功率需求")
    power_l_line, = ax_mid.plot([], [], label="功率上限")
    ax_mid.set_title("缓冲能量 / 功率")
    ax_mid.legend(loc="upper right")

    attack_line, = ax_bottom.plot([], [], label="进攻累计时长")
    defense_line, = ax_bottom.plot([], [], label="防御累计时长")
    move_line, = ax_bottom.plot([], [], label="移动累计时长")
    ax_bottom.set_title("姿态累计时长")
    ax_bottom.legend(loc="upper right")

    reason_text = ax_map.text(
        0.02,
        0.98,
        "",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    def update(_):
        sim.step(mpc)
        times = np.array(sim.logs["time"])
        x = np.array(sim.logs["x"])
        y = np.array(sim.logs["y"])
        ex = np.array(sim.logs["enemy_x"])
        ey = np.array(sim.logs["enemy_y"])
        tx = np.array(sim.logs["target_x"])
        ty = np.array(sim.logs["target_y"])
        hp = np.array(sim.logs["hp"])
        enemy_hp = np.array(sim.logs["enemy_hp"])
        heat = np.array(sim.logs["heat"])
        buffer = np.array(sim.logs["buffer"])
        power_demand = np.array(sim.logs["power_demand"])
        power_limit = np.array(sim.logs["power_limit"])
        posture = np.array(sim.logs["posture"])
        time_attack = np.array(sim.logs["time_attack"])
        time_defense = np.array(sim.logs["time_defense"])
        time_move = np.array(sim.logs["time_move"])

        if times.size == 0:
            return []

        sentry_dot.set_data([x[-1]], [y[-1]])
        enemy_dot.set_data([ex[-1]], [ey[-1]])
        target_dot.set_data([tx[-1]], [ty[-1]])

        ax_top.set_xlim(times[0], times[-1] + 1e-6)
        ax_top.set_ylim(0, max(hp.max(), enemy_hp.max(), heat.max()) * 1.1 + 1.0)
        hp_line.set_data(times, hp)
        enemy_hp_line.set_data(times, enemy_hp)
        heat_line.set_data(times, heat)

        ax_mid.set_xlim(times[0], times[-1] + 1e-6)
        ax_mid.set_ylim(
            0,
            max(buffer.max(), power_demand.max(), power_limit.max()) * 1.1 + 1.0,
        )
        buffer_line.set_data(times, buffer)
        power_d_line.set_data(times, power_demand)
        power_l_line.set_data(times, power_limit)

        ax_bottom.set_xlim(times[0], times[-1] + 1e-6)
        ax_bottom.set_ylim(
            0,
            max(time_attack.max(), time_defense.max(), time_move.max()) * 1.1 + 1.0,
        )
        attack_line.set_data(times, time_attack)
        defense_line.set_data(times, time_defense)
        move_line.set_data(times, time_move)

        posture_label = POSTURE_LABELS.get(int(posture[-1]), "未知")
        reason = sim.reason_log[-1] if sim.reason_log else ""
        reason_text.set_text(f"姿态: {posture_label}\n原因: {reason}")
        return [
            sentry_dot,
            enemy_dot,
            target_dot,
            hp_line,
            enemy_hp_line,
            heat_line,
            buffer_line,
            power_d_line,
            power_l_line,
            attack_line,
            defense_line,
            move_line,
            reason_text,
        ]

    global ANIM
    ANIM = FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    plt.show()


def main() -> int:
    stdout_reconfig = getattr(sys.stdout, "reconfigure", None)
    stderr_reconfig = getattr(sys.stderr, "reconfigure", None)
    if stdout_reconfig is not None:
        stdout_reconfig(encoding="utf-8")
    if stderr_reconfig is not None:
        stderr_reconfig(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Sentry posture simulator")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--duel", type=int, default=0)
    parser.add_argument("--duel-csv", type=str, default="")
    parser.add_argument("--duel-plot", type=str, default="")
    parser.add_argument("--duel-quiet", action="store_true")
    args = parser.parse_args()

    if args.seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")
    else:
        seed = args.seed

    sim_cfg = SimConfig(dt=args.dt, total_time=args.time, seed=seed)
    params = SentryParams()
    print(f"seed={seed}")

    if args.duel > 0:
        results = []
        for i in range(args.duel):
            round_seed = seed + i
            result = run_duel_round(sim_cfg, params, round_seed)
            results.append(result)
            if not args.duel_quiet:
                outcome = "击杀" if result["kill"] > 0.5 else "被击杀"
                print(
                    f"round {i + 1:03d}: {outcome}, time={result['duration']:.1f}s, "
                    f"switches={int(result['switches'])}, "
                    f"attack={result['attack_time'] / max(result['duration'], 1e-6):.2f}, "
                    f"defense={result['defense_time'] / max(result['duration'], 1e-6):.2f}, "
                    f"move={result['move_time'] / max(result['duration'], 1e-6):.2f}"
                )
        print(summarize_duels(results))
        if args.duel_csv:
            write_duel_csv(args.duel_csv, results)
            print(f"wrote {args.duel_csv}")
        if args.duel_plot:
            plot_duel_results(args.duel_plot, results)
            print(f"wrote {args.duel_plot}")
        return 0

    if args.no_viz:
        if args.time > 0:
            logs, reasons = run_sim(sim_cfg, params, args.time)
            print(summarize(logs))
            if args.csv:
                write_csv(args.csv, logs, reasons)
                print(f"wrote {args.csv}")
        else:
            mpc_cfg = MPCConfig(
                dt=sim_cfg.dt,
                horizon=10,
                max_hp=params.max_hp,
                max_heat=params.heat_limit,
                base_cooling=params.base_cooling,
                max_buffer=params.buffer_max,
                switch_cd_limit=params.switch_cd,
                degrade_time_limit=params.degrade_limit,
            )
            mpc = SentryMPC(mpc_cfg)
            sim = SentrySimulator(sim_cfg, params)
            try:
                while True:
                    sim.step(mpc)
            except KeyboardInterrupt:
                print(summarize(sim.logs))
    else:
        if args.time > 0:
            logs, reasons = run_sim(sim_cfg, params, args.time)
            print(summarize(logs))
            if args.csv:
                write_csv(args.csv, logs, reasons)
                print(f"wrote {args.csv}")
            visualize(logs, reasons, sim_cfg)
        else:
            visualize_live(sim_cfg, params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
