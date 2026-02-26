import numpy as np
from dataclasses import dataclass
from typing import List

# --- Constants ---
POSTURE_ATTACK = 1
POSTURE_DEFENSE = 2
POSTURE_MOVE = 3

# Rule Constraints
MAX_HP = 600.0          
MAX_HEAT = 500.0        
BASE_COOLING = 20.0     
MAX_BUFFER = 60.0       
SWITCH_CD_LIMIT = 5.0   
DEGRADE_TIME_LIMIT = 180.0 

@dataclass
class SentryState:
    """
    裁判系统反馈数据 + 内部维护变量
    """
    posture: int            # Protocol 0x020D
    hp: float               # Protocol 0x0201
    heat: float             # Protocol 0x0202
    buffer_energy: float    # Protocol 0x0202
    time_in_posture: float  # Internal Timer
    switch_cd: float        # Internal Timer

    def copy(self):
        return SentryState(
            self.posture, self.hp, self.heat, 
            self.buffer_energy, self.time_in_posture, self.switch_cd
        )

@dataclass
class SentryObservation:
    """
    视觉/感知系统输入 (当前时刻)
    """
    enemy_visible: bool     # 是否看到敌人
    enemy_distance: float   # 敌人距离 (m)
    is_aim_locked: bool     # 自瞄是否锁定
    
    # 导航距离: 距离目标点(路点)的距离
    # 距离越远 -> 机动需求越高
    target_distance: float  
    
    # 强制姿态控制: 0=Auto, 1=Attack, 2=Defense, 3=Move
    # 用于特殊路段强制切换
    force_posture: int = 0

@dataclass
class PredictionInput:
    """内部转换后的预测量"""
    incoming_dps: float     
    heat_demand: float      
    mobility_demand: float  

"""
w_survival: 生存代价权重。血量损失越多，惩罚越大；数值越高越保守、越倾向防御。
w_missed_fire: 错失开火代价权重。想打但因为热量/冷却限制没打出的惩罚；数值越高越倾向保持输出（更偏进攻/散热姿态）。
w_mobility_risk: 机动风险权重。机动需求高但功率倍率低时的惩罚；数值越高越倾向移动姿态。
w_switch: 姿态切换惩罚。数值越高越不爱切换，行为更稳定但反应更慢。
w_degrade: 姿态弱化惩罚（维持同姿态超过 180s 的罚分）。数值越高越避免长时间停留在同一姿态。

"""
class MPCConfig:
    def __init__(
        self,
        dt: float = 0.5,
        horizon: int = 10,
        
        w_survival: float = 20.0,
        w_missed_fire: float = 50.0,
        w_mobility_risk: float = 20.0,
        w_switch: float = 1.0,
        w_degrade: float = 500.0,
        max_hp: float = MAX_HP,
        max_heat: float = MAX_HEAT,
        base_cooling: float = BASE_COOLING,
        max_buffer: float = MAX_BUFFER,
        switch_cd_limit: float = SWITCH_CD_LIMIT,
        degrade_time_limit: float = DEGRADE_TIME_LIMIT,
    ):
        self.dt = dt
        self.horizon = horizon
        self.w_survival = w_survival
        self.w_missed_fire = w_missed_fire
        self.w_mobility_risk = w_mobility_risk
        self.w_switch = w_switch
        self.w_degrade = w_degrade
        self.max_hp = max_hp
        self.max_heat = max_heat
        self.base_cooling = base_cooling
        self.max_buffer = max_buffer
        self.switch_cd_limit = switch_cd_limit
        self.degrade_time_limit = degrade_time_limit

class SentryMPC:
    def __init__(self, config: MPCConfig):
        self.config = config

    def predict_horizon_from_obs(self, obs: SentryObservation) -> List[PredictionInput]:
        """
        核心预测器：将当前瞬时观测转换为未来N步的预测序列
        假设模型：Zero-Order Hold (假设未来一段时间环境保持不变)
        """
        # 1. 估算当前威胁度 (DPS)
        current_dps = 0.0
        if obs.enemy_visible:
            if obs.enemy_distance < 3.0: # 近战
                current_dps = 200.0
            elif obs.enemy_distance < 8.0: # 中距离
                current_dps = 50.0
            else: # 远距离
                current_dps = 10.0
        
        # 2. 估算开火需求 (Heat/s)
        current_heat_demand = 0.0
        if obs.is_aim_locked:
            current_heat_demand = 300.0 # 假设高射速
        elif obs.enemy_visible:
            current_heat_demand = 50.0 # 偶尔点射
            
        # 3. 估算机动需求 (0-1) 基于目标距离
        # 距离越远，需求越高。假设 20m 以外为满需求。
        current_mob_demand = np.clip(obs.target_distance / 20.0, 0.0, 1.0)
            
        # 生成预测序列
        predictions = []
        for _ in range(self.config.horizon):
            predictions.append(PredictionInput(
                incoming_dps=current_dps,
                heat_demand=current_heat_demand,
                mobility_demand=current_mob_demand
            ))
            
        return predictions

    def get_dynamics_coeffs(self, posture, time_in_posture):
        is_degraded = time_in_posture > self.config.degrade_time_limit
        
        cooling_mult = 1.0
        defense_mult = 1.0 
        power_mult = 1.0
        
        if posture == POSTURE_ATTACK:
            cooling_mult = 2.0 if is_degraded else 3.0
            defense_mult = 1.25
            power_mult = 0.5
        elif posture == POSTURE_DEFENSE:
            cooling_mult = 1.0 / 3.0
            defense_mult = 0.75 if is_degraded else 0.5
            power_mult = 0.5
        elif posture == POSTURE_MOVE:
            cooling_mult = 1.0 / 3.0
            defense_mult = 1.25
            power_mult = 1.2 if is_degraded else 1.5
            
        return cooling_mult, defense_mult, power_mult

    def step(self, state: SentryState, action: int, disturbance: PredictionInput):
        next_state = state.copy()
        dt = self.config.dt
        
        if action != state.posture and state.switch_cd <= 0:
            next_state.posture = action
            next_state.time_in_posture = 0.0
            next_state.switch_cd = self.config.switch_cd_limit
        else:
            next_state.posture = state.posture
            next_state.time_in_posture += dt
            next_state.switch_cd = max(0.0, state.switch_cd - dt)
            
        cool_mult, def_mult, power_mult = self.get_dynamics_coeffs(
            next_state.posture, next_state.time_in_posture
        )
        
        damage = disturbance.incoming_dps * def_mult * dt
        next_state.hp = max(0.0, state.hp - damage)
        
        cooling = self.config.base_cooling * cool_mult * dt
        max_fire_possible = self.config.max_heat - (state.heat - cooling)
        actual_fire = min(disturbance.heat_demand * dt, max_fire_possible)
        actual_fire = max(0.0, actual_fire)
        
        next_state.heat = max(0.0, state.heat + actual_fire - cooling)
        
        buffer_change = (power_mult - 1.0 - disturbance.mobility_demand) * 10.0 * dt
        next_state.buffer_energy = np.clip(
            state.buffer_energy + buffer_change,
            0.0,
            self.config.max_buffer,
        )
        
        return next_state, actual_fire

    def get_step_cost(self, state: SentryState, next_state: SentryState, 
                      disturbance: PredictionInput, actual_fire: float):
        cost = 0.0
        
        hp_loss = state.hp - next_state.hp
        hp_factor = 1.0 + (self.config.max_hp / (state.hp + 1.0))
        cost += hp_loss * self.config.w_survival * hp_factor
        
        desired_fire = disturbance.heat_demand * self.config.dt
        missed_fire = desired_fire - actual_fire
        if missed_fire > 0.1:
            cost += missed_fire * self.config.w_missed_fire
            
        _, _, power_mult = self.get_dynamics_coeffs(next_state.posture, next_state.time_in_posture)
        if disturbance.mobility_demand > 0.5:
            risk = (1.5 - power_mult) * disturbance.mobility_demand
            cost += risk * self.config.w_mobility_risk
            
        if next_state.time_in_posture > self.config.degrade_time_limit:
            cost += self.config.w_degrade
            
        return cost

    def solve(self, current_state: SentryState, current_obs: SentryObservation):
        """
        完全自主决策接口
        """
        # 0. 强制控制逻辑 (Force Logic)
        # 如果设置了 force_posture 且 CD 允许，则直接返回该姿态
        if current_obs.force_posture != 0:
            target = current_obs.force_posture
            if current_state.switch_cd <= 0:
                # CD 转好了，直接切
                return target, 0.0 
            elif current_state.posture == target:
                # 已经在该姿态，保持
                return target, 0.0
            else:
                # CD 没转好，没办法，只能先保持当前，下一帧再切
                # 这里返回当前姿态，但也无法做任何优化
                return current_state.posture, 0.0

        # 1. 内部预测
        predictions = self.predict_horizon_from_obs(current_obs)
        
        # 2. 求解最优动作
        best_cost = float('inf')
        best_action = current_state.posture
        
        candidates = [current_state.posture]
        if current_state.switch_cd <= 0:
            candidates = [POSTURE_ATTACK, POSTURE_DEFENSE, POSTURE_MOVE]
            
        for action in candidates:
            temp_state = current_state.copy()
            total_cost = 0.0
            
            if action != current_state.posture:
                total_cost += self.config.w_switch
                
            for t in range(self.config.horizon):
                dist = predictions[t]
                step_action = action if t == 0 else temp_state.posture
                next_state, actual_fire = self.step(temp_state, step_action, dist)
                total_cost += self.get_step_cost(temp_state, next_state, dist, actual_fire)
                temp_state = next_state
                
                if temp_state.hp <= 0:
                    total_cost += 100000.0
                    break
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_action = action
                
        return best_action, best_cost
