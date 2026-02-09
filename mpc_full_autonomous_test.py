import numpy as np
import pandas as pd
from sentry_mpc_detailed import SentryMPC, MPCConfig, SentryState, SentryObservation, POSTURE_ATTACK, POSTURE_DEFENSE, POSTURE_MOVE

def run_autonomous_tests(n_cases=100):
    config = MPCConfig()
    mpc = SentryMPC(config)
    
    results = []
    
    # 状态映射表
    posture_map = {1: '进攻(Attack)', 2: '防御(Defense)', 3: '移动(Move)'}
    
    print(f"Generating {n_cases} random test cases with Force Logic...")
    
    for i in range(n_cases):
        # --- 1. 随机生成当前机器人状态 (Referee System) ---
        posture = np.random.choice([POSTURE_ATTACK, POSTURE_DEFENSE, POSTURE_MOVE])
        hp = np.random.uniform(50, 600)
        heat = np.random.uniform(0, 500)
        buffer_energy = np.random.uniform(0, 60)
        
        # 30% 概率处于即将衰减的状态 (170s+)
        if np.random.random() < 0.3:
            time_in_posture = np.random.uniform(170, 185)
        else:
            time_in_posture = np.random.uniform(0, 100)
            
        switch_cd = 0.0 if np.random.random() > 0.2 else np.random.uniform(0, 5.0)
        
        state = SentryState(posture, hp, heat, buffer_energy, time_in_posture, switch_cd)
        
        # --- 2. 随机生成当前感知输入 (Vision/Sensor) ---
        enemy_visible = np.random.choice([True, False])
        enemy_dist = 20.0
        if enemy_visible:
            enemy_dist = np.random.uniform(1.0, 15.0)
            
        is_locked = False
        if enemy_visible and np.random.random() > 0.5:
            is_locked = True
            
        # 生成目标距离 (0-50m)
        target_dist = np.random.uniform(0.0, 50.0)
        
        # 生成强制控制指令 (10% 概率强制切换)
        force_posture = 0
        if np.random.random() < 0.1:
            force_posture = np.random.choice([POSTURE_ATTACK, POSTURE_DEFENSE, POSTURE_MOVE])
            
        obs = SentryObservation(enemy_visible, enemy_dist, is_locked, target_dist, force_posture)
        
        # --- 3. 运行 MPC ---
        opt_action, cost = mpc.solve(state, obs)
        
        # --- 4. 分析原因 (简易规则解释) ---
        reason = []
        if obs.force_posture != 0:
            if opt_action == obs.force_posture:
                reason.append(f"强制指令生效({posture_map[obs.force_posture]})")
            else:
                reason.append(f"强制指令失败(CD中)")
        else:
            if state.switch_cd > 0 and opt_action == state.posture:
                reason.append("冷却中")
            elif state.time_in_posture > 175:
                reason.append("即将衰减")
            
            if obs.target_distance > 20.0 and opt_action == POSTURE_MOVE:
                reason.append("长距离奔袭")
            
            if state.heat > 400 and obs.is_aim_locked and opt_action == POSTURE_ATTACK:
                reason.append("急需散热开火")
                
            if state.hp < 200 and opt_action == POSTURE_DEFENSE:
                reason.append("血量危急保命")
                
            if not reason:
                reason.append("综合最优")
            
        # --- 5. 记录结果 ---
        results.append({
            'ID': i+1,
            '当前姿态': posture_map[posture],
            '血量': f"{hp:.0f}",
            '维持时间': f"{time_in_posture:.1f}s",
            'CD': f"{switch_cd:.1f}s",
            '看见敌人': "是" if enemy_visible else "否",
            '目标距离': f"{target_dist:.1f}m",
            '强制指令': posture_map[force_posture] if force_posture!=0 else "-",
            'MPC输出': posture_map[opt_action],
            '决策原因': ",".join(reason)
        })
        
    # 转换为 DataFrame 并打印
    df = pd.DataFrame(results)
    
    # 设置 pandas 显示选项以完整打印表格
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.east_asian_width', True)
    
    print(df.to_string(index=False))
    
    # 保存为 CSV 方便查看
    df.to_csv('mpc_force_logic_results.csv', index=False, encoding='utf-8-sig')
    print(f"\nResults saved to mpc_force_logic_results.csv")

if __name__ == "__main__":
    run_autonomous_tests(100)
