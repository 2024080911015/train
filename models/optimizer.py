# models/optimizer.py
import numpy as np
import math
import random
from models.simulator import WPrimeBalanceSimulator


def generate_smart_strategy(params, course_data, rider_cp):
    """
    智能策略生成器：强制符合物理规律
    Params: [p_flat_ratio, p_climb_ratio] (相对于 CP 的倍率)
    """
    p_flat_ratio, p_climb_ratio = params

    # 强制逻辑：爬坡功率必须 >= 平路功率
    if p_climb_ratio < p_flat_ratio:
        p_climb_ratio = p_flat_ratio

    p_flat = rider_cp * p_flat_ratio
    p_climb = rider_cp * p_climb_ratio
    # 下坡功率强制为 0 或极低 (恢复)
    p_descent = 0.0

    strategy = []
    for seg in course_data:
        slope = seg['slope']

        # 逻辑判定
        if slope > 0.015:  # 坡度 > 1.5% 算爬坡
            strategy.append(p_climb)
        elif slope < -0.015:  # 坡度 < -1.5% 算下坡
            strategy.append(p_descent)
        else:
            strategy.append(p_flat)

    return np.array(strategy)


def objective_function(params, simulator, course_data):
    """
    目标函数
    """
    # 1. 生成符合常识的策略
    full_strategy = generate_smart_strategy(params, course_data, simulator.cyclist.cp)

    # 2. 运行仿真
    total_time, _, is_exhausted = simulator.run_segment_simulation(full_strategy, course_data)

    # 3. 惩罚逻辑
    if is_exhausted:
        # 严重惩罚，但带有梯度（鼓励跑完更多路程）
        # 这里简单处理：直接给一个极大值
        return total_time + 20000.0

    return total_time


def run_simulated_annealing(cyclist, course_data):
    """
    优化器：寻找最佳的 [平路系数, 爬坡系数]
    """
    sim = WPrimeBalanceSimulator(cyclist)

    # === 参数定义 ===
    # Params: [Flat_Ratio, Climb_Ratio]
    # 范围：平路 0.8~1.1 倍 CP，爬坡 1.0~2.5 倍 CP
    # 初始猜测：平路保本，爬坡稍微用力
    current_params = np.array([0.95, 1.1])

    # 计算初始分数
    current_score = objective_function(current_params, sim, course_data)
    best_params = current_params.copy()
    best_score = current_score

    # 退火参数
    T = 100.0
    alpha = 0.90
    iter_per_temp = 10

    print(f"  Start Opt: Flat={current_params[0]:.2f}xCP, Climb={current_params[1]:.2f}xCP")

    while T > 0.1:
        for _ in range(iter_per_temp):
            # 扰动
            new_params = current_params.copy()
            idx = random.randint(0, 1)
            change = random.uniform(-0.1, 0.1)
            new_params[idx] += change

            # 边界限制
            new_params[0] = np.clip(new_params[0], 0.5, 1.5)  # 平路系数
            new_params[1] = np.clip(new_params[1], 0.8, 3.0)  # 爬坡系数 (允许爆发)

            # 评估
            new_score = objective_function(new_params, sim, course_data)

            # 接受准则
            delta = new_score - current_score
            if delta < 0 or math.exp(-delta / T) > random.random():
                current_params = new_params
                current_score = new_score

                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()

        T *= alpha

    # 生成最终策略供返回
    final_strategy = generate_smart_strategy(best_params, course_data, cyclist.cp)

    print(f"  End Opt: Flat={best_params[0]:.2f}xCP ({best_params[0] * cyclist.cp:.0f}W), "
          f"Climb={best_params[1]:.2f}xCP ({best_params[1] * cyclist.cp:.0f}W)")

    return final_strategy, best_score