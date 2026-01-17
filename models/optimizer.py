# models/optimizer.py
import numpy as np
import math
import random
from models.simulator import WPrimeBalanceSimulator


def generate_smart_strategy(params, course_data, rider_cp):
    """
    智能策略生成器：强制符合物理规律
    Params: [p_flat_ratio, p_climb_ratio]
    """
    p_flat_ratio, p_climb_ratio = params

    # 强制逻辑：爬坡功率必须 >= 平路功率
    if p_climb_ratio < p_flat_ratio:
        p_climb_ratio = p_flat_ratio

    p_flat = rider_cp * p_flat_ratio
    p_climb = rider_cp * p_climb_ratio
    p_descent = 0.0

    strategy = []
    for seg in course_data:
        slope = seg['slope']
        if slope > 0.015:  # 爬坡
            strategy.append(p_climb)
        elif slope < -0.015:  # 下坡
            strategy.append(p_descent)
        else:
            strategy.append(p_flat)

    return np.array(strategy)


def objective_function(params, simulator, course_data, wind_speed=0):
    """
    目标函数：支持传入风速
    """
    # 1. 生成策略
    full_strategy = generate_smart_strategy(params, course_data, simulator.cyclist.cp)

    # 2. 运行仿真 (传入风速!)
    total_time, _, is_exhausted = simulator.run_segment_simulation(
        full_strategy, course_data, wind_speed=wind_speed
    )

    # 3. 惩罚逻辑
    if is_exhausted:
        return total_time + 20000.0

    return total_time


def run_simulated_annealing(cyclist, course_data, wind_speed=0):
    """
    优化器：支持针对特定风速寻找最优解
    :param wind_speed: 风速 (m/s)
    """
    sim = WPrimeBalanceSimulator(cyclist)

    # Params: [Flat_Ratio, Climb_Ratio]
    current_params = np.array([0.95, 1.1])

    # 计算初始分数 (带风速)
    current_score = objective_function(current_params, sim, course_data, wind_speed)
    best_params = current_params.copy()
    best_score = current_score

    # 退火参数
    T = 100.0
    alpha = 0.90
    iter_per_temp = 10

    # print(f"  [Opt Start] Wind={wind_speed:.1f} m/s | Flat={current_params[0]:.2f}x, Climb={current_params[1]:.2f}x")

    while T > 0.1:
        for _ in range(iter_per_temp):
            # 扰动
            new_params = current_params.copy()
            idx = random.randint(0, 1)
            change = random.uniform(-0.1, 0.1)
            new_params[idx] += change

            # 边界限制
            new_params[0] = np.clip(new_params[0], 0.5, 1.5)
            new_params[1] = np.clip(new_params[1], 0.8, 3.0)

            # 评估 (带风速)
            new_score = objective_function(new_params, sim, course_data, wind_speed)

            # 接受准则
            delta = new_score - current_score
            if delta < 0 or math.exp(-delta / T) > random.random():
                current_params = new_params
                current_score = new_score

                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()

        T *= alpha

    # 生成最终策略
    final_strategy = generate_smart_strategy(best_params, course_data, cyclist.cp)

    print(f"  > Opt Result (Wind {wind_speed:.1f}m/s): Flat={best_params[0]:.2f}x, Climb={best_params[1]:.2f}x, Time={best_score:.1f}s")

    return final_strategy, best_score