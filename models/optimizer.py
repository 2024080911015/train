import numpy as np
import math
import random

# 引入 Simulator 类 (在函数内部引用或在顶部引用均可)
from models.simulator import WPrimeBalanceSimulator


def objective_function(power_strategy, simulator, course_data):
    """
    目标函数：计算完赛时间。
    如果中途力竭 (W' < 0)，返回一个巨大的惩罚时间。
    """
    total_time, _, is_exhausted = simulator.run_segment_simulation(power_strategy, course_data)

    if is_exhausted:
        # 惩罚项：基础大数 + 耗尽的程度(可选)
        # 这里简单处理，直接返回 100小时，确保该解被淘汰
        return 360000.0

    return total_time


def run_simulated_annealing(cyclist, course_data):
    """
    执行模拟退火算法寻找最优功率策略
    """
    # 初始化仿真器
    sim = WPrimeBalanceSimulator(cyclist)
    n_segments = len(course_data)

    # === 1. 初始化策略 ===
    # 策略初值设为 85% CP (保守策略，确保能跑完，不容易一开始就力竭)
    current_strategy = np.ones(n_segments) * (cyclist.cp * 0.85)

    # 计算初始分数
    current_score = objective_function(current_strategy, sim, course_data)

    # 记录全局最优
    best_strategy = current_strategy.copy()
    best_score = current_score

    # === 2. 退火参数 (需根据赛道长度微调) ===
    T = 2000.0  # 初始温度
    T_min = 1.0  # 终止温度
    alpha = 0.99  # 降温速率 (越接近1越慢，搜得越细)
    iter_per_temp = 5  # 每个温度下的迭代次数

    print(f"Starting Annealing... Initial Score: {best_score:.2f}s")

    # === 3. 主循环 ===
    while T > T_min:
        for _ in range(iter_per_temp):
            # --- A. 产生新解 (Perturbation) ---
            new_strategy = current_strategy.copy()

            # 操作：随机选择一段，进行平滑扰动
            idx = random.randint(0, n_segments - 1)

            # 扰动幅度：随机增减 -30W 到 +30W
            change = random.uniform(-30, 30)

            # 技巧：同时修改相邻的 3 个点，保持功率曲线的平滑性 (符合人类习惯)
            # 比如 idx-1, idx, idx+1 都加上 change
            for k in range(max(0, idx - 1), min(n_segments, idx + 2)):
                new_val = new_strategy[k] + change
                # 限制功率范围 [0, P_max] (假设 P_max=1200)
                new_strategy[k] = np.clip(new_val, 0, 1200)

            # --- B. 评估新解 ---
            new_score = objective_function(new_strategy, sim, course_data)

            # --- C. 接受准则 (Metropolis Criteria) ---
            delta = new_score - current_score

            accepted = False
            if delta < 0:
                # 成绩变好了 (时间变短)，直接接受
                accepted = True
            else:
                # 成绩变差了，以一定概率接受 (跳出局部最优)
                # 防止溢出处理
                if T > 0.1:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accepted = True

            # --- D. 更新状态 ---
            if accepted:
                current_strategy = new_strategy
                current_score = new_score

                # 如果是历史最好，记录下来
                if current_score < best_score:
                    best_score = current_score
                    best_strategy = current_strategy.copy()
                    # print(f"  New Best: {best_score:.2f}s (Temp {T:.1f})")

        # 降温
        T *= alpha

    print(f"Optimization Finished. Optimal Time: {best_score:.2f}s")
    return best_strategy, best_score