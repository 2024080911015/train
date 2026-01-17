import numpy as np
import math
import random

# 引入 Simulator 类
from models.simulator import WPrimeBalanceSimulator


def generate_strategy_from_params(params, course_data):
    """
    [新增辅助函数] 根据少量关键参数生成全赛道的功率策略。
    这解决了"维度灾难"问题，使优化更容易收敛。

    :param params: list/array, [p_flat, p_climb, p_descent]
    :return: 完整的功率策略数组 (array)
    """
    p_flat, p_climb, p_descent = params
    strategy = []

    for seg in course_data:
        slope = seg['slope']

        # 定义简单的地形规则：
        # 坡度 > 2% (0.02) 视为爬坡
        # 坡度 < -2% (-0.02) 视为下坡
        # 其他视为平路
        if slope > 0.02:
            strategy.append(p_climb)
        elif slope < -0.02:
            strategy.append(p_descent)
        else:
            strategy.append(p_flat)

    return np.array(strategy)


def objective_function(params, simulator, course_data):
    """
    目标函数：计算完赛时间。
    [改进] 使用软惩罚 (Soft Penalty) 代替硬截断，帮助优化器找到临界最优解。
    """
    # 1. 将 3 个参数扩展为全赛道的功率策略
    full_strategy = generate_strategy_from_params(params, course_data)

    # 2. 运行仿真
    # 注意：确保 simulator.run_segment_simulation 的参数匹配 (如果修改了 simulator 支持风速，这里可能需要调整)
    total_time, _, is_exhausted = simulator.run_segment_simulation(full_strategy, course_data)

    if is_exhausted:
        # [改进] 软惩罚：
        # 返回 "实际时间 + 惩罚值"，而不是一个固定的巨大数字。
        # 这样算法能区分 "刚出发就力竭" 和 "终点前100米力竭" 的区别，从而引导解向可行域移动。
        return total_time + 10000.0

    return total_time


def run_simulated_annealing(cyclist, course_data):
    """
    执行参数化模拟退火算法寻找最优功率策略
    """
    # 初始化仿真器
    sim = WPrimeBalanceSimulator(cyclist)

    # 获取车手限制
    p_max_limit = cyclist.p_max if cyclist.p_max else 1200

    # === 1. 初始化策略参数 ===
    # 我们不再初始化几千个点的数组，而是只初始化 3 个核心参数
    # 格式: [平路功率, 爬坡功率, 下坡功率]
    # 初始猜测：CP 的 90%, 100%, 60%
    current_params = np.array([cyclist.cp * 0.9, cyclist.cp * 1.0, cyclist.cp * 0.6])

    # 确保不超标
    current_params = np.clip(current_params, 0, p_max_limit)

    # 计算初始分数
    current_score = objective_function(current_params, sim, course_data)

    # 记录全局最优
    best_params = current_params.copy()
    best_score = current_score

    # === 2. 退火参数 ===
    # 由于参数只有3个，搜索空间变小了，可以适当调整参数
    T = 1000.0  # 初始温度
    T_min = 1.0  # 终止温度
    alpha = 0.95  # 降温速率
    iter_per_temp = 20  # 每个温度下多搜几次，保证覆盖

    print(f"Starting Parametric Annealing...")
    print(
        f"  Initial Params: Flat={current_params[0]:.0f}W, Climb={current_params[1]:.0f}W, Descent={current_params[2]:.0f}W")
    print(f"  Initial Score: {best_score:.2f}s")

    # === 3. 主循环 ===
    while T > T_min:
        for _ in range(iter_per_temp):
            # --- A. 产生新解 (Perturbation) ---
            new_params = current_params.copy()

            # 操作：随机选择 1 个参数进行扰动
            idx = random.randint(0, 2)

            # 扰动幅度：随机增减 -20W 到 +20W
            change = random.uniform(-20, 20)

            new_params[idx] += change
            # 限制范围
            new_params[idx] = np.clip(new_params[idx], 0, p_max_limit)

            # --- B. 评估新解 ---
            new_score = objective_function(new_params, sim, course_data)

            # --- C. 接受准则 (Metropolis Criteria) ---
            delta = new_score - current_score

            accepted = False
            if delta < 0:
                # 成绩变好了 (时间变短)，直接接受
                accepted = True
            else:
                # 成绩变差了，按概率接受
                if T > 0.1:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accepted = True

            # --- D. 更新状态 ---
            if accepted:
                current_params = new_params
                current_score = new_score

                # 如果是历史最好，记录下来
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
                    # print(f"    New Best: {best_score:.2f}s (Params: {best_params.astype(int)})")

        # 降温
        T *= alpha

    print(f"Optimization Finished. Best Time: {best_score:.2f}s")
    print(f"Best Params: Flat={best_params[0]:.1f}W, Climb={best_params[1]:.1f}W, Descent={best_params[2]:.1f}W")

    # 最后，将最优参数转换回完整的策略数组返回，以便画图和存储
    final_full_strategy = generate_strategy_from_params(best_params, course_data)

    return final_full_strategy, best_score