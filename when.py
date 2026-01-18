import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 物理参数 (保持不变)
# ==========================================
# 动态风阻系数 (基于 6/5/4 人)
DYNAMIC_DRAG_DATA = {
    6: [0.975, 0.616, 0.497, 0.443, 0.426, 0.433],
    5: [0.975, 0.617, 0.498, 0.447, 0.443],
    4: [0.976, 0.618, 0.502, 0.466]
}
SWITCH_PENALTY_DRAG = 0.2

# 车手数据
STATS_TT = {'mass': 72, 'cp': 420, 'w_prime': 25000, 'cda': 0.23}
STATS_SPRINTER = {'mass': 75, 'cp': 340, 'w_prime': 35000, 'cda': 0.28}


class Rider:
    def __init__(self, id, stats, role="Finisher"):
        self.id = id
        self.mass = stats['mass']
        self.cp = stats['cp']
        self.w_prime_max = stats['w_prime']
        self.cda = stats['cda']
        self.w_bal = self.w_prime_max
        self.total_mass = self.mass + 8.0
        self.role = role
        self.rotations_completed = 0  # 记录已完成的轮换次数

    def reset(self):
        self.w_bal = self.w_prime_max
        self.rotations_completed = 0


# ==========================================
# 2. 物理函数
# ==========================================
def get_drag_factor(pos_idx, current_team_size):
    coeffs = DYNAMIC_DRAG_DATA.get(current_team_size, DYNAMIC_DRAG_DATA[4])
    if pos_idx < len(coeffs): return coeffs[pos_idx]
    return coeffs[-1]


def skiba_recovery(w_bal, w_max, cp, p_current, dt):
    if p_current > cp:
        new_w = w_bal - (p_current - cp) * dt
        return max(0, new_w)
    d_cp = cp - p_current
    tau = 546 * math.exp(-0.01 * max(-200, d_cp)) + 316
    new_w = w_max - (w_max - w_bal) * math.exp(-dt / tau)
    return min(new_w, w_max)


# ==========================================
# 3. 仿真引擎 (带最大轮换次数限制)
# ==========================================
def run_simulation(team, target_speed_kmh, race_dist_m, rotation_time_sec, max_sprinter_rotations):
    """
    max_sprinter_rotations: Sprinter 在掉队前允许参与的最大轮换次数
    """
    for r in team: r.reset()

    rho = 1.225
    g = 9.81
    crr = 0.004
    eta = 0.98
    target_v = target_speed_kmh / 3.6
    dt = 1.0

    current_dist = 0
    current_time = 0
    formation = team.copy()
    pull_timer = 0

    while current_dist < race_dist_m:

        # --- A. 掉队逻辑 ---
        if len(formation) > 4:
            leader = formation[0]
            # 只有 Sprinter 会掉队，且必须体力耗尽
            if leader.role == "Domestique" and leader.w_bal <= 0:
                formation.pop(0)
                pull_timer = 0
        elif len(formation) == 4:
            # 核心队员掉队 = 失败
            if any(r.w_bal <= 0 for r in formation): return None
        else:
            return None

        # --- B. 物理计算 ---
        current_size = len(formation)
        is_switching = (pull_timer == 0) and (current_time > 0)

        for pos, rider in enumerate(formation):
            alpha = get_drag_factor(pos, current_size)
            if is_switching and pos <= 1: alpha += SWITCH_PENALTY_DRAG

            # 关键：领骑时的 CdA 取决于领骑者是谁！
            # Sprinter 领骑 (0.28) 阻力远大于 TT 领骑 (0.23)
            f_drag = 0.5 * rho * (rider.cda * alpha) * (target_v ** 2)
            f_roll = rider.total_mass * g * crr
            p_req = ((f_drag + f_roll) * target_v) / eta

            rider.w_bal = skiba_recovery(rider.w_bal, rider.w_prime_max, rider.cp, p_req, dt)

        # --- C. 轮换逻辑 (混合模式) ---
        pull_timer += 1
        leader = formation[0]

        limit = rotation_time_sec  # 默认轮换时间

        if leader.role == "Domestique":
            # 检查 Sprinter 是否达到了轮换次数上限
            if leader.rotations_completed < max_sprinter_rotations:
                # 还没到上限，正常轮换（去队尾休息）
                pass
            else:
                # 到了上限，不再轮换，死命骑直到掉队 (Burnout Mode)
                limit = 9999

        if pull_timer >= limit:
            rider_to_move = formation.pop(0)
            rider_to_move.rotations_completed += 1  # 增加计数
            formation.append(rider_to_move)
            pull_timer = 0

        current_dist += target_v * dt
        current_time += dt

    return current_time


# ==========================================
# 4. 实验：遍历轮换次数
# ==========================================
def experiment_optimal_drop_time(team, race_dist):
    print("正在分析 Sprinter 最佳轮换次数...")

    # 设定一个固定的较高速度来测试耐久度 (例如 53 km/h)
    # 或者对每个 N 都进行速度寻优（更准确，但慢）-> 这里采用寻优

    results = []
    # 测试 Sprinter 参与 0 到 10 次轮换
    rotation_counts = range(0, 11)

    for n in rotation_counts:
        # 二分查找该 N 下的极限速度
        low, high = 45.0, 60.0
        best_time = None
        best_speed = 0

        for _ in range(10):
            mid = (low + high) / 2
            # 假设全队轮换周期为 30s
            t = run_simulation(team, mid, race_dist, 30, n)
            if t is not None:
                best_time = t
                best_speed = mid
                low = mid
            else:
                high = mid

        if best_time:
            print(f"Sprinter 参与 {n} 次轮换 -> 完赛时间: {best_time / 60:.2f} min (速度 {best_speed:.2f} km/h)")
            results.append((n, best_time))

    return results


if __name__ == "__main__":
    # 4 TT + 2 Sprinters
    team = [Rider(f"Sprinter_{i}", STATS_SPRINTER, "Domestique") for i in range(2)] + \
           [Rider(f"TT_{i}", STATS_TT, "Finisher") for i in range(4)]

    RACE_DIST = 44200

    data = experiment_optimal_drop_time(team, RACE_DIST)

    # 绘图
    ns = [x[0] for x in data]
    times = [x[1] / 60 for x in data]

    plt.figure(figsize=(10, 6))
    plt.plot(ns, times, 'o-', linewidth=2, color='tab:orange')
    plt.title("Impact of Sprinter Rotation Participation on Finish Time", fontsize=14)
    plt.xlabel("Number of Rotations per Sprinter before Burnout", fontsize=12)
    plt.ylabel("Finish Time (minutes)", fontsize=12)

    # 标注最优解
    min_idx = np.argmin(times)
    plt.plot(ns[min_idx], times[min_idx], 'r*', markersize=15, label='Optimal')

    plt.grid(True)
    plt.legend()
    plt.show()
