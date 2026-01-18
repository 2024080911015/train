import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 基础参数与数据
# ==========================================
# Bert Blocken 风阻数据 (6人队形)
DRAG_COEFFS_6 = [0.975, 0.616, 0.497, 0.443, 0.426, 0.433]

# 切换风阻惩罚
SWITCH_PENALTY_DRAG = 0.2

# 车手参数：6名完全一样的 TT 专家
STATS_TT = {'mass': 72, 'cp': 400, 'w_prime': 20000, 'cda': 0.23}


class Rider:
    def __init__(self, id, stats):
        self.id = id
        self.mass = stats['mass']
        self.cp = stats['cp']
        self.w_prime_max = stats['w_prime']
        self.cda = stats['cda']
        self.w_bal = self.w_prime_max
        self.total_mass = self.mass + 8.0

    def reset(self):
        self.w_bal = self.w_prime_max


# ==========================================
# 2. 物理与恢复模型
# ==========================================
def get_drag_factor(pos_idx):
    if pos_idx < len(DRAG_COEFFS_6):
        return DRAG_COEFFS_6[pos_idx]
    else:
        return 0.433


def skiba_recovery(w_bal, w_max, cp, p_current, dt):
    """Skiba 非线性恢复"""
    if p_current > cp:
        new_w = w_bal - (p_current - cp) * dt
        return max(0, new_w)

    d_cp = cp - p_current
    # 限制 d_cp 防止溢出，TT专家 CP 高，恢复能力强
    tau = 546 * math.exp(-0.01 * max(-200, d_cp)) + 316
    new_w = w_max - (w_max - w_bal) * math.exp(-dt / tau)
    return min(new_w, w_max)


# ==========================================
# 3. 仿真引擎 (全员轮换逻辑)
# ==========================================
def run_homogeneous_simulation(team, target_speed_kmh, race_dist_m, rotation_time_sec):
    for r in team: r.reset()

    rho = 1.225
    g = 9.81
    crr = 0.004
    eta = 0.98
    target_v = target_speed_kmh / 3.6
    dt = 1.0

    current_dist = 0
    current_time = 0
    formation = team.copy()  # 初始队列 [R1, R2, ..., R6]
    pull_timer = 0

    logs = {r.id: [] for r in team}

    while current_dist < race_dist_m:

        # --- A. 失败判定 ---
        # 只要有任何一人爆缸，策略即失败 (木桶效应)
        if any(r.w_bal <= 0 for r in formation):
            return None, None

        # --- B. 物理计算 ---
        is_switching = (pull_timer == 0) and (current_time > 0)

        for pos, rider in enumerate(formation):
            alpha = get_drag_factor(pos)

            # 切换成本：前两名车手受影响
            if is_switching and pos <= 1: alpha += SWITCH_PENALTY_DRAG

            f_drag = 0.5 * rho * (rider.cda * alpha) * (target_v ** 2)
            f_roll = rider.total_mass * g * crr
            p_req = ((f_drag + f_roll) * target_v) / eta

            rider.w_bal = skiba_recovery(rider.w_bal, rider.w_prime_max, rider.cp, p_req, dt)
            logs[rider.id].append(rider.w_bal)

        # --- C. 全员轮换逻辑 (Egalitarian Rotation) ---
        pull_timer += 1

        # 每个人都轮换，没有人“搭便车”
        limit = rotation_time_sec

        if pull_timer >= limit:
            # 领骑者 (Index 0) 移到 队尾 (Index 5)
            # R1 退下，R2 成为新 Leader
            formation.append(formation.pop(0))
            pull_timer = 0

        current_dist += target_v * dt
        current_time += dt

    return current_time, pd.DataFrame(logs)


# ==========================================
# 4. 双层优化器
# ==========================================
def optimize_homogeneous(team, race_dist):
    print("开始双层优化 (6 TT 全员轮换)...")

    # 外层：轮换时间
    rotation_options = range(20, 100, 10)

    summary_results = []
    best_global = {'time': float('inf'), 'speed': 0, 'rot': 0, 'log': None}

    for rot in rotation_options:
        # 内层：二分查找极限速度
        # TT 专家队非常强，速度上限较高
        low, high = 50.0, 70.0
        local_best = {'time': None, 'speed': 0}

        for _ in range(15):
            mid = (low + high) / 2
            t, log = run_homogeneous_simulation(team, mid, race_dist, rot)

            if t is not None:
                local_best = {'time': t, 'speed': mid, 'log': log}
                low = mid  # 尝试更快
            else:
                high = mid

        if local_best['time']:
            print(f"  轮换周期 {rot}s -> 极限速度 {local_best['speed']:.2f} km/h")
            summary_results.append((rot, local_best['time']))

            if local_best['time'] < best_global['time']:
                best_global = {
                    'time': local_best['time'],
                    'speed': local_best['speed'],
                    'rot': rot,
                    'log': local_best['log']
                }

    return best_global, summary_results


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # --- 组建车队 ---
    # 6名 TT 专家，能力完全一致
    team = []
    for i in range(6):
        team.append(Rider(f"TT_{i + 1}", STATS_TT))

    RACE_DIST = 44200  # Tokyo 赛道

    best, summary = optimize_homogeneous(team, RACE_DIST)

    print("\n" + "=" * 50)
    print(" 6人 TT 专家全员轮换策略 (Homogeneous) 结果")
    print("=" * 50)
    print(f"最佳轮换时间 (Optimal Rotation): {best['rot']} 秒")
    print(f"极限巡航速度 (Max Speed):       {best['speed']:.2f} km/h")
    print(f"完赛时间 (Finish Time):         {best['time'] / 60:.2f} min")

    # 绘图 1: 轮换时间分析
    rots = [x[0] for x in summary]
    times = [x[1] / 60 for x in summary]

    plt.figure(figsize=(10, 5))
    plt.plot(rots, times, 'o-', color='tab:green', linewidth=2)
    plt.title("6 TT Riders: Rotation Time Optimization", fontsize=14)
    plt.xlabel("Rotation Interval (s)", fontsize=12)
    plt.ylabel("Finish Time (min)", fontsize=12)
    plt.grid(True)
    plt.show()

    # 绘图 2: 能量曲线
    plt.figure(figsize=(12, 6))
    for r in team:
        # 所有人的曲线应该非常相似，交织在一起
        plt.plot(best['log'][r.id], label=r.id, linewidth=1.5, alpha=0.8)

    plt.title(f"Homogeneous Energy Dynamics (Speed: {best['speed']:.2f} km/h)\nSymmetric depletion pattern",
              fontsize=14)
    plt.ylabel("W' Balance (J)", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.axhline(0, color='black', linestyle=':')
    # 只显示前两个人的图例，避免遮挡，因为大家曲线都差不多
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.show()