import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 物理参数与车手数据
# ==========================================
# 动态风阻数据
DYNAMIC_DRAG_DATA = {
    6: [0.975, 0.616, 0.497, 0.443, 0.426, 0.433],
    5: [0.975, 0.617, 0.498, 0.447, 0.443],
    4: [0.976, 0.618, 0.502, 0.466]
}

# 切换风阻惩罚
SWITCH_PENALTY_DRAG = 0.2

# 车手参数
STATS_TT = {'mass': 72, 'cp': 400, 'w_prime': 20000, 'cda': 0.23}
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

    def reset(self):
        self.w_bal = self.w_prime_max


# ==========================================
# 2. 物理核心函数
# ==========================================
def get_drag_factor(pos_idx, current_team_size):
    coeffs = DYNAMIC_DRAG_DATA.get(current_team_size, DYNAMIC_DRAG_DATA[4])
    if pos_idx < len(coeffs):
        return coeffs[pos_idx]
    else:
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
# 3. 仿真引擎 (已修复返回值 Bug)
# ==========================================
def run_simulation(team, target_speed_kmh, race_dist_m, rotation_time_sec):
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

    logs = {r.id: [] for r in team}
    team_size_log = []

    while current_dist < race_dist_m:

        # --- A. 掉队逻辑 ---
        if len(formation) > 4:
            leader = formation[0]
            if leader.w_bal <= 0 and leader.role == "Domestique":
                formation.pop(0)
                pull_timer = 0
        elif len(formation) == 4:
            if any(r.w_bal <= 0 for r in formation):
                return None, None, None  # <--- 修复点：返回 3 个值
        else:
            return None, None, None  # <--- 修复点：返回 3 个值

        # --- B. 物理计算 ---
        current_size = len(formation)
        is_switching = (pull_timer == 0) and (current_time > 0)

        for pos, rider in enumerate(formation):
            alpha = get_drag_factor(pos, current_size)
            if is_switching and pos <= 1: alpha += SWITCH_PENALTY_DRAG

            f_drag = 0.5 * rho * (rider.cda * alpha) * (target_v ** 2)
            f_roll = rider.total_mass * g * crr
            p_req = ((f_drag + f_roll) * target_v) / eta

            rider.w_bal = skiba_recovery(rider.w_bal, rider.w_prime_max, rider.cp, p_req, dt)
            logs[rider.id].append(rider.w_bal)

        for r in team:
            if r not in formation: logs[r.id].append(0)

        team_size_log.append(current_size)

        # --- C. 轮转逻辑 ---
        pull_timer += 1
        leader = formation[0]

        if leader.role == "Domestique":
            limit = 9999
        else:
            limit = rotation_time_sec

        if pull_timer >= limit:
            formation.append(formation.pop(0))
            pull_timer = 0

        current_dist += target_v * dt
        current_time += dt

    return current_time, pd.DataFrame(logs), team_size_log


# ==========================================
# 4. 双层优化器
# ==========================================
def optimize_strategy(team, race_dist):
    print("开始双层优化: 寻找最佳轮换时间 (Outer Loop) 和 极限速度 (Inner Loop)...")

    rotation_options = range(20, 100, 10)
    summary_results = []
    best_global = {'time': float('inf'), 'speed': 0, 'rot': 0, 'log': None, 'size_log': None}

    for rot in rotation_options:
        low, high = 45.0, 75.0
        local_best = {'time': None, 'speed': 0}

        for _ in range(12):
            mid = (low + high) / 2
            # 这里的接收变量数量现在匹配了 (3个)
            t, log, size_log = run_simulation(team, mid, race_dist, rot)

            if t is not None:
                local_best = {'time': t, 'speed': mid, 'log': log, 'size_log': size_log}
                low = mid
            else:
                high = mid

        if local_best['time']:
            print(
                f"  轮换时间 {rot}s -> 极限速度 {local_best['speed']:.2f} km/h (完赛: {local_best['time'] / 60:.2f} min)")
            summary_results.append((rot, local_best['time']))

            if local_best['time'] < best_global['time']:
                best_global = {
                    'time': local_best['time'],
                    'speed': local_best['speed'],
                    'rot': rot,
                    'log': local_best['log'],
                    'size_log': local_best['size_log']
                }

    return best_global, summary_results


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    team = [Rider(f"Sprinter_{i + 1}", STATS_SPRINTER, "Domestique") for i in range(2)] + \
           [Rider(f"TT_{i + 1}", STATS_TT, "Finisher") for i in range(4)]

    RACE_DIST = 44200

    best, summary = optimize_strategy(team, RACE_DIST)

    print("\n" + "=" * 50)
    print(" 最终优化结果 (Final Optimal Strategy)")
    print("=" * 50)
    print(f"最佳轮换时间 (Optimal Rotation): {best['rot']} 秒")
    print(f"极限巡航速度 (Max Speed):       {best['speed']:.2f} km/h")
    print(f"完赛时间 (Finish Time):         {best['time'] / 60:.2f} min")

    # 绘图 1: 轮换时间敏感度
    rots = [x[0] for x in summary]
    times = [x[1] / 60 for x in summary]

    plt.figure(figsize=(10, 5))
    plt.plot(rots, times, 'o-', color='darkred', linewidth=2)
    plt.title("Sensitivity: Rotation Time vs. Race Time", fontsize=14)
    plt.xlabel("Rotation Interval (seconds)", fontsize=12)
    plt.ylabel("Finish Time (minutes)", fontsize=12)
    plt.grid(True)
    best_idx = np.argmin(times)
    plt.plot(rots[best_idx], times[best_idx], 'r*', markersize=15)
    plt.annotate(f"Optimal: {rots[best_idx]}s", xy=(rots[best_idx], times[best_idx]),
                 xytext=(rots[best_idx], times[best_idx] + 0.2))
    plt.show()

    # 绘图 2: 能量与掉队时序
    fig, ax1 = plt.subplots(figsize=(12, 6))
    for r in team:
        ls = '--' if r.role == "Domestique" else '-'
        lw = 2.5 if r.role == "Domestique" else 1.5
        alpha = 0.6 if r.role == "Domestique" else 1.0
        ax1.plot(best['log'][r.id], label=r.id, linestyle=ls, linewidth=lw, alpha=alpha)

    ax1.set_ylabel("W' Balance (J)", fontsize=12)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_title(f"Optimal Race Dynamics (Speed: {best['speed']:.2f} km/h)", fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(best['size_log'], color='black', linewidth=2, linestyle=':', label='Team Size')
    ax2.set_ylabel("Team Size (Riders)", color='black', fontsize=12)
    ax2.set_yticks([4, 5, 6])
    plt.show()