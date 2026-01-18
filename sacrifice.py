import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 物理参数
# ==========================================
DYNAMIC_DRAG_DATA = {
    6: [0.975, 0.616, 0.497, 0.443, 0.426, 0.433],
    5: [0.975, 0.617, 0.498, 0.447, 0.443],
    4: [0.976, 0.618, 0.502, 0.466]
}

SWITCH_PENALTY_DRAG = 0.2

STATS_TT = {'mass': 72, 'cp': 400, 'w_prime': 20000, 'cda': 0.23}
STATS_SPRINTER = {'mass': 75, 'cp': 340, 'w_prime': 35000, 'cda': 0.28}

# Sprinter 轮换限制
SPRINTER_ROTATION_LIMIT = 2


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
        self.rotations_done = 0

    def reset(self):
        self.w_bal = self.w_prime_max
        self.rotations_done = 0


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
# 3. 仿真引擎
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

    # 日志
    logs = {r.id: [] for r in team}
    team_size_log = []

    while current_dist < race_dist_m:
        # A. 掉队检查
        if len(formation) > 4:
            leader = formation[0]
            if leader.w_bal <= 0 and leader.role == "Domestique":
                formation.pop(0)
                pull_timer = 0
        elif len(formation) == 4:
            if any(r.w_bal <= 0 for r in formation): return None, None, None
        else:
            return None, None, None

        # B. 物理计算
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

        # C. 轮换逻辑
        pull_timer += 1
        leader = formation[0]

        current_limit = rotation_time_sec
        if leader.role == "Domestique":
            if leader.rotations_done >= SPRINTER_ROTATION_LIMIT:
                current_limit = 99999

        if pull_timer >= current_limit:
            rider_moving = formation.pop(0)
            rider_moving.rotations_done += 1
            formation.append(rider_moving)
            pull_timer = 0

        current_dist += target_v * dt
        current_time += dt

    return current_time, pd.DataFrame(logs), team_size_log


# ==========================================
# 4. 敏感度分析 (修改了范围)
# ==========================================
def analyze_rotation_sensitivity(team, race_dist):
    print(f"开始轮换时间敏感度分析 (扩展范围 5s-90s)...")

    # === 修改点：从 5秒 开始扫描，捕捉左侧的上升形态 ===
    rotation_options = range(5, 95, 5)

    results = []
    best_global = {'time': float('inf'), 'rot': 0, 'speed': 0}

    for rot in rotation_options:
        low, high = 48.0, 60.0
        local_best_time = None
        local_max_speed = 0

        for _ in range(12):
            mid = (low + high) / 2
            t, log, size_log = run_simulation(team, mid, race_dist, rot)

            if t is not None:
                local_best_time = t
                local_max_speed = mid
                low = mid
            else:
                high = mid

        if local_best_time:
            # 打印进度
            print(f"  Rot={rot:<2}s -> Time={local_best_time / 60:.3f} min")
            results.append((rot, local_best_time))

            if local_best_time < best_global['time']:
                best_global = {'time': local_best_time, 'rot': rot, 'speed': local_max_speed}
        else:
            print(f"  Rot={rot:<2}s -> Failed (Switching cost too high)")

    return best_global, results


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    team = [Rider(f"Sprinter_{i + 1}", STATS_SPRINTER, "Domestique") for i in range(2)] + \
           [Rider(f"TT_{i + 1}", STATS_TT, "Finisher") for i in range(4)]

    RACE_DIST = 44200

    # 1. 跑敏感度分析
    best, data = analyze_rotation_sensitivity(team, RACE_DIST)

    print("\n" + "=" * 50)
    print(f" 最优轮换时间: {best['rot']} 秒")
    print(f" 完赛时间: {best['time'] / 60:.3f} min")
    print("=" * 50)

    # --- 绘图 1: U型图 ---
    plt.figure(figsize=(10, 6))
    x_rot = [row[0] for row in data]
    y_time = [row[1] / 60 for row in data]

    plt.plot(x_rot, y_time, 'o-', color='tab:blue', linewidth=2)
    plt.plot(best['rot'], best['time'] / 60, 'r*', markersize=15, label='Optimal')

    plt.title("Figure 1: Sensitivity Analysis (U-Shape Visualization)", fontsize=14)
    plt.xlabel("Rotation Interval (s)", fontsize=12)
    plt.ylabel("Finish Time (min)", fontsize=12)
    plt.grid(True)

    # 标注左侧
    if len(x_rot) > 0 and x_rot[0] <= 10:
        plt.annotate('Inefficient Switching\n(Too frequent)',
                     xy=(x_rot[0], y_time[0]),
                     xytext=(x_rot[0] + 5, y_time[0]),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

    # 标注右侧
    plt.annotate('Deep Fatigue\n(Slow Recovery)',
                 xy=(x_rot[-1], y_time[-1]),
                 xytext=(x_rot[-1] - 25, y_time[-1]),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.legend()
    plt.show()

    # --- 跑一次最佳结果用于画能量图 ---
    print("\n生成能量图...")
    _, final_log, final_size_log = run_simulation(team, best['speed'], RACE_DIST, best['rot'])

    # --- 绘图 2: 能量图 ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    for r in team:
        ls = '--' if r.role == "Domestique" else '-'
        lw = 2.5 if r.role == "Domestique" else 1.5
        color = 'tab:red' if r.role == "Domestique" else None
        ax1.plot(final_log[r.id], label=r.id, linestyle=ls, linewidth=lw, color=color)

    ax1.set_title(f"Figure 2: Optimal Energy Dynamics (Rot={best['rot']}s)", fontsize=14)
    ax1.set_ylabel("W' Balance (J)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 叠加队伍人数
    ax2 = ax1.twinx()
    ax2.plot(final_size_log, 'k:', linewidth=2, label='Team Size')
    ax2.set_ylabel("Team Size")
    ax2.set_yticks([4, 5, 6])

    plt.tight_layout()
    plt.show()
