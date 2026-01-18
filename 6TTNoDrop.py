import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 基础参数与数据
# ==========================================
# Bert Blocken 风阻数据 (6人队形)
DRAG_COEFFS_6 = [0.975, 0.616, 0.497, 0.443, 0.426, 0.433]

# 切换风阻惩罚 (这是造成 U 型图左侧上升的元凶)
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
    # 限制 d_cp 防止溢出
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
    formation = team.copy()
    pull_timer = 0

    # 记录日志
    logs = {r.id: [] for r in team}

    while current_dist < race_dist_m:

        # --- A. 失败判定 ---
        if any(r.w_bal <= 0 for r in formation):
            return None, None

        # --- B. 物理计算 ---
        is_switching = (pull_timer == 0) and (current_time > 0)

        for pos, rider in enumerate(formation):
            alpha = get_drag_factor(pos)

            # 关键：这里施加切换惩罚，导致短轮换效率低
            if is_switching and pos <= 1: alpha += SWITCH_PENALTY_DRAG

            f_drag = 0.5 * rho * (rider.cda * alpha) * (target_v ** 2)
            f_roll = rider.total_mass * g * crr
            p_req = ((f_drag + f_roll) * target_v) / eta

            rider.w_bal = skiba_recovery(rider.w_bal, rider.w_prime_max, rider.cp, p_req, dt)
            logs[rider.id].append(rider.w_bal)

        # --- C. 全员轮换逻辑 ---
        pull_timer += 1
        limit = rotation_time_sec

        if pull_timer >= limit:
            formation.append(formation.pop(0))
            pull_timer = 0

        current_dist += target_v * dt
        current_time += dt

    return current_time, pd.DataFrame(logs)


# ==========================================
# 4. 敏感度分析 (U型图专用)
# ==========================================
def analyze_homogeneous_sensitivity(team, race_dist):
    print("开始轮换时间敏感度分析 (Homogeneous 6 TT)...")
    print("扫描范围: 5s - 95s (Step=5s)")

    # === 修改范围以捕捉 U 型左侧 ===
    rotation_options = range(5, 95, 5)

    results = []
    best_global = {'time': float('inf'), 'speed': 0, 'rot': 0}

    for rot in rotation_options:
        low, high = 50.0, 70.0
        local_best_time = None
        local_max_speed = 0

        for _ in range(12):
            mid = (low + high) / 2
            t, log = run_homogeneous_simulation(team, mid, race_dist, rot)

            if t is not None:
                local_best_time = t
                local_max_speed = mid
                low = mid
            else:
                high = mid

        if local_best_time:
            print(f"  Rot={rot:<2}s -> Time={local_best_time / 60:.3f} min")
            results.append((rot, local_best_time))

            if local_best_time < best_global['time']:
                best_global = {
                    'time': local_best_time,
                    'speed': local_max_speed,
                    'rot': rot
                }
        else:
            print(f"  Rot={rot:<2}s -> Failed (Too fast/inefficient)")

    return best_global, results


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # --- 组建车队 ---
    team = []
    for i in range(6):
        team.append(Rider(f"TT_{i + 1}", STATS_TT))

    RACE_DIST = 44200

    # 1. 运行敏感度分析
    best, data = analyze_homogeneous_sensitivity(team, RACE_DIST)

    print("\n" + "=" * 50)
    print(" 6人 TT 专家全员轮换 (Homogeneous) 最终结果")
    print("=" * 50)
    print(f"最佳轮换时间: {best['rot']} 秒")
    print(f"极限巡航速度: {best['speed']:.2f} km/h")
    print(f"完赛时间:     {best['time'] / 60:.3f} min")

    # --- 绘图 1: U型敏感度曲线 ---
    plt.figure(figsize=(10, 6))
    x_rot = [row[0] for row in data]
    y_time = [row[1] / 60 for row in data]

    plt.plot(x_rot, y_time, 'o-', color='tab:green', linewidth=2)
    plt.plot(best['rot'], best['time'] / 60, 'r*', markersize=15, label='Optimal')

    plt.title("Figure 1: Sensitivity Analysis (Homogeneous Strategy)\nFinish Time vs. Rotation Interval", fontsize=14)
    plt.xlabel("Rotation Interval (s)", fontsize=12)
    plt.ylabel("Finish Time (min)", fontsize=12)
    plt.grid(True)

    # 添加注释
    if len(x_rot) > 0:
        # 左侧标注
        plt.annotate('High Switching Cost',
                     xy=(x_rot[0], y_time[0]),
                     xytext=(x_rot[0] + 10, y_time[0]),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))
        # 右侧标注
        plt.annotate('Deep Fatigue',
                     xy=(x_rot[-1], y_time[-1]),
                     xytext=(x_rot[-1] - 20, y_time[-1]),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.legend()
    plt.show()

    # --- 绘图 2: 最佳状态下的能量图 ---
    print("\n生成能量动态图...")
    _, final_log = run_homogeneous_simulation(team, best['speed'], RACE_DIST, best['rot'])

    plt.figure(figsize=(12, 6))
    for r in team:
        # 6条线会非常对称地交织在一起
        plt.plot(final_log[r.id], label=r.id, linewidth=1.5, alpha=0.7)

    plt.title(f"Figure 2: Homogeneous Energy Dynamics (Rot={best['rot']}s, Speed={best['speed']:.2f} km/h)",
              fontsize=14)
    plt.ylabel("W' Balance (J)", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.axhline(0, color='black', linestyle=':')
    plt.legend(loc='upper right', ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()