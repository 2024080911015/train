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
        self.rotations_completed = 0

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
# 3. 仿真引擎 (修改返回值)
# ==========================================
def run_simulation(team, target_speed_kmh, race_dist_m, rotation_time_sec, max_sprinter_rotations):
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

    # 用于记录数据
    sprinter_w_final = 0  # 记录 Sprinter 掉队时的剩余体力

    while current_dist < race_dist_m:
        # A. 掉队逻辑
        if len(formation) > 4:
            leader = formation[0]
            if leader.role == "Domestique" and leader.w_bal <= 0:
                # Sprinter 耗尽体力掉队
                formation.pop(0)
                pull_timer = 0
        elif len(formation) == 4:
            if any(r.w_bal <= 0 for r in formation):
                return None, 0  # 失败
        else:
            return None, 0

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

        # C. 轮换逻辑
        pull_timer += 1
        leader = formation[0]

        limit = rotation_time_sec

        if leader.role == "Domestique":
            if leader.rotations_completed < max_sprinter_rotations:
                pass  # 正常轮换
            else:
                limit = 99999  # 死士模式

        if pull_timer >= limit:
            rider_to_move = formation.pop(0)
            rider_to_move.rotations_completed += 1
            formation.append(rider_to_move)
            pull_timer = 0

        current_dist += target_v * dt
        current_time += dt

    # === 新增：计算 Sprinter 平均实际完成轮换数 ===
    # 只要是活着做完的轮换都算
    sprinters = [r for r in team if r.role == "Domestique"]
    avg_actual_rotations = sum(r.rotations_completed for r in sprinters) / len(sprinters)

    return current_time, avg_actual_rotations


# ==========================================
# 4. 实验与分析
# ==========================================
def experiment_with_w_prime(team, race_dist):
    print("开始分析：战术目标 (Target) vs 生理极限 (Actual)...")

    results = []
    # 扫描 N=0 到 N=10
    target_rotations = range(0, 11)

    for n_target in target_rotations:
        low, high = 48.0, 60.0
        best_time = None
        best_actual_rot = 0

        # 二分查找
        for _ in range(12):
            mid = (low + high) / 2
            t, actual_rot = run_simulation(team, mid, race_dist, 30, n_target)

            if t is not None:
                best_time = t
                best_actual_rot = actual_rot
                low = mid
            else:
                high = mid

        if best_time:
            # 记录：目标N，完赛时间，实际完成轮换数
            results.append((n_target, best_time, best_actual_rot))
            print(f"Target N={n_target} -> Actual Rot={best_actual_rot:.1f} -> Time={best_time / 60:.2f} min")

    return results


if __name__ == "__main__":
    team = [Rider(f"Sprinter_{i + 1}", STATS_SPRINTER, "Domestique") for i in range(2)] + \
           [Rider(f"TT_{i + 1}", STATS_TT, "Finisher") for i in range(4)]

    RACE_DIST = 44200

    data = experiment_with_w_prime(team, RACE_DIST)

    # === 双轴绘图 ===
    n_targets = [row[0] for row in data]
    finish_times = [row[1] / 60 for row in data]
    actual_rots = [row[2] for row in data]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴：完赛时间 (Orange)
    color1 = 'tab:orange'
    ax1.set_xlabel('Tactical Target (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Finish Time (min)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(n_targets, finish_times, 'o-', color=color1, linewidth=3, label='Finish Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    # 右轴：实际完成轮换 (Blue) - 这代表了生理极限
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Actual Rotations Completed', color=color2, fontsize=12, fontweight='bold')

    # 画参考线 y=x (理想执行)
    ax2.plot(n_targets, n_targets, 'k:', alpha=0.3, label='Ideal Execution (y=x)')

    # 画实际数据
    line2 = ax2.plot(n_targets, actual_rots, 's--', color=color2, linewidth=3, label='Physiological Reality')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 关键标注：生理极限点
    # 找到第一个“指令 > 实际”的点
    for i, (target, actual) in enumerate(zip(n_targets, actual_rots)):
        if target > actual + 0.5:
            plt.annotate('Physiological Limit\n(W\' Depleted)',
                         xy=(target, actual),
                         xytext=(target + 1.5, actual - 1.5),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

            # 同时也标注 W' 耗尽
            plt.scatter([target], [actual], color='red', s=100, zorder=10)
            break

    plt.title("Constraint Analysis: Tactical Target vs. Physiological Reality", fontsize=14)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.tight_layout()
    plt.show()