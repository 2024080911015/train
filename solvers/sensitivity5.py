"""
sensitivity5.py - 团队策略敏感性分析

整合以下策略的扰动分析：
1. 6 TT No-Drop (6TTNoDrop.py) - 6名TT专家同质化策略
2. Sacrifice (sacrifice.py) - 4 TT + 2 Sprinter 牺牲策略
3. Optimal Sprinter Rotations (when.py) - 最优 Sprinter 轮换次数

使用蒙特卡洛方法对速度执行进行扰动，分析策略稳定性
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# === 路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ==========================================
# 1. 物理参数与车手类
# ==========================================
DYNAMIC_DRAG_DATA = {
    6: [0.975, 0.616, 0.497, 0.443, 0.426, 0.433],
    5: [0.975, 0.617, 0.498, 0.447, 0.443],
    4: [0.976, 0.618, 0.502, 0.466]
}
DRAG_COEFFS_6 = [0.975, 0.616, 0.497, 0.443, 0.426, 0.433]
SWITCH_PENALTY_DRAG = 0.2

# TT 专家参数
STATS_TT = {'mass': 72, 'cp': 400, 'w_prime': 20000, 'cda': 0.23}
# Sprinter 参数
STATS_SPRINTER = {'mass': 75, 'cp': 340, 'w_prime': 35000, 'cda': 0.28}

# Sprinter 轮换限制（sacrifice 策略）
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

    def copy(self):
        r = Rider(self.id, {'mass': self.mass, 'cp': self.cp, 
                           'w_prime': self.w_prime_max, 'cda': self.cda}, self.role)
        return r


# ==========================================
# 2. 物理函数
# ==========================================
def get_drag_factor_dynamic(pos_idx, current_team_size):
    coeffs = DYNAMIC_DRAG_DATA.get(current_team_size, DYNAMIC_DRAG_DATA[4])
    if pos_idx < len(coeffs): 
        return coeffs[pos_idx]
    return coeffs[-1]


def get_drag_factor_homogeneous(pos_idx):
    if pos_idx < len(DRAG_COEFFS_6):
        return DRAG_COEFFS_6[pos_idx]
    return 0.433


def skiba_recovery(w_bal, w_max, cp, p_current, dt):
    if p_current > cp:
        new_w = w_bal - (p_current - cp) * dt
        return max(0, new_w)
    d_cp = cp - p_current
    tau = 546 * math.exp(-0.01 * max(-200, d_cp)) + 316
    new_w = w_max - (w_max - w_bal) * math.exp(-dt / tau)
    return min(new_w, w_max)


def add_noise_to_speed(target_speed_kmh, sigma=0.05):
    """给目标速度添加随机噪声"""
    noise = np.random.normal(0, sigma)
    noisy_speed = target_speed_kmh * (1 + noise)
    return max(noisy_speed, 10.0)


# ==========================================
# 3. 仿真引擎：6 TT No-Drop 策略
# ==========================================
def run_6tt_nodrop_simulation(team, target_speed_kmh, race_dist_m, rotation_time_sec, sigma=0.0):
    """6 TT 专家同质化策略仿真（带噪声）"""
    for r in team: 
        r.reset()

    rho = 1.225
    g = 9.81
    crr = 0.004
    eta = 0.98
    dt = 1.0

    current_dist = 0
    current_time = 0
    formation = team.copy()
    pull_timer = 0
    
    # 记录每秒的累计时间（用于计算分段时间）
    time_at_distance = []
    distance_points = []

    while current_dist < race_dist_m:
        # 失败判定
        if any(r.w_bal <= 0 for r in formation):
            return None, None, None

        # 带噪声的速度
        if sigma > 0:
            actual_speed_kmh = add_noise_to_speed(target_speed_kmh, sigma)
        else:
            actual_speed_kmh = target_speed_kmh
        actual_v = actual_speed_kmh / 3.6

        # 物理计算
        is_switching = (pull_timer == 0) and (current_time > 0)

        for pos, rider in enumerate(formation):
            alpha = get_drag_factor_homogeneous(pos)
            if is_switching and pos <= 1: 
                alpha += SWITCH_PENALTY_DRAG

            f_drag = 0.5 * rho * (rider.cda * alpha) * (actual_v ** 2)
            f_roll = rider.total_mass * g * crr
            p_req = ((f_drag + f_roll) * actual_v) / eta

            rider.w_bal = skiba_recovery(rider.w_bal, rider.w_prime_max, rider.cp, p_req, dt)

        # 轮换逻辑
        pull_timer += 1
        if pull_timer >= rotation_time_sec:
            formation.append(formation.pop(0))
            pull_timer = 0

        current_dist += actual_v * dt
        current_time += dt
        
        # 每1km记录一次
        if len(distance_points) == 0 or current_dist - distance_points[-1] >= 1000:
            distance_points.append(current_dist)
            time_at_distance.append(current_time)

    return current_time, np.array(distance_points), np.array(time_at_distance)


# ==========================================
# 4. 仿真引擎：Sacrifice 策略 (4 TT + 2 Sprinter)
# ==========================================
def run_sacrifice_simulation(team, target_speed_kmh, race_dist_m, rotation_time_sec, sigma=0.0):
    """牺牲策略仿真（带噪声）"""
    for r in team: 
        r.reset()

    rho = 1.225
    g = 9.81
    crr = 0.004
    eta = 0.98
    dt = 1.0

    current_dist = 0
    current_time = 0
    formation = team.copy()
    pull_timer = 0
    
    time_at_distance = []
    distance_points = []

    while current_dist < race_dist_m:
        # 掉队检查
        if len(formation) > 4:
            leader = formation[0]
            if leader.w_bal <= 0 and leader.role == "Domestique":
                formation.pop(0)
                pull_timer = 0
        elif len(formation) == 4:
            if any(r.w_bal <= 0 for r in formation): 
                return None, None, None
        else:
            return None, None, None

        # 带噪声的速度
        if sigma > 0:
            actual_speed_kmh = add_noise_to_speed(target_speed_kmh, sigma)
        else:
            actual_speed_kmh = target_speed_kmh
        actual_v = actual_speed_kmh / 3.6

        # 物理计算
        current_size = len(formation)
        is_switching = (pull_timer == 0) and (current_time > 0)

        for pos, rider in enumerate(formation):
            alpha = get_drag_factor_dynamic(pos, current_size)
            if is_switching and pos <= 1: 
                alpha += SWITCH_PENALTY_DRAG

            f_drag = 0.5 * rho * (rider.cda * alpha) * (actual_v ** 2)
            f_roll = rider.total_mass * g * crr
            p_req = ((f_drag + f_roll) * actual_v) / eta

            rider.w_bal = skiba_recovery(rider.w_bal, rider.w_prime_max, rider.cp, p_req, dt)

        # 轮换逻辑
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

        current_dist += actual_v * dt
        current_time += dt
        
        if len(distance_points) == 0 or current_dist - distance_points[-1] >= 1000:
            distance_points.append(current_dist)
            time_at_distance.append(current_time)

    return current_time, np.array(distance_points), np.array(time_at_distance)


# ==========================================
# 5. 找最优速度（二分查找）
# ==========================================
def find_optimal_speed_6tt(rotation_time, race_dist):
    """为 6 TT No-Drop 策略找极限速度"""
    low, high = 50.0, 70.0
    best_speed = 0
    
    for _ in range(15):
        mid = (low + high) / 2
        team = [Rider(f"TT_{j+1}", STATS_TT, "Finisher") for j in range(6)]
        t, _, _ = run_6tt_nodrop_simulation(team, mid, race_dist, rotation_time, sigma=0.0)
        
        if t is not None:
            best_speed = mid
            low = mid
        else:
            high = mid
    
    return best_speed


def find_optimal_speed_sacrifice(rotation_time, race_dist):
    """为 Sacrifice 策略找极限速度"""
    low, high = 48.0, 60.0
    best_speed = 0
    
    for _ in range(15):
        mid = (low + high) / 2
        team = [Rider(f"Sprinter_{j+1}", STATS_SPRINTER, "Domestique") for j in range(2)] + \
               [Rider(f"TT_{j+1}", STATS_TT, "Finisher") for j in range(4)]
        t, _, _ = run_sacrifice_simulation(team, mid, race_dist, rotation_time, sigma=0.0)
        
        if t is not None:
            best_speed = mid
            low = mid
        else:
            high = mid
    
    return best_speed


# ==========================================
# 6. 蒙特卡洛模拟
# ==========================================
def run_monte_carlo_6tt(rotation_time, target_speed, race_dist, sigma=0.05, num_sims=100):
    """对 6 TT No-Drop 策略进行蒙特卡洛模拟"""
    finish_times = []
    dnf_count = 0
    split_times_list = []
    distances = None

    for i in range(num_sims):
        team = [Rider(f"TT_{j+1}", STATS_TT, "Finisher") for j in range(6)]
        t, dist, times = run_6tt_nodrop_simulation(team, target_speed, race_dist, rotation_time, sigma)

        if t is not None:
            finish_times.append(t)
            if dist is not None and len(dist) > 0:
                split_times_list.append(times)
                if distances is None:
                    distances = dist
        else:
            dnf_count += 1

    return finish_times, dnf_count, split_times_list, distances


def run_monte_carlo_sacrifice(rotation_time, target_speed, race_dist, sigma=0.05, num_sims=100):
    """对 Sacrifice 策略进行蒙特卡洛模拟"""
    finish_times = []
    dnf_count = 0
    split_times_list = []
    distances = None

    for i in range(num_sims):
        team = [Rider(f"Sprinter_{j+1}", STATS_SPRINTER, "Domestique") for j in range(2)] + \
               [Rider(f"TT_{j+1}", STATS_TT, "Finisher") for j in range(4)]
        t, dist, times = run_sacrifice_simulation(team, target_speed, race_dist, rotation_time, sigma)

        if t is not None:
            finish_times.append(t)
            if dist is not None and len(dist) > 0:
                split_times_list.append(times)
                if distances is None:
                    distances = dist
        else:
            dnf_count += 1

    return finish_times, dnf_count, split_times_list, distances


# ==========================================
# 7. 策略敏感性分析
# ==========================================
def analyze_strategy_sensitivity(strategy_name, run_monte_carlo_func, rotation_time, target_speed, race_dist):
    """
    对单个策略进行敏感性分析
    """
    print(f"\n=== Analyzing Sensitivity for: {strategy_name} ===")
    print(f"    Rotation: {rotation_time}s, Target Speed: {target_speed:.2f} km/h")
    
    sigma_levels = [0.02, 0.05, 0.10]
    num_sims = 200
    results_map = {}
    band_distances = None
    band_split_map = {}

    for sigma in sigma_levels:
        print(f"  > Running {num_sims} simulations with sigma={sigma * 100}%...")
        
        finish_times, dnf_count, split_times, distances = run_monte_carlo_func(
            rotation_time, target_speed, race_dist, sigma, num_sims
        )

        dnf_rate = (dnf_count / num_sims) * 100
        
        if len(finish_times) > 0:
            finish_times = np.array(finish_times)
            mean_time = np.mean(finish_times)
            std_time = np.std(finish_times)
        else:
            mean_time = float('nan')
            std_time = float('nan')

        results_map[f"sigma_{sigma}"] = {
            'finish_times': finish_times if len(finish_times) > 0 else np.array([]),
            'dnf_rate': dnf_rate
        }

        if len(split_times) > 0 and distances is not None:
            # 对齐所有 split_times 到相同长度
            min_len = min(len(st) for st in split_times)
            aligned_splits = [st[:min_len] for st in split_times]
            band_split_map[sigma] = aligned_splits
            band_distances = distances[:min_len]

        print(f"    Sigma {sigma*100:.0f}% -> Mean: {mean_time/60:.2f}m, Std: {std_time:.1f}s, DNF: {dnf_rate:.1f}%")

    return {
        'results_map': results_map,
        'band_distances': band_distances,
        'band_split_map': band_split_map
    }


# ==========================================
# 8. 绘图函数（与 sensitivity.py 格式一致）
# ==========================================
def plot_combined_box_and_gap(strategy_results, sigma_levels):
    """
    生成一张图，包含：
    - 左：两个箱线图（上下排列），分别绘制两种策略
    - 右：三个上下排列的 Gap 置信带子图
    """
    strategies = list(strategy_results.keys())
    colors = {
        strategies[0]: '#1f77b4',
        strategies[1]: '#d62728'
    }

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)
    
    # 左侧两个子图
    ax_box_1 = fig.add_subplot(gs[0:2, 0])
    ax_box_2 = fig.add_subplot(gs[2, 0])
    ax_gaps = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    # ===== 左上图：策略1 的 Boxplot =====
    strategy_1 = strategies[0]
    positions_1 = []
    data_1 = []
    for i, sigma in enumerate(sigma_levels):
        key = f"sigma_{sigma}"
        res = strategy_results[strategy_1]['results_map'].get(key, {})
        ft = res.get('finish_times', np.array([]))
        data_1.append(ft / 60.0 if len(ft) > 0 else np.array([]))
        positions_1.append(i)

    bp_1 = ax_box_1.boxplot(data_1, positions=positions_1, widths=0.5, patch_artist=True,
                            medianprops=dict(color='black', linewidth=1.5),
                            boxprops=dict(linewidth=1.2, facecolor=colors[strategy_1], alpha=0.6),
                            whiskerprops=dict(linewidth=1),
                            capprops=dict(linewidth=1))

    ax_box_1.set_ylabel("Finish Time (minutes)", fontsize=11)
    ax_box_1.set_title(f"Boxplot: {strategy_1}", fontsize=12, fontweight='bold')
    ax_box_1.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax_box_1.set_xticks(range(len(sigma_levels)))
    ax_box_1.set_xticklabels([f"{int(s * 100)}%" for s in sigma_levels], fontsize=10)

    # ===== 左下图：策略2 的 Boxplot =====
    strategy_2 = strategies[1]
    positions_2 = []
    data_2 = []
    for i, sigma in enumerate(sigma_levels):
        key = f"sigma_{sigma}"
        res = strategy_results[strategy_2]['results_map'].get(key, {})
        ft = res.get('finish_times', np.array([]))
        data_2.append(ft / 60.0 if len(ft) > 0 else np.array([]))
        positions_2.append(i)

    bp_2 = ax_box_2.boxplot(data_2, positions=positions_2, widths=0.5, patch_artist=True,
                            medianprops=dict(color='black', linewidth=1.5),
                            boxprops=dict(linewidth=1.2, facecolor=colors[strategy_2], alpha=0.6),
                            whiskerprops=dict(linewidth=1),
                            capprops=dict(linewidth=1))

    ax_box_2.set_ylabel("Finish Time (minutes)", fontsize=11)
    ax_box_2.set_xlabel("Execution Error (Sigma)", fontsize=11)
    ax_box_2.set_title(f"Boxplot: {strategy_2}", fontsize=12, fontweight='bold')
    ax_box_2.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax_box_2.set_xticks(range(len(sigma_levels)))
    ax_box_2.set_xticklabels([f"{int(s * 100)}%" for s in sigma_levels], fontsize=10)

    # ===== 右侧：三个 Gap band 子图，统一 Y 轴范围 =====
    global_min = 0
    global_max = 0
    
    for sigma in sigma_levels:
        for strategy_name in strategies:
            info = strategy_results[strategy_name]
            band_map = info.get('band_split_map', {})
            if sigma in band_map and len(band_map[sigma]) > 0:
                split_mat = np.vstack(band_map[sigma])
                mean_curve = np.mean(split_mat, axis=0)
                diff_mat = split_mat - mean_curve
                low_gap = np.percentile(diff_mat, 5, axis=0)
                high_gap = np.percentile(diff_mat, 95, axis=0)
                global_min = min(global_min, np.min(low_gap))
                global_max = max(global_max, np.max(high_gap))
    
    y_margin = (global_max - global_min) * 0.1 + 1
    ylim_min = global_min - y_margin
    ylim_max = global_max + y_margin
    
    for idx, sigma in enumerate(sigma_levels):
        ax = ax_gaps[idx]
        distances = None
        
        for strategy_name in strategies:
            info = strategy_results[strategy_name]
            band_map = info.get('band_split_map', {})
            if sigma in band_map and len(band_map[sigma]) > 0:
                if distances is None:
                    distances = info['band_distances'] / 1000.0
                split_mat = np.vstack(band_map[sigma])
                mean_curve = np.mean(split_mat, axis=0)
                diff_mat = split_mat - mean_curve
                low_gap = np.percentile(diff_mat, 5, axis=0)
                high_gap = np.percentile(diff_mat, 95, axis=0)
                ax.fill_between(distances, low_gap, high_gap, color=colors[strategy_name], alpha=0.35,
                                label=f"{strategy_name}")

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel("Gap (s)", fontsize=9)
        ax.set_title(f"Sigma={int(sigma * 100)}%", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(ylim_min, ylim_max)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
        if idx == 2:
            ax.set_xlabel("Distance (km)", fontsize=10)

    plt.suptitle("Team Strategy Sensitivity Analysis: Boxplot & Gap Confidence Bands", 
                fontsize=14, fontweight='bold', y=0.98)
    
    out_path = os.path.join(project_root, 'images', 'Q5_Sensitivity_Combined_Box_Gap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close(fig)


def plot_dnf_rate_comparison(strategy_results, sigma_levels):
    """
    绘制 DNF 率对比图
    """
    strategies = list(strategy_results.keys())
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.35
    x = np.arange(len(sigma_levels))
    
    for i, strategy_name in enumerate(strategies):
        dnf_rates = []
        for sigma in sigma_levels:
            key = f"sigma_{sigma}"
            res = strategy_results[strategy_name]['results_map'].get(key, {})
            dnf_rates.append(res.get('dnf_rate', 0))
        
        ax.bar(x + i * width, dnf_rates, width, label=strategy_name, color=colors[i], alpha=0.7)
    
    ax.set_ylabel("DNF Rate (%)", fontsize=12)
    ax.set_xlabel("Execution Error (Sigma)", fontsize=12)
    ax.set_title("DNF Rate Comparison by Strategy", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"{int(s * 100)}%" for s in sigma_levels], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    out_path = os.path.join(project_root, 'images', 'Q5_DNF_Rate_Comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_statistics_summary(strategy_results, sigma_levels):
    """
    绘制统计信息汇总表格图
    """
    strategies = list(strategy_results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 构建表格数据
    headers = ["Strategy", "Sigma", "Mean (min)", "Std (s)", "Min (min)", "Max (min)", "DNF (%)"]
    table_data = []
    
    for strategy_name in strategies:
        for sigma in sigma_levels:
            key = f"sigma_{sigma}"
            res = strategy_results[strategy_name]['results_map'].get(key, {})
            ft = res.get('finish_times', np.array([]))
            dnf = res.get('dnf_rate', 0)
            
            if len(ft) > 0:
                mean_t = np.mean(ft) / 60
                std_t = np.std(ft)
                min_t = np.min(ft) / 60
                max_t = np.max(ft) / 60
                table_data.append([
                    strategy_name, f"{int(sigma*100)}%",
                    f"{mean_t:.2f}", f"{std_t:.1f}",
                    f"{min_t:.2f}", f"{max_t:.2f}", f"{dnf:.1f}"
                ])
            else:
                table_data.append([
                    strategy_name, f"{int(sigma*100)}%",
                    "N/A", "N/A", "N/A", "N/A", f"{dnf:.1f}"
                ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表头颜色
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # 交替行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
    
    plt.title("Team Strategy Sensitivity Analysis - Statistics Summary", 
             fontsize=14, fontweight='bold', pad=20)
    
    out_path = os.path.join(project_root, 'images', 'Q5_Statistics_Summary.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ==========================================
# 9. 主程序
# ==========================================
def analyze_sensitivity():
    """
    对所有团队策略进行敏感性分析
    """
    print("=" * 60)
    print("Team Strategy Sensitivity Analysis")
    print("=" * 60)
    
    RACE_DIST = 44200  # Tokyo 赛道
    sigma_levels = [0.02, 0.05, 0.10]
    
    # ===== 策略1：6 TT No-Drop =====
    print("\n[Strategy 1] 6 TT No-Drop")
    rotation_6tt = 45  # 最优轮换时间约 45s
    speed_6tt = find_optimal_speed_6tt(rotation_6tt, RACE_DIST)
    # 使用 97% 的极限速度作为安全余量
    speed_6tt_safe = speed_6tt * 0.97
    print(f"  Optimal Speed: {speed_6tt:.2f} km/h, Using: {speed_6tt_safe:.2f} km/h (97%)")
    
    results_6tt = analyze_strategy_sensitivity(
        "6 TT No-Drop",
        run_monte_carlo_6tt,
        rotation_6tt, speed_6tt_safe, RACE_DIST
    )
    
    # ===== 策略2：Sacrifice (4 TT + 2 Sprinter) =====
    print("\n[Strategy 2] Sacrifice (4 TT + 2 Sprinter)")
    rotation_sacrifice = 30  # 最优轮换时间约 30s
    speed_sacrifice = find_optimal_speed_sacrifice(rotation_sacrifice, RACE_DIST)
    speed_sacrifice_safe = speed_sacrifice * 0.97
    print(f"  Optimal Speed: {speed_sacrifice:.2f} km/h, Using: {speed_sacrifice_safe:.2f} km/h (97%)")
    
    results_sacrifice = analyze_strategy_sensitivity(
        "Sacrifice (4TT+2Sp)",
        run_monte_carlo_sacrifice,
        rotation_sacrifice, speed_sacrifice_safe, RACE_DIST
    )
    
    # ===== 绘图 =====
    print("\n" + "=" * 60)
    print("Generating Plots...")
    print("=" * 60)
    
    strategy_results = {
        "6 TT No-Drop": results_6tt,
        "Sacrifice (4TT+2Sp)": results_sacrifice
    }
    
    # 绘制组合图（箱线图 + Gap 置信带）
    plot_combined_box_and_gap(strategy_results, sigma_levels)
    
    # 绘制 DNF 率对比图
    plot_dnf_rate_comparison(strategy_results, sigma_levels)
    
    # 绘制统计汇总表格
    plot_statistics_summary(strategy_results, sigma_levels)
    
    print("\n" + "=" * 60)
    print("Team Strategy Sensitivity Analysis Completed!")
    print("=" * 60)


if __name__ == "__main__":
    analyze_sensitivity()
