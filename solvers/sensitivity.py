import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# === 路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.cyclist import Cyclist
from models.simulator import WPrimeBalanceSimulator
from models.optimizer import generate_smart_strategy, run_simulated_annealing
from solvers.question_2 import load_custom_course_from_csv, load_real_course, get_all_riders


def add_noise_to_strategy(strategy, sigma=0.05):
    """
    给策略添加随机噪声，模拟执行偏差
    P_actual = P_target * (1 + N(0, sigma))
    """
    noise = np.random.normal(0, sigma, len(strategy))
    # 限制噪声幅度，防止出现负功率或离谱的数值
    noisy_strategy = strategy * (1 + noise)
    noisy_strategy = np.maximum(noisy_strategy, 0)  # 功率不能为负
    return noisy_strategy


def simulate_cumulative_times(sim, power_strategy, course_data, wind_speed=0.0):
    """
    使用与 WPrimeBalanceSimulator 相同的物理逻辑，返回每个赛段结束时的累计用时数组。
    """
    current_w = sim.cyclist.w_prime
    total_time = 0.0
    v_curr = 0.1
    is_exhausted = False
    DX_STEP = 50.0

    cumulative_times = [0.0]

    for i, segment in enumerate(course_data):
        p_target = power_strategy[i] if i < len(power_strategy) else power_strategy[-1]
        if is_exhausted:
            p_target = sim.cyclist.cp * 0.5

        seg_len = segment['length']
        slope = segment['slope']
        radius = segment['radius']

        dist_covered = 0.0
        while dist_covered < seg_len:
            dx = min(DX_STEP, seg_len - dist_covered)

            v_air = v_curr + wind_speed
            f_drag = 0.5 * sim.rho * sim.cd_a * (v_air ** 2)
            f_grav = sim.total_mass * sim.g * np.sin(slope)
            f_roll = sim.total_mass * sim.g * sim.mu_roll * np.cos(slope)
            f_resist = f_drag + f_grav + f_roll

            if v_curr < 0.1:
                v_curr = 0.1
            f_prop = p_target / v_curr
            acc = (f_prop - f_resist) / sim.total_mass

            v_next_sq = v_curr ** 2 + 2 * acc * dx
            v_curr = np.sqrt(v_next_sq) if v_next_sq > 0.1 else 0.1

            if radius < 100:
                v_limit = np.sqrt(sim.mu_tire * sim.g * radius)
                v_curr = min(v_curr, v_limit)

            dt = dx / v_curr
            total_time += dt

            if p_target > sim.cyclist.cp:
                current_w -= (p_target - sim.cyclist.cp) * dt
            else:
                rec_rate = (sim.cyclist.cp - p_target)
                current_w += rec_rate * dt * 0.5

            if current_w > sim.cyclist.w_prime:
                current_w = sim.cyclist.w_prime
            if current_w < 0:
                current_w = 0
                is_exhausted = True
                p_target = sim.cyclist.cp * 0.5

            dist_covered += dx

        cumulative_times.append(total_time)

    return np.array(cumulative_times), is_exhausted


def run_monte_carlo(rider, course_data, base_strategy, sigma=0.05, num_sims=100, wind_speed=0, collect_split_times=False):
    """
    运行蒙特卡洛模拟
    返回：(results, finish_times, dnf_times, dnf_count, split_times, distances)
    - finish_times: 仅包含未力竭选手的完赛时间
    - dnf_times: 力竭选手的时间（用于分析）
    - split_times: (可选) 每次仿真的累计计时轨迹列表，仅收集未力竭选手
    """
    sim = WPrimeBalanceSimulator(rider)
    finish_times = []  # 未力竭选手的时间
    dnf_times = []     # 力竭选手的时间
    dnf_count = 0
    results = []
    split_times = []
    distances = None

    if collect_split_times:
        dist_points = [0.0]
        for seg in course_data:
            dist_points.append(dist_points[-1] + seg['length'])
        distances = np.array(dist_points)

    print(f"  > Running {num_sims} simulations for {rider.name} with sigma={sigma * 100}% and wind={wind_speed:.1f}m/s...")

    for i in range(num_sims):
        # 1. 生成带噪声的实际执行策略
        noisy_strategy = add_noise_to_strategy(base_strategy, sigma)

        # 2. 运行仿真，必要时收集分段累计时间
        cum_times, is_exhausted = simulate_cumulative_times(sim, noisy_strategy, course_data, wind_speed=wind_speed)
        total_time = float(cum_times[-1])

        if is_exhausted:
            dnf_count += 1
            dnf_times.append(total_time)
        else:
            finish_times.append(total_time)
            if collect_split_times:
                split_times.append(cum_times)

        results.append({
            'sim_id': i,
            'time': total_time,
            'exhausted': is_exhausted,
            'strategy': noisy_strategy
        })

    return results, finish_times, dnf_times, dnf_count, split_times, distances


def plot_combined_box_and_gap(rider_results, sigma_levels, gap_sigma=0.05):
    """生成一张图，包含：
    - 左：两个箱线图（上下排列），分别绘制 TT 和 Sprinter 的不同 sigma 下的箱线图；
    - 右：三个上下排列的 Gap 置信带子图，每个对应一个 sigma，TT 与 Sprinter 用不同颜色阴影，且 Y 轴范围统一；
    左右排布，便于可视化。
    """
    riders = list(rider_results.keys())
    colors = {
        riders[0]: '#1f77b4',
        riders[1]: '#d62728'
    }

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)
    
    # 左侧两个子图：上半部分 TT，下半部分 Sprinter
    ax_box_tt = fig.add_subplot(gs[0:2, 0])  # 左侧上2/3
    ax_box_sprinter = fig.add_subplot(gs[2, 0])  # 左侧下1/3
    ax_gaps = [fig.add_subplot(gs[i, 1]) for i in range(3)]  # 右侧三个子图

    # ===== 左上图：TT 骑手的 Boxplot =====
    tt_rider = riders[0]
    positions_tt = []
    data_tt = []
    for i, sigma in enumerate(sigma_levels):
        key = f"sigma_{sigma}"
        res = rider_results[tt_rider]['results_map'].get(key, {})
        ft = res.get('finish_times', np.array([]))
        data_tt.append(ft / 60.0 if len(ft) > 0 else np.array([]))
        positions_tt.append(i)

    bp_tt = ax_box_tt.boxplot(data_tt, positions=positions_tt, widths=0.5, patch_artist=True,
                              medianprops=dict(color='black', linewidth=1.5),
                              boxprops=dict(linewidth=1.2, facecolor=colors[tt_rider], alpha=0.6),
                              whiskerprops=dict(linewidth=1),
                              capprops=dict(linewidth=1))

    ax_box_tt.set_ylabel("Finish Time (minutes)", fontsize=11)
    ax_box_tt.set_title(f"Boxplot: {tt_rider}", fontsize=12, fontweight='bold')
    ax_box_tt.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax_box_tt.set_xticks(range(len(sigma_levels)))
    ax_box_tt.set_xticklabels([f"{int(s * 100)}%" for s in sigma_levels], fontsize=10)

    # ===== 左下图：Sprinter 骑手的 Boxplot =====
    sprinter_rider = riders[1]
    positions_sprinter = []
    data_sprinter = []
    for i, sigma in enumerate(sigma_levels):
        key = f"sigma_{sigma}"
        res = rider_results[sprinter_rider]['results_map'].get(key, {})
        ft = res.get('finish_times', np.array([]))
        data_sprinter.append(ft / 60.0 if len(ft) > 0 else np.array([]))
        positions_sprinter.append(i)

    bp_sprinter = ax_box_sprinter.boxplot(data_sprinter, positions=positions_sprinter, widths=0.5, patch_artist=True,
                                          medianprops=dict(color='black', linewidth=1.5),
                                          boxprops=dict(linewidth=1.2, facecolor=colors[sprinter_rider], alpha=0.6),
                                          whiskerprops=dict(linewidth=1),
                                          capprops=dict(linewidth=1))

    ax_box_sprinter.set_ylabel("Finish Time (minutes)", fontsize=11)
    ax_box_sprinter.set_xlabel("Execution Error (Sigma)", fontsize=11)
    ax_box_sprinter.set_title(f"Boxplot: {sprinter_rider}", fontsize=12, fontweight='bold')
    ax_box_sprinter.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax_box_sprinter.set_xticks(range(len(sigma_levels)))
    ax_box_sprinter.set_xticklabels([f"{int(s * 100)}%" for s in sigma_levels], fontsize=10)

    # ===== 右侧：三个 Gap band 子图 (每个 sigma 一个)，统一 Y 轴范围 =====
    # 先计算所有 gap 的全局最小/最大值
    global_min = 0
    global_max = 0
    
    for sigma in sigma_levels:
        for rider_name in riders:
            info = rider_results[rider_name]
            band_map = info['band_split_map']
            if sigma in band_map and len(band_map[sigma]) > 0:
                split_mat = np.vstack(band_map[sigma])
                mean_curve = np.mean(split_mat, axis=0)
                diff_mat = split_mat - mean_curve
                low_gap = np.percentile(diff_mat, 5, axis=0)
                high_gap = np.percentile(diff_mat, 95, axis=0)
                global_min = min(global_min, np.min(low_gap))
                global_max = max(global_max, np.max(high_gap))
    
    # 为 Y 轴范围留出一些余量
    y_margin = (global_max - global_min) * 0.1
    ylim_min = global_min - y_margin
    ylim_max = global_max + y_margin
    
    for idx, sigma in enumerate(sigma_levels):
        ax = ax_gaps[idx]
        distances = None
        
        for rider_name in riders:
            info = rider_results[rider_name]
            band_map = info['band_split_map']
            if sigma in band_map and len(band_map[sigma]) > 0:
                if distances is None:
                    distances = info['band_distances'] / 1000.0
                split_mat = np.vstack(band_map[sigma])
                mean_curve = np.mean(split_mat, axis=0)
                diff_mat = split_mat - mean_curve
                low_gap = np.percentile(diff_mat, 5, axis=0)
                high_gap = np.percentile(diff_mat, 95, axis=0)
                ax.fill_between(distances, low_gap, high_gap, color=colors[rider_name], alpha=0.35,
                                label=f"{rider_name}")

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel("Gap (s)", fontsize=9)
        ax.set_title(f"Sigma={int(sigma * 100)}%", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(ylim_min, ylim_max)  # 统一 Y 轴范围
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
        if idx == 2:
            ax.set_xlabel("Distance (km)", fontsize=10)

    plt.suptitle("Sensitivity Analysis: Boxplot & Gap Confidence Bands", fontsize=14, fontweight='bold', y=0.98)
    
    out_path = os.path.join(project_root, 'images', 'Q3_Sensitivity_Combined_Box_Gap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_split_time_band(distances, split_times, rider_name, sigma):
    """
    绘制分段计时置信带：展示在给定误差水平下，每个计时点的时间分布。
    """
    if distances is None or len(split_times) == 0:
        print("[Info] No split-time data to plot.")
        return

    dist_km = distances / 1000.0
    split_mat = np.vstack(split_times)  # shape: (n_sims, n_points)

    mean_t = np.mean(split_mat, axis=0) / 60.0  # 转为分钟
    low_t = np.percentile(split_mat, 5, axis=0) / 60.0
    high_t = np.percentile(split_mat, 95, axis=0) / 60.0

    plt.figure(figsize=(12, 6))
    plt.plot(dist_km, mean_t, color='red', label='Mean Split Time')
    plt.fill_between(dist_km, low_t, high_t, color='red', alpha=0.2, label='90% Confidence Band (5-95%)')

    plt.title(f"Split Time Confidence Band | {rider_name} | Sigma={int(sigma*100)}%", fontsize=12)
    plt.xlabel("Distance (km)")
    plt.ylabel("Elapsed Time (minutes)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    sanitize_name = rider_name.replace(" ", "_")
    out_path = os.path.join(project_root, 'images', f'Q3_Split_Band_{sanitize_name}_sigma{int(sigma*100)}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def plot_split_time_gap_band(distances, split_times, rider_name, sigma):
    """
    绘制相对均值的时间差置信带（Gap Plot），更直观展示秒级波动。
    """
    if distances is None or len(split_times) == 0:
        print("[Info] No split-time data to plot (gap).")
        return

    dist_km = distances / 1000.0
    split_mat = np.vstack(split_times)  # seconds

    mean_curve = np.mean(split_mat, axis=0)
    diff_mat = split_mat - mean_curve  # seconds

    low_gap = np.percentile(diff_mat, 5, axis=0)
    high_gap = np.percentile(diff_mat, 95, axis=0)

    plt.figure(figsize=(12, 6))
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Mean Reference')
    plt.fill_between(dist_km, low_gap, high_gap, color='red', alpha=0.25, label='90% Confidence Band (Gap)')
    plt.plot(dist_km, np.zeros_like(dist_km), color='red', linewidth=1.5)

    plt.title(f"Split Time Gap Confidence Band | {rider_name} | Sigma={int(sigma*100)}%", fontsize=12)
    plt.xlabel("Distance (km)")
    plt.ylabel("Time Gap vs Mean (seconds)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    sanitize_name = rider_name.replace(" ", "_")
    out_path = os.path.join(project_root, 'images', f'Q3_Split_GapBand_{sanitize_name}_sigma{int(sigma*100)}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def plot_split_time_gap_band_multi(distances, split_times_map, rider_name, sigmas):
    """
    将多个 sigma 的 gap 置信带绘制在一张图里（上下三个子图）。
    """
    if distances is None or not split_times_map:
        print("[Info] No split-time data to plot (gap multi).")
        return

    dist_km = distances / 1000.0
    fig, axes = plt.subplots(len(sigmas), 1, figsize=(12, 10), sharex=True)

    # 预先计算全局 y 轴范围，保证三个子图比例一致
    global_min = np.inf
    global_max = -np.inf
    cache = {}
    for sigma in sigmas:
        st = split_times_map.get(sigma, [])
        if st:
            split_mat = np.vstack(st)
            diff_mat = split_mat - np.mean(split_mat, axis=0)
            low_gap = np.percentile(diff_mat, 5, axis=0)
            high_gap = np.percentile(diff_mat, 95, axis=0)
            cache[sigma] = (low_gap, high_gap)
            global_min = min(global_min, low_gap.min())
            global_max = max(global_max, high_gap.max())
        else:
            cache[sigma] = None

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        print("[Info] No split-time data to plot (gap multi).")
        return

    # 对称并留一点边距
    max_abs = max(abs(global_min), abs(global_max))
    padding = max_abs * 0.05 + 0.5  # 至少 0.5s 边距
    y_min, y_max = -max_abs - padding, max_abs + padding

    for idx, sigma in enumerate(sigmas):
        ax = axes[idx] if len(sigmas) > 1 else axes
        data = cache.get(sigma)
        if data is None:
            ax.text(0.5, 0.5, f"No data for sigma={int(sigma*100)}%", ha='center', va='center')
            ax.axis('off')
            continue

        low_gap, high_gap = data
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.fill_between(dist_km, low_gap, high_gap, color='red', alpha=0.25, label='90% Band')
        ax.plot(dist_km, np.zeros_like(dist_km), color='red', linewidth=1.2)

        ax.set_ylabel(f"Gap (s) @ {int(sigma*100)}%")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper left')

    axes[-1].set_xlabel("Distance (km)")
    fig.suptitle(f"Split Time Gap Bands | {rider_name}", fontsize=13, y=1.02)
    plt.tight_layout()

    sanitize_name = rider_name.replace(" ", "_")
    out_path = os.path.join(project_root, 'images', f'Q3_Split_GapBand_Combined_{sanitize_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def analyze_rider_sensitivity(rider, course_data, wind_speed=0.0):
    """
    封装单车手敏感性分析逻辑
    """
    print(f"\n=== Analyzing Sensitivity for: {rider.name} ===")
    
    # 1. 获取（或生成）基准策略
    # 这里我们复用 run_simulated_annealing 来获得在该赛道和风速下的优解
    print(f"  [Step 1] Calculating Optimal Strategy (Wind={wind_speed} m/s)...")
    base_strategy, _ = run_simulated_annealing(rider, course_data, wind_speed=wind_speed)
    
    # 稍微保留余量（安全系数），模拟真实比赛中选手不会顶着极限跑，防止稍微一抖动就爆缸
    base_strategy = base_strategy * 0.97

    # 2. 设定误差水平
    sigma_levels = [0.02, 0.05, 0.10]
    band_sigmas = [0.02, 0.05, 0.10]  # 需要绘制分段计时置信带的误差水平
    results_map = {}
    num_sims = 200
    band_distances = None
    band_split_map = {}

    # 3. 蒙特卡洛模拟循环
    print("  [Step 2] Starting Monte Carlo Simulations...")
    for sigma in sigma_levels:
        collect_band = sigma in band_sigmas
        _, finish_times, dnf_times, dnf_count, split_times, distances = run_monte_carlo(
            rider, course_data, base_strategy, sigma, num_sims=num_sims, wind_speed=wind_speed, collect_split_times=collect_band
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
            'dnf_times': np.array(dnf_times),
            'dnf_rate': dnf_rate
        }

        if collect_band and len(split_times) > 0 and distances is not None:
            band_split_map[sigma] = split_times
            band_distances = distances

        print(f"    Sigma {sigma*100:.0f}% -> Mean: {mean_time/60:.2f}m, DNF: {dnf_rate:.1f}%")

    # 4. 绘图
    return {
        'results_map': results_map,
        'band_distances': band_distances,
        'band_split_map': band_split_map
    }


def analyze_sensitivity():
    print("=== Running Sensitivity Analysis (Two Riders Comparison) ===")

    # 1. 加载东京赛道数据
    course_path = os.path.join(project_root, 'data', 'course_tokyo.csv')
    if not os.path.exists(course_path):
        print("Error: Course data not found.")
        return
    course_data = load_real_course(course_path)
    
    # 设定无风条件
    wind_speed = 0.0
    print(f"  [Condition] Global Wind Speed: {wind_speed:.2f} m/s")

    # 2. 定义两个车手：TT Specialist 和 Sprinter
    rider_tt = Cyclist("Male TT", "TT Specialist", "Male", cp=400, w_prime=20000, mass=72, cd_area=0.23, p_max=1200)
    rider_sprinter = Cyclist("Male Sprinter", "Sprinter", "Male", cp=340, w_prime=35000, mass=75, cd_area=0.28, p_max=1600)

    # 3. 分别分析并收集结果
    res_tt = analyze_rider_sensitivity(rider_tt, course_data, wind_speed)
    res_sp = analyze_rider_sensitivity(rider_sprinter, course_data, wind_speed)

    # 4. 合并输出一张总图
    plot_combined_box_and_gap({rider_tt.name: res_tt, rider_sprinter.name: res_sp}, [0.02, 0.05, 0.10], gap_sigma=0.05)

    print("\nSensitivity Analysis Comparisons Completed.")


if __name__ == "__main__":
    analyze_sensitivity()
