"""
风速影响对比分析：顺风 vs 逆风
生成顺风图片，并将顺风和逆风的曲线画在一张图里
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# === 路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.cyclist import Cyclist
from models.simulator import WPrimeBalanceSimulator
from models.optimizer import run_simulated_annealing


# === 数据加载函数 ===
def load_real_course(csv_path, step_size=100.0):
    if not os.path.exists(csv_path): 
        return None
    df = pd.read_csv(csv_path)
    original_dist = df['distance'].values
    original_elev = df['elevation'].values
    total_len = original_dist[-1]
    new_dist = np.arange(0, total_len, step_size)
    f_interp = interp1d(original_dist, original_elev, kind='linear')
    new_elev = f_interp(new_dist)
    course_data = []
    for i in range(len(new_dist) - 1):
        length = new_dist[i + 1] - new_dist[i]
        elev_diff = new_elev[i + 1] - new_elev[i]
        slope = np.arctan(elev_diff / length)
        course_data.append({'length': length, 'slope': slope, 'radius': 9999.0, 'elevation_start': new_elev[i]})
    return course_data


def get_all_riders():
    return [
        Cyclist("Male TT", "TT Specialist", "Male", cp=400, w_prime=20000, mass=72, cd_area=0.23, p_max=1200),
        Cyclist("Male Sprinter", "Sprinter", "Male", cp=340, w_prime=35000, mass=75, cd_area=0.28, p_max=1600),
        Cyclist("Female TT", "TT Specialist", "Female", cp=260, w_prime=15000, mass=60, cd_area=0.20, p_max=900),
        Cyclist("Female Sprinter", "Sprinter", "Female", cp=220, w_prime=25000, mass=62, cd_area=0.24, p_max=1100)
    ]


RIDER_COLORS = {'Male TT': 'tab:blue', 'Male Sprinter': 'navy', 'Female TT': 'tab:red', 'Female Sprinter': 'darkred'}


def reconstruct_simulation_traces(rider, course_data, power_strategy, wind_speed=0):
    """
    重新运行仿真以获取详细轨迹：功率、速度、时间 vs 距离
    wind_speed: 正值=逆风(增加阻力), 负值=顺风(减少阻力)
    """
    sim = WPrimeBalanceSimulator(rider)
    current_w = sim.cyclist.w_prime
    total_time = 0.0
    v_curr = 0.1
    is_exhausted = False
    DX_STEP = 50.0

    distances = [0.0]
    velocities = [v_curr]
    times = [0.0]
    
    current_dist = 0.0

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
            
            # 物理计算
            # wind_speed > 0: 逆风，增加空气阻力
            # wind_speed < 0: 顺风，减少空气阻力
            v_air = v_curr + wind_speed
            if v_air < 0:
                v_air = 0  # 顺风不能超过车速时，视为零相对风速
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

            # W' 平衡
            if p_target > sim.cyclist.cp:
                loss = (p_target - sim.cyclist.cp) * dt
                current_w -= loss
            else:
                rec_rate = (sim.cyclist.cp - p_target)
                current_w += rec_rate * dt * 0.5
            
            if current_w > sim.cyclist.w_prime: 
                current_w = sim.cyclist.w_prime
            if current_w < 0:
                current_w = 0
                is_exhausted = True

            dist_covered += dx
            
        current_dist += seg_len
        distances.append(current_dist)
        velocities.append(v_curr)
        times.append(total_time)
        
    return np.array(distances), np.array(velocities), np.array(times)


def plot_tailwind_impact(course_name, rider, course_data, data_no_wind, data_tailwind, wind_speed_val):
    """
    生成顺风影响图（类似逆风图）
    """
    dist_nw, vel_nw, time_nw = reconstruct_simulation_traces(
        rider, course_data, data_no_wind['power_strategy'], wind_speed=0)
    dist_tw, vel_tw, time_tw = reconstruct_simulation_traces(
        rider, course_data, data_tailwind['power_strategy'], wind_speed=-abs(wind_speed_val))  # 负值=顺风
    
    p_nw = np.array(data_no_wind['power_strategy'])
    p_tw = np.array(data_tailwind['power_strategy'])
    
    dist_km = dist_nw / 1000.0
    vel_nw_kph = vel_nw * 3.6
    vel_tw_kph = vel_tw * 3.6
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. 功率对比
    ax1.plot(dist_km[:-1], p_nw, label='No Wind Power', color='tab:blue', alpha=0.8, linewidth=2)
    ax1.plot(dist_km[:-1], p_tw, label='Tailwind Power', color='tab:green', alpha=0.8, linestyle='--', linewidth=2)
    ax1.fill_between(dist_km[:-1], p_nw, p_tw, where=(p_tw < p_nw), color='green', alpha=0.15, interpolate=True)
    ax1.fill_between(dist_km[:-1], p_nw, p_tw, where=(p_tw >= p_nw), color='blue', alpha=0.15, interpolate=True)
    ax1.set_ylabel('Power (Watts)', fontsize=11)
    ax1.set_title(f'Rider: {rider.name} - Power Strategy (Tailwind Impact: {abs(wind_speed_val)*3.6:.1f} km/h)', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 速度对比
    ax2.plot(dist_km, vel_nw_kph, label='No Wind Speed', color='tab:blue', linewidth=2)
    ax2.plot(dist_km, vel_tw_kph, label=f'Tailwind Speed ({abs(wind_speed_val)*3.6:.1f} km/h)', color='tab:green', linestyle='--', linewidth=2)
    ax2.fill_between(dist_km, vel_nw_kph, vel_tw_kph, color='lightgreen', alpha=0.3, label='Speed Gain')
    ax2.set_ylabel('Speed (km/h)', fontsize=11)
    ax2.set_title('Speed Profile', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. 时间差累积（0在中间，负值=顺风更快，画在下方）
    time_diff = time_tw - time_nw  # 负值表示顺风更快
    ax3.plot(dist_km, time_diff, color='green', label='Cumulative Time Difference', linewidth=2)
    ax3.fill_between(dist_km, 0, time_diff, color='green', alpha=0.15)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # 设置对称的y轴范围，0在中间
    max_abs_val = max(abs(time_diff.min()), abs(time_diff.max()))
    ax3.set_ylim(-max_abs_val * 1.1, max_abs_val * 1.1)
    
    ax3.set_ylabel('Time Difference (s)\n(Negative = Faster)', fontsize=11)
    ax3.set_xlabel('Distance (km)', fontsize=11)
    ax3.set_title(f'Cumulative Time Difference (Total: {time_diff[-1]:.1f} s)', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    out_path = os.path.join(project_root, 'images', f'Q2_Tailwind_Impact_{course_name}_{rider.name.replace(" ", "_")}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  > Generated Tailwind Impact Plot: {out_path}")


def plot_combined_wind_comparison(course_name, rider, course_data, data_no_wind, data_headwind, data_tailwind, wind_speed_val):
    """
    将顺风和逆风的曲线画在一张图里
    """
    # 重构三种情况的轨迹
    dist_nw, vel_nw, time_nw = reconstruct_simulation_traces(
        rider, course_data, data_no_wind['power_strategy'], wind_speed=0)
    dist_hw, vel_hw, time_hw = reconstruct_simulation_traces(
        rider, course_data, data_headwind['power_strategy'], wind_speed=abs(wind_speed_val))  # 逆风
    dist_tw, vel_tw, time_tw = reconstruct_simulation_traces(
        rider, course_data, data_tailwind['power_strategy'], wind_speed=-abs(wind_speed_val))  # 顺风
    
    p_nw = np.array(data_no_wind['power_strategy'])
    p_hw = np.array(data_headwind['power_strategy'])
    p_tw = np.array(data_tailwind['power_strategy'])
    
    dist_km = dist_nw / 1000.0
    vel_nw_kph = vel_nw * 3.6
    vel_hw_kph = vel_hw * 3.6
    vel_tw_kph = vel_tw * 3.6
    
    wind_kmh = abs(wind_speed_val) * 3.6
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    
    # === 1. 功率对比 ===
    ax1 = axes[0]
    ax1.plot(dist_km[:-1], p_nw, label='No Wind', color='tab:blue', linewidth=2.5, alpha=0.9)
    ax1.plot(dist_km[:-1], p_hw, label=f'Headwind ({wind_kmh:.0f} km/h)', color='tab:red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.plot(dist_km[:-1], p_tw, label=f'Tailwind ({wind_kmh:.0f} km/h)', color='tab:green', linestyle=':', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Power (Watts)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Rider: {rider.name} - Power Strategy Comparison\n(No Wind vs Headwind vs Tailwind)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # === 2. 速度对比 ===
    ax2 = axes[1]
    ax2.plot(dist_km, vel_nw_kph, label='No Wind', color='tab:blue', linewidth=2.5)
    ax2.plot(dist_km, vel_hw_kph, label=f'Headwind ({wind_kmh:.0f} km/h)', color='tab:red', linestyle='--', linewidth=2)
    ax2.plot(dist_km, vel_tw_kph, label=f'Tailwind ({wind_kmh:.0f} km/h)', color='tab:green', linestyle=':', linewidth=2)
    
    # 填充区域显示差异
    ax2.fill_between(dist_km, vel_nw_kph, vel_hw_kph, color='red', alpha=0.1, label='Headwind Speed Loss')
    ax2.fill_between(dist_km, vel_nw_kph, vel_tw_kph, color='green', alpha=0.1, label='Tailwind Speed Gain')
    
    ax2.set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
    ax2.set_title('Speed Profile Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # === 3. 累计时间差（0在中间，正值在上，负值在下）===
    ax3 = axes[2]
    time_diff_hw = time_hw - time_nw  # 逆风时间损失（正值 = 更慢，画在上方）
    time_diff_tw = time_tw - time_nw  # 顺风时间差（负值 = 更快，画在下方）
    
    ax3.plot(dist_km, time_diff_hw, label=f'Headwind (Total: +{time_diff_hw[-1]:.1f}s)', 
             color='tab:red', linewidth=2.5)
    ax3.plot(dist_km, time_diff_tw, label=f'Tailwind (Total: {time_diff_tw[-1]:.1f}s)', 
             color='tab:green', linewidth=2.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)
    
    ax3.fill_between(dist_km, 0, time_diff_hw, color='red', alpha=0.15)
    ax3.fill_between(dist_km, 0, time_diff_tw, color='green', alpha=0.15)
    
    # 设置对称的y轴范围，0在中间
    max_abs_val = max(abs(time_diff_hw.max()), abs(time_diff_tw.min()))
    ax3.set_ylim(-max_abs_val * 1.15, max_abs_val * 1.15)
    
    ax3.set_ylabel('Time Difference (s)\n(Positive = Slower, Negative = Faster)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Time Impact (Relative to No Wind)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(project_root, 'images', f'Q2_Wind_Combined_{course_name}_{rider.name.replace(" ", "_")}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  > Generated Combined Wind Comparison Plot: {out_path}")
    
    return out_path


def combine_male_wind_comparison_images(project_root):
    """
    将东京赛道的男性 TT 与 Sprinter 风速对比图并排合并为一张图
    """
    img_dir = os.path.join(project_root, 'images')
    fname_tt = os.path.join(img_dir, 'Q2_Wind_Combined_Tokyo_Olympic_Male_TT.png')
    fname_sp = os.path.join(img_dir, 'Q2_Wind_Combined_Tokyo_Olympic_Male_Sprinter.png')
    
    if not (os.path.exists(fname_tt) and os.path.exists(fname_sp)):
        print("[Warn] Tokyo male wind comparison images not found; skip combining.")
        return
    
    img_tt = plt.imread(fname_tt)
    img_sp = plt.imread(fname_sp)
    
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    
    axes[0].imshow(img_tt)
    axes[0].axis('off')
    axes[0].set_title('Male TT Specialist', fontsize=16, fontweight='bold', pad=20)
    
    axes[1].imshow(img_sp)
    axes[1].axis('off')
    axes[1].set_title('Male Sprinter', fontsize=16, fontweight='bold', pad=20)
    
    plt.suptitle('Tokyo Olympic Course - Wind Impact Comparison (Male Riders)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    out_path = os.path.join(img_dir, 'Q2_Wind_Combined_Tokyo_Male_All.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  > Generated Combined Male Wind Comparison: {out_path}")


def solve_wind_comparison():
    """
    主函数：生成顺风图片，并将顺风逆风合并到一张图
    """
    print("=== Wind Comparison Analysis: Headwind vs Tailwind ===\n")
    
    riders = get_all_riders()
    
    # 加载赛道
    tokyo_data = load_real_course(os.path.join(project_root, 'data', 'course_tokyo.csv'))
    flanders_m = load_real_course(os.path.join(project_root, 'data', 'flanders_men.csv'))
    
    if not tokyo_data:
        print("Error: Tokyo course data not found!")
        return
    
    # 风速设置（与原代码一致）
    WIND_SPEED = 4.0 / 3.6  # 4 km/h 转换为 m/s
    
    configs = [
        {'name': 'Tokyo_Olympic', 'data': tokyo_data},
    ]
    
    if flanders_m:
        configs.append({'name': 'Flanders_WorldChamp', 'data': flanders_m})
    
    results_summary = []
    
    for conf in configs:
        course_name = conf['name']
        course_data = conf['data']
        print(f"\n{'='*60}")
        print(f"Processing Course: {course_name}")
        print(f"{'='*60}")
        
        for rider in riders:
            # 只处理男性车手（与原图一致），女性可以取消注释
            if rider.gender != 'Male':
                continue
                
            print(f"\n  Rider: {rider.name}")
            
            # 1. 无风优化
            print("    Optimizing for No Wind...")
            p_no_wind, t_no_wind = run_simulated_annealing(rider, course_data, wind_speed=0)
            
            # 2. 逆风优化
            print("    Optimizing for Headwind...")
            p_headwind, t_headwind = run_simulated_annealing(rider, course_data, wind_speed=WIND_SPEED)
            
            # 3. 顺风优化（负风速）
            print("    Optimizing for Tailwind...")
            p_tailwind, t_tailwind = run_simulated_annealing(rider, course_data, wind_speed=-WIND_SPEED)
            
            data_no_wind = {'power_strategy': p_no_wind, 'total_time': t_no_wind}
            data_headwind = {'power_strategy': p_headwind, 'total_time': t_headwind}
            data_tailwind = {'power_strategy': p_tailwind, 'total_time': t_tailwind}
            
            # 生成顺风单独的影响图
            print("    Generating Tailwind Impact Plot...")
            plot_tailwind_impact(course_name, rider, course_data, data_no_wind, data_tailwind, WIND_SPEED)
            
            # 生成顺风+逆风合并对比图
            print("    Generating Combined Wind Comparison Plot...")
            plot_combined_wind_comparison(course_name, rider, course_data, 
                                          data_no_wind, data_headwind, data_tailwind, WIND_SPEED)
            
            # 记录结果
            results_summary.append({
                'Course': course_name,
                'Rider': rider.name,
                'No Wind Time (s)': round(t_no_wind, 2),
                'Headwind Time (s)': round(t_headwind, 2),
                'Tailwind Time (s)': round(t_tailwind, 2),
                'Headwind Loss (s)': round(t_headwind - t_no_wind, 2),
                'Tailwind Gain (s)': round(t_no_wind - t_tailwind, 2)
            })
    
    # 保存结果摘要
    df = pd.DataFrame(results_summary)
    out_csv = os.path.join(project_root, 'data', 'Q2_Wind_Comparison_Results.csv')
    df.to_csv(out_csv, index=False)
    print(f"\n\nResults saved to: {out_csv}")
    print("\n" + "="*60)
    print("Summary Table:")
    print("="*60)
    print(df.to_string(index=False))
    
    # 合并东京男性车手的对比图为单张并排图
    print("\n" + "="*60)
    print("Combining Male Riders Comparison Images...")
    print("="*60)
    combine_male_wind_comparison_images(project_root)
    
    print("\n\nAll plots generated successfully!")
    print(f"Check the 'images' folder for:")
    print("  - Q2_Tailwind_Impact_*.png (顺风影响图)")
    print("  - Q2_Wind_Combined_*.png (顺风+逆风合并对比图)")
    print("  - Q2_Wind_Combined_Tokyo_Male_All.png (男性车手合并对比图)")


if __name__ == "__main__":
    solve_wind_comparison()
