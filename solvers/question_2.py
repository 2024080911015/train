import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === 1. 路径设置，确保能导入 models ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.cyclist import Cyclist
from models.simulator import WPrimeBalanceSimulator
from models.optimizer import run_simulated_annealing


# === 2. 赛道加载工具 ===

def load_custom_course_from_csv(csv_path):
    """
    加载由 generate_custom_track.py 生成的高精度赛道数据。
    该 CSV 已包含计算好的 radius, slope, length，直接读取即可。
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Custom course file '{csv_path}' not found.")
        return None

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded custom course with {len(df)} segments.")
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    course_data = []

    for _, row in df.iterrows():
        course_data.append({
            'length': row['length'],
            'slope': row['slope'],
            'radius': row['radius']
        })

    return course_data


def load_real_course(csv_path):
    """
    [修改版] 读取真实赛道 CSV 文件 (Tokyo / Flanders)。
    **已移除自动插值逻辑**：直接按 CSV 中的行读取分段。
    请确保 CSV 文件中的点足够密集（例如每 100m 一个点）。
    CSV 格式需为: distance, elevation
    """
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} not found. Skipping this course.")
        return None

    try:
        df = pd.read_csv(csv_path)
        # 确保按距离排序
        df = df.sort_values(by='distance').reset_index(drop=True)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    course_data = []

    # === 直接读取 CSV 分段 ===
    # 遍历每一行，计算与下一行的差值作为一段
    for i in range(len(df) - 1):
        dist_curr = df.iloc[i]['distance']
        elev_curr = df.iloc[i]['elevation']

        dist_next = df.iloc[i + 1]['distance']
        elev_next = df.iloc[i + 1]['elevation']

        length = dist_next - dist_curr

        # 忽略距离极小或为0的段（防止除零错误）
        if length <= 0.001:
            continue

        # 计算坡度 (弧度)
        slope = np.arctan((elev_next - elev_curr) / length)

        # 真实赛道默认设为直道 (如果 CSV 里没有 radius 列)
        radius = 9999.0

        course_data.append({
            'length': length,
            'slope': slope,
            'radius': radius
        })

    print(f"Loaded real course from {csv_path}: {len(course_data)} segments.")
    return course_data


# === 3. 定义车手 ===
def get_all_riders():
    """返回四类典型车手对象"""
    return [
        Cyclist("Male TT", "TT Specialist", "Male",
                cp=400, w_prime=20000, mass=72, cd_area=0.23, p_max=1200),
        Cyclist("Male Sprinter", "Sprinter", "Male",
                cp=340, w_prime=35000, mass=75, cd_area=0.28, p_max=1600),
        Cyclist("Female TT", "TT Specialist", "Female",
                cp=260, w_prime=15000, mass=60, cd_area=0.20, p_max=900),
        Cyclist("Female Sprinter", "Sprinter", "Female",
                cp=220, w_prime=25000, mass=62, cd_area=0.24, p_max=1100)
    ]


# === 4. 绘图与可视化 ===

# 定义4个选手的颜色方案
RIDER_COLORS = {
    'Male TT': 'tab:blue',
    'Male Sprinter': 'navy',
    'Female TT': 'tab:red',
    'Female Sprinter': 'darkred'
}

RIDER_LINESTYLES = {
    'Male TT': '-',
    'Male Sprinter': '--',
    'Female TT': '-',
    'Female Sprinter': '--'
}


def plot_comparison_result(course_name, rider_results):
    """
    策略一：对比绘图 - 一张图展示4个选手的功率策略和W'变化
    :param course_name: 赛道名称
    :param rider_results: 列表，每个元素包含 {rider, course_data, power_strategy, w_history, total_time}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 用于记录时间结果（图例中显示）
    time_labels = []

    # === 新增: 准备地形数据 (使用最长的赛道作为背景) ===
    longest_result = max(rider_results, key=lambda r: sum(s['length'] for s in r['course_data']))
    l_course_data = longest_result['course_data']
    
    l_lengths = [s['length'] for s in l_course_data]
    l_distances = np.cumsum(l_lengths) / 1000.0  # km
    l_distances = np.insert(l_distances, 0, 0)[:-1]
    
    l_elevations = [0]
    curr_ele = 0
    for s in l_course_data:
        curr_ele += s['length'] * np.sin(s['slope'])
        l_elevations.append(curr_ele)
    l_elevations = l_elevations[:-1]
    
    l_n = len(l_course_data)
    l_distances = l_distances[:l_n]
    l_elevations = l_elevations[:l_n]
    
    # 绘制海拔背景 (双轴)
    ax1_ele = ax1.twinx()
    color_e = 'tab:green'
    ax1_ele.set_ylabel('Elevation (m)', color=color_e, fontsize=12)
    ax1_ele.fill_between(l_distances, min(l_elevations), l_elevations, color=color_e, alpha=0.2, label='Elevation')
    
    # 标记急弯 (R < 100m) - 基于最长赛道
    sharp_turn_indices = [i for i, s in enumerate(l_course_data) if s['radius'] < 100 and i < len(l_distances)]
    for idx in sharp_turn_indices:
        ax1.axvline(x=l_distances[idx], color='red', linestyle=':', alpha=0.15)  # 颜色淡一点以免干扰曲线

    for result in rider_results:
        rider = result['rider']
        course_data = result['course_data']
        power_strategy = result['power_strategy']
        w_history = result['w_history']
        total_time = result['total_time']
        
        # 数据准备
        lengths = [s['length'] for s in course_data]
        distances = np.cumsum(lengths) / 1000.0  # km
        distances = np.insert(distances, 0, 0)[:-1]
        
        # 对齐
        n_segments = len(course_data)
        distances = distances[:n_segments]
        w_history = w_history[:n_segments]
        power_strategy = power_strategy[:n_segments]
        
        # 获取该选手的颜色和线型
        color = RIDER_COLORS.get(rider.name, 'gray')
        linestyle = RIDER_LINESTYLES.get(rider.name, '-')
        
        # 标签包含时间
        label = f"{rider.name} ({total_time/60:.1f} min)"
        time_labels.append((rider.name, total_time))
        
        # --- 子图 1: 功率曲线对比 ---
        ax1.plot(distances, power_strategy, color=color, linestyle=linestyle,
                 label=label, alpha=0.8, linewidth=1.5)
        
        # --- 子图 2: W' 余额对比 ---
        ax2.plot(distances, w_history, color=color, linestyle=linestyle,
                 label=f"{rider.name}", linewidth=1.5)
    
    # 子图1 设置
    ax1.set_ylabel('Power (Watts)', fontsize=12)
    ax1.set_title(f"Course: {course_name} - Power Strategy Comparison (4 Riders)", fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2 设置
    ax2.set_xlabel('Distance (km)', fontsize=12)
    ax2.set_ylabel("W' Balance (Joules)", fontsize=12)
    ax2.axhline(y=0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_dir = os.path.join(project_root, 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    safe_course = course_name.replace(" ", "_")
    filename = f'Q2_{safe_course}_comparison.png'
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=150)
    print(f"  -> Comparison graph saved: {filename}")
    plt.close()
    
    return time_labels


# === 5. 主求解逻辑 ===

def solve_q2():
    print("=== Running Question 2: Multi-Course Optimization ===")

    # 1. 加载所有车手
    riders = get_all_riders()
    print(f"Loaded {len(riders)} riders.")

    # 2. 预加载赛道数据
    # (A) 自主设计的赛道 (男女通用)
    custom_path = os.path.join(project_root, 'data', 'course_custom.csv')
    print(f"Loading Custom Course from {custom_path}...")
    custom_data = load_custom_course_from_csv(custom_path)

    # (B) 东京奥运会赛道 (男子完整，女子取一半距离)
    tokyo_path = os.path.join(project_root, 'data', 'course_tokyo.csv')
    tokyo_data_full = load_real_course(tokyo_path)
    
    # 计算女子赛道 (取前一半距离)
    tokyo_data_female = None
    if tokyo_data_full:
        total_length = sum(s['length'] for s in tokyo_data_full)
        half_length = total_length / 2.0
        accumulated = 0
        tokyo_data_female = []
        for s in tokyo_data_full:
            if accumulated >= half_length:
                break
            tokyo_data_female.append(s)
            accumulated += s['length']
        print(f"Tokyo Female course: {len(tokyo_data_female)} segments ({accumulated/1000:.1f} km)")

    # (C) Flanders 赛道 (男女分别读取不同文件)
    flanders_men_path = os.path.join(project_root, 'data', 'flanders_men.csv')
    flanders_women_path = os.path.join(project_root, 'data', 'flanders_women.csv')
    flanders_data_male = load_real_course(flanders_men_path)
    flanders_data_female = load_real_course(flanders_women_path)

    # 3. 开始优化循环
    results = []

    # 定义赛道配置：每个赛道根据性别选择不同数据
    course_configs = [
        # {'name': 'Designed_Custom_Track', 'male': custom_data, 'female': custom_data},
        {'name': 'Tokyo_Olympic', 'male': tokyo_data_full, 'female': tokyo_data_female},
        {'name': 'Flanders_WorldChamp', 'male': flanders_data_male, 'female': flanders_data_female}
    ]

    for config in course_configs:
        c_name = config['name']
        print(f"\n{'='*50}")
        print(f"Processing Course: {c_name}")
        print(f"{'='*50}")
        
        # 收集该赛道所有选手的结果（用于对比绘图）
        rider_results_for_plot = []
        
        for rider in riders:
            # 根据性别选择对应的赛道数据
            c_data = config['male'] if rider.gender == 'Male' else config['female']
            
            if c_data is None:
                print(f"Skipping {c_name} for {rider.name} (data not available)")
                continue
            
            print(f"\n--- Optimizing for {rider.name} ({rider.rider_type}) | Segments: {len(c_data)} ---")

            # A. 运行模拟退火
            best_strategy, best_time = run_simulated_annealing(rider, c_data)
            print(f"  > Best Time: {best_time:.2f} s ({best_time/60:.1f} min)")

            # B. 结果回放
            sim = WPrimeBalanceSimulator(rider)
            _, w_history, _ = sim.run_segment_simulation(best_strategy, c_data)

            # C. 收集用于绘图的数据
            rider_results_for_plot.append({
                'rider': rider,
                'course_data': c_data,
                'power_strategy': best_strategy,
                'w_history': w_history,
                'total_time': best_time
            })

            # D. 收集结果数据
            results.append({
                "Course": c_name,
                "Rider": rider.name,
                "Type": rider.rider_type,
                "Gender": rider.gender,
                "Time (s)": round(best_time, 2),
                "Time (min)": round(best_time / 60, 2)
            })
        
        # E. 绘制该赛道的4选手对比图
        if rider_results_for_plot:
            print(f"\n--- Generating comparison plot for {c_name} ---")
            plot_comparison_result(c_name, rider_results_for_plot)

    # 4. 打印最终结果
    print("\n=== Final Results Summary ===")
    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res[["Course", "Rider", "Type", "Gender", "Time (s)", "Time (min)"]]
        print(df_res)

        csv_out = os.path.join(project_root, 'data', 'Q2_Final_Results.csv')
        df_res.to_csv(csv_out, index=False)
        print(f"\nSummary saved to: {csv_out}")
    else:
        print("No courses loaded.")


if __name__ == "__main__":
    solve_q2()