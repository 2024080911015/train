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

def plot_individual_result(rider, course_name, course_data, power_strategy, w_history, total_time):
    """绘制单人策略结果并保存"""
    # 数据准备
    lengths = [s['length'] for s in course_data]
    distances = np.cumsum(lengths) / 1000.0  # km
    distances = np.insert(distances, 0, 0)[:-1]

    elevations = [0]
    curr_ele = 0
    for s in course_data:
        # [修复] 坡度是弧度，垂直分量应使用 sin
        curr_ele += s['length'] * np.sin(s['slope'])
        elevations.append(curr_ele)
    elevations = elevations[:-1]

    # 对齐 (注意: w_history 长度是 n_segments + 1，取前 n_segments 个)
    n_segments = len(course_data)
    distances = distances[:n_segments]
    w_history = w_history[:n_segments]  # 取每段开始时的 W' 值
    power_strategy = power_strategy[:n_segments]
    elevations = elevations[:n_segments]

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- 子图 1: 功率与海拔 ---
    color_p = 'tab:blue' if rider.gender == 'Male' else 'tab:red'

    ax1.set_ylabel('Power (Watts)', color=color_p, fontsize=12)
    ax1.plot(distances, power_strategy, color=color_p, label=f'{rider.name} Power', alpha=0.8, linewidth=1)
    ax1.axhline(y=rider.cp, color='gray', linestyle='--', alpha=0.5, label='CP')

    # 双轴海拔
    ax1_ele = ax1.twinx()
    color_e = 'tab:green'
    ax1_ele.set_ylabel('Elevation (m)', color=color_e, fontsize=12)
    ax1_ele.fill_between(distances, min(elevations), elevations, color=color_e, alpha=0.2, label='Elevation')

    # 标记急弯 (R < 100m)
    sharp_turn_indices = [i for i, s in enumerate(course_data) if s['radius'] < 100 and i < len(distances)]
    for idx in sharp_turn_indices:
        ax1.axvline(x=distances[idx], color='red', linestyle=':', alpha=0.3)

    ax1.set_title(f"Course: {course_name} | Rider: {rider.name} | Time: {total_time:.0f}s ({total_time / 60:.1f}min)",
                  fontsize=14)
    ax1.legend(loc='upper left')

    # --- 子图 2: W' 余额 ---
    color_w = 'tab:orange'
    ax2.set_xlabel('Distance (km)', fontsize=12)
    ax2.set_ylabel("W' Balance (Joules)", color=color_w, fontsize=12)
    ax2.plot(distances, w_history, color=color_w, label="W' Balance", linewidth=2)
    ax2.fill_between(distances, 0, w_history, color=color_w, alpha=0.1)
    ax2.axhline(y=0, color='red', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_dir = os.path.join(project_root, 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    safe_course = course_name.replace(" ", "_")
    safe_rider = rider.name.replace(" ", "_")
    filename = f'Q2_{safe_course}_{safe_rider}.png'
    save_path = os.path.join(output_dir, filename)

    plt.savefig(save_path, dpi=150)
    print(f"  -> Graph saved: {filename}")
    plt.close()


# === 5. 主求解逻辑 ===

def solve_q2():
    print("=== Running Question 2: Multi-Course Optimization ===")

    # 1. 加载所有车手
    riders = get_all_riders()
    print(f"Loaded {len(riders)} riders.")

    # 2. 定义赛道列表
    courses = []

    # (A) 加载自主设计的赛道 (course_custom.csv)
    custom_path = os.path.join(project_root, 'data', 'course_custom.csv')
    print(f"Loading Custom Course from {custom_path}...")
    custom_data = load_custom_course_from_csv(custom_path)

    if custom_data:
        courses.append({'name': 'Designed_Custom_Track', 'data': custom_data})
    else:
        print("Skipping Custom Track (File not found or error).")

    # (B) 加载真实赛道 (Tokyo / Flanders)
    # 直接读取文件，不进行插值
    tokyo_path = os.path.join(project_root, 'data', 'course_tokyo.csv')
    tokyo_data = load_real_course(tokyo_path)
    if tokyo_data:
        courses.append({'name': 'Tokyo_Olympic', 'data': tokyo_data})

    flanders_path = os.path.join(project_root, 'data', 'course_flanders.csv')
    flanders_data = load_real_course(flanders_path)
    if flanders_data:
        courses.append({'name': 'Flanders_WorldChamp', 'data': flanders_data})

    # 3. 开始优化循环
    results = []

    for course in courses:
        c_name = course['name']
        c_data = course['data']
        print(f"\n==========================================")
        print(f"Processing Course: {c_name} (Segments: {len(c_data)})")
        print(f"==========================================")

        for rider in riders:
            print(f"\n--- Optimizing for {rider.name} ({rider.rider_type}) ---")

            # A. 运行模拟退火
            best_strategy, best_time = run_simulated_annealing(rider, c_data)
            print(f"  > Best Time: {best_time:.2f} s")

            # B. 结果回放
            sim = WPrimeBalanceSimulator(rider)
            _, w_history, _ = sim.run_segment_simulation(best_strategy, c_data)

            # C. 绘图
            plot_individual_result(rider, c_name, c_data, best_strategy, w_history, best_time)

            # D. 收集数据
            results.append({
                "Course": c_name,
                "Rider": rider.name,
                "Type": rider.rider_type,
                "Gender": rider.gender,
                "Time (s)": round(best_time, 2),
                "Time (min)": round(best_time / 60, 2)
            })

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