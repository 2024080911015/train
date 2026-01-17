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


# === 2. 赛道生成与加载工具 ===

def generate_designed_course():
    """
    生成符合题目要求的'自选赛道' (Designed Course)。
    特征：20km长，包含起伏，人为设置了 4 个急转弯。
    """
    segment_length = 100.0
    num_segments = 200  # 20km

    course_data = []

    # 定义急转弯: (位置索引, 半径m)
    # R=15m -> 安全速度上限 ≈ 10.8 m/s
    sharp_turns = {
        40: 15.0,  # 4km
        90: 15.0,  # 9km
        140: 15.0,  # 14km
        190: 15.0  # 19km
    }

    for i in range(num_segments):
        radius = 9999.0  # 默认为直道
        slope = 0.0

        # --- 地形设计 (起伏) ---
        if 50 <= i < 100:
            slope = 0.04  # 上坡
        elif 100 <= i < 150:
            slope = -0.04  # 下坡
        elif 150 <= i < 200:
            slope = 0.02 * np.sin(i / 10.0)  # 起伏

        # --- 急弯插入 ---
        if i in sharp_turns:
            radius = sharp_turns[i]
            slope = 0.0  # 弯道通常较平

        course_data.append({
            'length': segment_length,
            'slope': slope,
            'radius': radius
        })

    return course_data


def load_real_course(csv_path):
    """
    读取真实赛道 CSV 文件 (Tokyo / Flanders) 并处理。
    CSV 格式需为: distance, elevation
    """
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} not found. Skipping this course.")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    course_data = []

    # 遍历数据点计算坡度
    # 假设 CSV 数据点够密集，或者你需要在这里做插值(Interpolation)
    for i in range(len(df) - 1):
        dist_curr = df.iloc[i]['distance']
        elev_curr = df.iloc[i]['elevation']

        dist_next = df.iloc[i + 1]['distance']
        elev_next = df.iloc[i + 1]['elevation']

        length = dist_next - dist_curr
        if length <= 0: continue

        # 计算坡度 (弧度)
        slope = np.arctan((elev_next - elev_curr) / length)

        # 真实数据 CSV 通常没有半径信息，默认设为直道
        # *进阶*: 如果你想在东京赛道增加急弯限制，可以在这里根据 distance 手动判断
        radius = 9999.0

        course_data.append({
            'length': length,
            'slope': slope,
            'radius': radius
        })

    return course_data


# === 3. 定义车手 (复用 Q1 定义) ===
def get_all_riders():
    """返回四类典型车手对象"""
    return [
        # --- 男性车手 ---
        Cyclist("Male TT", "TT Specialist", "Male",
                cp=400, w_prime=20000, mass=72, cd_area=0.23, p_max=1200),
        Cyclist("Male Sprinter", "Sprinter", "Male",
                cp=340, w_prime=35000, mass=75, cd_area=0.28, p_max=1600),

        # --- 女性车手 ---
        Cyclist("Female TT", "TT Specialist", "Female",
                cp=260, w_prime=15000, mass=60, cd_area=0.20, p_max=900),
        Cyclist("Female Sprinter", "Sprinter", "Female",
                cp=220, w_prime=25000, mass=62, cd_area=0.24, p_max=1100)
    ]


# === 4. 绘图与可视化 ===

def plot_individual_result(rider, course_name, course_data, power_strategy, w_history, total_time):
    """
    绘制单人策略结果并保存
    文件名格式: Q2_{Course}_{RiderName}.png
    """
    # 数据准备
    lengths = [s['length'] for s in course_data]
    distances = np.cumsum(lengths) / 1000.0  # km
    distances = np.insert(distances, 0, 0)[:-1]

    elevations = [0]
    curr_ele = 0
    for s in course_data:
        curr_ele += s['length'] * np.tan(s['slope'])
        elevations.append(curr_ele)
    elevations = elevations[:-1]

    w_history = w_history[:len(distances)]

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- 子图 1: 功率与海拔 ---
    color_p = 'tab:blue' if rider.gender == 'Male' else 'tab:red'

    ax1.set_ylabel('Power (Watts)', color=color_p, fontsize=12)
    ax1.plot(distances, power_strategy, color=color_p, label=f'{rider.name} Power', alpha=0.8, linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color_p)
    ax1.axhline(y=rider.cp, color='gray', linestyle='--', alpha=0.5, label='CP')

    # 双轴显示海拔
    ax1_ele = ax1.twinx()
    color_e = 'tab:green'
    ax1_ele.set_ylabel('Elevation (m)', color=color_e, fontsize=12)
    ax1_ele.fill_between(distances, min(elevations), elevations, color=color_e, alpha=0.2, label='Elevation')
    ax1_ele.tick_params(axis='y', labelcolor=color_e)

    # 标记急转弯 (仅当赛道中有设定急弯时)
    sharp_turn_indices = [i for i, s in enumerate(course_data) if s['radius'] < 100]
    for idx in sharp_turn_indices:
        ax1.axvline(x=distances[idx], color='red', linestyle=':', alpha=0.5)

    ax1.set_title(f"Course: {course_name} | Rider: {rider.name} | Time: {total_time:.0f}s ({total_time / 60:.1f}min)",
                  fontsize=14)
    ax1.legend(loc='upper left')

    # --- 子图 2: W' 余额 ---
    color_w = 'tab:orange'
    ax2.set_xlabel('Distance (km)', fontsize=12)
    ax2.set_ylabel("W' Balance (Joules)", color=color_w, fontsize=12)
    ax2.plot(distances, w_history, color=color_w, label="W' Balance", linewidth=2)
    ax2.fill_between(distances, 0, w_history, color=color_w, alpha=0.1)
    ax2.axhline(y=0, color='red', linewidth=1)  # 力竭线
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_dir = os.path.join(project_root, 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 文件名处理
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

    # 2. 定义所有赛道
    # 字典列表结构：{'name': 显示名称, 'data': 赛道数据列表}
    courses = []

    # (A) 自选赛道
    print("Generating Designed Course...")
    courses.append({
        'name': 'Designed_Course',
        'data': generate_designed_course()
    })

    # (B) 东京奥运会赛道 (需确保文件存在)
    tokyo_path = os.path.join(project_root, 'data', 'course_tokyo.csv')
    tokyo_data = load_real_course(tokyo_path)
    if tokyo_data:
        courses.append({'name': 'Tokyo_Olympic', 'data': tokyo_data})
        print(f"Loaded Tokyo Course: {len(tokyo_data)} segments")

    # (C) 比利时赛道 (需确保文件存在)
    flanders_path = os.path.join(project_root, 'data', 'course_flanders.csv')
    flanders_data = load_real_course(flanders_path)
    if flanders_data:
        courses.append({'name': 'Flanders_WorldChamp', 'data': flanders_data})
        print(f"Loaded Flanders Course: {len(flanders_data)} segments")

    # 3. 双层循环：遍历赛道 -> 遍历车手
    results = []

    for course in courses:
        c_name = course['name']
        c_data = course['data']
        print(f"\n==========================================")
        print(f"Processing Course: {c_name}")
        print(f"==========================================")

        for rider in riders:
            print(f"\n--- Optimizing for {rider.name} ({rider.rider_type}) ---")

            # A. 运行模拟退火
            # 注意：真实赛道可能很长，initial_guess 可以优化
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

    # 4. 打印最终对比表
    print("\n=== Final Results Summary ===")
    if results:
        df_res = pd.DataFrame(results)
        # 调整列顺序
        df_res = df_res[["Course", "Rider", "Type", "Gender", "Time (s)", "Time (min)"]]
        print(df_res)

        # 保存汇总 CSV
        csv_out = os.path.join(project_root, 'data', 'Q2_Final_Results.csv')
        df_res.to_csv(csv_out, index=False)
        print(f"\nSummary saved to: {csv_out}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    solve_q2()