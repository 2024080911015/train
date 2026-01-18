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


def run_monte_carlo(rider, course_data, base_strategy, sigma=0.05, num_sims=100, wind_speed=0):
    """
    运行蒙特卡洛模拟
    返回：(results, finish_times, dnf_times, dnf_count)
    - finish_times: 仅包含未力竭选手的完赛时间
    - dnf_times: 力竭选手的时间（用于分析）
    """
    sim = WPrimeBalanceSimulator(rider)
    finish_times = []  # 未力竭选手的时间
    dnf_times = []     # 力竭选手的时间
    dnf_count = 0
    results = []

    print(f"  > Running {num_sims} simulations for {rider.name} with sigma={sigma * 100}% and wind={wind_speed:.1f}m/s...")

    for i in range(num_sims):
        # 1. 生成带噪声的实际执行策略
        noisy_strategy = add_noise_to_strategy(base_strategy, sigma)

        # 2. 运行仿真
        # 注意：这里我们假设风速会有微小波动，或者仅考虑选手的功率波动
        # 这里仅模拟选手功率波动，风速设为固定值
        total_time, _, is_exhausted = sim.run_segment_simulation(noisy_strategy, course_data, wind_speed=wind_speed)

        if is_exhausted:
            dnf_count += 1
            dnf_times.append(total_time)
        else:
            finish_times.append(total_time)

        results.append({
            'sim_id': i,
            'time': total_time,
            'exhausted': is_exhausted,
            'strategy': noisy_strategy
        })

    return results, finish_times, dnf_times, dnf_count


def plot_sensitivity_boxplot(results_dict, sigma_levels):
    """
    绘制箱线图：不同误差水平下的成绩分布
    仅绘制未力竭选手的时间分布，并在标签中显示DNF率
    """
    plt.figure(figsize=(12, 7))

    data_to_plot = []
    labels = []
    dnf_rates = []

    for sigma in sigma_levels:
        key = f"sigma_{sigma}"
        if key in results_dict:
            finish_times = results_dict[key]['finish_times']
            dnf_rate = results_dict[key]['dnf_rate']
            dnf_rates.append(dnf_rate)
            
            if len(finish_times) > 0:
                # 将秒转换为分钟
                times_min = np.array(finish_times) / 60.0
                data_to_plot.append(times_min)
            else:
                # 如果全部力竭，用空数组占位
                data_to_plot.append(np.array([]))
            
            labels.append(f"{int(sigma * 100)}% Error\n(DNF: {dnf_rate:.1f}%)")

    # 绘制箱线图
    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor="lightblue"),
                     medianprops=dict(color="red", linewidth=2))
    
    # 根据DNF率调整箱体颜色（DNF率越高越红）
    colors = []
    for dnf_rate in dnf_rates:
        # 从蓝色渐变到红色
        r = min(1.0, dnf_rate / 50.0)  # DNF率50%时全红
        b = max(0.0, 1.0 - dnf_rate / 50.0)
        colors.append((r, 0.3, b, 0.6))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.title("Sensitivity Analysis: Impact of Execution Error on Finish Time\n(Excludes DNF/Exhausted Riders)", fontsize=12)
    plt.ylabel("Finish Time (minutes)")
    plt.xlabel("Rider Execution Error Level (Sigma)")
    plt.grid(True, linestyle='--', alpha=0.3)

    # 保存
    out_path = os.path.join(project_root, 'images', 'Q3_Sensitivity_Boxplot.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def analyze_sensitivity():
    print("=== Running Sensitivity Analysis (Stochastic Modeling) ===")

    # 1. 准备基础数据 (以自定义赛道 + 男性计时赛车手为例)
    # 你也可以循环所有车手，这里为了演示只取一个典型例子
    course_path = os.path.join(project_root, 'data', 'course_tokyo.csv')
    if not os.path.exists(course_path):
        print("Error: Course data not found. Run generate_custom_track.py first.")
        return

    course_data = load_real_course(course_path)
    rider = Cyclist("Male TT", "TT Specialist", "Male", cp=400, w_prime=20000, mass=72, cd_area=0.23, p_max=1200)

    # 2. 获取理论最优策略 (Baseline)
    print("\n[Step 1] Calculating Theoretical Optimal Strategy...")
    # 这里为了快，可以直接用简单的逻辑生成，或者调用模拟退火
    # 为了演示效果，我们先快速生成一个“伪最优”策略 (爬坡加力，平路巡航)
    params = [0.95, 1.1]  # Flat 95% CP, Climb 110% CP
    #base_strategy = generate_smart_strategy(params, course_data, rider.cp)
    
    # 设定无风条件
    wind_speed = 0.0
    print(f"  [Condition] Using No Wind Scenario: {wind_speed:.2f} m/s")

    # 或者如果你想更严谨，可以先跑一次模拟退火 (耗时较长)
    base_strategy, _ = run_simulated_annealing(rider, course_data, wind_speed=wind_speed)

    # 3. 设定不同的误差水平 (Sigma)
    # 2% (职业选手), 5% (业余高手), 10% (普通人)
    sigma_levels = [0.02, 0.05, 0.10]

    results_map = {}

    # 4. 循环进行蒙特卡洛模拟
    num_sims = 200
    print("\n[Step 2] Starting Monte Carlo Simulations...")
    for sigma in sigma_levels:
        sim_results, finish_times, dnf_times, dnf_count = run_monte_carlo(rider, course_data, base_strategy, sigma, num_sims=num_sims, wind_speed=wind_speed)

        # 统计分析 (仅统计未力竭选手)
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

        print(
            f"  [Result] Sigma {sigma * 100}%: Finishers={len(finish_times)}, Mean Time={mean_time / 60:.2f}min, Std Dev={std_time:.2f}s, DNF Rate={dnf_rate:.1f}%")

    # 5. 绘图：箱线图
    plot_sensitivity_boxplot(results_map, sigma_levels)

    print("\nSensitivity Analysis Completed.")


if __name__ == "__main__":
    analyze_sensitivity()
