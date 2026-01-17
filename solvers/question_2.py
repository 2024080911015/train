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
from models.optimizer import run_simulated_annealing  # 确保导入的是修改后的版本


# === 数据加载函数 (保持不变) ===
def load_custom_course_from_csv(csv_path):
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    course_data = []
    for _, row in df.iterrows():
        course_data.append({'length': row['length'], 'slope': row['slope'], 'radius': row['radius']})
    return course_data


def load_real_course(csv_path, step_size=100.0):
    if not os.path.exists(csv_path): return None
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


# === 绘图函数 ===
RIDER_COLORS = {'Male TT': 'tab:blue', 'Male Sprinter': 'navy', 'Female TT': 'tab:red', 'Female Sprinter': 'darkred'}
RIDER_LINESTYLES = {'Male TT': '-', 'Male Sprinter': '--', 'Female TT': '-', 'Female Sprinter': '--'}



def reconstruct_simulation_traces(rider, course_data, power_strategy, wind_speed=0):
    """
    Re-run simulation to get detailed traces for plotting: Power, Speed, Time over distance.
    Returns: (distances, velocities, times) arrays corresponding to segment boundaries.
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
            
            # Physics
            v_air = v_curr + wind_speed
            f_drag = 0.5 * sim.rho * sim.cd_a * (v_air ** 2)
            f_grav = sim.total_mass * sim.g * np.sin(slope)
            f_roll = sim.total_mass * sim.g * sim.mu_roll * np.cos(slope)
            f_resist = f_drag + f_grav + f_roll
            
            if v_curr < 0.1: v_curr = 0.1
            f_prop = p_target / v_curr
            acc = (f_prop - f_resist) / sim.total_mass
            
            v_next_sq = v_curr ** 2 + 2 * acc * dx
            v_curr = np.sqrt(v_next_sq) if v_next_sq > 0.1 else 0.1
            
            if radius < 100:
                v_limit = np.sqrt(sim.mu_tire * sim.g * radius)
                v_curr = min(v_curr, v_limit)
                
            dt = dx / v_curr
            total_time += dt

            # W' Balance
            if p_target > sim.cyclist.cp:
                loss = (p_target - sim.cyclist.cp) * dt
                current_w -= loss
            else:
                rec_rate = (sim.cyclist.cp - p_target)
                current_w += rec_rate * dt * 0.5
            
            if current_w > sim.cyclist.w_prime: current_w = sim.cyclist.w_prime
            if current_w < 0:
                current_w = 0
                is_exhausted = True

            dist_covered += dx
            
        current_dist += seg_len
        distances.append(current_dist)
        velocities.append(v_curr)
        times.append(total_time)
        
    return np.array(distances), np.array(velocities), np.array(times)

def plot_wind_impact_vivid(course_name, rider_results):
    """
    Generate vivid comparison plots for each rider: Wind vs No Wind.
    Top: Speed comparison (Area diff)
    Bottom: Power comparison
    """
    for res_pair in rider_results:
        # Group by rider to find pair (Wind vs NoWind)
        pass # implemented in the main logic loop better

def plot_rider_impact_detailed(course_name, rider, data_no_wind, data_wind):
    dist_nw, vel_nw, time_nw = reconstruct_simulation_traces(rider, data_no_wind['course_data'], data_no_wind['power_strategy'], wind_speed=0)
    dist_w, vel_w, time_w = reconstruct_simulation_traces(rider, data_wind['course_data'], data_wind['power_strategy'], wind_speed=16.0/3.6) # assuming 16km/h from context
    
    # Power strategies aligned with distance steps (segments)
    # Strategy is per segment. Step plot logic needs alignment.
    # reconstruct_simulation_traces returns len(course_data)+1 points.
    
    # Create aligned power arrays for plotting (step)
    p_nw = np.array(data_no_wind['power_strategy'])
    p_w = np.array(data_wind['power_strategy'])

    # Convert to km and km/h
    dist_km = dist_nw / 1000.0
    vel_nw_kph = vel_nw * 3.6
    vel_w_kph = vel_w * 3.6
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. Power Comparison
    # We use step plot logic
    x_steps = dist_km[:-1] # Start of segments
    # Actually step plot: 'post' means [i] applies to interval [i, i+1].
    # But dist_km has N+1 points.
    
    # Let's use fill_between for power to show difference vivid
    # But steps are tricky with fill_between. Interpolate or just step.
    ax1.plot(dist_km[:-1], p_nw, label='No Wind Power', color='tab:blue', alpha=0.8)
    ax1.plot(dist_km[:-1], p_w, label='Wind Power', color='tab:red', alpha=0.8, linestyle='--')
    ax1.fill_between(dist_km[:-1], p_nw, p_w, where=(p_w > p_nw), color='red', alpha=0.1, interpolate=True, step='post')
    ax1.fill_between(dist_km[:-1], p_nw, p_w, where=(p_w <= p_nw), color='blue', alpha=0.1, interpolate=True, step='post')
    
    ax1.set_ylabel('Power (Watts)')
    ax1.set_title(f'Rider: {rider.name} - Power Strategy (Wind Impact)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Speed Comparison
    ax2.plot(dist_km, vel_nw_kph, label='No Wind Speed', color='tab:blue')
    ax2.plot(dist_km, vel_w_kph, label='Wind Speed (16km/h Headwind)', color='tab:red', linestyle='--')
    
    # Fill speed loss
    ax2.fill_between(dist_km, vel_nw_kph, vel_w_kph, color='gray', alpha=0.2, label='Speed Loss')
    
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_title('Speed Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Time Gap Accumulation
    # Time gap = Time_Wind - Time_NoWind (how much slower)
    # Important: Times are simulated segment by segment.
    # The arrays are aligned by segment index.
    time_gap = time_w - time_nw
    
    ax3.plot(dist_km, time_gap, color='purple', label='Cumulative Time Loss', linewidth=2)
    ax3.fill_between(dist_km, 0, time_gap, color='purple', alpha=0.1)
    
    ax3.set_ylabel('Time Loss (s)')
    ax3.set_xlabel('Distance (km)')
    ax3.set_title(f'Cumulative Time Loss (Total: {time_gap[-1]:.1f} s)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(project_root, 'images', f'Q2_Impact_{course_name}_{rider.name.replace(" ", "_")}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  > Generated Impact Plot: {out_path}")


def plot_comparison_result(course_name, rider_results):
    fig, ax1 = plt.subplots(figsize=(16, 9))

    # 背景
    longest = max(rider_results, key=lambda r: sum(s['length'] for s in r['course_data']))
    l_data = longest['course_data']
    l_dist = np.insert(np.cumsum([s['length'] for s in l_data]) / 1000.0, 0, 0)

    # 海拔
    ax2 = ax1.twinx()
    l_elev = [0]
    curr = 0
    for s in l_data:
        curr += s['length'] * np.sin(s['slope'])
        l_elev.append(curr)
    ax2.fill_between(l_dist, min(l_elev), l_elev, color='#2ca02c', alpha=0.15, label='Elevation')
    ax2.set_ylabel('Elevation (m)', color='#2ca02c')

    # 功率曲线
    for res in rider_results:
        rider = res['rider']
        p_strat = res['power_strategy']
        dist = np.insert(np.cumsum([s['length'] for s in res['course_data']]) / 1000.0, 0, 0)
        p_plot = np.append(p_strat, p_strat[-1]) if len(p_strat) > 0 else p_strat
        n = min(len(dist), len(p_plot))

        # 区分有风/无风样式
        is_wind = res['is_wind']
        color = RIDER_COLORS.get(rider.name, 'gray')
        ls = ':' if is_wind else RIDER_LINESTYLES.get(rider.name, '-')
        label = f"{rider.name} {'(Wind Opt)' if is_wind else '(No Wind)'}"
        alpha = 0.7 if is_wind else 1.0

        ax1.step(dist[:n], p_plot[:n], where='post', color=color, linestyle=ls, label=label, alpha=alpha, linewidth=2)

    ax1.set_title(f"Optimization: {course_name} (Re-optimized for Wind)", fontsize=16)
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Power (Watts)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    out = os.path.join(project_root, 'images', f'Q2_{course_name}_ReOptimized.png')
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")
    plt.close()


def plot_combined_results(results_map):
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    targets = ['Tokyo_Olympic', 'Flanders_WorldChamp']

    for i, name in enumerate(targets):
        res_list = results_map.get(name)
        if not res_list: continue
        ax = axes[i]

        # 简单绘制逻辑，只为展示
        longest = max(res_list, key=lambda r: sum(s['length'] for s in r['course_data']))
        l_dist = np.insert(np.cumsum([s['length'] for s in longest['course_data']]) / 1000.0, 0, 0)

        for res in res_list:
            rider = res['rider']
            p_strat = res['power_strategy']
            dist = np.insert(np.cumsum([s['length'] for s in res['course_data']]) / 1000.0, 0, 0)
            p_plot = np.append(p_strat, p_strat[-1]) if len(p_strat) > 0 else p_strat
            n = min(len(dist), len(p_plot))

            is_wind = res['is_wind']
            c = RIDER_COLORS.get(rider.name, 'gray')
            ls = ':' if is_wind else RIDER_LINESTYLES.get(rider.name, '-')

            ax.step(dist[:n], p_plot[:n], where='post', color=c, linestyle=ls, alpha=0.8)

        ax.set_title(f"{name} (Solid=No Wind, Dotted=Wind Optimized)")
        ax.set_xlim(0, l_dist[-1])

    out = os.path.join(project_root, 'images', 'Q2_Combined_ReOptimized.png')
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")
    plt.close()


# === 主逻辑 ===
def solve_q2():
    print("=== Re-Running Q2: Optimization with & without Wind ===")

    riders = get_all_riders()

    # 加载赛道
    custom_data = load_custom_course_from_csv(os.path.join(project_root, 'data', 'course_custom.csv'))
    tokyo_data = load_real_course(os.path.join(project_root, 'data', 'course_tokyo.csv'))

    tokyo_female = []
    if tokyo_data:
        half = sum(s['length'] for s in tokyo_data) / 2
        acc = 0
        for s in tokyo_data:
            if acc >= half: break
            tokyo_female.append(s)
            acc += s['length']

    flanders_m = load_real_course(os.path.join(project_root, 'data', 'flanders_men.csv'))
    flanders_f = load_real_course(os.path.join(project_root, 'data', 'flanders_women.csv'))

    configs = [
        {'name': 'Designed_Custom_Track', 'm': custom_data, 'f': custom_data},
        {'name': 'Tokyo_Olympic', 'm': tokyo_data, 'f': tokyo_female},
        {'name': 'Flanders_WorldChamp', 'm': flanders_m, 'f': flanders_f}
    ]

    results_csv = []
    plot_data_map = {}

    # 定义两种环境
    conditions = [
        {'label': 'No Wind', 'speed': 0.0},
        {'label': 'Wind 16km/h', 'speed': 16.0 / 3.6}  # 4.44 m/s
    ]

    for conf in configs:
        c_name = conf['name']
        print(f"\nProcessing {c_name}...")
        plot_list = []

        for rider in riders:
            c_data = conf['m'] if rider.gender == 'Male' else conf['f']
            if not c_data: continue

            print(f"  Rider: {rider.name} ({rider.rider_type})")

            for cond in conditions:
                w_speed = cond['speed']
                label = cond['label']

                # === 核心修改：针对当前风速，重新运行优化算法 ===
                # 这会寻找专门适应这个风速的最佳策略
                best_strategy, best_time = run_simulated_annealing(rider, c_data, wind_speed=w_speed)

                # 记录用于绘图
                plot_list.append({
                    'rider': rider,
                    'course_data': c_data,
                    'power_strategy': best_strategy,
                    'total_time': best_time,
                    'is_wind': (w_speed > 0)
                })

                # 记录到表格
                results_csv.append({
                    'Condition': label,
                    'Course': c_name,
                    'Rider': rider.name,
                    'Time (s)': round(best_time, 2),
                    'Time (min)': round(best_time / 60, 2)
                })

        plot_data_map[c_name] = plot_list
        plot_comparison_result(c_name, plot_list)

        # === 新增：为每位车手生成详细的风阻影响对比图 ===
        rider_groups = {}
        for p in plot_list:
            r_name = p['rider'].name
            if r_name not in rider_groups: rider_groups[r_name] = {}
            if p['is_wind']:
                rider_groups[r_name]['wind'] = p
            else:
                rider_groups[r_name]['no_wind'] = p

        for r_name, group in rider_groups.items():
            if 'wind' in group and 'no_wind' in group:
                print(f"  Generating vivid comparison chart for {r_name}...")
                plot_rider_impact_detailed(c_name, group['no_wind']['rider'], group['no_wind'], group['wind'])

    plot_combined_results(plot_data_map)

    df = pd.DataFrame(results_csv)
    # 调整顺序
    df = df[['Condition', 'Course', 'Rider', 'Time (s)', 'Time (min)']]
    out_csv = os.path.join(project_root, 'data', 'Q2_Final_ReOptimized.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nDone. Results saved to {out_csv}")
    print(df)


if __name__ == "__main__":
    solve_q2()