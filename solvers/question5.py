import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 基础参数与数据
# ==========================================
# Bert Blocken 风阻数据 (6人队形)
# 位置 1-6 的阻力系数
DRAG_COEFFS_6 = [0.975, 0.616, 0.497, 0.443, 0.426, 0.433]

# 切换风阻惩罚
SWITCH_PENALTY_DRAG = 0.2

# 车手参数配置
# 假设我们有 4 名强力 TT 车手和 2 名相对较弱的车手 (或高爆发低耐力的 Sprinter)
STATS_STRONG = {'mass': 72, 'cp': 420, 'w_prime': 25000, 'cda': 0.23}
STATS_WEAK = {'mass': 75, 'cp': 340, 'w_prime': 35000, 'cda': 0.28}  # 弱势/冲刺手


class Rider:
    def __init__(self, id, stats, role="Puller"):
        self.id = id
        self.mass = stats['mass']
        self.cp = stats['cp']
        self.w_prime_max = stats['w_prime']
        self.cda = stats['cda']
        self.w_bal = self.w_prime_max
        self.total_mass = self.mass + 8.0
        self.role = role  # "Puller" (干活的), "Passenger" (坐车的)

    def reset(self):
        self.w_bal = self.w_prime_max


# ==========================================
# 2. 物理与恢复模型
# ==========================================
def get_drag_factor(pos_idx):
    if pos_idx < len(DRAG_COEFFS_6):
        return DRAG_COEFFS_6[pos_idx]
    else:
        return 0.433  # 队尾保护


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
# 3. 仿真引擎 (No-Drop 专用逻辑)
# ==========================================
def run_no_drop_simulation(team, target_speed_kmh, race_dist_m, rotation_time_sec):
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

    logs = {r.id: [] for r in team}

    while current_dist < race_dist_m:

        # --- A. 失败判定 (No-Drop) ---
        # 只要有任何一人(包括弱者)爆缸，策略即失败
        if any(r.w_bal <= 0 for r in formation):
            return None, None

        # --- B. 物理计算 ---
        is_switching = (pull_timer == 0) and (current_time > 0)

        for pos, rider in enumerate(formation):
            alpha = get_drag_factor(pos)

            # 切换成本：只有参与轮转的人受罚 (前两人)
            if is_switching and pos <= 1: alpha += SWITCH_PENALTY_DRAG

            f_drag = 0.5 * rho * (rider.cda * alpha) * (target_v ** 2)
            f_roll = rider.total_mass * g * crr
            p_req = ((f_drag + f_roll) * target_v) / eta

            rider.w_bal = skiba_recovery(rider.w_bal, rider.w_prime_max, rider.cp, p_req, dt)
            logs[rider.id].append(rider.w_bal)

        # --- C. 保护性轮转逻辑 (核心!) ---
        pull_timer += 1
        leader = formation[0]

        # 设定领骑时间限制
        if leader.role == "Passenger":
            # 弱者完全不领骑，轮到他时立刻换下去 (0秒)
            limit = 0
        else:
            # 强者承担所有工作，按优化时间轮转
            limit = rotation_time_sec

        if pull_timer >= limit:
            formation.append(formation.pop(0))  # 移到队尾
            pull_timer = 0

        current_dist += target_v * dt
        current_time += dt

    return current_time, pd.DataFrame(logs)


# ==========================================
# 4. 双层优化器
# ==========================================
def optimize_no_drop(team, race_dist):
    print("开始双层优化 (No-Drop 策略)...")

    # 外层：轮换时间 (强者的轮换时间)
    rotation_options = range(20, 100, 10)

    summary_results = []
    best_global = {'time': float('inf'), 'speed': 0, 'rot': 0, 'log': None}

    for rot in rotation_options:
        # 内层：二分查找极限速度
        # No-Drop 的速度上限通常比牺牲策略低，范围设低一点
        low, high = 40.0, 60.0
        local_best = {'time': None, 'speed': 0}

        for _ in range(15):
            mid = (low + high) / 2
            t, log = run_no_drop_simulation(team, mid, race_dist, rot)

            if t is not None:
                local_best = {'time': t, 'speed': mid, 'log': log}
                low = mid  # 尝试更快
            else:
                high = mid

        if local_best['time']:
            print(f"  轮换周期 {rot}s -> 极限速度 {local_best['speed']:.2f} km/h")
            summary_results.append((rot, local_best['time']))

            if local_best['time'] < best_global['time']:
                best_global = {
                    'time': local_best['time'],
                    'speed': local_best['speed'],
                    'rot': rot,
                    'log': local_best['log']
                }

    return best_global, summary_results


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # --- 组建车队 ---
    # 4名 强力车手 (Puller) + 2名 弱势车手 (Passenger)
    team = []
    for i in range(4):
        team.append(Rider(f"Strong_{i + 1}", STATS_STRONG, "Puller"))
    for i in range(2):
        team.append(Rider(f"Weak_{i + 1}", STATS_WEAK, "Passenger"))

    RACE_DIST = 44200

    best, summary = optimize_no_drop(team, RACE_DIST)

    print("\n" + "=" * 50)
    print(" 全员完赛策略 (No-Drop) 最终结果")
    print("=" * 50)
    print(f"最佳轮换时间 (Optimal Rotation): {best['rot']} 秒")
    print(f"极限巡航速度 (Max Cruising Speed): {best['speed']:.2f} km/h")
    print(f"完赛时间 (Finish Time):          {best['time'] / 60:.2f} min")

    # 绘图 1: 轮换时间分析
    rots = [x[0] for x in summary]
    times = [x[1] / 60 for x in summary]

    plt.figure(figsize=(10, 5))
    plt.plot(rots, times, 'o-', color='tab:blue', linewidth=2)
    plt.title("No-Drop Strategy: Rotation Time Optimization", fontsize=14)
    plt.xlabel("Rotation Interval for Strong Riders (s)", fontsize=12)
    plt.ylabel("Finish Time (min)", fontsize=12)
    plt.grid(True)
    plt.show()

    # 绘图 2: 能量曲线
    plt.figure(figsize=(12, 6))
    for r in team:
        # 弱者用虚线，强者用实线
        ls = '--' if r.role == "Passenger" else '-'
        width = 2.5 if r.role == "Passenger" else 1.5
        plt.plot(best['log'][r.id], label=r.id, linestyle=ls, linewidth=width)

    plt.title(f"No-Drop Energy Dynamics (Speed: {best['speed']:.2f} km/h)\nWeak riders (Dashed) do zero pulling",
              fontsize=14)
    plt.ylabel("W' Balance (J)", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.axhline(0, color='black', linestyle=':')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()