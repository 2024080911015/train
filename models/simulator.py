# models/simulator.py
import numpy as np


class WPrimeBalanceSimulator:
    def __init__(self, cyclist):
        self.cyclist = cyclist
        self.g = 9.81
        self.rho = 1.225
        # 优先使用车手专属 CdA，否则默认
        self.cd_a = getattr(self.cyclist, 'cd_area', 0.25)
        self.mu_roll = 0.003
        self.mu_tire = 0.8

        # 质量 = 车手 + 8kg 车重
        self.mass = getattr(self.cyclist, 'mass', 75.0)
        self.total_mass = self.mass + 8.0

    def run_segment_simulation(self, power_strategy, course_data, wind_speed=0, wind_deg=0):
        """
        执行赛道仿真 (高精度分步积分版)
        """
        current_w = self.cyclist.w_prime
        total_time = 0.0

        # 用于记录每一“大段”结束时的状态，方便画图
        w_history = []

        # 初始速度
        v_curr = 0.1

        # 是否力竭标记
        is_exhausted = False

        # 定义仿真子步长 (米)
        # 将长赛段切碎，防止 5km 长坡一步算完导致物理失真
        DX_STEP = 50.0

        for i, segment in enumerate(course_data):
            # 获取该段的目标功率
            p_target = power_strategy[i] if i < len(power_strategy) else power_strategy[-1]

            # 如果已经力竭，强制将目标功率降为恢复功率 (例如 100W 或 CP的一半)
            if is_exhausted:
                p_target = self.cyclist.cp * 0.5

            seg_len = segment['length']
            slope = segment['slope']
            radius = segment['radius']

            # --- 子步长积分循环 ---
            dist_covered = 0.0

            while dist_covered < seg_len:
                # 确定当前步长 (处理最后剩余的一点距离)
                dx = min(DX_STEP, seg_len - dist_covered)

                # 1. 物理受力计算
                # 简化风速模型 (假设正面风)
                v_air = v_curr + wind_speed

                f_drag = 0.5 * self.rho * self.cd_a * (v_air ** 2)
                f_grav = self.total_mass * self.g * np.sin(slope)
                f_roll = self.total_mass * self.g * self.mu_roll * np.cos(slope)
                f_resist = f_drag + f_grav + f_roll

                # 2. 驱动力 (防止除零)
                if v_curr < 0.1: v_curr = 0.1
                f_prop = p_target / v_curr

                # 3. 加速度 a = F_net / m
                acc = (f_prop - f_resist) / self.total_mass

                # 4. 更新速度 (v^2 = u^2 + 2as)
                v_next_sq = v_curr ** 2 + 2 * acc * dx

                if v_next_sq <= 0.1:
                    v_curr = 0.1  # 爬不动了，甚至要倒退，强制最低速
                else:
                    v_curr = np.sqrt(v_next_sq)

                # 5. 弯道限速检查 (仅在进入弯道前或弯道中生效)
                # 简化处理：如果这一段半径很小，全段限速
                if radius < 100:
                    v_limit = np.sqrt(self.mu_tire * self.g * radius)
                    v_curr = min(v_curr, v_limit)

                # 6. 累加时间 dt = dx / v_avg
                # 简单算术平均
                dt = dx / v_curr
                total_time += dt

                # 7. W' (无氧能量) 结算模型
                if p_target > self.cyclist.cp:
                    # 消耗: (P - CP) * t
                    loss = (p_target - self.cyclist.cp) * dt
                    current_w -= loss
                else:
                    # 恢复: Skiba 模型简化版
                    # recovery_power = (CP - P)
                    # 实际恢复通常是非线性的，这里使用标准指数衰减或简化线性
                    # 简化：W_bal 恢复速率与 (CP - P) 成正比
                    # 时间常数 tau 通常约 500s 左右，这里用线性简化避免过度复杂
                    t_w = 546.0  # 典型恢复常数
                    # D_cp = CP - P
                    # W_exp = W' - W_bal
                    # dW/dt = (W_exp / t_w) * (D_cp / CP) ... 比较复杂的公式
                    # 我们用简单的线性恢复作为竞赛近似:
                    rec_rate = (self.cyclist.cp - p_target)
                    # 限制恢复效率 (人不能像电池一样完美充电)
                    current_w += rec_rate * dt * 0.5

                    # 8. 边界检查
                if current_w > self.cyclist.w_prime:
                    current_w = self.cyclist.w_prime

                if current_w < 0:
                    current_w = 0
                    is_exhausted = True
                    # 一旦力竭，本段剩余部分和后续所有段都只能用低功率蹭
                    # 可以在这里 break 子循环，或者让它继续跑但 P 已被置为低值

                dist_covered += dx

            # 记录这一大段结束时的 W'
            w_history.append(current_w)

        return total_time, w_history, is_exhausted