import numpy as np


class WPrimeBalanceSimulator:
    def __init__(self, cyclist):
        self.cyclist = cyclist

        # --- 1. 物理常数定义 ---
        self.g = 9.81
        self.rho = 1.225  # 空气密度 (kg/m^3)
        # [修复] 使用车手自己的 CdA 参数，如果没有则默认 0.32
        self.cd_a = getattr(self.cyclist, 'cd_area', 0.32)

        # [修复点] 这里统一命名为 mu_roll，防止报错
        self.mu_roll = 0.003  # 滚动阻力系数 (Rolling Resistance)
        self.mu_tire = 0.8  # 轮胎侧向摩擦系数 (用于急弯限制)

        # --- 2. 质量计算 ---
        # 尝试从车手对象获取体重，如果没有则默认 75kg
        self.mass = getattr(self.cyclist, 'mass', 75.0)
        # 人车总质量 (假设车重 8kg)
        self.total_mass = self.mass + 8.0

    def solve_velocity_with_limit(self, power, slope, radius):
        """
        核心物理引擎：计算在给定功率、坡度和弯道半径下的最终速度。
        包含: 1. 空气阻力+重力+滚阻平衡; 2. 弯道离心力安全限制。
        """
        # --- A. 求解物理动力学方程 (A*v^3 + B*v - P = 0) ---
        # 1. 准备系数
        A = 0.5 * self.rho * self.cd_a
        # [使用 mu_roll] 重力分量 + 滚动阻力分量
        B = self.total_mass * self.g * (np.sin(slope) + self.mu_roll * np.cos(slope))
        C = -power

        # 2. 求解三次方程实根
        # numpy.roots 可能会返回复数，需过滤
        roots = np.roots([A, 0, B, C])
        real_roots = roots[np.isreal(roots)].real
        # 取大于0的根 (物理意义上的速度)
        valid_v = real_roots[real_roots > 0]

        if len(valid_v) == 0:
            # 异常情况(如陡坡功率不足)，给一个极小的非零速度防止除零
            v_phys = 0.1
        else:
            # [修复] 取最小的正实根（对应稳态速度）
            v_phys = min(valid_v)

        # --- B. 求解安全极限 (急转弯限制) ---
        # 公式: v_safe <= sqrt(mu * g * R)
        if radius >= 9999:  # 9999代表直道
            v_safe = float('inf')
        else:
            v_safe = np.sqrt(self.mu_tire * self.g * radius)

        # --- C. 取两者较小值 (木桶效应) ---
        final_v = min(v_phys, v_safe)

        return final_v

    def run_segment_simulation(self, power_strategy, course_data, wind_speed=0, wind_deg=0):
        """
        :param wind_speed: 风速 (m/s)
        :param wind_deg: 风向 (0度为正北，与赛道方向的夹角)
        """
        current_w = self.cyclist.w_prime
        total_time = 0
        w_history = [current_w]
        speed_history = []  # 记录速度以便分析

        # 初始速度 (假设从静止开始，或者给一个初速度 10m/s)
        v_curr = 0.1

        is_exhausted = False
        cp_rel = self.cyclist.cp / self.mass

        for i, segment in enumerate(course_data):
            # 获取当前段的目标功率（如果不优化每一段，可以取策略参数计算）
            # 注意：如果插值后段数变多，power_strategy 数组长度也要匹配
            p_target = power_strategy[i] if i < len(power_strategy) else power_strategy[-1]

            slope = segment['slope']
            length = segment['length']
            radius = segment['radius']

            # === 核心修复：动力学迭代 ===
            # 将一段拆分为更小的时间步 (dt) 来模拟加速度，或者简单地整段模拟
            # 这里用欧拉法做简单积分：v_next = v_curr + a * dt

            # 1. 计算当前受力
            # 风速模型：假设简单的逆风/顺风 (headwind_component)
            # 实际上应该结合赛道方向计算，这里简化假设 wind_deg 是相对于前进方向
            v_air = v_curr + wind_speed * np.cos(np.radians(wind_deg))

            f_drag = 0.5 * self.rho * self.cd_a * (v_air ** 2)
            f_grav = self.total_mass * self.g * np.sin(slope)
            f_roll = self.total_mass * self.g * self.mu_roll * np.cos(slope)

            f_resist = f_drag + f_grav + f_roll

            # 2. 计算驱动力 F_prop = P / v
            # 避免除以零
            v_safe_div = max(v_curr, 0.1)
            f_prop = p_target / v_safe_div

            # 3. 计算净力与加速度
            f_net = f_prop - f_resist
            acc = f_net / self.total_mass

            # 4. 更新速度 (v^2 = v0^2 + 2ax)
            # 这种公式比时间积分更适合定长路段
            v_next_sq = v_curr ** 2 + 2 * acc * length

            if v_next_sq < 0:
                v_next = 0.1  # 无法爬坡，强制最小速度
            else:
                v_next = np.sqrt(v_next_sq)

            # === 安全限制 (弯道减速) ===
            # 如果这是一个急弯段，必须限制进入速度
            if radius < 100:
                v_limit = np.sqrt(self.mu_tire * self.g * radius)
                v_next = min(v_next, v_limit)

            # 5. 计算这一段耗时
            # 平均速度 approximation
            v_avg = (v_curr + v_next) / 2
            dt = length / max(v_avg, 0.1)
            total_time += dt

            # 更新状态供下一段使用
            v_curr = v_next
            speed_history.append(v_curr)

            # === W' 生理模型 (与原来保持一致) ===
            if p_target > self.cyclist.cp:
                loss = (p_target - self.cyclist.cp) * dt
                current_w -= loss
            else:
                p_rel = p_target / self.mass
                recovery_rate = cp_rel - (0.0879 * p_rel + 2.9214)  # 简化的Skiba公式
                if recovery_rate > 0:
                    current_w += recovery_rate * self.mass * dt

            if current_w > self.cyclist.w_prime: current_w = self.cyclist.w_prime
            if current_w < 0:
                current_w = 0
                is_exhausted = True

            w_history.append(current_w)

        return total_time, w_history, is_exhausted