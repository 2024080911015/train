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

    def run_segment_simulation(self, power_strategy, course_data):
        """
        运行整场比赛的仿真
        :param power_strategy: 每一段的目标功率列表
        :param course_data: 赛道数据列表
        :return: (total_time, w_history, is_exhausted)
        """
        current_w = self.cyclist.w_prime
        total_time = 0
        w_history = [current_w]
        is_exhausted = False

        # 预计算相对 CP (W/kg)，用于恢复公式
        cp_rel = self.cyclist.cp / self.mass

        for i, p in enumerate(power_strategy):
            # 1. 获取赛道信息
            segment = course_data[i]
            slope = segment['slope']
            length = segment['length']
            radius = segment.get('radius', 9999)  # 默认为直道

            # 2. 物理层：计算速度和耗时
            v = self.solve_velocity_with_limit(p, slope, radius)
            dt = length / v
            total_time += dt

            # 3. 生理层：计算 W' 变化
            if p > self.cyclist.cp:
                # [消耗阶段] P > CP
                loss = (p - self.cyclist.cp) * dt
                current_w -= loss
            else:
                # [恢复阶段] P < CP (使用线性恢复模型)
                p_rel = p / self.mass
                # 恢复率公式: CP_rel - (0.0879 * P_rel + 2.9214)
                recovery_rate_rel = cp_rel - (0.0879 * p_rel + 2.9214)

                if recovery_rate_rel > 0:
                    recovery_joules = recovery_rate_rel * self.mass * dt
                    current_w += recovery_joules

            # 4. 边界检查
            if current_w > self.cyclist.w_prime:
                current_w = self.cyclist.w_prime  # 满油

            if current_w < 0:
                is_exhausted = True
                current_w = 0  # 避免画图时出现负数

            w_history.append(current_w)

        return total_time, w_history, is_exhausted