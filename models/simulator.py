import numpy as np

class WPrimeBalanceSimulator:
    def __init__(self, cyclist):
        self.cyclist = cyclist

    def calculate_tau(self, power):
        """计算恢复时间常数 Tau """
        d_cp = self.cyclist.cp - power
        # 如果 d_cp < 0 (即 P > CP)，这个函数不适用，但在循环中会被过滤
        if d_cp < 0: return np.inf
        return 546 * np.exp(-0.01 * d_cp) + 316

    def simulate_race(self, power_series, time_step=1.0):
        """
        模拟整个比赛过程中的 W' 余额
        输入: power_series (每秒的功率数组)
        输出: w_prime_balance_series (每秒的 W' 余额)
        """
        w_bal = [self.cyclist.w_prime]  # 初始状态满油
        current_w = self.cyclist.w_prime
        cp = self.cyclist.cp

        for p in power_series:
            if p > cp:
                # 消耗阶段: 线性减少
                expenditure = (p - cp) * time_step
                current_w -= expenditure
            else:
                # 恢复阶段: 指数恢复  (Skiba 2 微分逻辑)
                tau = self.calculate_tau(p)
                # 恢复公式: W_bal(t) = W' - (W' - W_bal_prev) * e^(-dt/tau)
                # 注意：这里是指“剩余的亏空”会随时间衰减
                w_deficit = self.cyclist.w_prime - current_w
                current_w = self.cyclist.w_prime - w_deficit * np.exp(-time_step / tau)

            # 物理限制：W' 不能超过上限，也不能透支（假设透支即力竭）
            if current_w > self.cyclist.w_prime:
                current_w = self.cyclist.w_prime

            w_bal.append(current_w)

            if current_w < 0:
                print("警告: 车手力竭！W' 耗尽。")

        return np.array(w_bal)
