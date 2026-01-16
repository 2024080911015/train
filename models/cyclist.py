import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

class Cyclist:
    def __init__(self, name, rider_type, gender, cp, w_prime, p_max=None):
        self.name = name
        self.rider_type = rider_type
        self.gender = gender
        self.cp = cp  # 临界功率 (Watts) [cite: 847]
        self.w_prime = w_prime  # 无氧做功容量 (Joules) [cite: 847]
        self.p_max = p_max  # 瞬间峰值功率 (Watts) [cite: 281]
    def get_theoretical_power(self, durations):
        durations = np.array(durations)
        powers = self.cp + self.w_prime / durations
        return powers
    def __repr__(self):
        return f"Cyclist({self.name} [{self.gender} {self.rider_type}]: CP={self.cp}W, W'={self.w_prime}J)"


class PowerDurationModel:
    @staticmethod
    def fit_2_parameter_model(durations, powers):
        """
        基于双参数模型 P(t) = W'/t + CP 拟合数据 [cite: 16]
        输入: durations (秒列表), powers (瓦特列表)
        输出: cp, w_prime
        """
        # 线性化处理: Power = W' * (1/t) + CP
        # y = mx + c  -> Power = W' * (1/Time) + CP
        # 斜率是 W', 截距是 CP [cite: 448]

        x = 1 / np.array(durations)
        y = np.array(powers)

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        w_prime = slope
        cp = intercept
        return cp, w_prime

    @staticmethod
    def fit_apr_model(durations, powers, p_max):
        """
        针对短时间爆发的 APR 模型 (极端强度域) [cite: 281]
        P(t) = P_3min + (P_max - P_3min) * e^(-k*t)
        通常用于拟合 < 3分钟的数据
        """
        # 这里需要非线性最小二乘法 (scipy.optimize.curve_fit)
        # 具体实现取决于题目给的数据点密度
        pass
