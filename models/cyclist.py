import numpy as np


class Cyclist:
    def __init__(self, name, rider_type, gender, cp, w_prime, mass, cd_area, p_max=None):
        """
        初始化车手对象
        :param name: 姓名/ID
        :param rider_type: 类型 (TT Specialist / Sprinter)
        :param gender: 性别 (Male / Female)
        :param cp: 临界功率 Critical Power (Watts)
        :param w_prime: 无氧做功容量 W' (Joules)
        :param mass: 体重 (kg) -> 用于计算重力 F_gravity
        :param cd_area: 风阻有效面积 CdA (m^2) -> 用于计算风阻 F_aero
        :param p_max: 峰值功率 (Watts, 可选)
        """
        self.name = name
        self.rider_type = rider_type
        self.gender = gender
        self.cp = cp
        self.w_prime = w_prime
        self.mass = mass  # 新增参数
        self.cd_area = cd_area  # 新增参数
        self.p_max = p_max

    def get_theoretical_power(self, durations):
        """
        根据 CP 模型计算给定持续时间下的最大功率 P(t) = CP + W' / t
        """
        durations = np.array(durations)
        # 避免除以零
        with np.errstate(divide='ignore'):
            powers = self.cp + self.w_prime / durations

        # 如果定义了 p_max，进行截断
        if self.p_max:
            powers = np.minimum(powers, self.p_max)
        return powers

    def __repr__(self):
        return (f"Cyclist({self.name}: {self.gender} {self.rider_type}, "
                f"CP={self.cp}W, W'={self.w_prime}J, "
                f"Mass={self.mass}kg, CdA={self.cd_area}m^2)")