import sys
import os
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# Ensure we can import from models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.cyclist import PowerDurationModel, Cyclist


def solve_q1():
    print("Running Question 1: Power Duration Model Fitting...")
    riders = [
        # 男性车手
        Cyclist("Rider A", "TT Specialist", "Male", cp=400, w_prime=20000, p_max=1200),
        Cyclist("Rider B", "Sprinter", "Male", cp=320, w_prime=35000, p_max=1600),

        # 女性车手
        Cyclist("Rider C", "TT Specialist", "Female", cp=270, w_prime=15000, p_max=900),
        Cyclist("Rider D", "Sprinter", "Female", cp=220, w_prime=22000, p_max=1100)
    ]
    durations = np.linspace(5, 3600, 1000)

    # 3. 绘图设置
    plt.figure(figsize=(12, 8))

    # 颜色映射
    colors = {'Male': 'blue', 'Female': 'red'}
    line_styles = {'TT Specialist': '-', 'Sprinter': '--'}

    for rider in riders:
        # 计算功率曲线
        powers = rider.get_theoretical_power(durations)

        # 绘图
        label_text = f"{rider.gender} {rider.rider_type} (CP={rider.cp}W, W'={rider.w_prime}J)"
        color = colors[rider.gender]
        ls = line_styles[rider.rider_type]

        plt.plot(durations, powers, label=label_text, color=color, linestyle=ls, linewidth=2)

    # 4. 图表美化
    plt.title("Power Profiles: Time Trial Specialists vs. Sprinters", fontsize=16)
    plt.xlabel("Duration (seconds)", fontsize=12)
    plt.ylabel("Maximum Sustainable Power (Watts)", fontsize=12)

    # 使用对数坐标轴往往能更清晰地展示短时间和长时间的差异（可选）
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(fontsize=10)

    # 5. 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, 'Q1_Power_Curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to: {save_path}")

    # 显示图表
    plt.show()


if __name__ == "__main__":
    solve_q1()


