import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# === 路径修复 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.cyclist import Cyclist


def annotate_w_prime(ax, rider, durations):
    """
    辅助函数：在图表上标注 W' (绘制虚线框和文字)
    选取 t=60s 作为一个典型的无氧做功时间点进行展示
    """
    demo_time = 60  # 在 1分钟 处展示
    # 计算该时间点的功率
    demo_power = rider.get_theoretical_power([demo_time])[0]

    # 1. 垂直虚线 (从曲线落到 CP 线)
    ax.vlines(x=demo_time, ymin=rider.cp, ymax=demo_power,
              colors='black', linestyles='--', alpha=0.6, linewidth=1.5)

    # 2. 水平虚线 (从曲线连到 Y轴)
    ax.hlines(y=demo_power, xmin=0, xmax=demo_time,
              colors='black', linestyles='--', alpha=0.4, linewidth=1)

    # 3. 标注文字 (放在矩形区域中间偏右)
    # 计算矩形中心位置
    mid_power = (rider.cp + demo_power) / 2
    ax.text(demo_time + 10, mid_power,
            f"  W' Area\n  ({rider.w_prime} J)",
            fontsize=9, color='black', verticalalignment='center', fontweight='bold')

    # 4. 在 Y轴上标记该点的功率值 (可选，增加可读性)
    ax.text(0, demo_power, f"{int(demo_power)}W ",
            ha='right', va='center', fontsize=8, color='black')


def solve_q1():
    print("Running Question 1: Plotting Separate Male/Female Power Profiles...")

    # === 创建车手数据 ===
    riders = [
        # --- Males ---
        Cyclist("Male TT", "TT Specialist", "Male", cp=400, w_prime=20000, mass=72, cd_area=0.23, p_max=1200),
        Cyclist("Male Sprinter", "Sprinter", "Male", cp=340, w_prime=35000, mass=75, cd_area=0.28, p_max=1600),
        # --- Females ---
        Cyclist("Female TT", "TT Specialist", "Female", cp=260, w_prime=15000, mass=60, cd_area=0.20, p_max=900),
        Cyclist("Female Sprinter", "Sprinter", "Female", cp=220, w_prime=25000, mass=62, cd_area=0.24, p_max=1100)
    ]

    # === 绘图设置 ===
    durations = np.linspace(5, 1800, 1000)  # 关注前 30分钟 (1800s) 以便看清 W' 区域

    # 创建 1行2列 的子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)  # sharey=False 因为男女功率差异大，独立坐标轴看细节更清楚

    # 定义分组和样式
    groups = {
        'Male': {'ax': axes[0], 'color_base': 'blue'},
        'Female': {'ax': axes[1], 'color_base': 'red'}
    }

    line_styles = {'TT Specialist': '-', 'Sprinter': '--'}

    # 遍历车手进行绘图
    for rider in riders:
        group_info = groups[rider.gender]
        ax = group_info['ax']
        base_color = group_info['color_base']

        # 区分同一性别下的不同类型颜色深浅或样式
        # 这里用样式区分：TT是实线，Sprinter是虚线
        # 颜色统一用该性别的色系
        if rider.rider_type == 'TT Specialist':
            plot_color = 'tab:blue' if rider.gender == 'Male' else 'tab:red'
        else:
            plot_color = 'navy' if rider.gender == 'Male' else 'darkred'  # 冲刺手用深一点的颜色

        powers = rider.get_theoretical_power(durations)

        # 1. 绘制功率曲线
        ax.plot(durations, powers,
                label=f"{rider.rider_type} (CP={rider.cp}W)",
                color=plot_color, linestyle=line_styles[rider.rider_type], linewidth=2.5)

        # 2. 绘制 CP 线 (基准线)
        ax.axhline(y=rider.cp, color=plot_color, linestyle=':', alpha=0.5, linewidth=1.5)
        ax.text(durations[-1], rider.cp, f" CP={rider.cp}", va='center', fontsize=9, color=plot_color,
                fontweight='bold')

        # 3. 标注 W' (仅对每组的 TT Specialist 进行标注，避免图表太乱)
        if rider.rider_type == 'TT Specialist':
            annotate_w_prime(ax, rider, durations)

    # === 美化图表 ===

    # 设置 Male 子图
    axes[0].set_title("Male Cyclists Power Profile", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Power Output (Watts)", fontsize=12)
    axes[0].set_xlabel("Duration (seconds)", fontsize=12)
    axes[0].set_ylim(0, 1650)
    axes[0].set_xlim(0, 1800)
    axes[0].grid(True, which='both', linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper right', frameon=True, shadow=True)

    # 设置 Female 子图
    axes[1].set_title("Female Cyclists Power Profile", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Power Output (Watts)", fontsize=12)
    axes[1].set_xlabel("Duration (seconds)", fontsize=12)
    axes[1].set_ylim(0, 1200)  # 哪怕女性功率低一些，我们也调低上限，让曲线充满画面
    axes[1].set_xlim(0, 1800)
    axes[1].grid(True, which='both', linestyle='--', alpha=0.3)
    axes[1].legend(loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()

    # 保存
    images_dir = os.path.join(project_root, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    save_path = os.path.join(images_dir, 'Q1_Power_Profiles_Split_Gender.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nGraph saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    solve_q1()