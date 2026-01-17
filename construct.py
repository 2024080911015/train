import matplotlib.pyplot as plt
import numpy as np

# === 1. 准备数据 ===
labels = ['Tokyo (Men)', 'Flanders (Men)', 'Flanders (Women)', 'Tokyo (Women)']
real_times = [55.07, 47.80, 36.09, 30.22]
sim_tt_times = [57.71, 52.05, 40.12, 32.93]
sim_spr_times = [64.75, 57.86, 44.67, 37.51]

# === 2. 构建雷达图 ===
N = len(labels)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})

# --- 关键修改 1: 旋转方向 (正北起步，顺时针) ---
ax.set_theta_offset(np.pi / 2)  # 起点设为 90度 (12点钟方向)
ax.set_theta_direction(-1)      # 顺时针方向 (符合常规阅读习惯)

# --- 关键修改 2: 大幅增加文字间距 ---
ax.tick_params(pad=35)

# 设置标签
plt.xticks(angles[:-1], labels, color='black', size=12, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=10)

# --- 关键修改 3: 进一步扩大显示范围 (留白) ---
plt.ylim(0, 90)

# --- 绘制数据 ---
values_real = real_times + real_times[:1]
ax.plot(angles, values_real, linewidth=2, linestyle='solid', label='Real Winner', color='#2ca02c')
ax.fill(angles, values_real, '#2ca02c', alpha=0.15)

values_tt = sim_tt_times + sim_tt_times[:1]
ax.plot(angles, values_tt, linewidth=2, linestyle='dashed', label='Simulated (TT)', color='#1f77b4')
ax.fill(angles, values_tt, '#1f77b4', alpha=0.1)

values_spr = sim_spr_times + sim_spr_times[:1]
ax.plot(angles, values_spr, linewidth=2, linestyle='dotted', label='Simulated (Sprinter)', color='#d62728')
ax.fill(angles, values_spr, '#d62728', alpha=0.05)

# === 3. 布局微调 ===
# 将标题抬高，避开正上方的 Tokyo (Men)
plt.title('Performance Comparison:\nReal vs. Simulation (Time in Minutes)', weight='bold', size=16, y=1.15)

# 图例保持在底部
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=11)

plt.tight_layout()
plt.show()