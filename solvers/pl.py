import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

# === 1. 赛道数据 ===
segments = [
    {'length': 5000.0, 'slope': 0.0, 'radius': 10000.0, 'desc': 'Leg1_Flat_North'},
    {'length': 31.416, 'slope': 0.0, 'radius': 20.0, 'desc': 'Turn1'},
    {'length': 5000.0, 'slope': 0.04, 'radius': 10000.0, 'desc': 'Leg2_Climb_East'},
    {'length': 31.416, 'slope': 0.0, 'radius': 20.0, 'desc': 'Turn2'},
    {'length': 5000.0, 'slope': -0.04, 'radius': 10000.0, 'desc': 'Leg3_Descent_South'},
    {'length': 31.416, 'slope': 0.0, 'radius': 20.0, 'desc': 'Turn3'},
    {'length': 5000.0, 'slope': 0.0, 'radius': 10000.0, 'desc': 'Leg4_Flat_West'},
    {'length': 31.416, 'slope': 0.0, 'radius': 20.0, 'desc': 'Turn4'}
]

# === 2. 生成坐标 ===
x, y, z = 0.0, 0.0, 0.0
heading = np.pi / 2
xs, ys, zs, slopes = [x], [y], [z], [0.0]
step_size = 10.0
corner_indices = []

for seg in segments:
    if 'Turn' in seg['desc']:
        corner_indices.append(len(xs) - 1)
    num_steps = int(np.ceil(seg['length'] / step_size))
    if seg['radius'] < 50: num_steps = 20
    dist_step = seg['length'] / num_steps
    z_step = dist_step * seg['slope']
    turn_dir = -1
    for i in range(num_steps):
        if seg['radius'] > 5000: d_theta = 0
        else: d_theta = (seg['length'] / seg['radius'] / num_steps) * turn_dir
        heading += d_theta
        x += dist_step * np.cos(heading)
        y += dist_step * np.sin(heading)
        z += z_step
        xs.append(x); ys.append(y); zs.append(z); slopes.append(seg['slope'])

xs = np.array(xs); ys = np.array(ys); zs = np.array(zs); slopes = np.array(slopes)

# === 3. 绘图 ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 定义离散颜色
COLOR_FLAT = '#32CD32'   # 绿色
COLOR_CLIMB = '#FF4500'  # 红色
COLOR_DESCENT = '#00BFFF' # 蓝色

# 绘制线条
for i in range(len(xs)-1):
    s = slopes[i]
    if s > 0.01: color = COLOR_CLIMB
    elif s < -0.01: color = COLOR_DESCENT
    else: color = COLOR_FLAT
    ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2], color=color, linewidth=3)

# 标注拐角
for i, idx in enumerate(corner_indices):
    cx, cy, cz = xs[idx], ys[idx], zs[idx]
    label = f"Corner {i+1}"
    color = 'black'
    if i == 3:
        label = "Start/Finish"
        cx, cy, cz = xs[0], ys[0], zs[0]
        color = 'green'
    ax.text(cx, cy, cz + 50, label, color=color, fontsize=10, fontweight='bold', ha='center')
    ax.scatter(cx, cy, cz, color=color, s=50)

# 设置
ax.set_box_aspect((1, 1, 0.3))
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Elevation (m)')
ax.set_title('3D Course Profile: Rectangular Loop', fontsize=15, y=0.9)

# === 添加离散图例 (右下角) ===
legend_elements = [
    Patch(facecolor=COLOR_FLAT, label='Flat (0%)'),
    Patch(facecolor=COLOR_CLIMB, label='Climb (+4%)'),
    Patch(facecolor=COLOR_DESCENT, label='Descent (-4%)')
]

ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.0, 0.05), title="Slope Category")

ax.view_init(elev=30, azim=-60)
plt.tight_layout()
plt.show()