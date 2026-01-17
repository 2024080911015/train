import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import splprep, splev


def calculate_menger_curvature(p1, p2, p3):
    """
    [数学核心] 三点定圆法计算曲率半径 R
    输入: p1, p2, p3 为 (x, y) 坐标
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # 1. 计算三边长
    a = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    b = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

    # 2. 计算面积 (叉乘法，带绝对值)
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    # 3. 计算半径 R = abc / 4S
    if area < 1e-6:  # 防止除零 (直线)
        return 10000.0

    R = (a * b * c) / (4 * area)
    return min(R, 10000.0)  # 设置上限


def generate_custom_course_data(total_points=2000):
    """
    生成自主设计的赛道数据
    :param total_points: 生成的数据点数量 (至少1000)
    :return: DataFrame (包含 distance, slope, radius, x, y)
    """
    # === 1. 定义赛道骨架 (关键点) ===
    # 设计一个复杂的环形赛道：20km左右
    # 坐标单位：米
    waypoints = np.array([
        [0, 0],
        [2000, 0],  # 长直道 (2km)
        [2500, 500],  # 缓弯
        [2500, 1500],  # 直道
        [2200, 1800],  # 急弯入口
        [2000, 1500],  # >> 急弯1 (发卡弯) <<
        [1500, 1500],  # 短直道
        [1500, 2500],  # 直道
        [1800, 2800],  # >> 急弯2 (直角) <<
        [3000, 2800],  # 长直道
        [3500, 2000],  # 缓弯
        [4000, 1000],  # >> 急弯3 (S弯上部) <<
        [3800, 800],  # >> 急弯4 (S弯下部) <<
        [1000, -500],  # 长直道回程
        [0, 0]  # 闭环
    ])

    # === 2. B样条插值 (使轨迹平滑) ===
    # k=3 (三次样条), s=0 (强制经过关键点), per=True (闭环)
    tck, u = splprep(waypoints.T, u=None, s=0.0, per=True, k=3)

    # 生成密集的点
    u_new = np.linspace(u.min(), u.max(), total_points)
    x_new, y_new = splev(u_new, tck, der=0)
    path_points = np.vstack((x_new, y_new)).T

    # === 3. 计算物理属性 ===
    course_list = []
    total_distance = 0

    for i in range(len(path_points)):
        # 索引处理 (闭环)
        idx_prev = (i - 1) % len(path_points)
        idx_curr = i
        idx_next = (i + 1) % len(path_points)

        p_prev = path_points[idx_prev]
        p_curr = path_points[idx_curr]
        p_next = path_points[idx_next]

        # A. 这一小段的长度
        # 实际上是当前点到下一个点的直线距离近似
        seg_len = np.linalg.norm(path_points[idx_next] - p_curr)

        # B. 这一点的曲率半径
        radius = calculate_menger_curvature(p_prev, p_curr, p_next)

        # C. 这一点的坡度 (自主设计)
        # 模拟：0-5km平路，5-10km上坡，10-15km下坡，15-20km起伏
        # [修复] 坡度统一使用弧度制，与 load_real_course 保持一致
        dist_km = total_distance / 1000.0
        slope_percent = 0.0
        if 5.0 <= dist_km < 10.0:
            slope_percent = 0.04  # 4% 上坡
        elif 10.0 <= dist_km < 15.0:
            slope_percent = -0.03  # 3% 下坡
        elif 15.0 <= dist_km < 20.0:
            slope_percent = 0.02 * np.sin(dist_km * 5)  # 起伏
        # 转换为弧度: arctan(坡度百分比)
        slope = np.arctan(slope_percent)

        course_list.append({
            'distance_from_start': total_distance,
            'length': seg_len,
            'slope': slope,
            'radius': radius,
            'x': p_curr[0],
            'y': p_curr[1]
        })

        total_distance += seg_len

    df = pd.DataFrame(course_list)
    return df


def visualize_course(df):
    """画出赛道形状和急弯位置"""
    plt.figure(figsize=(10, 8))

    # 散点图，颜色代表半径 (越红越急)
    # 限制颜色映射范围在 0-100米，让急弯更明显
    sc = plt.scatter(df['x'], df['y'], c=df['radius'],
                     cmap='jet_r', s=5, vmin=0, vmax=100)

    plt.colorbar(sc, label='Radius (m) [Red=Sharp Turn]')
    plt.title(f"Custom Designed Track\nTotal Length: {df['distance_from_start'].iloc[-1] / 1000:.2f} km")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    # 标记出半径 < 20米的急弯点
    sharp_turns = df[df['radius'] < 20]
    plt.scatter(sharp_turns['x'], sharp_turns['y'], color='black', marker='x', label='Sharp Turn (<20m)')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # 1. 生成数据
    df_track = generate_custom_course_data(total_points=2000)

    # 2. 保存 CSV (供 question_2.py 使用)
    import os

    if not os.path.exists('models/data'): os.makedirs('models/data')
    df_track.to_csv('data/course_custom.csv', index=False)
    print(f"赛道数据已生成: data/course_custom.csv (共 {len(df_track)} 个点)")

    # 3. 打印急弯统计
    sharp_count = len(df_track[df_track['radius'] < 20])
    print(f"检测到 {sharp_count} 个急转弯点 (R < 20m)")

    # 4. 可视化
    visualize_course(df_track)