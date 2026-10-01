# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import gymutil, gymapi
from math import sqrt


def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None,):
    """
    生成随机起伏地形（rough terrain）：先在粗网格上随机采样高度，再插值上采样成平滑的随机起伏

    Parameters
        terrain (SubTerrain): 地形对象
        min_height (float): 地形的最小高度 [米]
        max_height (float): 地形的最大高度 [米]
        step (float): 两点之间的最小高度变化量 [米]
        downsampled_scale (float): 两个随机采样点之间的间距（必须大于等于 terrain.horizontal_scale）

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # 把参数从"米"换算成离散格子单位
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
        terrain.length * terrain.horizontal_scale / downsampled_scale)))

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw += z_upsampled.astype(np.int16)
    return terrain


def sloped_terrain(terrain, slope=1):
    """
    生成斜坡地形：整块地形沿 X 方向线性抬升（slope 为负则下降）

    Parameters:
        terrain (SubTerrain): 地形对象
        slope (int): 坡度（正值上坡、负值下坡）
    Returns:
        terrain (SubTerrain): 更新后的地形
    """

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
    terrain.height_field_raw[:, np.arange(terrain.length)] += (max_height * xx / terrain.width).astype(terrain.height_field_raw.dtype)
    return terrain


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.):
    """
    生成金字塔斜坡地形：高度从四周向中心逐层抬升，中心保留一块平地

    Parameters:
        terrain (terrain): 地形对象
        slope (int): 坡度（正值或负值）
        platform_size (float): 地形中心平地的边长 [米]
    Returns:
        terrain (SubTerrain): 更新后的地形
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x-xx)) / center_x
    yy = (center_y - np.abs(center_y-yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain


def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
    """
    生成离散障碍物地形：随机摆放若干矩形凸台/凹坑，机器人需要绕行或跨越

    Parameters:
        terrain (terrain): 地形对象
        max_height (float): 障碍物的最大高度（实际取值来自 [-max, -max/2, max/2, max]） [米]
        min_size (float): 矩形障碍物的最小尺寸 [米]
        max_size (float): 矩形障碍物的最大尺寸 [米]
        num_rects (int): 随机生成的障碍物数量
        platform_size (float): 地形中心平地的边长 [米]
    Returns:
        terrain (SubTerrain): 更新后的地形
    """
    # 把参数从"米"换算成离散格子单位
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def wave_terrain(terrain, num_waves=1, amplitude=1.):
    """
    生成波浪地形：高度随 sin/cos 周期性起伏，像连绵的波浪

    Parameters:
        terrain (terrain): 地形对象
        num_waves (int): 地形长度方向上正弦波的个数
    Returns:
        terrain (SubTerrain): 更新后的地形
    """
    amplitude = int(0.5*amplitude / terrain.vertical_scale)
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        terrain.height_field_raw += (amplitude*np.cos(yy / div) + amplitude*np.sin(xx / div)).astype(
            terrain.height_field_raw.dtype)
    return terrain


def stairs_terrain(terrain, step_width, step_height):
    """
    生成台阶地形：沿 X 方向像楼梯一样逐级升高

    Parameters:
        terrain (terrain): 地形对象
        step_width (float):  每级台阶的宽度 [米]
        step_height (float): 每级台阶的高度 [米]
    Returns:
        terrain (SubTerrain): 更新后的地形
    """
    # 把参数从"米"换算成离散格子单位
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] += height
        height += step_height
    return terrain


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):
    """
    生成金字塔台阶地形：四周低、向中心逐级升高的环形台阶，中心保留平地（常用于训练上下台阶）

    Parameters:
        terrain (terrain): 地形对象
        step_width (float): 每级台阶的宽度 [米]
        step_height (float): 每级台阶的高度 [米]
        platform_size (float): 地形中心平地的边长 [米]
    Returns:
        terrain (SubTerrain): 更新后的地形
    """
    # 把参数从"米"换算成离散格子单位
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain


def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-10):
    """
    生成踏脚石地形：稀疏散布的"石块"，石块之间是深坑，用于训练机器人的精确落足能力

    Parameters:
        terrain (terrain): 地形对象
        stone_size (float): 石块的水平尺寸 [米]
        stone_distance (float): 石块之间的间距（即坑洞的尺寸） [米]
        max_height (float): 石块的最大高度（可正可负） [米]
        platform_size (float): 地形中心平地的边长 [米]
        depth (float): 坑洞的深度（默认=-10.） [米]
    Returns:
        terrain (SubTerrain): 更新后的地形
    """
    # 把参数从"米"换算成离散格子单位
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # 填充行首的第一块石块（考虑随机偏移）
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # 沿 X 方向逐块填充这一行
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # 填充列首的第一块石块（考虑随机偏移）
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # 沿 Y 方向逐块填充这一列
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    将高度场数组转换为"顶点 + 三角形"表示的三角网格。

    Isaac Gym 的碰撞体只认三角网格、不认高度场，因此地形生成后必须经过这一步转换。
    可选功能：把坡度超过阈值的斜面"竖直化"，形成垂直墙面：

        若 (y2-y1)/(x2-x1) > slope_threshold -> 把 A 点移到 A'（即令 x1 = x2）。对所有方向都做同样处理。
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): 输入的高度场
        horizontal_scale (float): 高度场的水平缩放（每个格子的边长） [米]
        vertical_scale (float): 高度场的垂直缩放（高度值 1 对应的米数） [米]
        slope_threshold (float): 坡度阈值，超过该阈值的表面会被竖直化。若为 None 则不做修正（默认：None）
    Returns:
        vertices (np.array(float)): 形状为 (num_vertices, 3) 的数组，每行是一个顶点的空间坐标 [米]
        triangles (np.array(int)): 形状为 (num_triangles, 3) 的数组，每行是构成一个三角形的 3 个顶点的索引。
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # 由高度场网格生成三角网格的顶点与三角形（每个格子沿对角线切成 2 个三角形）
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles


class SubTerrain:
    """子地形类：保存一块 height_field_raw 高度场及其尺度参数。

    horizontal_scale 是水平分辨率（一个格子代表多少米），
    vertical_scale 是垂直分辨率（高度值 +1 代表升高多少米）。
    """

    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        """初始化一块全平（高度全为 0）的地形，网格尺寸为 width x length 个格子"""
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
