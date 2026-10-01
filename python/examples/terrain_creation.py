"""
Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Terrain（地形）创建示例
-------------------------
演示地形网格的用法。
按 'R' 键重置仿真
"""

import numpy as np
from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
import os

from isaacgym import gymutil, gymapi
from isaacgym.terrain_utils import *
from math import sqrt


# 初始化 gym：获取 Gym 接口实例（对 Isaac Gym 一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments()

# 配置仿真参数（Z 轴向上，重力沿 -Z 方向）
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_FLEX:
    print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
    args.physics_engine = gymapi.SIM_PHYSX
sim_params.substeps = 2
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# 加载小球资产（asset），后面把大量小球撒到地形上滚动
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "assets")
asset_file = "urdf/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# 设置环境（env）网格布局：800 个小球，每行 80 个，间距 0.56 米
num_envs = 800
num_per_row = 80
env_spacing = 0.56
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
pose = gymapi.Transform()
pose.r = gymapi.Quat(0, 0, 0, 1)
pose.p.z = 1.
pose.p.x = 3.

envs = []
# 设置随机种子（保证每次运行生成相同的小球颜色）
np.random.seed(17)
for i in range(num_envs):
    # 创建环境（env：一块独立的仿真空间）
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # 生成随机亮色给小球
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # 创建小球 actor（演员：资产的实例）
    ahandle = gym.create_actor(env, asset, pose, None, 0, 0)
    gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# 保存整个仿真初始刚体状态的副本，按 R 键时写回去即可重置
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# 依次生成 8 种可用地形，拼成一个大高度场（heightfield）
num_terains = 8
terrain_width = 12.
terrain_length = 12.
horizontal_scale = 0.25  # 水平分辨率（米/格）
vertical_scale = 0.005  # 垂直分辨率（每个高度单位对应的米数）
num_rows = int(terrain_width/horizontal_scale)
num_cols = int(terrain_length/horizontal_scale)
heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16)


def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)


# 每种地形占高度场的 num_rows 行，从上到下依次为：
# ① 随机起伏 ② 斜坡 ③ 金字塔斜坡 ④ 离散障碍块 ⑤ 波浪 ⑥ 台阶 ⑦ 金字塔台阶 ⑧ 踏脚石
heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.2, step=0.2, downsampled_scale=0.5).height_field_raw
heightfield[num_rows:2*num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
heightfield[2*num_rows:3*num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
heightfield[3*num_rows:4*num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(), max_height=0.5, min_size=1., max_size=5., num_rects=20).height_field_raw
heightfield[4*num_rows:5*num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=2., amplitude=1.).height_field_raw
heightfield[5*num_rows:6*num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
heightfield[6*num_rows:7*num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
heightfield[7*num_rows:8*num_rows, :] = stepping_stones_terrain(new_sub_terrain(), stone_size=1.,
                                                                stone_distance=1., max_height=0.5, platform_size=0.).height_field_raw

# 把高度场转换成三角网格并添加到仿真中作为地形
# slope_threshold 用于平滑过陡的斜坡边缘
vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = -1.
tm_params.transform.p.y = -1.
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

# 创建可视化窗口（viewer），使用默认相机属性
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(-5, -5, 15)
cam_target = gymapi.Vec3(0, 0, 10)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 订阅 R 键事件用于重置
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

while not gym.query_viewer_has_closed(viewer):

    # 从可视化窗口获取输入事件并相应处理
    for evt in gym.query_viewer_action_events(viewer):
        # 按下 R 键：把所有刚体状态恢复为初始状态
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # 推进物理仿真一步，并取回计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新渲染画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待真实时间走完一个 dt，
    # 让物理仿真与渲染帧率同步（实时观看效果）
    gym.sync_frame_time(sim)

# 销毁资源：先关闭可视化窗口，再销毁仿真器
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
