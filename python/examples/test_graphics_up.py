"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Isaac Gym 图形（上行轴）示例
--------------------------
本示例演示如何更改 Isaac Gym 使用的"向上轴"（up axis）。
默认以 Y 轴为向上。通过 gymapi.UpAxis.UP_AXIS_Z 可以
把 Viewer 和相机传感器都改成 Z 轴向上。

需要 Pillow（旧名 PIL）才能从 Python 写出图片。运行 `pip install pillow`
 安装 Pillow。
"""


import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# 获取 Gym 接口实例（对 Isaac Gym 一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数（headless 无窗口运行；--save_images 保存图片；--up_axis_z 切换 Z 轴向上）
args = gymutil.parse_arguments(description="Graphics Example",
                               headless=True,
                               custom_parameters=[
                                   {"name": "--save_images", "action": "store_true", "help": "Write RGB and Depth Images To Disk"},
                                   {"name": "--up_axis_z", "action": "store_true", "help": ""}])

if args.save_images:
    from PIL import Image as im

# 获取默认仿真参数
sim_params = gymapi.SimParams()
if args.up_axis_z:
    # 选择了 Z 轴向上：相应调整向上轴和重力方向
    sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.flex.shape_collision_margin = 0.04

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# 创建仿真器：指定计算设备、渲染设备、物理引擎和仿真参数
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# 创建默认地面平面（Z 轴向上时地面的法向也要跟着改）
plane_params = gymapi.PlaneParams()
if args.up_axis_z:
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.distance = 2.0
gym.add_ground(sim, plane_params)

if not args.headless:
    # 用默认相机属性创建可视化窗口（viewer）
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')

# 设置环境（env）网格布局：本例只用 1 个环境
num_envs = 1
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 本示例使用的资产列表（重复加载 8 个小球资产）
repeat_assets = 8
asset_files = []
asset_root = "../../assets"
for i in range(repeat_assets):
    asset_files.append("urdf/ball.urdf")

# 加载全部资产
assets = []
for i in range(len(asset_files)):
    asset_handle = gym.load_asset(sim, asset_root, asset_files[i])
    assets.append(asset_handle)

# 准备存放 actor 句柄和环境句柄的列表
actor_handles = [[]]
envs = []

# 创建环境并把小球按网格摆放
for i in range(num_envs):
    actor_handles.append([])
    env = gym.create_env(sim, env_lower, env_upper, 4)
    envs.append(env)
    num_assets = len(assets)
    for j in range(num_assets):
        # 把资产在环境内按网格摆放。让 0 号 actor 从 10 米高处落下，
        # 这样输出图片里能明显看出相机是否跟随物体运动
        grid = np.ceil(np.sqrt(num_assets))
        d = 2 * spacing / grid
        x = d * (0.5 + np.mod(j, grid))
        y = 6.0
        z = d * (0.5 + np.floor(j/grid))
        actor_pose = gymapi.Transform()
        # 根据向上轴的不同交换 y/z 分量（Z 轴向上时"高度"放在 z 上）
        if sim_params.up_axis == gymapi.UpAxis.UP_AXIS_Z:
            actor_pose.p = gymapi.Vec3(x, z, y)
        else:
            actor_pose.p = gymapi.Vec3(x, y, z)
        actor_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        asset_name = "asset_%d" % j
        handle = gym.create_actor(env, assets[j], actor_pose, asset_name, i, 1, 0)
        actor_handles[i].append(handle)


# 创建相机传感器：从固定位置观看环境原点
# （注释里的第二个相机：可附着在 0 号 actor 的 0 号刚体上并随之移动，本例未启用）
if sim_params.up_axis == gymapi.UpAxis.UP_AXIS_Z:
    camera_position = gymapi.Vec3(4.0, 14.0, 10.0)
else:
    camera_position = gymapi.Vec3(14.0, 10.0, 4.0)
camera_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, camera_position, camera_target)

camera_handles = [[]]
for i in range(num_envs):
    camera_handles.append([])
    # 相机传感器属性：分辨率 360×240
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 360
    camera_properties.height = 240

    # 给第一个相机设置固定位置和观察目标，
    # 位置和目标点都是环境坐标系下的坐标
    h1 = gym.create_camera_sensor(envs[i], camera_properties)
    gym.set_camera_location(h1, envs[i], camera_position, camera_target)
    camera_handles[i].append(h1)

# 创建图片输出目录（若不存在）
if not os.path.exists("graphics_images"):
    os.mkdir("graphics_images")
frame_count = 0

# 主仿真循环
while True:
    # 推进物理仿真一步，并取回计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 把物理状态同步给图形系统
    gym.step_graphics(sim)

    # 渲染所有相机传感器
    gym.render_all_camera_sensors(sim)

    if args.save_images and np.mod(frame_count, 30) == 0:
        # 每 30 帧保存一次图片
        for i in range(num_envs):
            for j in range(1):
                # 把图片写到磁盘的 gym 工具仅推荐用于 RGB 图像。
                rgb_filename = "graphics_images/rgb_env%d_cam%d_frame%d.png" % (i, j, frame_count)
                gym.write_camera_image_to_file(sim, envs[i], camera_handles[i][j], gymapi.IMAGE_COLOR, rgb_filename)

    if not args.headless:
        # 渲染可视化窗口
        gym.draw_viewer(viewer, sim, True)

        # 等待真实时间走完一个 dt，使 viewer 与
        # 仿真速率同步。headless（无窗口）模式下不需要。
        gym.sync_frame_time(sim)

        # 检查退出条件——用户关闭了 viewer 窗口
        if gym.query_viewer_has_closed(viewer):
            break

    frame_count = frame_count + 1

print('Done')

# 清理资源：先关闭可视化窗口，再销毁仿真器
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
