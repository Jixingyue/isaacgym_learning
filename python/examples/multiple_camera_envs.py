"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


多环境多相机示例
--------------------
演示在场景的多个环境（env）中创建多个相机传感器的能力
"""
import os

import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# 获取 Gym 接口实例（一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(description="Multiple Cameras Example",
                               custom_parameters=[
                                   {"name": "--save_images", "action": "store_true", "help": "Write RGB and Depth Images To Disk"},
                                   {"name": "--up_axis_z", "action": "store_true", "help": ""}])

# 配置仿真参数与地面参数
sim_params = gymapi.SimParams()
plane_params = gymapi.PlaneParams()

# 默认竖直向上是 Y 轴（可用 --up_axis_z 改为 Z 轴向上）
sim_params.up_axis = gymapi.UP_AXIS_Y
if args.up_axis_z:
    sim_params.up_axis = gymapi.UP_AXIS_Z

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# 创建仿真器（这里计算设备和渲染设备都用 0 号）
sim = gym.create_sim(0, 0, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# 添加地面
gym.add_ground(sim, plane_params)

# 创建环境（env：一块独立的仿真空间），这里建 2 个
n_envs = 2
spacing = 1
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
num_per_row = int(np.sqrt(n_envs))

env_ptrs = [gym.create_env(sim, lower, upper, num_per_row) for _ in range(n_envs)]
env_idxs = [i for i in range(n_envs)]

# 配置相机属性：水平视场角 75 度，分辨率 1920x1080
camera_props = gymapi.CameraProperties()
camera_props.horizontal_fov = 75.0
camera_props.width = 1920
camera_props.height = 1080


# 用字典记录相机句柄（handle：后续操作这个相机的"凭证"），结构为 {环境编号: {相机名: 句柄}}
ch_map = {idx: {} for idx in env_idxs}

# 第一个相机 cam0 的摆放位置
pos = gymapi.Vec3(1.38, 1.0, 0)
name = 'cam0'

# 在每个环境里各创建一个 cam0
for env_idx in env_idxs:
    env_ptr = env_ptrs[env_idx]
    # 在环境中创建相机传感器，返回相机句柄
    ch = gym.create_camera_sensor(env_ptr, camera_props)
    ch_map[env_idx][name] = ch
    # 设置相机的位姿（位置 + 朝向）
    gym.set_camera_transform(ch, env_ptr, gymapi.Transform(p=pos, r=gymapi.Quat()))
    print("Added {} with handle {} in env {} | View matrix: {}".format(name, ch, env_idx, gym.get_camera_view_matrix(sim, env_ptr, ch)))

# 第二个相机 cam1 的摆放位置（换一个角度看）
pos = gymapi.Vec3(0.5, 9.0, .8)
name = 'cam1'

# 在每个环境里再各创建一个 cam1
for env_idx in env_idxs:
    env_ptr = env_ptrs[env_idx]
    ch = gym.create_camera_sensor(env_ptr, camera_props)
    ch_map[env_idx][name] = ch
    gym.set_camera_transform(ch, env_ptr, gymapi.Transform(p=pos, r=gymapi.Quat()))
    print("Added {} with handle {} in env {} | View matrix: {}".format(name, ch, env_idx, gym.get_camera_view_matrix(sim, env_ptr, ch)))

# 如果指定了 --save_images，先创建用于保存图片的目录
if args.save_images and not os.path.exists("multiple_camera_images"):
    os.mkdir("multiple_camera_images")

# 一次性渲染所有相机传感器的画面
gym.render_all_camera_sensors(sim)

# 逐环境处理每个相机：打印视图矩阵，需要的话把彩色画面保存成 png 文件
for env_idx in env_idxs:
    env_ptr = env_ptrs[env_idx]

    name = 'cam0'
    ch = ch_map[env_idx][name]
    rgb_filename = "multiple_camera_images/rgb_env%d_cam0.png" % (env_idx)
    if args.save_images:
        # 把相机的彩色画面写入文件
        gym.write_camera_image_to_file(sim, env_ptr, ch, gymapi.IMAGE_COLOR, rgb_filename)

    print("View matrix of {} with handle {} in env {}: {}".format(name, ch, env_idx, gym.get_camera_view_matrix(sim, env_ptr, ch)))

    name = 'cam1'
    ch = ch_map[env_idx][name]
    rgb_filename = "multiple_camera_images/rgb_env%d_cam1.png" % (env_idx)

    if args.save_images:
        # 把相机的彩色画面写入文件
        gym.write_camera_image_to_file(sim, env_ptr, ch, gymapi.IMAGE_COLOR, rgb_filename)

    print("View matrix of {} with handle {} in env {}: {}".format(name, ch, env_idx, gym.get_camera_view_matrix(sim, env_ptr, ch)))

# 推进物理仿真一步，并取回结果
gym.simulate(sim)
gym.fetch_results(sim, True)
