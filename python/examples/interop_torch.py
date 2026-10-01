"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


PyTorch 张量互操作(Tensor API)示例 —— RL 训练的核心接口
---------------------------------------------------------
演示 Isaac Gym 的张量 API:用 PyTorch 张量批量读写仿真状态,
数据直接留在 GPU 显存中,无需逐环境循环拷贝 —— 这是大规模强化学习
训练的高速数据通道(IsaacGymEnvs 等训练框架正建立在此接口之上)。
- 创建 16 个环境,每个环境一个下落弹跳的球
- 演示如何把相机图像获取为 GPU 张量
- 演示如何获取刚体状态张量(位置/姿态/速度)并按列切片
- 周期性打印状态、保存相机图像、统计 FPS
"""

import numpy as np
import imageio
import math
import os

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch


gym = gymapi.acquire_gym()

# 解析命令行参数(--headless 表示无窗口运行,纯数据吞吐模式)
args = gymutil.parse_arguments(description="PyTorch tensor interop example",
                               custom_parameters=[
                                   {"name": "--headless", "action": "store_true", "help": ""}])

# 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 8
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.4
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

# 该开关决定物理张量放在 CPU 还是 GPU 上(本例强制使用 GPU 流水线,充分发挥张量 API 的优势)
sim_params.use_gpu_pipeline = True
if not args.use_gpu_pipeline:
    print("Warning: Forcing GPU pipeline.")

# 创建仿真器:参数依次为计算设备 ID、图形设备 ID、物理引擎类型、仿真参数
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

ball_asset = gym.create_sphere(sim, 0.5, None)

use_viewer = not args.headless

# 创建可视化窗口(仅非 headless 模式)
if use_viewer:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
else:
    viewer = None

# 添加地面平面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 设置环境(env)网格布局:16 个环境排成 4x4 方阵
num_envs = 16
envs_per_row = int(math.sqrt(num_envs))
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs = []
cams = []
cam_tensors = []

# 创建环境
for i in range(num_envs):
    # 创建环境:参数为仿真器、包围盒下界、上界、每行环境数
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    # 添加 actor(actor:演员,即资产放入环境后生成的实例),球放在 5 米高处下落
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 5.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle = gym.create_actor(env, ball_asset, pose, "ball", i, 0)

    # 设置球的弹性系数(落到地面会弹起来)
    props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    props[0].restitution = 0.9
    gym.set_actor_rigid_shape_properties(env, actor_handle, props)

    # 给球随机设置一种颜色
    c = 0.5 + 0.5 * np.random.random(3)
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(c[0], c[1], c[2]))

    # 添加传感器相机;enable_tensors=True 才允许用张量 API 读取相机图像
    cam_props = gymapi.CameraProperties()
    cam_props.width = 128
    cam_props.height = 128
    cam_props.enable_tensors = True
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, gymapi.Vec3(5, 1, 0), gymapi.Vec3(0, 1, 0))
    cams.append(cam_handle)

    # 获取相机图像 GPU 张量:图像数据直接留在 GPU 显存里,不用拷贝到内存
    cam_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
    print("Got camera tensor with shape", cam_tensor.shape)

    # 把相机张量包装成 PyTorch 张量:wrap_tensor 返回的张量直接指向底层显存,
    # 零拷贝 —— 对张量的读写就是直接读写仿真数据
    torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
    cam_tensors.append(torch_cam_tensor)
    print("  Torch camera tensor device:", torch_cam_tensor.device)
    print("  Torch camera tensor shape:", torch_cam_tensor.shape)

# 把 viewer 观察相机对准中间的环境
cam_pos = gymapi.Vec3(8, 2, 6)
cam_target = gymapi.Vec3(-8, 0, -6)
middle_env = envs[num_envs // 2 + envs_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# 使用张量 API 前必须先调用 prepare_sim:它会分配好各状态张量的底层缓冲区
gym.prepare_sim(sim)

# 获取 GPU 上的刚体状态张量:一次性拿到"所有环境的所有刚体"的状态,
# 无需逐环境循环拷贝。每行 13 列 = 位置(3) + 姿态四元数(4) + 线速度(3) + 角速度(3)。
# 同族 API 还有 acquire_actor_root_state_tensor(所有 actor 的根状态)等,
# 写回物理引擎则用 set_actor_root_state_tensor(_indexed) 系列
state_tensor = gym.acquire_rigid_body_state_tensor(sim)
print("Gym state tensor shape:", state_tensor.shape)
print("Gym state tensor data @ 0x%x" % state_tensor.data_address)

# 把物理状态张量包装成 PyTorch 张量:wrap_tensor 返回的张量直接指向 GPU 显存,
# 对它切片、运算就是直接操作仿真数据 —— 这是 RL 训练的核心接口,
# 观测(读取状态)和控制(写回目标)都通过这类张量完成
rb_states = gymtorch.wrap_tensor(state_tensor)
print("Torch state tensor device:", rb_states.device)
print("Torch state tensor shape:", rb_states.shape)
print("Torch state tensor data @ 0x%x" % rb_states.data_ptr())

# 按列切出各种状态的"视图"(切片不复制数据,与 rb_states 共享同一块显存)
num_bodies = rb_states.shape[0]
rb_positions = rb_states[:, 0:3]       # 位置
rb_orientations = rb_states[:, 3:7]    # 姿态(四元数)
rb_linvels = rb_states[:, 7:10]        # 线速度
rb_angvels = rb_states[:, 10:13]       # 角速度

# 创建保存相机图像的目录
img_dir = "interop_images"
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

frame_count = 0
next_fps_report = 2.0
t1 = 0

# 主循环:headless 模式下 viewer 为 None,一直运行(Ctrl+C 退出);
# 有窗口时则运行到用户关闭窗口为止
while viewer is None or not gym.query_viewer_has_closed(viewer):

    frame_no = gym.get_frame_count(sim)

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 从物理引擎把最新状态同步进张量:simulate 更新的是底层缓冲区,
    # 必须 refresh 之后张量里的数据才是最新的。
    # 同族 API 还有 refresh_actor_root_state_tensor(刷新 actor 根状态张量)等
    gym.refresh_rigid_body_state_tensor(sim)

    gym.step_graphics(sim)

    # 渲染传感器相机并准备读取图像张量:
    # 在 start/end_access_image_tensors 之间才能安全地访问相机图像张量
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    # 在开头的一小段时间里,每隔 10 帧打印一次状态、保存一次相机图像
    if frame_no < 60 and frame_no % 10 == 0:

        print("========= Frame %d ==========" % frame_no)

        # 打印状态张量(这里只打印位置,其余可按需取消注释)
        # print("RB states:")
        # print(rb_states.cpu().detach().numpy())
        print("RB positions:")
        print(rb_positions)
        # print("RB orientations:")
        # print(rb_orientations)
        # print("RB linear velocities:")
        # print(rb_linvels)
        # print("RB angular velocities:")
        # print(rb_angvels)

        for i in range(num_envs):
            # 把相机图像张量保存为图片文件
            fname = os.path.join(img_dir, "cam-%04d-%04d.png" % (frame_no, i))
            cam_img = cam_tensors[i].cpu().numpy()
            imageio.imwrite(fname, cam_img)

    # 每 2 秒统计并打印一次帧率(括号里的数字是"帧率 x 环境数"的等效吞吐量)
    t = gym.get_elapsed_time(sim)
    if t >= next_fps_report:
        t2 = gym.get_elapsed_time(sim)
        fps = frame_count / (t2 - t1)
        print("FPS %.1f (%.1f)" % (fps, fps * num_envs))
        frame_count = 0
        t1 = gym.get_elapsed_time(sim)
        next_fps_report = t1 + 2.0

    # 结束访问相机图像张量(与 start_access 配对使用)
    gym.end_access_image_tensors(sim)

    if viewer is not None:
        # 绘制 viewer 画面,并按真实时间同步帧率
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    frame_count += 1
