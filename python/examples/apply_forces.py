"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


施加力（apply_forces.py）
----------------------------
本示例演示如何使用张量 API（tensor API）对刚体施加力和扭矩。
"""

from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch

# 获取 Gym 接口实例（一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(
    description="Example of applying forces and torques to bodies")

# 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
elif args.physics_engine == gymapi.SIM_FLEX and not args.use_gpu_pipeline:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
else:
    raise Exception("GPU pipeline is only available with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# 创建仿真器：指定计算设备、渲染设备、物理引擎和仿真参数
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# 添加地面（法线朝 +Z，因为本例的竖直向上方向是 Z 轴）
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 创建观察者窗口，用于实时观看仿真画面
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# 加载蚂蚁机器人资产（asset：从 MJCF 文件加载的模型模板）
asset_root = "../../assets"
asset_file = "mjcf/nv_ant.xml"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# 查询资产包含的刚体数量
num_bodies = gym.get_asset_rigid_body_count(asset)
print('num_bodies', num_bodies)

# 默认摆放位姿（沿 Z 方向抬高 1 米）
pose = gymapi.Transform()
pose.p.z = 1.0

# 设置环境网格布局
num_envs = 4
num_per_row = int(np.sqrt(num_envs))
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# 设置随机种子，保证每次运行结果一致
np.random.seed(17)

envs = []
handles = []
for i in range(num_envs):
    # 创建环境（env：一块独立的仿真空间）
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # 生成随机亮色，用于给蚂蚁上色
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # 创建演员（actor：放进环境里的具体物体实例），返回的 handle 是后续操作的"凭证"
    ahandle = gym.create_actor(env, asset, pose, "actor", i, 1)
    handles.append(ahandle)
    gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# 设置观察者相机的视角
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 20, 5), gymapi.Vec3(0, 0, 1))

# 仿真前的最后准备（初始化内部缓冲区，调用张量 API 之前必须执行）
gym.prepare_sim(sim)

# 扭矩大小（每轮正负交替，让蚂蚁往不同方向翻转）
torque_amt = 100

frame_count = 0
# 主循环：只要观察者窗口没被关闭就一直运行
while not gym.query_viewer_has_closed(viewer):

    # 每 200 帧触发一次施力（在第 99、299、499... 帧时执行）
    if (frame_count - 99) % 200 == 0:
        # 为蚂蚁的根刚体（每个环境的 0 号刚体）设置力和扭矩
        forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
        torques = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
        forces[:, 0, 2] = 300          # 给根刚体一个 Z 方向 300N 的向上推力
        torques[:, 0, 2] = torque_amt  # 给根刚体一个绕 Z 轴的扭矩
        # 用张量 API 一次性给所有环境的所有刚体施加力和扭矩
        gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

        torque_amt = -torque_amt  # 扭矩正负交替，改变旋转方向

    # 推进物理仿真一步
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新观察者窗口的画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待真实时间走完一个 dt。
    # 这样可以让物理仿真与渲染帧率同步（避免跑得过快）。
    gym.sync_frame_time(sim)

    frame_count += 1

# 销毁观察者窗口
gym.destroy_viewer(viewer)
# 销毁仿真器，释放资源
gym.destroy_sim(sim)
