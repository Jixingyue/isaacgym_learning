"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


在指定位置施力（apply_forces_at_pos.py）
----------------------------
本示例演示如何使用张量 API（tensor API）在给定位置对刚体施加力
（当施力点不在重心上时，还会产生扭矩效果）。

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
    description="Example of applying forces to bodies at given positions")

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

mid = 0.5 * env_spacing * (num_per_row - 1)
# 设置观察者相机的视角（对准环境网格的中心）
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, mid, 5), gymapi.Vec3(0, mid, 1))

# 仿真前的最后准备（初始化内部缓冲区，调用张量 API 之前必须执行）
gym.prepare_sim(sim)

# 获取刚体状态张量，并用 PyTorch 张量包装，之后就能直接用 torch 读写它
rb_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(rb_tensor)
# 只取位置分量（每个刚体状态的前 3 个数），整理成 [环境数, 刚体数, 3] 的形状
rb_positions = rb_states[:, 0:3].view(num_envs, num_bodies, 3)

# 施力点相对刚体中心的偏移量（每轮正负交替，扭矩方向随之改变）
force_offset = 0.2

frame_count = 0
# 主循环：只要观察者窗口没被关闭就一直运行
while not gym.query_viewer_has_closed(viewer):

    # 每 200 帧触发一次施力（在第 99、299、499... 帧时执行）
    if (frame_count - 99) % 200 == 0:

        # 先刷新刚体状态张量，拿到最新的位置数据
        gym.refresh_rigid_body_state_tensor(sim)

        # 为蚂蚁的根刚体（每个环境的 0 号刚体）设置力和施力点位置
        forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
        force_positions = rb_positions.clone()
        forces[:, 0, 2] = 400                       # 给根刚体一个 Z 方向 400N 的向上推力
        force_positions[:, 0, 1] += force_offset    # 施力点沿 Y 方向偏移一点（偏离中心）
        # 在指定位置施力：因为施力点偏离中心，会产生扭矩让蚂蚁翻转
        gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)

        force_offset = -force_offset  # 偏移方向交替，扭矩方向随之改变

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
