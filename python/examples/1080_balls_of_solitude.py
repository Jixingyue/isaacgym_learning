"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


1080 个孤独的球
-------------------------
演示碰撞过滤（collision filtering）的用法：
- 限制碰撞只发生在同一环境内的演员（actor）之间；
- 模拟所有碰撞（包括不同环境中的演员之间）；
- 或者完全不模拟演员之间的碰撞——但它们仍会与地面发生碰撞。

模式可以通过命令行参数设置：
    --no_collisions  让所有演员之间互不碰撞
    --all_collisions 让所有演员（即使来自不同环境）都参与碰撞

按 'R' 键重置仿真
"""

import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt

# 获取 Gym 接口实例（一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 36, "help": "Number of environments to create"},
        {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
        {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"}])

# 配置仿真参数
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# 创建仿真器：指定计算设备、渲染设备、物理引擎和仿真参数
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# 添加地面（所有环境共用）
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 创建观察者窗口，用于实时观看仿真画面
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 加载球的资产（asset：从 URDF/MJCF 文件加载的物体模型，相当于创建实例的"模板"）
asset_root = "../../assets"
asset_file = "urdf/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# 设置环境网格布局（环境会按每排 num_per_row 个摆放）
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
env_spacing = 1.25
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []

# 订阅观察者窗口的键盘事件（这里监听 R 键，用于重置仿真）
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

# 设置随机种子，保证每次运行结果一致
np.random.seed(17)

for i in range(num_envs):
    # 创建环境（env：一块独立的仿真空间，可并行创建成百上千个）
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # 生成随机亮色，用于给这个环境里的球上色
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # 在环境中搭建金字塔形球堆
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    n = 4
    radius = 0.2
    ball_spacing = 2.5 * radius
    min_coord = -0.5 * (n - 1) * ball_spacing
    y = min_coord+4
    while n > 0:
        z = min_coord
        for j in range(n):
            x = min_coord
            for k in range(n):
                pose.p = gymapi.Vec3(x, 1.5 + y, z)

                # 设置碰撞过滤（collision filtering）
                if args.all_collisions:
                    # 所有物体都要碰撞：
                    # 把所有演员放进同一分组，过滤掩码设为 0（不过滤任何碰撞）
                    collision_group = 0
                    collision_filter = 0

                elif args.no_collisions:
                    # 所有演员之间都不碰撞：
                    # 给所有演员设置相同的过滤掩码，即可屏蔽它们之间的碰撞。
                    # 这种情况下分组无所谓。
                    # 另一种做法是把每个演员放进不同的分组。
                    collision_group = 0
                    collision_filter = 1

                else:
                    # 同一环境内的球要碰撞，但与其它环境的球不碰撞：
                    # 每个环境单独一个分组，过滤掩码设为 0
                    collision_group = i
                    collision_filter = 0

                # 创建演员（actor：放进环境里的具体物体实例），并传入碰撞分组和过滤掩码
                ahandle = gym.create_actor(env, asset, pose, None, collision_group, collision_filter)
                # 给这个演员的 0 号刚体上色（视觉网格与碰撞网格都改）
                gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

                x += ball_spacing
            z += ball_spacing
        y += ball_spacing
        n -= 1
        min_coord = -0.5 * (n - 1) * ball_spacing

# 设置观察者相机的视角：从 (20, 5, 20) 望向 (0, 1, 0)
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))

# 保存一份初始状态（所有刚体的位姿/速度等），重置时可以直接恢复回去
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))


# 主循环：只要观察者窗口没被关闭就一直运行
while not gym.query_viewer_has_closed(viewer):

    # 获取观察者窗口的输入事件并做相应处理
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            # 按下 R 键：把所有刚体恢复到初始状态
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # 推进物理仿真一步
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新观察者窗口的画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待真实时间走完一个 dt。
    # 这样可以让物理仿真与渲染帧率同步（避免跑得过快）。
    gym.sync_frame_time(sim)

# 销毁观察者窗口
gym.destroy_viewer(viewer)
# 销毁仿真器，释放资源
gym.destroy_sim(sim)
