"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Projectiles（抛射物）示例
-------------------
演示如何生成并移动物体，展示碰撞过滤（collision filtering）的用法，
以及如何用可视化窗口（viewer）与物理仿真交互：
按空格键或鼠标左键发射抛射物，按 R 键重置仿真。
"""

import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt

# 初始化 gym：获取 Gym 接口实例（对 Isaac Gym 一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数（--num_envs 指定环境数量）
args = gymutil.parse_arguments(
    description="Projectiles Example: Press SPACE to fire a projectile. Press R to reset the simulation.",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 16, "help": "Number of environments to create"}])

# 配置仿真参数
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.05
    sim_params.flex.num_inner_iterations = 6
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# 添加地面平面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 创建可视化窗口（viewer）。本例必须有 viewer（要靠键盘/鼠标交互）
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 订阅输入事件（键盘/鼠标），这样就能用输入与仿真交互
# 订阅空格键→"space_shoot"，R 键→"reset"，鼠标左键→"mouse_shoot"
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "space_shoot")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")

# 加载资产（asset：机器人的模型描述，这里是 MJCF 格式的蚂蚁模型）
asset_root = "../../assets"
asset_file = "mjcf/nv_ant.xml"

asset_options = gymapi.AssetOptions()
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 设置环境（env）网格布局：每行的环境数取总数的平方根，间距 2 米
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
spacing = 2.0

envs = []
actor_handles = []

for i in range(num_envs):
    # 创建环境（env：一块独立的仿真空间）
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, lower, upper, num_per_row)
    envs.append(env)

    # 把 actor（演员：资产的实例）放进环境，pose 为初始位姿（位置+四元数姿态）
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    ahandle = gym.create_actor(env, asset, pose, "ant", i, 1)
    actor_handles.append(ahandle)

    # 覆盖 MJCF 里默认的 DOF 属性，让蚂蚁的腿能自由摆动（去掉驱动刚度和阻尼）
    props = gym.get_actor_dof_properties(env, ahandle)
    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, ahandle, props)

# 额外创建一个专门存放抛射物的环境（proj_env），并创建
# 20 个轮流使用的抛射物（发射一次换下一个，循环复用）
proj_env = gym.create_env(sim, lower, upper, 4)
proj_asset_options = gymapi.AssetOptions()
proj_asset_options.density = 10.
proj_asset = gym.create_box(sim, 0.3, 0.3, 0.3, proj_asset_options)
projectiles = []

for i in range(20):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(i * 0.5, 1.0, 50)
    pose.r = gymapi.Quat(0, 0, 0, 1)

    # 创建抛射物 actor：分组参数传 -1，表示它能与任何环境里的 actor 发生碰撞
    # （普通环境里的 actor 属于各自分组，互相之间不碰撞——这就是碰撞过滤）
    ahandle = gym.create_actor(proj_env, proj_asset, pose, "projectile" + str(i), -1, 0)

    # 给每个抛射物设置不同的随机颜色
    c = 0.5 + 0.5 * np.random.random(3)
    gym.set_rigid_body_color(proj_env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(c[0], c[1], c[2]))

    projectiles.append(ahandle)

# 保存整个仿真的初始刚体状态副本，按 R 键时可以写回去实现重置
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

proj_index = 0

while not gym.query_viewer_has_closed(viewer):

    # 推进物理仿真一步，并取回计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 从可视化窗口获取输入事件并相应处理
    for evt in gym.query_viewer_action_events(viewer):

        # 按下 R 键：把所有刚体状态恢复为初始状态（重置仿真）
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

        # 按下空格键或鼠标左键：发射一个抛射物
        elif (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0:
            if evt.action == "mouse_shoot":
                # 鼠标发射时，打印点击位置在窗口中的像素坐标
                pos = gym.get_viewer_mouse_position(viewer)
                window_size = gym.get_viewer_size(viewer)
                xcoord = round(pos.x * window_size.x)
                ycoord = round(pos.y * window_size.y)
                print(f"Fired projectile with mouse at coords: {xcoord} {ycoord}")

            # 获取相机位姿，用相机的朝向作为发射方向
            cam_pose = gym.get_viewer_camera_transform(viewer, proj_env)
            cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))

            # 从相机所在位置出发，以 25 m/s 的初速度沿视线方向发射
            spawn = cam_pose.p
            speed = 25
            vel = cam_fwd * speed

            # 随机生成角速度，让抛射物旋转着飞出去
            angvel = 1.57 - 3.14 * np.random.random(3)

            # 轮流取一个抛射物，直接改写它的刚体状态（位置+速度）实现"发射"
            proj_handle = projectiles[proj_index]
            state = gym.get_actor_rigid_body_states(proj_env, proj_handle, gymapi.STATE_NONE)
            state['pose']['p'].fill((spawn.x, spawn.y, spawn.z))
            state['pose']['r'].fill((0, 0, 0, 1))
            state['vel']['linear'].fill((vel.x, vel.y, vel.z))
            state['vel']['angular'].fill((angvel[0], angvel[1], angvel[2]))
            gym.set_actor_rigid_body_states(proj_env, proj_handle, state, gymapi.STATE_ALL)

            # 指向下一个抛射物（20 个循环使用）
            proj_index = (proj_index + 1) % len(projectiles)

    # 更新渲染画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

    # 等待真实时间走完一个 dt，
    # 让物理仿真与渲染帧率同步（实时观看效果）
    gym.sync_frame_time(sim)

# 销毁资源：先关闭可视化窗口，再销毁仿真器
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
