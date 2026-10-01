"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Franka Attractor（Franka 机械臂吸引子示例）
--------------------------------------------
通过一个「吸引子目标点」对 Franka Panda 机械臂做位置控制，机械臂会努力去够到这个目标点。

吸引子（attractor）就像一个虚拟的磁铁：我们在空间中放置一个目标点，
物理引擎会自动生成朝向该点的力，牵引机械臂末端跟过去，从而实现最简单的机械臂控制。
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil

# 初始化 gym 接口（Isaac Gym 中的所有操作都通过这个 gym 句柄调用）
gym = gymapi.acquire_gym()

# 解析命令行参数（如物理引擎、是否用 GPU 等）
args = gymutil.parse_arguments(description="Franka Attractor Example")

# 配置仿真参数（SimParams 相当于物理世界的"世界设定"）
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
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
# create_sim：创建物理仿真世界，返回一个 sim 句柄供后续使用

if sim is None:
    print("*** Failed to create sim")
    quit()

# 创建可视化窗口（viewer），用于在屏幕上渲染场景
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 添加地面平面（防止机械臂掉下去）
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 加载 Franka 机械臂的模型资源（asset，即 URDF 文件描述的机器人本体）
asset_root = "../../assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
franka_asset = gym.load_asset(
    sim, asset_root, franka_asset_file, asset_options)

# 设置环境网格（多个相同的场景按方格排列，便于一次训练/展示很多个机械臂）
num_envs = 36
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
# 注意：这里只定义了网格的范围，真正的环境靠后面 create_env 一个个创建

# 一些后续会用到的句柄（handle）列表（句柄相当于对象的"身份证编号"）
envs = []
franka_handles = []
franka_hand = "panda_hand"

# 吸引子（attractor）的初始化设置
attractor_handles = []
attractor_properties = gymapi.AttractorProperties()
attractor_properties.stiffness = 5e5
# stiffness（刚度）越大，吸引子像弹簧拉得越"硬"，机械臂被拽得越使劲
attractor_properties.damping = 5e3
# damping（阻尼）越大，运动越"柔"、越不容易震荡，像减震器一样消耗多余能量

# 让吸引子在所有轴向上都起作用（位置 + 姿态六个自由度都被吸引）
attractor_properties.axes = gymapi.AXIS_ALL
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# 创建用于可视化显示的辅助几何体
# 创建一个线框坐标轴（用于标示吸引子目标点的位置和朝向）
axes_geom = gymutil.AxesGeometry(0.1)
# 创建一个线框小球
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

for i in range(num_envs):
    # 创建单个环境（场景）
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # 往环境里添加 Franka 机械臂（把资产实例化成一个 actor）
    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 2)
    body_dict = gym.get_actor_rigid_body_dict(env, franka_handle)
    # body_dict：刚体名字到索引的映射，用于查找"手"这个刚体
    props = gym.get_actor_rigid_body_states(env, franka_handle, gymapi.STATE_POS)
    hand_handle = body = gym.find_actor_rigid_body_handle(env, franka_handle, franka_hand)
    # find_actor_rigid_body_handle：按名字找到"panda_hand"（机械手）这个刚体的句柄

    # 初始化吸引子：把目标点先放在机械手当前的位置，再稍微偏移
    attractor_properties.target = props['pose'][:][body_dict[franka_hand]]
    attractor_properties.target.p.y -= 0.1
    attractor_properties.target.p.z = 0.1
    attractor_properties.rigid_handle = hand_handle
    # rigid_handle：告诉吸引子"你去吸引哪个刚体"（这里吸引的是机械手）

    # 在吸引子目标点处画出坐标轴和小球，便于观察
    gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

    franka_handles.append(franka_handle)
    attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)
    # create_rigid_body_attractor：真正创建并注册这个吸引子，让它开始"吸"机械手
    attractor_handles.append(attractor_handle)

# 获取 Franka 各关节（dof，自由度）的限位与行程范围
franka_dof_props = gym.get_actor_dof_properties(envs[0], franka_handles[0])
# get_actor_dof_properties：获取关节属性（限位、刚度、阻尼、驱动模式等）
franka_lower_limits = franka_dof_props['lower']
franka_upper_limits = franka_dof_props['upper']
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
franka_num_dofs = len(franka_dof_props)

# 覆盖默认的刚度和阻尼值（让机械臂各关节转起来柔和一点）
franka_dof_props['stiffness'].fill(1000.0)
franka_dof_props['damping'].fill(1000.0)

# 前两个关节用"位置控制"模式，让机械臂站得更稳
franka_dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS

# 第 8 个及之后的关节（夹爪手指）也用位置控制，并给很高的刚度
franka_dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
franka_dof_props['stiffness'][7:] = 1e10
franka_dof_props['damping'][7:] = 1.0

for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], franka_handles[i], franka_dof_props)
    # set_actor_dof_properties：把上面配置好的关节属性应用回机械臂


def update_franka(t):
    gym.clear_lines(viewer)
    for i in range(num_envs):
        # 根据当前时间 t 更新吸引子目标点（让目标点沿着一条摆线轨迹移动）
        attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i])
        # get_attractor_properties：读取当前吸引子的属性，方便修改目标点
        pose = attractor_properties.target
        pose.p.x = 0.2 * math.sin(1.5 * t - math.pi * float(i) / num_envs)
        pose.p.y = 0.7 + 0.1 * math.cos(2.5 * t - math.pi * float(i) / num_envs)
        pose.p.z = 0.2 * math.cos(1.5 * t - math.pi * float(i) / num_envs)

        gym.set_attractor_target(envs[i], attractor_handles[i], pose)
        # set_attractor_target：设置吸引子的新目标点，机械臂会跟着它走

        # 在吸引子目标点处画出坐标轴和小球
        gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
        gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)


for i in range(num_envs):
    # 应用更新后的刚度和阻尼属性
    gym.set_actor_dof_properties(envs[i], franka_handles[i], franka_dof_props)

    # 设置 Franka 的初始姿态：让每个关节都处于其行程范围的中间位置
    franka_dof_states = gym.get_actor_dof_states(envs[i], franka_handles[i], gymapi.STATE_NONE)
    for j in range(franka_num_dofs):
        franka_dof_states['pos'][j] = franka_mids[j]
    gym.set_actor_dof_states(envs[i], franka_handles[i], franka_dof_states, gymapi.STATE_POS)
    # set_actor_dof_states：设置关节状态（这里是关节角度位置）

# 把相机对准环境中心
cam_pos = gymapi.Vec3(-4.0, 4.0, -1.0)
cam_target = gymapi.Vec3(0.0, 2.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 移动机械臂之前先等待的时间（秒）
next_franka_update_time = 1.5

while not gym.query_viewer_has_closed(viewer):
    # 每隔 0.01 秒更新一次吸引子目标点的位姿
    t = gym.get_sim_time(sim)
    if t >= next_franka_update_time:
        update_franka(t)
        next_franka_update_time += 0.01

    # Step the physics —— 前进一步物理仿真
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering —— 前进一步渲染
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    # sync_frame_time：让渲染帧率与真实时间保持同步

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
