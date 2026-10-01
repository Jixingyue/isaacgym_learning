"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF(Degree of Freedom,自由度)控制方法示例
------------------------------------------
演示对自由度的多种控制方式。DOF 指关节可动的方向,比如小车沿滑轨平移、
摆杆绕轴转动,各算一个自由度。本例演示:
- 从 URDF 文件加载倒立摆(cartpole)资产
- 读取/设置 DOF 属性(控制模式、刚度、阻尼)
- 设置 DOF 的位置目标和速度目标
- 读取 DOF 的当前位置
- 对 DOF 直接施加力矩(effort)
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil

# 初始化 gym:获取 Gym 接口实例(一切操作的入口)
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(description="Joint control Methods Example")

# 配置仿真参数
sim_params = gymapi.SimParams()
# 每个物理步细分为 2 个子步;dt 为步长(1/60 秒,即 60Hz)
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# 创建仿真器:参数依次为计算设备 ID、图形设备 ID、物理引擎类型、仿真参数
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# 创建可视化窗口(viewer),使用默认相机属性
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# 添加地面平面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# 设置环境(env)网格布局:4 个环境排成一行
num_envs = 4
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# 加载倒立摆 URDF 资产(asset:模型文件加载到内存后的表示)
asset_root = "../../assets"
asset_file = "urdf/cartpole.urdf"

# 加载资产并指定所有关节的默认控制模式为位置模式
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # 固定底座(滑轨不动)
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 倒立摆 actor 的初始根部位姿:p 是位置,r 是旋转四元数(这里绕 X 轴转 -90 度让它立起来)
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# 创建环境 0
# 小车:用位置目标模式固定在原点不动。
# 摆杆:用位置目标模式保持在 45 度角。
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole0 = gym.create_actor(env0, cartpole_asset, initial_pose, 'cartpole', 0, 1)
# 配置 DOF 属性:元组里两个值分别对应小车滑轨和摆杆两个自由度
# driveMode 是控制模式;stiffness(刚度)越大越"硬",像弹簧的劲度;
# damping(阻尼)抑制运动,防止来回震荡
props = gym.get_actor_dof_properties(env0, cartpole0)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
props["stiffness"] = (5000.0, 5000.0)
props["damping"] = (100.0, 100.0)
gym.set_actor_dof_properties(env0, cartpole0, props)
# 设置驱动目标:按名称查找 DOF 句柄(handle,引用关节的编号)
# 小车位置目标为 0;摆杆位置目标为 0.25*π(即 45 度)
cart_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'slider_to_cart')
pole_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'cart_to_pole')
gym.set_dof_target_position(env0, cart_dof_handle0, 0)
gym.set_dof_target_position(env0, pole_dof_handle0, 0.25 * math.pi)

# 创建环境 1
# 小车:用位置目标模式固定不动。
# 摆杆:用速度目标模式匀速旋转。
env1 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole1 = gym.create_actor(env1, cartpole_asset, initial_pose, 'cartpole', 1, 1)
# 配置 DOF 属性:摆杆为速度模式(DOF_MODE_VEL),速度模式下靠阻尼来跟踪目标速度
props = gym.get_actor_dof_properties(env1, cartpole1)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL)
props["stiffness"] = (5000.0, 0.0)
props["damping"] = (100.0, 200.0)
gym.set_actor_dof_properties(env1, cartpole1, props)
# 设置驱动目标:小车位置目标为 0;摆杆速度目标为 -2π rad/s(每秒反向转一圈)
cart_dof_handle1 = gym.find_actor_dof_handle(env1, cartpole1, 'slider_to_cart')
pole_dof_handle1 = gym.find_actor_dof_handle(env1, cartpole1, 'cart_to_pole')
gym.set_dof_target_position(env1, cart_dof_handle1, 0)
gym.set_dof_target_velocity(env1, pole_dof_handle1, -2.0 * math.pi)

# 创建环境 2
# 小车:用速度目标模式左右来回移动。
# 摆杆:用位置目标模式保持竖直。
env2 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole2 = gym.create_actor(env2, cartpole_asset, initial_pose, 'cartpole', 2, 1)
# 配置 DOF 属性:小车为速度模式,摆杆为位置模式
props = gym.get_actor_dof_properties(env2, cartpole2)
props["driveMode"] = (gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_POS)
props["stiffness"] = (0.0, 5000.0)
props["damping"] = (200.0, 100.0)
gym.set_actor_dof_properties(env2, cartpole2, props)
# 设置驱动目标:小车速度目标 1 m/s;摆杆位置目标 0(竖直)
cart_dof_handle2 = gym.find_actor_dof_handle(env2, cartpole2, 'slider_to_cart')
pole_dof_handle2 = gym.find_actor_dof_handle(env2, cartpole2, 'cart_to_pole')
gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)
gym.set_dof_target_position(env2, pole_dof_handle2, 0.0)

# 创建环境 3
# 小车:位置模式固定不动,但会被施加的力矩推动。
# 摆杆:用力矩模式(DOF_MODE_EFFORT)直接施加力矩来保持直立。
env3 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole3 = gym.create_actor(env3, cartpole_asset, initial_pose, 'cartpole', 3, 1)
# 配置 DOF 属性:摆杆为力矩模式,需自己算出要施加多大的力
props = gym.get_actor_dof_properties(env3, cartpole3)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_EFFORT)
props["stiffness"] = (5000.0, 0.0)
props["damping"] = (100.0, 0.0)
gym.set_actor_dof_properties(env3, cartpole3, props)
# 设置驱动目标并直接施加 200 的力矩(effort)
cart_dof_handle3 = gym.find_actor_dof_handle(env3, cartpole3, 'slider_to_cart')
pole_dof_handle3 = gym.find_actor_dof_handle(env3, cartpole3, 'cart_to_pole')
gym.set_dof_target_position(env3, cart_dof_handle3, 0.0)
gym.apply_dof_effort(env3, pole_dof_handle3, 200)

# 把观察相机对准第一个环境
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 主循环:只要用户没有关闭 viewer 窗口就一直运行
while not gym.query_viewer_has_closed(viewer):

    # 推进一步物理仿真;fetch_results 等待并取回这一步的计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新渲染:step_graphics 把物理状态同步给图形系统,draw_viewer 绘制画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 环境 0 无需处理(目标都是固定值)

    # 环境 1 无需处理(目标都是固定值)

    # 更新环境 2:读取小车当前位置,到达边界(±0.5)时反转目标速度,实现左右往返
    pos = gym.get_dof_position(env2, cart_dof_handle2)
    if pos >= 0.5:
        gym.set_dof_target_velocity(env2, cart_dof_handle2, -1.0)
    elif pos <= -0.5:
        gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)

    # 更新环境 3:读取摆杆角度,按比例施加反向力矩(简单的 P 控制器)让它保持直立
    pos = gym.get_dof_position(env3, pole_dof_handle3)
    gym.apply_dof_effort(env3, pole_dof_handle3, -pos * 50)

    # 等待真实时间走过一个 dt(时间步长)。
    # 这样可以让物理仿真与渲染帧率同步,按真实时间观看效果。
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
