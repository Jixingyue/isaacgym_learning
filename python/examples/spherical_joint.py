"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Spherical Joint（球关节演示）
------------
- 演示球关节（spherical joint，可 3 个转动自由度）的使用方法。
- 周期性地给出随机目标姿态，用位置控制驱动球关节跟踪该姿态。
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# 简单的资产描述符类，用于从列表中选择要加载的机器人模型


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("urdf/spherical_joint.urdf", False),
    # AssetDesc("mjcf/spherical_joint.xml", False),
]

# 解析命令行参数
args = gymutil.parse_arguments(
    description="Spherical Joint: Show example of controlling a spherical joint robot.",
)

# 初始化 gym：获取 Gym 接口实例（对 Isaac Gym 一切操作的入口）
gym = gymapi.acquire_gym()

# 配置仿真参数（本例无重力，Z 轴向上）
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0  # 仿真步长：1/60 秒
sim_params.gravity = gymapi.Vec3(0, 0, 0)  # 重力设为 0（机构悬浮，只看关节运动）
sim_params.up_axis = gymapi.UP_AXIS_Z  # 以 Z 轴为向上方向

if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()


# 创建可视化窗口（viewer），使用默认相机属性
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 加载资产（asset：机器人的模型描述，这里是带球关节的 URDF）
asset_root = "../../assets"
asset_file = asset_descriptors[0].file_name

# asset_options：加载资产时的选项
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # 固定底座，机构悬空不动
asset_options.flip_visual_attachments = asset_descriptors[0].flip_visual_attachments
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 获取所有 DOF（自由度，可理解为"关节"）的名称数组
dof_names = gym.get_asset_dof_names(asset)

# 获取 DOF 属性数组（刚度、阻尼、限位等）
dof_props = gym.get_asset_dof_properties(asset)

# 创建 DOF 状态数组（每个元素包含位置 pos 和速度 vel），之后用它更新 actor
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# 获取每个 DOF 的类型列表（旋转 DOF 还是平移 DOF）
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# 取出 DOF 状态数组中的位置切片（numpy 视图），修改它就等于修改 dof_states
dof_positions = dof_states['pos']

# 取出 DOF 属性数组中与限位相关的各切片
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# 初始化默认位置和限位（确保都在合理范围内）
defaults = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # 确保默认位置落在限位范围内（若 0 不在范围内，就取最接近的限位值）
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        # 给没有限位的关节人为设定合理的动画范围
        if dof_types[i] == gymapi.DOF_ROTATION:
            # 无限位的旋转关节：范围取 ±π（半圈）
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # 无限位的滑动关节：范围取 ±1 米
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
        else:
            print("Unknown DOF type!")
            exit()
    # 把该 DOF 的位置设为默认值
    dof_positions[i] = defaults[i]

# 打印所有 DOF 的属性信息（名称、类型、刚度、阻尼、限位等）
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# 设置环境（env）网格布局：本例只用 1 个环境
num_envs = 1
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 设置观察相机的位置和看向的目标点
cam_pos = gymapi.Vec3(1.0, 1.0, 1)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 缓存有用的句柄（handle：整数 ID，之后用它指代对应的 env / actor）
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # 创建环境（env：一块独立的仿真空间）
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # 创建 actor（演员：把资产实例化放进环境）
    # pose 是初始位姿：p 为位置，r 为四元数表示的姿态
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # 覆盖 DOF 属性：6 个 DOF 全部设为位置控制模式（DOF_MODE_POS），
    # 并设置位置控制用的刚度（stiffness）和阻尼（damping）
    props = gym.get_actor_dof_properties(env, actor_handle)
    props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
    props["stiffness"] = (50.0, 50.0, 50.0, 50.0, 50.0, 50.0)
    props["damping"] = (10.0, 10.0, 10.0, 100.0, 100.0, 100.0)

    gym.set_actor_dof_properties(env, actor_handle, props)

    # 设置 actor 的初始 DOF 状态
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)


def random_quaternion():
    """生成 (x, y, z, w) 形式的均匀随机四元数（代表任意朝向）。

    Returns:
        np.ndarray: 4 元素数组。
    """
    r1, r2, r3 = np.random.random(3)

    q1 = math.sqrt(1.0 - r1) * (math.sin(2 * math.pi * r2))
    q2 = math.sqrt(1.0 - r1) * (math.cos(2 * math.pi * r2))
    q3 = math.sqrt(r1) * (math.sin(2 * math.pi * r3))
    q4 = math.sqrt(r1) * (math.cos(2 * math.pi * r3))

    quat_xyzw = np.array([q2, q3, q4, q1])

    if quat_xyzw[-1] < 0:
        quat_xyzw = -quat_xyzw

    return quat_xyzw


def quat2expcoord(q):
    """把四元数转换为指数坐标（旋转向量：旋转轴 × 旋转角）。
    球关节的后 3 个 DOF 用指数坐标来描述 3 个转动自由度。

    Args:
        q (np.ndarray): [x, y, z, w] 形式的 4 元素四元数。

    Returns:
        np.ndarray: 3 元素的指数坐标。
    """
    if (q[-1] < 0):
        q = -q

    theta = 2. * math.atan2(np.linalg.norm(q[:-1]), q[-1])
    w = (1. / np.sin(theta/2.0)) * q[:-1]

    return w * theta


# 用于可视化目标姿态的辅助几何体（一个线框坐标轴）
axes_geom = gymutil.AxesGeometry(0.5)

cnt = 0
while not gym.query_viewer_has_closed(viewer):

    # 推进物理仿真一步，并取回计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 每隔 1000 步设置一个新的随机目标姿态
    if cnt % 1000 == 0:
        # 生成随机目标四元数
        goal_quat = random_quaternion()

        print("New goal orientation:", goal_quat)

        # 清掉上一帧画的线
        gym.clear_lines(viewer)

        # 在环境中画出目标姿态对应的坐标轴（线框表示）
        goal_viz_T = gymapi.Transform(r=gymapi.Quat(*goal_quat))
        gymutil.draw_lines(axes_geom, gym, viewer, env, goal_viz_T)

        # 前 3 个 DOF 目标清零，后 3 个 DOF 写入目标姿态的指数坐标，
        # 让球关节通过位置控制去跟踪这个随机姿态
        dof_positions[:] = 0.0
        dof_positions[3:] = quat2expcoord(goal_quat)

        for i in range(num_envs):
            # 下发 DOF 位置目标（位置控制模式）
            gym.set_actor_dof_position_targets(envs[i], actor_handles[i], dof_positions)

    # 更新渲染画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待真实时间走完一个 dt，
    # 让物理仿真与渲染帧率同步（实时观看效果）
    gym.sync_frame_time(sim)

    cnt += 1

print("Done")

# 销毁资源：先关闭可视化窗口，再销毁仿真器
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
