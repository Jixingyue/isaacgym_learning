"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Joint Monkey（关节动画演示）
------------
- 为指定资产（asset，机器人模型）逐个播放自由度（DOF）的运动范围。
- 演示 DOF 属性与状态的读取和设置。
- 演示用画线工具可视化 DOF 坐标系（原点和轴）。
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


# 可选的机器人模型列表（运行时通过 --asset_id 参数选择加载哪一个）
asset_descriptors = [
    AssetDesc("mjcf/nv_humanoid.xml", False),
    AssetDesc("mjcf/nv_ant.xml", False),
    AssetDesc("urdf/cartpole.urdf", False),
    AssetDesc("urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf", False),
    AssetDesc("urdf/franka_description/robots/franka_panda.urdf", True),
    AssetDesc("urdf/kinova_description/urdf/kinova.urdf", False),
    AssetDesc("urdf/anymal_b_simple_description/urdf/anymal.urdf", True),
]


# 解析命令行参数（--asset_id 选模型、--speed_scale 调动画速度、--show_axis 显示关节轴）
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 0, "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
    quit()


# 初始化 gym：获取 Gym 接口实例（对 Isaac Gym 一切操作的入口）
gym = gymapi.acquire_gym()

# 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0  # 仿真步长：1/60 秒（对应 60 FPS）
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

# 添加地面平面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 创建可视化窗口（viewer），使用默认相机属性
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 加载资产（asset：机器人的模型描述，相当于"模板"，还不是场景中的实体）
asset_root = "../../assets"
asset_file = asset_descriptors[args.asset_id].file_name

# asset_options：加载资产时的选项（是否固定底座、是否翻转视觉网格等）
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # 固定底座，让机器人悬空做关节动画
asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 获取所有 DOF（自由度，可理解为"关节"）的名称数组
dof_names = gym.get_asset_dof_names(asset)

# 获取 DOF 属性数组（刚度、阻尼、限位等）
dof_props = gym.get_asset_dof_properties(asset)

# 创建 DOF 状态数组（每个元素包含位置 pos 和速度 vel），之后用它驱动各个 actor
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

# 初始化默认位置、限位和动画速度（确保都在合理范围内）
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
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
    # 把该 DOF 的位置设为默认值
    dof_positions[i] = defaults[i]
    # 根据 DOF 类型和运动范围确定动画速度（范围越大动得越快）
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

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

# 设置环境（env）网格布局：共 36 个环境排成网格，每行 6 个，间距 2.5 米
num_envs = 36
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 设置观察相机的位置和看向的目标点
cam_pos = gymapi.Vec3(17.2, 2.0, 16)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 缓存有用的句柄（handle：整数 ID，之后用它指代对应的 env / actor）
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # 创建环境（env：一块独立的仿真空间，里面可以放各种 actor）
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # 创建 actor（演员：把资产实例化放进环境，相当于场景里的一台机器人）
    # pose 是初始位姿：p 为位置，r 为四元数表示的姿态（旋转）
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # 设置 actor 的初始 DOF 状态（位置和速度一起写入）
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

# 关节动画状态机的 4 个状态，流转顺序为：
# ANIM_SEEK_LOWER（运动到下限）→ ANIM_SEEK_UPPER（运动到上限）
# → ANIM_SEEK_DEFAULT（回到默认位置）→ ANIM_FINISHED（本关节完成，切换到下一个关节后重新开始）
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

# 初始化动画状态：从"运动到下限"开始，先驱动第 0 个关节
anim_state = ANIM_SEEK_LOWER
current_dof = 0
print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

while not gym.query_viewer_has_closed(viewer):

    # 推进物理仿真一步，并取回计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    speed = speeds[current_dof]

    # 驱动当前关节做动画（状态机流转：每次只有一个关节在动，其余保持默认位置）
    if anim_state == ANIM_SEEK_LOWER:
        # 状态 1：向下限匀速运动，到达下限后切换到"运动到上限"
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= lower_limits[current_dof]:
            dof_positions[current_dof] = lower_limits[current_dof]
            anim_state = ANIM_SEEK_UPPER
    elif anim_state == ANIM_SEEK_UPPER:
        # 状态 2：向上限匀速运动，到达上限后切换到"回默认位置"
        dof_positions[current_dof] += speed * dt
        if dof_positions[current_dof] >= upper_limits[current_dof]:
            dof_positions[current_dof] = upper_limits[current_dof]
            anim_state = ANIM_SEEK_DEFAULT
    if anim_state == ANIM_SEEK_DEFAULT:
        # 状态 3：从上限往回运动到默认位置，到达后进入"完成"状态
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= defaults[current_dof]:
            dof_positions[current_dof] = defaults[current_dof]
            anim_state = ANIM_FINISHED
    elif anim_state == ANIM_FINISHED:
        # 状态 4：当前关节动画完成 → 切换到下一个关节（最后一个之后循环回第 0 个），重新进入状态 1
        dof_positions[current_dof] = defaults[current_dof]
        current_dof = (current_dof + 1) % num_dofs
        anim_state = ANIM_SEEK_LOWER
        print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

    if args.show_axis:
        # 开启 --show_axis 时，先清掉上一帧画的线
        gym.clear_lines(viewer)

    # 把这份 DOF 状态同步（克隆）到所有环境中的 actor（每个环境里的机器人做一模一样的动作）
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

        if args.show_axis:
            # 获取当前 DOF 的坐标系（原点 origin 和轴方向 axis）
            dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i], current_dof)
            frame = gym.get_dof_frame(envs[i], dof_handle)

            # 从 DOF 原点沿关节轴方向画一条红色线段，直观显示关节的旋转轴
            p1 = frame.origin
            p2 = frame.origin + frame.axis * 0.7
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

    # 更新渲染画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待真实时间走完一个 dt，
    # 让物理仿真与渲染帧率同步（这样看到的是实时的动画效果）
    gym.sync_frame_time(sim)

print("Done")

# 销毁资源：先关闭可视化窗口，再销毁仿真器
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
