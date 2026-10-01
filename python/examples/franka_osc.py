"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Franka Operational Space Control（Franka 操作空间控制 OSC）
------------------------------------------------------------
对 Franka 机械臂做「操作空间控制」，用于演示雅可比矩阵（Jacobian）和质量矩阵（Mass Matrix）的张量 API。

OSC（Operational Space Control，任务空间/操作空间控制）的核心思想：
不直接指定每个关节的角度，而是在机械臂末端（笛卡尔空间）设计控制律——
比如"想让末端朝某个方向移动、朝某个姿态旋转"，
再通过雅可比矩阵把末端的误差映射回每个关节需要施加的力矩。
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    # orientation_error 计算姿态误差：用四元数做"目标姿态 - 当前姿态"，
    # 得到的是一个旋转向量的近似（方向 + 大小），用来当作转动的"误差信号"


# Parse arguments —— 解析命令行参数（多加了环境数量、是否画圆轨迹、是否随机姿态等参数）
args = gymutil.parse_arguments(description="Franka Tensor OSC Example",
                               custom_parameters=[
                                   {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
                                   {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                   {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])

# Initialize gym —— 初始化 gym 接口
gym = gymapi.acquire_gym()

# configure sim —— 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

# Create viewer —— 创建可视化窗口
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Add ground plane —— 添加地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Load franka asset —— 加载 Franka 模型资产
asset_root = "../../assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01
asset_options.disable_gravity = True

print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
franka_asset = gym.load_asset(
    sim, asset_root, franka_asset_file, asset_options)

# get joint limits and ranges for Franka —— 获取 Franka 各关节的限位与行程范围
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
# get_asset_dof_properties：直接从资产（模型）上读取关节属性（无需先创建具体环境）
franka_lower_limits = franka_dof_props['lower']
franka_upper_limits = franka_dof_props['upper']
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
franka_num_dofs = len(franka_dof_props)

# set default DOF states —— 设置默认关节状态（前 7 个电机关节放在中间角度）
default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"][:7] = franka_mids[:7]

# set DOF control properties (except grippers) —— 设置关节控制属性（前 7 个=机械臂本体）
franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
# DOF_MODE_EFFORT：力矩控制模式——不自动做位置闭环，而是直接接受我们算出的力矩（这正是 OSC 需要的）
franka_dof_props["stiffness"][:7].fill(0.0)
franka_dof_props["damping"][:7].fill(0.0)
# 机械臂本体不给刚度/阻尼，完全由我们的 OSC 控制律输出力矩

# set DOF control properties for grippers —— 设置夹爪（手指）的控制属性
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)
# 夹爪用位置控制（带刚度和阻尼），保持张开状态即可

# Set up the env grid —— 设置环境网格
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default franka pose —— 默认的 Franka 初始位姿
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
pose.r = gymapi.Quat(0, 0, 0, 1)

print("Creating %d environments" % num_envs)

envs = []
hand_idxs = []
init_pos_list = []
init_orn_list = []

for i in range(num_envs):
    # Create env —— 创建单个环境
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Add franka —— 添加 Franka 机械臂
    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 1)

    # Set initial DOF states —— 设置初始关节状态
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # Set DOF control properties —— 应用关节控制属性
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # Get inital hand pose —— 读取机械手（末端）的初始位姿
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # Get global index of hand in rigid body state tensor
    # 拿到机械手在"刚体状态张量"里的全局索引，后面用张量批量读取所有环境的数据
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

# Point camera at middle env —— 相机对准中间环境
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors ===== —— 准备张量
# from now on, we will use the tensor API to access and control the physics simulation
# 从这一步开始，改用「张量 API」读写和控制物理仿真（可批量、可在 GPU 上运行）
gym.prepare_sim(sim)

# initial hand position and orientation tensors —— 机械手初始位置/姿态的张量
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3)
init_orn = torch.Tensor(init_orn_list).view(num_envs, 4)

if args.use_gpu_pipeline:
    init_pos = init_pos.to('cuda:0')
    init_orn = init_orn.to('cuda:0')

# desired hand positions and orientations —— 期望（目标）的机械手位置与姿态
pos_des = init_pos.clone()
orn_des = init_orn.clone()

# Prepare jacobian tensor —— 准备雅可比矩阵张量
# 雅可比矩阵：把「关节速度」映射成「末端速度」的线性变换矩阵，是 OSC/IK 的桥梁
# For franka, tensor shape is (num_envs, 10, 6, 9)
# 对 Franka（含夹爪和世界坐标系共 10 个刚体），张量形状为 (环境数, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)
# acquire_jacobian_tensor：获取雅可比矩阵张量（每个刚体一个 6×9 的雅可比）

# Jacobian entries for end effector —— 取出末端执行器（机械手）对应的雅可比块
hand_index = gym.get_asset_rigid_body_dict(franka_asset)["panda_hand"]
j_eef = jacobian[:, hand_index - 1, :]

# Prepare mass matrix tensor —— 准备质量矩阵张量
# 质量矩阵描述机械臂各关节的"惯量耦合"，OSC 需要它把末端力换算回关节力矩
# For franka, tensor shape is (num_envs, 9, 9)
# 对 Franka（9 个关节），形状为 (环境数, 9, 9)
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)

kp = 5
kv = 2 * math.sqrt(kp)
# kp 是位置/姿态误差的"比例增益"（相当于弹簧硬度），kv 是速度阻尼增益

# Rigid body state tensor —— 刚体状态张量（位置、姿态、速度等）
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# DOF state tensor —— 关节状态张量（位置、速度）
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
# acquire_dof_state_tensor：获取关节状态张量（第 0 维是位置，第 1 维是速度）
dof_vel = dof_states[:, 1].view(num_envs, 9, 1)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)

itr = 0
while not gym.query_viewer_has_closed(viewer):

    # Randomize desired hand orientations —— 周期性随机化目标姿态
    if itr % 250 == 0 and args.orn_control:
        orn_des = torch.rand_like(orn_des)
        orn_des /= torch.norm(orn_des)

    itr += 1

    # Update jacobian and mass matrix —— 刷新雅可比矩阵与质量矩阵（它们随姿态变化）
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    # Get current hand poses —— 读取机械手当前位姿
    pos_cur = rb_states[hand_idxs, :3]
    orn_cur = rb_states[hand_idxs, 3:7]

    # Set desired hand positions —— 设定目标位置（在 XZ 平面画圆轨迹）
    if args.pos_control:
        pos_des[:, 0] = init_pos[:, 0] - 0.1
        pos_des[:, 1] = math.sin(itr / 50) * 0.2
        pos_des[:, 2] = init_pos[:, 2] + math.cos(itr / 50) * 0.2

    # Solve for control (Operational Space Control) —— 计算操作空间控制律
    # 下面就是 OSC 的核心公式，一步步拆解：
    m_inv = torch.inverse(mm)
    # 1) 先算末端在笛卡尔空间下的"等效质量矩阵" m_eef：
    #    m_eef = (J * M^-1 * J^T)^-1，其中 J 是雅可比，M 是关节质量矩阵
    m_eef = torch.inverse(j_eef @ m_inv @ torch.transpose(j_eef, 1, 2))
    orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
    orn_err = orientation_error(orn_des, orn_cur)
    # 2) 位置误差：目标位置减去当前位置，再乘以比例增益 kp（拉得越远力越大）
    pos_err = kp * (pos_des - pos_cur)

    if not args.pos_control:
        pos_err *= 0

    # 3) 把位置误差和姿态误差拼成 6 维的"末端速度误差" dpose
    dpose = torch.cat([pos_err, orn_err], -1)

    # 4) 核心公式 u = J^T * m_eef * (kp * dpose) - kv * M * q_dot
    #    第一项：把末端误差通过雅可比转置映射回关节力矩（让末端朝目标走）
    #    第二项：加一个与关节速度成正比的阻尼项，防止震荡
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose).unsqueeze(-1) - kv * mm @ dof_vel

    # Set tensor action —— 把算好的关节力矩下发给物理引擎
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))
    # set_dof_actuation_force_tensor：直接给关节施加力矩（不经过内置位置驱动器）

    # Step the physics —— 前进一步物理仿真
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering —— 前进一步渲染
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    # gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
