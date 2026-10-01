"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick（Franka 抓取方块）
----------------
使用雅可比矩阵 + 逆运动学（IK）控制 Franka 机械臂去抓取一个方块。

本示例展示两种控制器：
- IK（逆运动学，Inverse Kinematics）：已知末端目标位姿，反解出各关节应该转到什么角度。
- OSC（操作空间控制）：在末端笛卡尔空间设计控制律，再通过雅可比矩阵映射回关节力矩。

IK 使用的是「阻尼最小二乘法（Damped Least Squares）」，参考论文：
https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time


def quat_axis(q, axis=0):
    # 取出四元数 q 旋转矩阵的第 axis 个轴向量（把某根坐标轴转到指定方向）
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    # 计算姿态误差：用"目标四元数 - 当前四元数"得到近似的旋转向量误差
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    # 返回抓取方块所需的水平旋转（yaw 偏航）：根据方块当前朝向计算该绕竖直轴转多少才能夹稳
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares —— 用阻尼最小二乘法求解 IK
    # 原理：把"末端误差 dpose"通过雅可比矩阵反解成"关节角度增量"。
    # 公式：Δq = J^T * (J*J^T + λ^2*I)^-1 * dpose
    # 其中 λ（damping）用来让矩阵可逆、避免奇异点附近关节速度爆炸（所以叫"阻尼"最小二乘）
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    # 操作空间控制：先把末端误差映射成关节力矩，再加上"零空间"力矩保持末端姿态不变
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    # 核心公式：τ = J^T * Λ * (kp*dpose - kd*ẋ)
    # 其中 Λ=m_eef 是末端等效质量矩阵，kp*dpose 是"拉力"项，kd*ẋ 是阻尼项
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    # 零空间控制力矩 u_null：用来防止关节位形大幅漂移，同时不影响末端位姿
    # 它被投射到 OSC 的零空间里，这样既保持末端姿态恒定，又能"顺便"让关节回到默认角度
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    # (I - J^T*J_inv) 是零空间投影算子：只保留不影响末端位姿的关节运动分量
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


# set random seed —— 固定随机种子，保证结果可复现
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface —— 获取 gym 接口
gym = gymapi.acquire_gym()

# parse arguments —— 解析命令行参数

# Add custom arguments —— 添加自定义参数（选择控制器类型 ik/osc、环境数量）
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller —— 取出控制器选项（ik 或 osc）
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device —— 设置 torch 计算设备（用 GPU pipeline 时在 GPU 上，否则 CPU）
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim —— 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters —— 设置控制器参数
# IK params —— IK 参数（阻尼系数）
damping = 0.05

# OSC params —— OSC 参数
kp = 150.        # 位置/姿态误差的比例增益（拉力大小）
kd = 2.0 * np.sqrt(kp)   # 末端速度的阻尼增益
kp_null = 10.    # 零空间位置增益（把关节拉回默认角度）
kd_null = 2.0 * np.sqrt(kp_null)  # 零空间速度阻尼

# create sim —— 创建仿真
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer —— 创建可视化窗口
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "../../assets"

# create table asset —— 创建"桌子"资产（一个长方体盒子）
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create box asset —— 创建"方块"资产（机械爪要去抓的小方块）
box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# load franka asset —— 加载 Franka 模型资产
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs —— 配置 Franka 的关节
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

# use position drive for all dofs —— IK 模式下所有关节用位置驱动
if controller == "ik":
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    # IK 控制器输出的是"目标关节角度"，所以用位置驱动（内置闭环自己算力矩）
    franka_dof_props["stiffness"][:7].fill(400.0)
    franka_dof_props["damping"][:7].fill(40.0)
else:       # osc —— OSC 模式下用力矩驱动
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    # OSC 控制器直接输出力矩，所以设成力矩驱动、刚度阻尼归零
    franka_dof_props["stiffness"][:7].fill(0.0)
    franka_dof_props["damping"][:7].fill(0.0)
# grippers —— 夹爪始终用位置驱动
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets —— 默认关节状态与位置目标
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]
# grippers open —— 夹爪初始保持张开（放在上极限）
default_dof_pos[7:] = franka_upper_limits[7:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch —— 转成 torch 张量
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
# 拿到"panda_hand"（机械手）这个刚体的索引，把它作为末端执行器
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["panda_hand"]

# configure env grid —— 配置环境网格
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

box_pose = gymapi.Transform()

envs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []

# add ground plane —— 添加地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env —— 创建环境
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table —— 添加桌子
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add box —— 在桌面随机位置放置待抓取的方块
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    box_pose.p.z = table_dims.z + 0.5 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor —— 拿方块在刚体状态张量里的全局索引
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    # add franka —— 添加 Franka 机械臂
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)

    # set dof properties —— 应用关节属性
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states —— 设置初始关节状态
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets —— 设置初始关节位置目标
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)
    # set_actor_dof_position_targets：给位置驱动模式的关节下达"目标角度"

    # get inital hand pose —— 读取机械手初始位姿
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor —— 拿机械手全局索引
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

# point camera at middle env —— 相机对准中间环境
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors ===== —— 准备张量
# from now on, we will use the tensor API that can run on CPU or GPU
# 从这一步起改用「张量 API」，既能在 CPU 也能在 GPU 上运行
gym.prepare_sim(sim)

# initial hand position and orientation tensors —— 机械手初始位姿张量
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping —— 抓取时机械手的朝向（手掌朝下）
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw —— 方块角点坐标，用来计算抓取时的偏航角
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis —— 竖直向下方向的单位轴
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor —— 获取雅可比矩阵张量
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
# 固定基座的 Franka，张量形状为 (环境数, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to franka hand —— 只取机械手对应的雅可比块（前 7 个关节）
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# get mass matrix tensor —— 获取质量矩阵张量
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm（只需机械臂前 7 关节的子块）

# get rigid body state tensor —— 获取刚体状态张量
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor —— 获取关节状态张量
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)
dof_vel = dof_states[:, 1].view(num_envs, 9, 1)

# Create a tensor noting whether the hand should return to the initial position
# 创建一个张量，标记"机械手是否需要先回到初始位置"
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors —— 初始化动作张量
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

# simulation loop —— 仿真主循环
while not gym.query_viewer_has_closed(viewer):

    # step the physics —— 前进一步物理仿真
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors —— 刷新张量
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]

    # 机械手到方块的方向向量、距离、以及"是否在方块正上方"的判断基础
    to_box = box_pos - hand_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # how far the hand should be from box for grasping —— 抓取时手离方块应保持的偏移高度
    grasp_offset = 0.11 if controller == "ik" else 0.10

    # determine if we're holding the box (grippers are closed and box is near)
    # 判断是否已经抓住方块（夹爪闭合 且 方块离手很近）
    gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
    gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * box_size)

    yaw_q = cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = quat_axis(yaw_q, 0)
    hand_yaw_dir = quat_axis(hand_rot, 0)
    yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    # 判断机械手是否已回到初始位置；若是，才允许它开始向方块移动
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    # 如果手已经悬停在方块正上方，就下降到抓取高度；否则先移动到方块上方的更高位置
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
    grasp_pos = box_pos.clone()
    grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)

    # compute goal position and orientation —— 计算目标位置与姿态
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

    # compute position and orientation error —— 计算位置误差与姿态误差
    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    # Deploy control based on type —— 根据控制器类型下发控制量
    if controller == "ik":
        # IK：关节目标角 = 当前角 + 反解出来的角度增量
        pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
    else:       # osc
        # OSC：直接输出关节力矩
        effort_action[:, :7] = control_osc(dpose)

    # gripper actions depend on distance between hand and box
    # 夹爪是否闭合取决于手离方块的距离
    close_gripper = (box_dist < grasp_offset + 0.02) | gripped
    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    # 当手抬到一定高度就张开夹爪，放下方块、从头开始
    hand_restart = hand_restart | (box_pos[:, 2] > 0.6)
    keep_going = torch.logical_not(hand_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * num_envs).to(device), torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
    pos_action[:, 7:9] = grip_acts

    # Deploy actions —— 下发动作
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    # update viewer —— 更新渲染画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup —— 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
