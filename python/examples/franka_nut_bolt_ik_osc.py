"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka 拧螺丝（螺母 + 螺栓）
-----------------------------
本文件演示用雅可比矩阵 + 逆运动学（IK）控制 Franka 机械臂，把螺母旋拧到螺栓上。
核心是一个「有限状态机（FSM，Finite State Machine）」：把拧螺丝这件复杂任务拆成
一连串有序的小步骤（悬停到螺母上方 → 下降 → 夹紧 → 抬起 → 对准螺栓 → 旋入 …），
每个状态只负责一个小目标，到达后自动切换到下一个状态。

（注意：这里使用阻尼最小二乘法求 IK，参考论文：
https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf）
"""

from inspect import Attribute
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch


def orientation_error(desired, current):
    # 计算姿态误差：目标四元数与当前四元数的"差"，近似为一个旋转向量
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[0:3] * torch.sign(q_r[3])


def control_ik(dpose, damping, j_eef, num_envs):
    # solve damped least squares —— 用阻尼最小二乘法求解逆运动学（IK）
    # 已知末端误差 dpose，反解各关节需要的角度增量：
    # Δq = J^T * (J*J^T + λ^2*I)^-1 * dpose
    # λ 是阻尼系数，避免奇异点处关节速度爆炸
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


class ScrewFSM:
    """拧螺丝任务的有限状态机：记录当前状态，每帧根据位姿误差决定是否切换到下一状态。"""

    def __init__(self, sim_dt, nut_height, bolt_height, screw_speed, screw_limit_angle, device, env_idx):
        self._sim_dt = sim_dt
        self._nut_height = nut_height
        self._bolt_height = bolt_height
        self._screw_speed = screw_speed
        self._screw_limit_angle = screw_limit_angle
        self.device = device
        self.env_idx = env_idx

        # states: —— 当前状态，初始为"悬停到螺母上方"
        self._state = "go_above_nut"

        # control / position constants: —— 各状态用到的位置/控制常量（都是相对螺母/螺栓的偏移）
        # 悬停在螺母正上方的高度偏移
        self._above_offset = torch.tensor([0, 0, 0.08 + self._bolt_height], dtype=torch.float32, device=self.device)
        # 下降到"准备夹紧"高度（略高于螺母顶面）
        self._grip_offset = torch.tensor([0, 0, 0.12 + self._nut_height], dtype=torch.float32, device=self.device)
        # 拎起螺母后的抬起高度
        self._lift_offset = torch.tensor([0, 0, 0.15 + self._bolt_height], dtype=torch.float32, device=self.device)
        # 在螺栓上方悬停的高度
        self._above_bolt_offset = torch.tensor([0, 0, self._bolt_height], dtype=torch.float32, device=self.device) + self._grip_offset
        # 套到螺栓上（下压一部分）后的高度
        self._on_bolt_offset = torch.tensor([0, 0, 0.8 * self._bolt_height], dtype=torch.float32, device=self.device) + self._grip_offset
        # "手掌朝下"的四元数（机械手朝下的基准姿态）
        self._hand_down_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        # 抓取螺母时的倾斜角（绕竖直轴转 30°），用来更稳地夹住六角螺母
        grab_angle = torch.tensor([np.pi / 6.0], dtype=torch.float32, device=self.device)
        grab_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)
        grab_quat = quat_from_angle_axis(grab_angle, grab_axis).squeeze()
        self._nut_grab_q = quat_mul(grab_quat, self._hand_down_quat)
        # 当前累计的旋入角度，以及旋入轴（绕竖直轴旋转）
        self._screw_angle = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        self._screw_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)

        # 6 维末端位姿误差缓存（前三位置、后三姿态）
        self._dpose = torch.zeros(6, dtype=torch.float32, device=self.device)
        # 夹爪开合程度
        self._gripper_separation = 0.0

    def get_dp_from_target(self, target_pos, target_quat, hand_pose) -> float:
        # 计算"末端当前位姿"到"目标位姿"的 6 维误差，并返回其模长（误差大小）
        self._dpose[:3] = target_pos - hand_pose[:3]
        self._dpose[3:] = orientation_error(target_quat, hand_pose[3:])
        return torch.norm(self._dpose, p=2)

    # returns
    def update(self, nut_pose, bolt_pose, hand_pose, current_gripper_sep):
        # 每帧调用：根据当前状态设定目标位姿；一旦误差足够小就切换到下一状态
        newState = self._state
        if self._state == "go_above_nut":
            # 状态1：张开夹爪，先悬停到螺母正上方
            self._gripper_separation = 0.08
            target_pos = nut_pose[:3] + self._above_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "prep_grip"
        elif self._state == "prep_grip":
            # 状态2：保持张开，下降到准备夹紧的高度
            pass
            self._gripper_separation = 0.08
            target_pos = nut_pose[:3] + self._grip_offset
            targetQ = quat_mul(nut_pose[3:], self._nut_grab_q)
            error = self.get_dp_from_target(target_pos, targetQ, hand_pose)
            if error < 2e-3:
                newState = "grip"
        elif self._state == "grip":
            # 状态3：合拢夹爪，夹住螺母（等夹爪实际闭合到位）
            self._gripper_separation = 0.0
            target_pos = nut_pose[:3] + self._grip_offset
            targetQ = quat_mul(nut_pose[3:], self._nut_grab_q)
            error = self.get_dp_from_target(target_pos, targetQ, hand_pose)
            gripped = (current_gripper_sep < 0.035)
            if error < 1e-2 and gripped:
                newState = "lift"
        elif self._state == "lift":
            # 状态4：夹着螺母向上提起
            self._gripper_separation = 0.0
            target_pos = nut_pose[:3]
            target_pos[2] = bolt_pose[2] + 0.004
            target_pos = target_pos + self._lift_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "go_above_bolt"
        elif self._state == "go_above_bolt":
            # 状态5：平移到螺栓的正上方
            self._gripper_separation = 0.0
            target_pos = bolt_pose[:3]
            target_pos = target_pos + self._above_bolt_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "go_on_bolt"
        elif self._state == "go_on_bolt":
            # 状态6：向下把螺母套到螺栓上（到螺栓高度）
            self._gripper_separation = 0.0
            target_pos = bolt_pose[:3]
            target_pos[2] = bolt_pose[2]
            target_pos = target_pos + self._on_bolt_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "loosen_grip"
        elif self._state == "loosen_grip":
            # 状态7：略微松开夹爪（留一点间隙，让螺母能旋转）
            target_sep = 0.037
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos = target_pos + self._on_bolt_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            un_gripped = current_gripper_sep > target_sep * 0.98
            if error < 2e-3 and un_gripped:
                self._screw_angle[0] = 0.0
                newState = "screw_motion"
        elif self._state == "screw_motion":
            # 状态8：旋入！夹爪一边保持"半松"夹持，一边绕竖直轴旋转，把螺母拧进螺栓
            target_sep = 0.037
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos[2] = nut_pose[2]
            target_pos = target_pos + self._grip_offset
            # 每个仿真步把旋入角往前推进一点（转速 × 步长）
            self._screw_angle[0] = self._screw_angle[0] - self._sim_dt * self._screw_speed
            screw_quat = quat_from_angle_axis(self._screw_angle, self._screw_axis).squeeze()
            self.get_dp_from_target(target_pos, quat_mul(screw_quat, self._hand_down_quat), hand_pose)
            if self._screw_angle[0] < -self._screw_limit_angle:
                newState = "ungrip_screw"
        elif self._state == "ungrip_screw":
            # 状态9：旋到目标角度后，完全松开夹爪
            target_sep = 0.06
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos[2] = nut_pose[2]
            target_pos = target_pos + self._grip_offset
            screw_quat = quat_from_angle_axis(self._screw_angle, self._screw_axis).squeeze()
            self.get_dp_from_target(target_pos, quat_mul(screw_quat, self._hand_down_quat), hand_pose)
            un_gripped = current_gripper_sep > target_sep * 0.98
            if un_gripped:
                newState = "rotate_back"
        elif self._state == "rotate_back":
            # 状态10：松手后，反向把机械手转回初始角（为下一次拧紧做准备）
            target_sep = 0.06
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos[2] = nut_pose[2]
            target_pos = target_pos + self._grip_offset
            self._screw_angle[0] = self._screw_angle[0] + self._sim_dt * 2.0 * self._screw_speed
            screw_quat = quat_from_angle_axis(self._screw_angle, self._screw_axis).squeeze()
            self.get_dp_from_target(target_pos, quat_mul(screw_quat, self._hand_down_quat), hand_pose)
            if self._screw_angle[0] > 0.99 * self._screw_limit_angle:
                newState = "back_to_screw_grip"
        elif self._state == "back_to_screw_grip":
            # 状态11：重新收紧夹爪，准备继续旋入（拧螺丝通常要"拧一下、松一下、再拧"）
            target_sep = 0.037
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos[2] = nut_pose[2]
            target_pos = target_pos + self._grip_offset
            screw_quat = quat_from_angle_axis(self._screw_angle, self._screw_axis).squeeze()
            error = self.get_dp_from_target(target_pos, quat_mul(screw_quat, self._hand_down_quat), hand_pose)
            gripped = (current_gripper_sep < target_sep * 1.01)
            if error < 2e-3 and gripped:
                self._screw_angle[0] = self._screw_limit_angle
                newState = "screw_motion"

        if newState != self._state:
            # 状态发生变化时更新并打印，方便观察流程走到哪一步
            self._state = newState
            print(f"Env {self.env_idx} going to state {newState}")

    @property
    def d_pose(self):
        return self._dpose

    @property
    def gripper_separation(self):
        return self._gripper_separation


# set random seed —— 固定随机种子
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface —— 获取 gym 接口
gym = gymapi.acquire_gym()

# parse arguments —— 解析命令行参数

# Add custom arguments —— 添加自定义参数（环境数量）
custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 4, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) Nut-Bolt Screwing",
    custom_parameters=custom_parameters,
)

# Force GPU: —— 强制使用 GPU（本示例的 SDF 不支持 CPU 仿真）
if not args.use_gpu or args.use_gpu_pipeline:
    print("Forcing GPU sim - CPU sim not supported by SDF")
    args.use_gpu = True
    args.use_gpu_pipeline = True

# set torch device —— 设置 torch 设备
device = args.sim_device

# configure sim —— 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 32
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.005
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters —— 设置控制器参数
# IK params —— IK 参数（阻尼系数）
damping = 0.15

# create sim —— 创建仿真
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer —— 创建可视化窗口
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "../../assets"

# create table asset —— 创建桌子资产
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create bolt asset —— 创建螺栓资产（拧螺丝的"公件"）
bolt_file = "urdf/nut_bolt/bolt_m4_tight_SI_5x.urdf"
bolt_options = gymapi.AssetOptions()
bolt_options.flip_visual_attachments = False  # default = False
bolt_options.fix_base_link = True
# 下面这些是螺栓的物理材质参数（摩擦、密度、速度上限等），让螺栓稳稳立在桌面上
bolt_options.thickness = 0.0  # default = 0.02 (not overridden in .cpp)
bolt_options.density = 800.0  # 7850.0
bolt_options.armature = 0.0  # default = 0.0
bolt_options.linear_damping = 0.0  # default = 0.0
bolt_options.max_linear_velocity = 1000.0  # default = 1000.0
bolt_options.angular_damping = 0.0  # default = 0.5
bolt_options.max_angular_velocity = 1000.0  # default = 64.0
bolt_options.disable_gravity = False  # default = False
bolt_options.enable_gyroscopic_forces = True  # default = True
bolt_asset = gym.load_asset(sim, asset_root, bolt_file, bolt_options)

# create nut asset —— 创建螺母资产（拧螺丝的"母件"，要被夹起来套上去）
nut_file = "urdf/nut_bolt/nut_m4_tight_SI_5x.urdf"
nut_options = gymapi.AssetOptions()
nut_options.flip_visual_attachments = False  # default = False
nut_options.fix_base_link = False
nut_options.thickness = 0.0  # default = 0.02 (not overridden in .cpp)
nut_options.density = 800  # 7850.0  # default = 1000
nut_options.armature = 0.0  # default = 0.0
nut_options.linear_damping = 0.0  # default = 0.0
nut_options.max_linear_velocity = 1000.0  # default = 1000.0
nut_options.angular_damping = 0.0  # default = 0.5
nut_options.max_angular_velocity = 1000.0  # default = 64.0
nut_options.disable_gravity = False  # default = False
nut_options.enable_gyroscopic_forces = True  # default = True
nut_asset = gym.load_asset(sim, asset_root, nut_file, nut_options)

# create box asset ——（这里原本可创建方块资产，现已注释掉，本示例用不到）

#asset_options = gymapi.AssetOptions()
#box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# load franka asset —— 加载 Franka 模型资产
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs —— 配置 Franka 关节
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

# use position drive for all dofs —— 本示例统一用位置驱动（IK 输出目标角度）
franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][:7].fill(400.0)
franka_dof_props["damping"][:7].fill(40.0)
# grippers —— 夹爪
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets —— 默认关节状态与位置目标
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]
# grippers open —— 夹爪初始张开
default_dof_pos[7:] = franka_upper_limits[7:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch —— 转成 torch 张量
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
# 拿到"panda_hand"（机械手）刚体索引，作为末端执行器
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
bolt_pose = gymapi.Transform()
nut_pose = gymapi.Transform()

# fsm parameters: —— 状态机相关参数（FSM 在 CPU 上运行）
fsm_device = 'cpu'

envs = []
nut_idxs = []
bolt_idxs = []
hand_idxs = []
fsms = []

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

    # add bolt —— 在桌面随机位置放置螺栓
    bolt_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
    bolt_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.0)
    bolt_pose.p.z = table_dims.z
    bolt_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    bolt_handle = gym.create_actor(env, bolt_asset, bolt_pose, "bolt", i, 0)
    bolt_props = gym.get_actor_rigid_shape_properties(env, bolt_handle)
    #bolt_props[0].filter = imesh
    # 把螺栓的摩擦等参数调低，方便旋转拧入
    bolt_props[0].friction = 0.0  # default = ?
    bolt_props[0].rolling_friction = 0.0  # default = 0.0
    bolt_props[0].torsion_friction = 0.0  # default = 0.0
    bolt_props[0].restitution = 0.0  # default = ?
    bolt_props[0].compliance = 0.0  # default = 0.0
    bolt_props[0].thickness = 0.0  # default = 0.0
    gym.set_actor_rigid_shape_properties(env, bolt_handle, bolt_props)

    # get global index of box in rigid body state tensor —— 拿螺栓全局索引
    bolt_idx = gym.get_actor_rigid_body_index(env, bolt_handle, 0, gymapi.DOMAIN_SIM)
    bolt_idxs.append(bolt_idx)

    # add nut —— 在螺栓附近随机放置螺母
    nut_pose.p.x = bolt_pose.p.x + np.random.uniform(-0.04, 0.04)
    nut_pose.p.y = bolt_pose.p.y + 0.2 + np.random.uniform(-0.04, 0.04)
    nut_pose.p.z = table_dims.z + 0.02
    nut_handle = gym.create_actor(env, nut_asset, nut_pose, "nut", i, 0)
    nut_props = gym.get_actor_rigid_shape_properties(env, nut_handle)
    #nut_props[0].filter = i
    # 螺母保留一定摩擦，避免夹起后轻易滑落
    nut_props[0].friction = 0.2  # default = ?
    nut_props[0].rolling_friction = 0.0  # default = 0.0
    nut_props[0].torsion_friction = 0.0  # default = 0.0
    nut_props[0].restitution = 0.0  # default = ?
    nut_props[0].compliance = 0.0  # default = 0.0
    nut_props[0].thickness = 0.0  # default = 0.0
    gym.set_actor_rigid_shape_properties(env, nut_handle, nut_props)

    # get global index of box in rigid body state tensor —— 拿螺母全局索引
    nut_idx = gym.get_actor_rigid_body_index(env, nut_handle, 0, gymapi.DOMAIN_SIM)
    nut_idxs.append(nut_idx)

    # add franka —— 添加 Franka 机械臂
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 0)

    # set dof properties —— 应用关节属性
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states —— 设置初始关节状态
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets —— 设置初始关节位置目标
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get global index of hand in rigid body state tensor —— 拿机械手全局索引
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    # create env's fsm - run them on CPU
    # 为每个环境创建一个状态机（在 CPU 上运行）
    fsms.append(ScrewFSM(sim_params.dt, 0.016, 0.1, 30.0 / 180.0 * np.pi, 60.0/180.0 * np.pi, fsm_device, i))

# point camera at middle env —— 相机对准中间环境
cam_pos = gymapi.Vec3(1, 0, 0.6)
cam_target = gymapi.Vec3(-1, 0, 0.5)
middle_env = envs[0]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors ===== —— 准备张量
# from now on, we will use the tensor API that can run on CPU or GPU
# 从这一步起改用「张量 API」，既能在 CPU 也能在 GPU 上运行
gym.prepare_sim(sim)

# get jacobian tensor —— 获取雅可比矩阵张量
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
# 固定基座 Franka，形状为 (环境数, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to franka hand —— 取机械手对应的雅可比块（前 7 关节）
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# get rigid body state tensor —— 获取刚体状态张量
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor —— 获取关节状态张量
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)

# Set action tensors —— 初始化动作张量
pos_action = torch.zeros_like(dof_pos).squeeze(-1)

# dp and gripper sep tensors: —— 位姿误差与夹爪开合度的张量（供 FSM 使用）
d_pose = torch.zeros((num_envs, 6), dtype=torch.float32, device=fsm_device)
grip_sep = torch.zeros((num_envs, 1), dtype=torch.float32, device=fsm_device)

# simulation loop —— 仿真主循环
while not gym.query_viewer_has_closed(viewer):

    # step the physics —— 前进一步物理仿真
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors —— 刷新张量
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    # 把状态张量搬到 CPU，供状态机逐环境读取（FSM 在 CPU 上跑）
    rb_states_fsm = rb_states.to(fsm_device)
    nut_poses = rb_states_fsm[nut_idxs, :7]
    bolt_poses = rb_states_fsm[bolt_idxs, :7]
    hand_poses = rb_states_fsm[hand_idxs, :7]
    dof_pos_fsm = dof_pos.to(fsm_device)
    cur_grip_sep_fsm = dof_pos_fsm[:, 7] + dof_pos_fsm[:, 8]
    for env_idx in range(num_envs):
        # 每个环境独立运行自己的状态机，输出目标位姿误差与夹爪目标开合度
        fsms[env_idx].update(nut_poses[env_idx, :], bolt_poses[env_idx, :], hand_poses[env_idx, :], cur_grip_sep_fsm[env_idx])
        d_pose[env_idx, :] = fsms[env_idx].d_pose
        grip_sep[env_idx] = fsms[env_idx].gripper_separation

    # 用 IK 把"末端位姿误差"反解成"关节角度增量"，得到新的关节目标角
    pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(d_pose.unsqueeze(-1).to(device), damping, j_eef, num_envs)
    # gripper actions depend on distance between hand and box —— 夹爪的开关取决于离目标物体的距离

    # 把夹爪目标开合度拆成两根手指各自的目标角度
    grip_acts = torch.cat((0.5 * grip_sep, 0.5 * grip_sep), 1).to(device)
    pos_action[:, 7:9] = grip_acts

    # Deploy actions —— 下发动作（关节位置目标）
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

    # update viewer —— 更新渲染画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup —— 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
