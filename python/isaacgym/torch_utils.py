"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

PyTorch 数学工具库：四元数/欧拉角/刚体变换、随机采样、缩放映射等运算，
是机器人强化学习最常用的一组张量工具（均为批量化操作，可同时对整个环境批次计算）。
"""

import torch
import numpy as np


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    """把输入数据转成 PyTorch 张量（可指定数据类型、所在设备、是否需要梯度）"""
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def quat_mul(a, b):
    """四元数乘法（姿态复合）：结果 = 先应用 b 的旋转、再应用 a 的旋转。

    机器人学中用于把两个旋转叠加成一个新姿态，例如"关节旋转 ∘ 初始姿态"。
    四元数排列格式为 (x, y, z, w)，其中 w 是实部。
    """
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    """沿最后一维做 L2 归一化（变成单位向量），eps 用于防止除零"""
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    """用四元数 a 旋转三维向量 b，返回旋转后的向量（效果等价于"旋转矩阵 × 向量"，但计算更快）"""
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    """用四元数 q 正向旋转向量 v：把 v 从 q 的局部（本体）坐标系转到世界坐标系"""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    """用四元数 q 反向旋转向量 v：把 v 从世界坐标系转回 q 的局部（本体）坐标系"""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_conjugate(a):
    """求四元数共轭：向量部分取反、实部保持不变。

    单位四元数的共轭即为它的逆，常用于求逆姿态或做坐标系之间的变换
    （例如"世界 → 本体"与"本体 → 世界"的旋转互为共轭）。
    """
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    """把四元数归一化成单位四元数（只有单位四元数才表示一个纯旋转）"""
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    """由"绕单位轴 axis 旋转 angle 弧度"构造单位四元数（轴角 → 四元数）"""
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    """把任意角度归一化到 (-π, π] 区间内"""
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    """求刚体变换（q 旋转, t 平移）的逆变换：旋转取共轭，平移取反并旋转回原坐标系"""
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    """对点 v 应用刚体变换 (q, t)：先旋转、再平移"""
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    """对方向向量 v 只应用旋转部分（方向向量不受平移影响）"""
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    """复合两个刚体变换：先做 (q2, t2)、再做 (q1, t1)，得到一个等效的变换"""
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    """求单位向量 v 经姿态 q 旋转后的结果，即 v 在世界系下的基向量（如求本体的"上/前"方向）"""
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
    """按轴索引构造 Vec 类型的参数：在指定轴上填 value、其余维填 0（第 0 维固定填 x_value）。

    常用于设置"上轴"这类只有某一维需要非零值的向量参数。
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    """返回 |a| * sign(b)：取 a 的绝对值、再配上 b 的正负号"""
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    """四元数 → 欧拉角 (roll, pitch, yaw)，是机器人学中最直观的姿态表示。

    roll 为绕 X 轴的横滚角、pitch 为绕 Y 轴的俯仰角、yaw 为绕 Z 轴的偏航角；
    pitch 分支用 where 处理 ±90° 处的万向锁奇异问题。返回值均对 2π 取模。
    """
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll（绕 X 轴旋转，横滚角）
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch（绕 Y 轴旋转，俯仰角）
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw（绕 Z 轴旋转，偏航角）
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    """欧拉角 (roll, pitch, yaw) → 四元数 (x, y, z, w)，是 get_euler_xyz 的逆运算"""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    """在 [lower, upper) 内均匀采样，返回指定形状的随机张量（RL 中常用于随机初始化和域随机化）"""
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def torch_random_dir_2(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    """随机采样 2D 单位方向向量：先均匀随机一个角度，再取其 (cos, sin) 得到单位向量"""
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    """把张量 t 逐元素夹紧到 [min_t, max_t] 区间内（上下限本身也可以是张量）"""
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    """把 [-1, 1] 内的值线性映射到 [lower, upper]（例如把策略输出的动作映射到关节力矩范围）"""
    return (0.5 * (x + 1.0) * (upper - lower) + lower)


@torch.jit.script
def unscale(x, lower, upper):
    """scale 的逆运算：把 [lower, upper] 内的值映射回 [-1, 1]"""
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    """unscale 的 Numpy 版本：把 [lower, upper] 内的值映射回 [-1, 1]"""
    return (2.0 * x - upper - lower) / (upper - lower)
