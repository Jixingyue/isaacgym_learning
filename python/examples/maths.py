"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Gym API 数学工具
------------
演示 Gym API 提供的各种数学运算
并演示如何与 numpy 数据类型互相转换
"""


import numpy as np
import math
from isaacgym import gymapi
# 定义两个三维向量（Vec3：Gym API 的基础向量类型，有 x/y/z 三个分量）
a = gymapi.Vec3(2, 3, 4)
b = gymapi.Vec3(0.5, 0.25, 0.125)
print("(%f, %f, %f)" % (a.x, a.y, a.z))

# 向量的加减、数乘、数除
c = a + b
d = a - b
e = a * 2
f = a / 2.0
print("(%f, %f, %f)" % (c.x, c.y, c.z))
print("(%f, %f, %f)" % (d.x, d.y, d.z))
print("(%f, %f, %f)" % (e.x, e.y, e.z))
print("(%f, %f, %f)" % (f.x, f.y, f.z))
print()

# 从轴-角创建四元数（Quat：表示旋转的方式，即绕指定轴旋转指定角度）
qx1 = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
qy1 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.5 * math.pi)
qz1 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * math.pi)
print("(%f, %f, %f, %f)" % (qx1.x, qx1.y, qx1.z, qx1.w))
print("(%f, %f, %f, %f)" % (qy1.x, qy1.y, qy1.z, qy1.w))
print("(%f, %f, %f, %f)" % (qz1.x, qz1.y, qz1.z, qz1.w))
print()

# 从横滚(roll)、俯仰(pitch)、偏航(yaw)欧拉角创建四元数
qx2 = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
qy2 = gymapi.Quat.from_euler_zyx(0, 0.5 * math.pi, 0)
qz2 = gymapi.Quat.from_euler_zyx(0, 0, 0.5 * math.pi)
print("(%f, %f, %f, %f)" % (qx2.x, qx2.y, qx2.z, qx2.w))
print("(%f, %f, %f, %f)" % (qy2.x, qy2.y, qy2.z, qy2.w))
print("(%f, %f, %f, %f)" % (qz2.x, qz2.y, qz2.z, qz2.w))
print()

# 从四元数反算横滚、俯仰、偏航角
r1, p1, y1 = qx2.to_euler_zyx()
r2, p2, y2 = qy2.to_euler_zyx()
r3, p3, y3 = qz2.to_euler_zyx()
print("%f, %f, %f" % (r1, p1, y1))
print("%f, %f, %f" % (r2, p2, y2))
print("%f, %f, %f" % (r3, p3, y3))
print()

# 四元数乘法（组合两次旋转）与归一化
qxy = (qx1 * qy2).normalize()

# 用四元数旋转三维向量
c = qx1.rotate(a)
d = qy1.rotate(a)
e = qxy.rotate(a)
print("(%f, %f, %f)" % (a.x, a.y, a.z))
print("(%f, %f, %f)" % (c.x, c.y, c.z))
print("(%f, %f, %f)" % (d.x, d.y, d.z))
print("(%f, %f, %f)" % (e.x, e.y, e.z))
print()

# 四元数求逆（两次求逆就还原回原四元数）
qx1_inv = qx1.inverse()
qx1_inv_inv = qx1_inv.inverse()
print(qx1)
print(qx1_inv)
print(qx1_inv_inv)
print()

# 向量的长度和方向（normalize 后得到单位方向向量）
length = a.length()
d = a.normalize()
print("%f" % length)
print("(%f, %f, %f)" % (d.x, d.y, d.z))
print()

# 点积和叉积
dp = a.dot(b)
cp = a.cross(b)
print("%f" % dp)
print("(%f, %f, %f)" % (cp.x, cp.y, cp.z))
print()

# 变换（Transform：位置 + 朝向，常用来描述物体的位姿）
tx = gymapi.Transform(b, qx1)
# 变换点（受平移影响）与变换向量（只被旋转，不受平移影响）
c = tx.transform_point(a)
e = tx.transform_vector(d)
print("(%f, %f, %f)" % (c.x, c.y, c.z))
print("(%f, %f, %f)" % (e.x, e.y, e.z))
print()

# 逆变换
tx_inv = tx.inverse()
tx_inv_inv = tx_inv.inverse()
print(tx.p, tx.r)
print(tx_inv.p, tx_inv.r)
print(tx_inv_inv.p, tx_inv_inv.r)
print()

# 把 Gym 数据类型当作 numpy 结构化数组访问（dtype 是它们对应的 numpy 类型）
print(gymapi.Vec3.dtype)
print(gymapi.Quat.dtype)
print(gymapi.Transform.dtype)
print()

# 用 Gym 数据类型创建 numpy 数组（这里创建含 2 个 Transform 的数组）
arr = np.zeros(2, dtype=gymapi.Transform.dtype)
print(arr)
print(arr['p'])
print(arr['r'])
print()

# 从变换中取出平移和旋转分量，写入结构化数组
arr[0] = ((tx.p.x, tx.p.y, tx.p.z), (tx.r.x, tx.r.y, tx.r.z, tx.r.w))
arr['p'][1] = (1, 2, 3)
arr['r'][1] = (0, 0, 0, 1)
print(arr)
print(arr[0])
print(arr[1])
print()

# 从结构化数组读回 Gym 类型（from_buffer 直接包装数组内存，零拷贝）
pose1 = gymapi.Transform.from_buffer(arr[0])
pose2 = gymapi.Transform.from_buffer(arr[1])
print("(%f, %f, %f) (%f, %f, %f, %f)" % (pose1.p.x, pose1.p.y, pose1.p.z,
                                         pose1.r.x, pose1.r.y, pose1.r.z, pose1.r.w))
print("(%f, %f, %f) (%f, %f, %f, %f)" % (pose2.p.x, pose2.p.y, pose2.p.z,
                                         pose2.r.x, pose2.r.y, pose2.r.z, pose2.r.w))
print()
