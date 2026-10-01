"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Gym 工具集：命令行参数解析（parse_arguments）+ 调试绘制（draw_line / 线框几何体）等实用功能，
Isaac Gym 的所有示例程序都会 import 本模块
"""

from __future__ import print_function, division, absolute_import

from abc import ABC, abstractmethod
import math
import numpy as np
import argparse
from bisect import bisect

from . import gymapi


class LineGeometry(ABC):
    """线段几何基类：抽象出"一组线段"的顶点与颜色，供调试绘制（draw_lines）使用"""

    @abstractmethod
    def vertices(self):
        """ 返回 Vec3 类型的 Numpy 数组，形状为 (num_lines(), 2)，每行是一条线段的两个端点 """

    @abstractmethod
    def colors(self):
        """ 返回 Vec3 类型的 Numpy 数组，长度为 num_lines()，每项对应一条线段的颜色 """

    def num_lines(self):
        """返回线段的数量"""
        return self.vertices().shape[0]

    # 按指定姿态（pose）对顶点做刚体变换，返回变换后的顶点实例
    def instance_verts(self, pose=None):
        if pose is not None:
            return pose.transform_points(self.vertices())
        else:
            return np.copy(self.vertices())


class AxesGeometry(LineGeometry):
    """坐标轴几何体：用红/绿/蓝三条线段分别表示 X/Y/Z 轴，常用于可视化坐标系朝向"""

    def __init__(self, scale=1.0, pose=None):
        verts = np.empty((3, 2), gymapi.Vec3.dtype)
        verts[0][0] = (0, 0, 0)
        verts[0][1] = (scale, 0, 0)
        verts[1][0] = (0, 0, 0)
        verts[1][1] = (0, scale, 0)
        verts[2][0] = (0, 0, 0)
        verts[2][1] = (0, 0, scale)

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        colors = np.empty(3, gymapi.Vec3.dtype)
        colors[0] = (1.0, 0.0, 0.0)
        colors[1] = (0.0, 1.0, 0.0)
        colors[2] = (0.0, 0.0, 1.0)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class WireframeBoxGeometry(LineGeometry):
    """线框长方体几何体：用 12 条线段画出一个长方体的棱，常用于可视化包围盒/碰撞体"""

    def __init__(self, xdim=1, ydim=1, zdim=1, pose=None, color=None):
        if color is None:
            color = (1, 0, 0)

        num_lines = 12

        x = 0.5 * xdim
        y = 0.5 * ydim
        z = 0.5 * zdim

        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        # 顶面
        verts[0][0] = (x, y, z)
        verts[0][1] = (x, y, -z)
        verts[1][0] = (-x, y, z)
        verts[1][1] = (-x, y, -z)
        verts[2][0] = (x, y, z)
        verts[2][1] = (-x, y, z)
        verts[3][0] = (x, y, -z)
        verts[3][1] = (-x, y, -z)
        # 底面
        verts[4][0] = (x, -y, z)
        verts[4][1] = (x, -y, -z)
        verts[5][0] = (-x, -y, z)
        verts[5][1] = (-x, -y, -z)
        verts[6][0] = (x, -y, z)
        verts[6][1] = (-x, -y, z)
        verts[7][0] = (x, -y, -z)
        verts[7][1] = (-x, -y, -z)
        # 竖直棱
        verts[8][0] = (x, y, z)
        verts[8][1] = (x, -y, z)
        verts[9][0] = (x, y, -z)
        verts[9][1] = (x, -y, -z)
        verts[10][0] = (-x, y, z)
        verts[10][1] = (-x, -y, z)
        verts[11][0] = (-x, y, -z)
        verts[11][1] = (-x, -y, -z)

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class WireframeBBoxGeometry(LineGeometry):
    """线框包围盒几何体：根据 2x3 的最小/最大坐标矩阵（bbox）生成对应的线框长方体"""

    def __init__(self, bbox, pose=None, color=None):
        if bbox.shape != (2, 3):
            raise ValueError('Expected bbox to be a matrix of 2 by 3!')

        if color is None:
            color = (1, 0, 0)

        num_lines = 12

        min_x, min_y, min_z = bbox[0]
        max_x, max_y, max_z = bbox[1]

        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        # 顶面
        verts[0][0] = (max_x, max_y, max_z)
        verts[0][1] = (max_x, max_y, min_z)
        verts[1][0] = (min_x, max_y, max_z)
        verts[1][1] = (min_x, max_y, min_z)
        verts[2][0] = (max_x, max_y, max_z)
        verts[2][1] = (min_x, max_y, max_z)
        verts[3][0] = (max_x, max_y, min_z)
        verts[3][1] = (min_x, max_y, min_z)

        # 底面
        verts[4][0] = (max_x, min_y, max_z)
        verts[4][1] = (max_x, min_y, min_z)
        verts[5][0] = (min_x, min_y, max_z)
        verts[5][1] = (min_x, min_y, min_z)
        verts[6][0] = (max_x, min_y, max_z)
        verts[6][1] = (min_x, min_y, max_z)
        verts[7][0] = (max_x, min_y, min_z)
        verts[7][1] = (min_x, min_y, min_z)

        # 竖直棱
        verts[8][0] = (max_x, max_y, max_z)
        verts[8][1] = (max_x, min_y, max_z)
        verts[9][0] = (max_x, max_y, min_z)
        verts[9][1] = (max_x, min_y, min_z)
        verts[10][0] = (min_x, max_y, max_z)
        verts[10][1] = (min_x, min_y, max_z)
        verts[11][0] = (min_x, max_y, min_z)
        verts[11][1] = (min_x, min_y, min_z)

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class WireframeSphereGeometry(LineGeometry):
    """线框球体几何体：用经纬线勾勒出球面（调试时可视化球形目标/碰撞范围用）"""

    def __init__(self, radius=1.0, num_lats=8, num_lons=8, pose=None, color=None, color2=None):
        if color is None:
            color = (1, 0, 0)

        if color2 is None:
            color2 = color

        num_lines = 2 * num_lats * num_lons

        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        idx = 0

        ustep = 2 * math.pi / num_lats
        vstep = math.pi / num_lons

        u = 0.0
        for i in range(num_lats):
            v = 0.0
            for j in range(num_lons):
                x1 = radius * math.sin(v) * math.sin(u)
                y1 = radius * math.cos(v)
                z1 = radius * math.sin(v) * math.cos(u)

                x2 = radius * math.sin(v + vstep) * math.sin(u)
                y2 = radius * math.cos(v + vstep)
                z2 = radius * math.sin(v + vstep) * math.cos(u)

                x3 = radius * math.sin(v + vstep) * math.sin(u + ustep)
                y3 = radius * math.cos(v + vstep)
                z3 = radius * math.sin(v + vstep) * math.cos(u + ustep)

                verts[idx][0] = (x1, y1, z1)
                verts[idx][1] = (x2, y2, z2)
                colors[idx] = color

                idx += 1

                verts[idx][0] = (x2, y2, z2)
                verts[idx][1] = (x3, y3, z3)
                colors[idx] = color2

                idx += 1

                v += vstep
            u += ustep

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


def draw_lines(geom, gym, viewer, env, pose):
    """
    把一组线段几何体添加到查看器（viewer）中显示，用于调试可视化

    :param geom: LineGeometry 的一个实例（要绘制的几何体）。
    :param gym: Gym API 对象。
    :param viewer: GymViewer 查看器对象。
    :param env: 若不为 None，pose 位于该环境的局部坐标系中；若为 None，pose 位于全局坐标系中。
    :param pose: 待绘制几何体的姿态（位置 + 朝向）。
    """
    verts = geom.instance_verts(pose)
    gym.add_lines(viewer, env, geom.num_lines(), verts, geom.colors())


def draw_line(p1, p2, color, gym, viewer, env):
    """在两点 p1、p2 之间画一条调试线段，颜色由 color 指定"""
    verts = np.empty((1, 2), dtype=gymapi.Vec3.dtype)
    verts[0][0] = (p1.x, p1.y, p1.z)
    verts[0][1] = (p2.x, p2.y, p2.z)
    colors = np.empty(1, dtype=gymapi.Vec3.dtype)
    colors[0] = (color.x, color.y, color.z)
    gym.add_lines(viewer, env, 1, verts, colors)


def parse_device_str(device_str):
    """解析形如 'cpu'、'cuda' 或 'cuda:0' 的设备字符串，返回 (设备类型, 设备编号) 元组"""
    # 默认值
    device = 'cpu'
    device_id = 0

    if device_str == 'cpu' or device_str == 'cuda':
        device = device_str
        device_id = 0
    else:
        device_args = device_str.split(':')
        assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id

# 解析所有 Gym 示例通用的命令行参数并返回给调用者
# 注意：args.physics_engine 中保存的是所选物理引擎对应的 gymapi 常量值


def parse_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    """解析命令行参数——所有 Isaac Gym 示例程序的入口都会调用这个函数。

    通用参数包括：仿真设备（--sim_device）、张量流水线（--pipeline）、物理引擎（--flex/--physx）等；
    还可通过 custom_parameters 添加自定义参数，每个元素是含 name/type/default/action/help 键的字典。
    """
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # 默认使用 PhysX 物理引擎
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # 指定了 --nographics 就必然没有窗口，隐含 --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args

# 把 sim_cfg 中提供的仿真选项解析填充到 gym api 的 sim_options 对象中

# 注意：此函数已弃用，未来版本将不再支持。


def parse_sim_config(sim_cfg: dict, sim_options: gymapi.SimParams):
    # 待办:: 增加参数值合法性检查
    opt = "dt"
    if opt in sim_cfg:
        val = float(sim_cfg[opt])
        sim_options.dt = val

    opt = "substeps"
    if opt in sim_cfg:
        val = int(sim_cfg[opt])
        sim_options.substeps = val

    opt = "up_axis"
    if opt in sim_cfg:
        val = gymapi.UpAxis(sim_cfg[opt])
        sim_options.up_axis = val

    opt = "gravity"
    if opt in sim_cfg:
        val = tuple(sim_cfg[opt])
        sim_options.gravity = gymapi.Vec3(val[0], val[1], val[2])

    opt = "use_gpu_pipeline"
    if opt in sim_cfg:
        val = bool(sim_cfg[opt])
        sim_options.use_gpu_pipeline = val

    if "flex" in sim_cfg:
        parse_flex_config(sim_cfg["flex"], sim_options)

    if "physx" in sim_cfg:
        parse_physx_config(sim_cfg["physx"], sim_options)

# 解析 FleX 物理引擎的仿真选项

# 注意：此函数已弃用，未来版本将不再支持。


def parse_flex_config(flex_cfg: dict, sim_options: gymapi.SimParams):
    ints = ["solver_type", "num_outer_iterations", "num_inner_iterations", "friction_mode"]
    floats = ["relaxation", "warm_start", "contact_regularization",
              "geometric_stiffness", "shape_collision_distance", "shape_collision_margin",
              "dynamic_friction", "static_friction", "particle_friction"]
    bools = ["deterministic_mode"]

    params = {"bool": bools, "int": ints, "float": floats}

    parse_float_int_bool(flex_cfg, sim_options.flex, params)

# 解析 PhysX 物理引擎的仿真选项

# 注意：此函数已弃用，未来版本将不再支持。


def parse_physx_config(physx_cfg: dict, sim_options: gymapi.SimParams):
    ints = ["num_threads", "solver_type", "num_position_iterations", "num_velocity_iterations", "max_gpu_contact_pairs", "num_subscenes", "contact_collection"]
    floats = ["contact_offset", "rest_offset", "bounce_threshold_velocity", "max_depenetration_velocity",
              "default_buffer_size_multiplier", "friction_correlation_distance", "friction_offset_threshold"]
    bools = ["use_gpu", "always_use_articulations"]

    params = {"bool": bools, "int": ints, "float": floats}

    parse_float_int_bool(physx_cfg, sim_options.physx, params)

# 解析 cfg 中的 float/int/bool 类型字段（按 float_int_bool 中列出的清单）并写入 opts

# 注意：此函数已弃用，未来版本将不再支持。


def parse_float_int_bool(cfg: dict, opts: object, float_int_bool: dict):
    if "float" in float_int_bool:
        for opt in float_int_bool["float"]:
            if opt in cfg:
                val = float(cfg[opt])
                setattr(opts, opt, val)

    if "int" in float_int_bool:
        for opt in float_int_bool["int"]:
            if opt in cfg:
                if opt == "contact_collection":
                    val = gymapi.ContactCollection(cfg[opt])
                else:
                    val = int(cfg[opt])
                setattr(opts, opt, val)

    if "bool" in float_int_bool:
        for opt in float_int_bool["bool"]:
            if opt in cfg:
                val = bool(cfg[opt])
                setattr(opts, opt, val)


def parse_bool(v):
    """把 bool/int/字符串等多种形式的输入统一解析成布尔值（供 argparse 使用）"""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v == 1:
            return True
        elif v == 0:
            return False
    if isinstance(v, str):
        if v.lower() in ("true", "yes", "t", "y", "1"):
            return True
        elif v.lower() in ("false", "no", "f", "n", "0"):
            return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_property_setter_map(gym):
    """返回"属性名 → 对应 Gym 设置函数"的映射表（域随机化时批量设置属性用）"""
    property_to_setters = {
        "dof_properties": gym.set_actor_dof_properties,
        "tendon_properties": gym.set_actor_tendon_properties,
        "rigid_body_properties": gym.set_actor_rigid_body_properties,
        "rigid_shape_properties": gym.set_actor_rigid_shape_properties,
        "sim_params": gym.set_sim_params,
    }

    return property_to_setters


def get_property_getter_map(gym):
    """返回"属性名 → 对应 Gym 读取函数"的映射表（域随机化时批量读取属性用）"""
    property_to_getters = {
        "dof_properties": gym.get_actor_dof_properties,
        "tendon_properties": gym.get_actor_tendon_properties,
        "rigid_body_properties": gym.get_actor_rigid_body_properties,
        "rigid_shape_properties": gym.get_actor_rigid_shape_properties,
        "sim_params": gym.get_sim_params,
    }

    return property_to_getters


def get_default_setter_args(gym):
    """返回各属性对应的设置函数所需的默认附加参数表"""
    property_to_setter_args = {
        "dof_properties": [],
        "tendon_properties": [],
        "rigid_body_properties": [True],
        "rigid_shape_properties": [],
        "sim_params": [],
    }

    return property_to_setter_args


def generate_random_samples(attr_randomization_params, shape, randomization_ct,
                            extern_sample=None):
    """按随机化配置生成一组随机采样值。

    支持 gaussian（高斯）/loguniform（对数均匀）/uniform（均匀）三种分布，
    additive（加法）与 scaling（乘法）两种作用方式，
    以及 linear（线性）/constant（阶梯）两种随训练步数逐步生效的调度策略。
    """
    rand_range = attr_randomization_params['range']
    distribution = attr_randomization_params['distribution']
    sched_type = attr_randomization_params['schedule'] if 'schedule' in attr_randomization_params else None
    sched_step = attr_randomization_params['schedule_steps'] if 'schedule' in attr_randomization_params else None
    operation = attr_randomization_params['operation']
    if sched_type == 'linear':
        sched_scaling = 1 / sched_step * min(randomization_ct, sched_step)
    elif sched_type == 'constant':
        sched_scaling = 0 if randomization_ct < sched_step else 1
    else:
        sched_scaling = 1

    if extern_sample is not None:
        sample = extern_sample
        if operation == 'additive':
            sample *= sched_scaling
        elif operation == 'scaling':
            sample = sample * sched_scaling + 1 * (1 - sched_scaling)
    elif distribution == "gaussian":
        mu, var = rand_range
        if operation == 'additive':
            mu *= sched_scaling
            var *= sched_scaling
        elif operation == 'scaling':
            var = var * sched_scaling  # 方差随时间逐步增大
            mu = mu * sched_scaling + 1 * (1 - sched_scaling)  # 对均值做线性插值
        sample = np.random.normal(mu, var, shape)
    elif distribution == "loguniform":
        lo, hi = rand_range
        if operation == 'additive':
            lo *= sched_scaling
            hi *= sched_scaling
        elif operation == 'scaling':
            lo = lo * sched_scaling + 1 * (1 - sched_scaling)
            hi = hi * sched_scaling + 1 * (1 - sched_scaling)
        sample = np.exp(np.random.uniform(np.log(lo), np.log(hi), shape))
    elif distribution == "uniform":
        lo, hi = rand_range
        if operation == 'additive':
            lo *= sched_scaling
            hi *= sched_scaling
        elif operation == 'scaling':
            lo = lo * sched_scaling + 1 * (1 - sched_scaling)
            hi = hi * sched_scaling + 1 * (1 - sched_scaling)
        sample = np.random.uniform(lo, hi, shape)
    return sample


def get_bucketed_val(new_prop_val, attr_randomization_params):
    """把随机化后的属性值归入离散的"桶"中（物理引擎的材质数量有限，用分桶限制材质种数）"""
    if attr_randomization_params['distribution'] == 'uniform':
        # 桶的取值范围由均匀分布的区间决定
        lo, hi = attr_randomization_params['range'][0], attr_randomization_params['range'][1]
    else:
        # 高斯分布时，桶的取值范围取均值 ± 2 倍标准差
        lo = attr_randomization_params['range'][0] - 2 * np.sqrt(attr_randomization_params['range'][1])
        hi = attr_randomization_params['range'][0] + 2 * np.sqrt(attr_randomization_params['range'][1])
    num_buckets = attr_randomization_params['num_buckets']
    buckets = [(hi - lo) * i / num_buckets + lo for i in range(num_buckets)]
    return buckets[bisect(buckets, new_prop_val) - 1]


def apply_random_samples(prop, og_prop, attr, attr_randomization_params,
                         randomization_ct, extern_sample=None):
    """生成随机采样值并应用到对象 prop 的 attr 属性上（在原始值 og_prop 的基础上做加法或乘法）"""
    if isinstance(prop, gymapi.SimParams):
        if attr == 'gravity':
            sample = generate_random_samples(attr_randomization_params, 3, randomization_ct)
            if attr_randomization_params['operation'] == 'scaling':
                prop.gravity.x = og_prop['gravity'].x * sample[0]
                prop.gravity.y = og_prop['gravity'].y * sample[1]
                prop.gravity.z = og_prop['gravity'].z * sample[2]
            elif attr_randomization_params['operation'] == 'additive':
                prop.gravity.x = og_prop['gravity'].x + sample[0]
                prop.gravity.y = og_prop['gravity'].y + sample[1]
                prop.gravity.z = og_prop['gravity'].z + sample[2]
    elif isinstance(prop, np.ndarray):
        sample = generate_random_samples(attr_randomization_params, prop[attr].shape,
                                         randomization_ct, extern_sample)
        if attr_randomization_params['operation'] == 'scaling':
            new_prop_val = og_prop[attr] * sample
        elif attr_randomization_params['operation'] == 'additive':
            new_prop_val = og_prop[attr] + sample

        if 'num_buckets' in attr_randomization_params and attr_randomization_params['num_buckets'] > 0:
            new_prop_val = get_bucketed_val(new_prop_val, attr_randomization_params)
        prop[attr] = new_prop_val
    else:
        sample = generate_random_samples(attr_randomization_params, 1,
                                         randomization_ct, extern_sample)
        cur_attr_val = og_prop[attr]
        if attr_randomization_params['operation'] == 'scaling':
            new_prop_val = cur_attr_val * sample
        elif attr_randomization_params['operation'] == 'additive':
            new_prop_val = cur_attr_val + sample

        if 'num_buckets' in attr_randomization_params and attr_randomization_params['num_buckets'] > 0:
            new_prop_val = get_bucketed_val(new_prop_val, attr_randomization_params)
        setattr(prop, attr, new_prop_val)


def check_buckets(gym, envs, dr_params):
    """检查域随机化配置中的材质分桶是否超限：总桶数与形状总数都不得超过 64K 上限"""
    total_num_buckets = 0
    for actor, actor_properties in dr_params["actor_params"].items():
        cur_num_buckets = 0

        if 'rigid_shape_properties' in actor_properties.keys():
            prop_attrs = actor_properties['rigid_shape_properties']
            if 'restitution' in prop_attrs and 'num_buckets' in prop_attrs['restitution']:
                cur_num_buckets = prop_attrs['restitution']['num_buckets']
            if 'friction' in prop_attrs and 'num_buckets' in prop_attrs['friction']:
                if cur_num_buckets > 0:
                    cur_num_buckets *= prop_attrs['friction']['num_buckets']
                else:
                    cur_num_buckets = prop_attrs['friction']['num_buckets']
            total_num_buckets += cur_num_buckets

    assert total_num_buckets <= 64000, 'Explicit material bucketing has been specified, but the provided total bucket count exceeds 64K: {} specified buckets'.format(
        total_num_buckets)

    shape_ct = 0

    # 单独用一层循环来统计形状数，因为不能假设每个环境里都存在所有类型的演员
    for env in envs:
        for i in range(gym.get_actor_count(env)):
            actor_handle = gym.get_actor_handle(env, i)
            actor_name = gym.get_actor_name(env, actor_handle)
            if actor_name in dr_params["actor_params"] and 'rigid_shape_properties' in dr_params["actor_params"][actor_name]:
                shape_ct += gym.get_actor_rigid_shape_count(env, actor_handle)

    assert shape_ct <= 64000 or total_num_buckets > 0, 'Explicit material bucketing is not used but the total number of shapes exceeds material limit. Please specify bucketing to limit material count.'


def _indent_xml(elem, level=0):
    """递归地为 XML 元素添加换行和缩进，让导出的 XML 文件排版美观"""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _indent_xml(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
