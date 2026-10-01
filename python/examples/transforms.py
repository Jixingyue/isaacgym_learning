"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Transform（坐标变换）可视化
--------------------
计算并可视化柜子上不同位置的相对变换
- 在柜子的不同刚体上施加局部偏移，计算抓取点处的变换
- 用线框坐标轴/线框球体把这些变换可视化出来

背景知识：Transform = 位置 p（Vec3）+ 旋转 r（四元数 Quat），描述一个刚体的位姿，
也就是"坐标系"（原点 + 三个轴的方向）。机器人学中经常用到它的两种运算：
- 变换的乘法（T1 * T2）= 两个坐标系变换的复合（先做 T2 再做 T1）
- 变换求逆 inv()：例如"物体在机械臂末端坐标系下的位姿" = 末端位姿的逆 × 物体世界位姿
- transform_point(v)：把刚体局部坐标系下的点 v 变换到世界坐标系（p_world = R·v + t）
"""

import os
import math
from isaacgym import gymapi
from isaacgym import gymutil

# 初始化 gym：获取 Gym 接口实例（对 Isaac Gym 一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(description="Visualize Transforms")

# 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.5
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

# 加载资产（asset：柜子的模型描述，路径相对于当前工作目录）
print("Working directory: %s" % os.getcwd())
asset_root = "../../assets"
asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
# 固定柜子的底座，使柜身不会移动
asset_options.fix_base_link = True
asset_options.use_mesh_materials = True
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 设置环境（env）网格布局：共 2 个环境（每个放一个柜子），间距 1 米
num_envs = 2
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)


# 柜子辅助类：根据刚体状态计算抽屉/柜门各个抓取点处的变换
class Cabinet:

    # 各个把手在对应刚体"局部坐标系"下的偏移量（即抓取点相对把手原点的位置）
    DRAWER_GRASP = gymapi.Vec3(0.3, 0.0, 0.01)
    LEFT_DOOR_GRASP = gymapi.Vec3(0.03, 0.35, 0.185)
    RIGHT_DOOR_GRASP = gymapi.Vec3(0.03, -0.35, 0.185)

    # 抽屉/柜门把手所对应的刚体在 actor 刚体数组中的索引
    TOP_DRAWER_INDEX = 8
    BOTTOM_DRAWER_INDEX = 6
    RIGHT_DOOR_INDEX = 4
    LEFT_DOOR_INDEX = 2

    def __init__(self, env, actor):
        self.env = env
        self.actor = actor

    def get_grasp_points(self):

        # 一次性读取 actor 所有刚体的位姿（pose = 位置 + 旋转）
        poses = gym.get_actor_rigid_body_states(self.env, self.actor, gymapi.STATE_POS)['pose']

        # 取出各个把手刚体的位姿（世界坐标系下）
        top_drawer_handle_pose = gymapi.Transform.from_buffer(poses[self.TOP_DRAWER_INDEX])
        bottom_drawer_handle_pose = gymapi.Transform.from_buffer(poses[self.BOTTOM_DRAWER_INDEX])
        left_door_handle_pose = gymapi.Transform.from_buffer(poses[self.LEFT_DOOR_INDEX])
        right_door_handle_pose = gymapi.Transform.from_buffer(poses[self.RIGHT_DOOR_INDEX])

        # 在把手位姿上施加局部偏移，算出各抓取点在世界坐标系下的位置
        # transform_point(v) 的数学含义：p_world = R·v + t（先旋转再平移）
        # 相当于复合变换 T_handle * T_offset 作用在原点上
        top_drawer_point = top_drawer_handle_pose.transform_point(self.DRAWER_GRASP)
        bottom_drawer_point = bottom_drawer_handle_pose.transform_point(self.DRAWER_GRASP)
        left_door_handle_point = left_door_handle_pose.transform_point(self.LEFT_DOOR_GRASP)
        right_door_handle_point = right_door_handle_pose.transform_point(self.RIGHT_DOOR_GRASP)

        # 用抓取点位置 + 把手旋转，构造完整的抓取变换（位置和姿态）
        top_drawer_grasp = gymapi.Transform(top_drawer_point, top_drawer_handle_pose.r)
        bottom_drawer_grasp = gymapi.Transform(bottom_drawer_point, bottom_drawer_handle_pose.r)
        left_door_handle_grasp = gymapi.Transform(left_door_handle_point, left_door_handle_pose.r)
        right_door_handle_grasp = gymapi.Transform(right_door_handle_point, right_door_handle_pose.r)

        return top_drawer_grasp, bottom_drawer_grasp, left_door_handle_grasp, right_door_handle_grasp


cabinets = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # 创建环境（env：一块独立的仿真空间）
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # 把柜子模型放进环境：初始位姿为沿 Y 偏移 0.41 米、绕 X 轴转 -90°
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.41, 0.0)
    pose.r = gymapi.Quat.from_euler_zyx(-0.5 * math.pi, 0, 0)
    ahandle = gym.create_actor(env, asset, pose, "cabinet", i, 1)

    # 把 env / actor 保存到 Cabinet 对象里，稍后引用
    cab = Cabinet(env, ahandle)
    dof_props = gym.get_actor_dof_properties(env, ahandle)

    # 设置关节驱动的刚度、阻尼，并切换为位置控制模式（让抽屉/柜门保持在初始位置）
    dof_props['stiffness'].fill(1000000.0)
    dof_props['damping'].fill(500.0)
    dof_props["driveMode"] = gymapi.DOF_MODE_POS
    gym.set_actor_dof_properties(env, ahandle, dof_props)
    cabinets.append(cab)

# 让相机看向第一个环境
cam_pos = gymapi.Vec3(3, 1.5, 3)
cam_target = gymapi.Vec3(0, 0.5, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 创建用于可视化的辅助几何体
# 线框坐标轴（用来显示变换的姿态/朝向）
axes_geom = gymutil.AxesGeometry(0.1)
# 线框球体（用来标记抓取点的位置）
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 1, 0))

while not gym.query_viewer_has_closed(viewer):

    # 推进物理仿真一步，并取回计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 每帧先清空上一帧画的线
    gym.clear_lines(viewer)

    for i in range(num_envs):
        # 取出该环境对应的柜子（每个环境只有一个柜子）
        cab = cabinets[i]

        # 获取我们想要可视化的各个变换（抓取点变换）
        top_drawer_grasp, bottom_drawer_grasp, left_door_handle_grasp, right_door_handle_grasp = cab.get_grasp_points()

        # 顶层抽屉：在抓取点处画线框坐标轴 + 线框球
        gymutil.draw_lines(axes_geom, gym, viewer, cab.env, top_drawer_grasp)
        gymutil.draw_lines(sphere_geom, gym, viewer, cab.env, top_drawer_grasp)

        # 底层抽屉
        gymutil.draw_lines(axes_geom, gym, viewer, cab.env, bottom_drawer_grasp)
        gymutil.draw_lines(sphere_geom, gym, viewer, cab.env, bottom_drawer_grasp)

        # 左柜门
        gymutil.draw_lines(axes_geom, gym, viewer, cab.env, left_door_handle_grasp)
        gymutil.draw_lines(sphere_geom, gym, viewer, cab.env, left_door_handle_grasp)

        # 右柜门
        gymutil.draw_lines(axes_geom, gym, viewer, cab.env, right_door_handle_grasp)
        gymutil.draw_lines(sphere_geom, gym, viewer, cab.env, right_door_handle_grasp)

    # 更新渲染画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待真实时间走完一个 dt，
    # 让物理仿真与渲染帧率同步（实时观看效果）
    gym.sync_frame_time(sim)

print("Exiting")

# 销毁资源：先关闭可视化窗口，再销毁仿真器
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
