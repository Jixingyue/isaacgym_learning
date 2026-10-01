"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Soft Body（软体仿真）
---------
简单演示导入一个带软体连杆的 URDF，以及一个刚体压板机构
"""

import math
import random
from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np

# 初始化 gym：获取 Gym 接口实例（对 Isaac Gym 一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数（软体仿真只支持 FleX 物理引擎）
args = gymutil.parse_arguments(description="FEM Soft Body Example")
if args.physics_engine != gymapi.SIM_FLEX:
    print("*** Soft body example only supports FleX")
    print("*** Run example with --flex flag")
    quit()

# 固定随机种子，保证每次运行时随机化的材料参数一致
random.seed(7)

# 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 3
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.75
sim_params.flex.shape_collision_margin = 0.1

# 开启冯·米塞斯（Von-Mises）应力可视化，并设定应力映射到颜色的范围
sim_params.stress_visualization = True
sim_params.stress_visualization_min = 0.0
sim_params.stress_visualization_max = 1.e+5

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.compute_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# 添加地面平面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 创建可视化窗口（viewer）
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 加载用于创建软体的球体（icosphere）URDF 资产（asset：模型模板）
asset_root = "../../assets"
soft_asset_file = "urdf/icosphere.urdf"

soft_thickness = 0.1    # 给软体设置一定厚度很重要，可以避免相互穿透

# asset_options：加载资产时的选项（固定底座、厚度、DOF 驱动模式等）
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.thickness = soft_thickness
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

asset_soft_body_count = gym.get_asset_soft_body_count(soft_asset)
asset_soft_materials = gym.get_asset_soft_materials(soft_asset)

# 打印资产的软体材料属性
print('Soft Material Properties:')
for i in range(asset_soft_body_count):
    mat = asset_soft_materials[i]
    print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')

# 设置环境（env）网格布局：4 个环境排成 2×2，间距 3 米
num_envs = 4
spacing = 3.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 缓存一些常用句柄（handle：之后用来指代对象的整数 ID）
envs = []
soft_actors = []

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
for i in range(num_envs):

    # 创建环境（env：一块独立的仿真空间）
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 2.0, 0.0)

    # 把"软体 + 导轨"actor（演员）放进环境，初始高度 2 米
    soft_actor = gym.create_actor(env, soft_asset, pose, "soft", i, 1)
    soft_actors.append(soft_actor)

    # 在默认值附近随机化软体材料参数（杨氏模量、泊松比、阻尼），
    # 让 4 个环境里的软体软硬程度各不相同
    actor_default_soft_materials = gym.get_actor_soft_materials(env, soft_actor)
    actor_soft_materials = gym.get_actor_soft_materials(env, soft_actor)
    for j in range(asset_soft_body_count):
        # 杨氏模量（越大越硬）在默认值的 0.2~2.4 倍之间随机
        youngs = actor_soft_materials[j].youngs
        actor_soft_materials[j].youngs = random.uniform(youngs * 0.2, youngs * 2.4)

        # 泊松比在默认值的 0.8~1.2 倍之间随机
        poissons = actor_soft_materials[j].poissons
        actor_soft_materials[j].poissons = random.uniform(poissons * 0.8, poissons * 1.2)

        damping = actor_soft_materials[j].damping
        # 默认阻尼为 0，这里直接从零开始随机取值
        actor_soft_materials[j].damping = random.uniform(0.0, 0.08)**2

        gym.set_actor_soft_materials(env, soft_actor, actor_soft_materials)

    # 给导轨关节启用 PD 位置控制，这样就能通过 GUI
    # 控制压板下压（压向软体）
    gym.set_joint_target_position(env, gym.get_joint_handle(env, "soft", "rail"), 0.0)

# 把相机对准环境
cam_pos = gymapi.Vec3(-4.0, 2.8, -1.2)
cam_target = gymapi.Vec3(0.0, 1.4, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 功能开关（演示用：是否绘制软体接触点、是否计算接触压强，默认都关闭）
flag_draw_contacts = False
flag_compute_pressure = False

while not gym.query_viewer_has_closed(viewer):

    # 推进物理仿真一步，并取回计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if (flag_draw_contacts):
        # 开启时：清空旧线条，并画出所有软体接触点
        gym.clear_lines(viewer)
        for env in envs:
            gym.draw_env_soft_contacts(viewer, env, gymapi.Vec3(0.7, 0.2, 0.15), 0.0005, True, True)

    if (flag_compute_pressure):
        # 从仿真中读取四面体网格和三角形网格数据
        (tet_indices, tet_stress) = gym.get_sim_tetrahedra(sim)
        (tri_indices, tri_parents, tri_normals) = gym.get_sim_triangles(sim)

        for env in envs:

            # 获取每个软体 actor 的数据范围（起始索引 + 数量，假设每个 actor 只有一个软体）
            tet_range = gym.get_actor_tetrahedra_range(env, 0, 0)
            tri_range = gym.get_actor_triangle_range(env, 0, 0)

            for i in range(tri_range.start, tri_range.start + tri_range.count):

                # 该三角形所属的"父"四面体
                parent = tri_parents[i]
                parent_stress = tet_stress[parent]

                # 把应力张量转成 numpy 矩阵
                stress = np.matrix([(parent_stress.x.x, parent_stress.y.x, parent_stress.z.x),
                                    (parent_stress.x.y, parent_stress.y.y, parent_stress.z.y),
                                    (parent_stress.x.z, parent_stress.y.z, parent_stress.z.z)])

                normal = np.array((tri_normals[i].x, tri_normals[i].y, tri_normals[i].z))

                # 计算面力密度（法向量已归一化，应力 × 法向量 = 单位面积上的接触力）
                force = np.dot(stress, normal)

    # 更新渲染画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

    # 等待真实时间走完一个 dt，
    # 让物理仿真与渲染帧率同步（实时观看效果）
    gym.sync_frame_time(sim)

# 销毁资源：先关闭可视化窗口，再销毁仿真器
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
