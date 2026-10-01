"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


大小质量比测试（Large mass-ratio test）
-------------------------------
测试箱子堆叠在质量差异很大时的仿真稳定性。共有两种场景可选：
- 等尺寸箱子：每往上一层的箱子，质量是它正下方箱子的 10 倍
- 倒金字塔箱子：所有箱子密度相同，但每往上一层尺寸放大 4 倍，质量随之变为 4*4*4 = 64 倍
"""

from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt

# 初始化 gym：获取 Gym 接口实例（对 Isaac Gym 一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数（--num_envs 指定环境数量，--inverted_pyramid_test 启用倒金字塔场景）
args = gymutil.parse_arguments(
    description="Large Mass Ratio Test",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 100, "help": "Number of environments to create"},
        {"name": "--inverted_pyramid_test", "action": "store_true", "help": "Run test with stack of boxes of increasing sizes but the same density"}])

inverted_pyramid_test = args.inverted_pyramid_test

# 配置仿真参数（本例需要较多的求解迭代次数，才能让大质量比堆叠保持稳定）
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.substeps = 6
    sim_params.flex.num_inner_iterations = 5
    sim_params.flex.num_inner_iterations = 24
    sim_params.flex.relaxation = 0.7
    sim_params.flex.warm_start = 0.25
    displacement = 0.02
elif args.physics_engine is gymapi.SIM_PHYSX:
    sim_params.substeps = 6
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 100
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.rest_offset = 0.001
    displacement = 0.002

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# 用默认相机属性创建可视化窗口（viewer）
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# 添加地面平面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# 设置环境（env）网格布局：每行的环境数取总数的平方根，间距 4.5 米
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
spacing = 4.5
if inverted_pyramid_test:
    spacing = 12.0

# 设置相机位置并看向第一个环境
cam_pos = gymapi.Vec3(8.0, 6.5, 3.0)
cam_target = gymapi.Vec3(10.0, 6.0, 10.0)
if not inverted_pyramid_test:
    cam_pos.y = 3.0
    cam_target.y = 1.9

gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

box_color = gymapi.Vec3(0.1, 0.9, 0.05)
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 创建列表，保存环境句柄和资产/actor 句柄（handle：之后用来指代对象的整数 ID）
envs = []
box_handles = []
box_sizes = []
actor_handles = []

# 创建不同密度的箱子资产（asset：模型模板，密度单位 kg/m^3）
box_size = 0.5
box_density = 1

# AssetOptions 用于给资产设置各种物理属性，
# 包括 density（密度）、angularDamping（角阻尼）、maxAngularVelocity（最大角速度）、
# linearDamping（线阻尼）、maxLinearVelocity（最大线速度）等
asset_options = gymapi.AssetOptions()
asset_options.thickness = 0.005

if not inverted_pyramid_test:
    # 场景一：等尺寸箱子堆叠。共 5 层，自下而上每层密度 ×10（即质量 ×10）
    stack_height = 5

    for dx in range(stack_height):
        asset_options.density = box_density
        # 创建一个正方体资产（尺寸相同，但每层密度不同 → 质量不同）
        asset_box = gym.create_box(sim, box_size, box_size, box_size, asset_options)
        box_handles.append(asset_box)
        box_density *= 10

    for i in range(num_envs):
        # 创建环境（env：一块独立的仿真空间）
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        box_color = gymapi.Vec3(0.2, 0.75, 0.18)
        color_step = box_color.y / (stack_height - 1)

        # 把箱子逐层放入环境并向上堆叠（displacement 是微小间隙，避免初始互相穿透）
        for b in range(stack_height):
            name = 'box_{}'.format(b)
            actor_handles.append(gym.create_actor(env, box_handles[b], gymapi.Transform(
                p=gymapi.Vec3(0., 0.5 * box_size + (box_size + displacement) * b + 0.001, 0.)), name, i, 0))
            gym.set_rigid_body_color(env, actor_handles[-1], 0, gymapi.MESH_VISUAL, box_color)
            box_color.x += 0.85 * color_step * b
            box_color.y -= color_step * b
            box_color.z -= 0.04 * b
else:
    # 场景二：倒金字塔堆叠。所有箱子密度相同（100 kg/m^3），每往上一层边长 ×2，质量变为 8 倍
    stack_height = 5
    scale_factor = 2
    color_step = box_color.x / (stack_height - 1)

    new_box_size = box_size
    for dx in range(stack_height):
        asset_options.density = 100
        # 逐层创建尺寸递增的箱子资产
        asset_box = gym.create_box(sim, new_box_size, new_box_size, new_box_size, asset_options)
        box_handles.append(asset_box)
        box_sizes.append(new_box_size)
        new_box_size *= scale_factor

    print('Creating %d environments' % num_envs)
    for i in range(num_envs):
        # 创建环境（env：一块独立的仿真空间）
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        start_height = 0.001

        box_color = gymapi.Vec3(0.9, 0.25, 0.15)
        color_step = box_color.x / (stack_height - 1)

        # 把箱子逐层放入环境并向上堆叠（大箱子在下、小箱子在上，形成倒金字塔）
        for b in range(stack_height):
            name = 'box_{}'.format(b)
            start_height += (0.5 * box_sizes[b] + displacement)
            actor_handles.append(gym.create_actor(env, box_handles[b], gymapi.Transform(p=gymapi.Vec3(0., start_height, 0.)), name, i, 0))

            box_color.x -= 0.9 * color_step * b
            box_color.y -= 0.02 * color_step * b
            box_color.z += 0.8 * color_step * b
            gym.set_rigid_body_color(env, actor_handles[-1], 0, gymapi.MESH_VISUAL, box_color)

            start_height += 0.5 * box_sizes[b]

while not gym.query_viewer_has_closed(viewer):

    # 推进物理仿真一步，并取回计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新渲染画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待真实时间走完一个 dt，
    # 让物理仿真与渲染帧率同步（实时观看效果）
    gym.sync_frame_time(sim)

print('Done')

# 销毁资源：先关闭可视化窗口，再销毁仿真器
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
