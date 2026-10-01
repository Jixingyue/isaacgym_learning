"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


刚体物理属性示例
----------------
演示如何加载刚体、修改刚体的物理属性并对其施加各种操作。
具体通过三个场景展示以下内容:
- 加载具有不同物理属性的刚体资产(asset:模型文件加载到内存后的表示)
- 修改刚体形状属性(摩擦系数、弹性系数等)
- 修改刚体视觉属性(颜色等)
- 对刚体施加作用力
- 设置刚体线速度
"""

from isaacgym import gymutil
from isaacgym import gymapi

# 初始化 gym:获取 Gym 接口实例(一切操作的入口)
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(description="Body Physics Properties Example")

# 配置仿真参数
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.relaxation = 0.9
    sim_params.flex.dynamic_friction = 0.0
    sim_params.flex.static_friction = 0.0
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# 创建仿真器:参数依次为计算设备 ID、图形设备 ID、物理引擎类型、仿真参数
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# 创建可视化窗口(viewer),使用默认相机属性
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# 添加地面平面,静摩擦/动摩擦设为 0(让箱子能在地面上自由滑动)
plane_params = gymapi.PlaneParams()
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0

gym.add_ground(sim, plane_params)

# 设置环境(env)网格布局:每个环境是一块独立的仿真空间,env_lower/env_upper 定义环境的包围盒范围
num_envs = 3
spacing = 1.8
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 创建列表保存环境句柄和资产句柄(handle:句柄,用于后续引用某个对象的整数编号)
envs = []
box_handles = []
actor_handles = []

# 创建不同密度的方块资产(密度单位 kg/m^3,密度越大质量越大)
box_size = 0.2
box_densities = [8., 32., 1024.]
for dx in range(3):
    # AssetOptions(资产选项)用于以不同的物理属性加载资产
    # 可配置的属性包括:density(密度)、angularDamping(角阻尼)、maxAngularVelocity(最大角速度)、linearDamping(线阻尼)、maxLinearVelocity(最大线速度)等
    asset_options = gymapi.AssetOptions()
    asset_options.density = box_densities[dx]
    # 用内置 API 创建一个立方体资产(边长 box_size)
    asset_box = gym.create_box(sim, box_size, box_size, box_size, asset_options)
    box_handles.append(asset_box)

# 创建胶囊体资产
asset_options = gymapi.AssetOptions()
asset_options.density = 100.
asset_capsule = gym.create_capsule(sim, 0.2, 0.2, asset_options)

# 创建禁用重力的球资产(从 URDF 模型文件加载)
asset_root = "../../assets"
asset_file = "urdf/ball.urdf"
asset_options = gymapi.AssetOptions()
asset_options.disable_gravity = True
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset_ball = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 创建静态方块资产(fix_base_link=True 表示固定底座,不会被撞动)
asset_options.fix_base_link = True
asset_box = gym.create_box(sim, 0.5, 0.1, 0.5, asset_options)

print('Creating %d environments' % num_envs)
for i in range(num_envs):
    # 创建环境:参数为仿真器、包围盒下界、上界、每行环境数
    env = gym.create_env(sim, env_lower, env_upper, 1)
    envs.append(env)

    # 场景 1:运动的箱子撞向不同密度的目标箱(密度越大越难被撞动)
    if i == 0:
        # 把运动箱子放进环境(actor:演员,即资产放入环境后生成的实例)
        for dx in range(3):
            name = 'moving_box_{}'.format(dx)
            actor_handles.append(gym.create_actor(env, box_handles[0], gymapi.Transform(p=gymapi.Vec3(dx, 0.1, 0.)), name, i, 0))
            # 读取运动箱子的形状物理属性
            shape_props = gym.get_actor_rigid_shape_properties(env, actor_handles[-1])
            # set_actor_rigid_shape_properties 用于把形状属性写回刚体
            # 属性包括:friction(滑动摩擦)、rolling_friction(滚动摩擦)、torsion_friction(扭转摩擦)、restitution(弹性恢复系数)等
            shape_props[0].friction = 0.
            shape_props[0].rolling_friction = 0.
            shape_props[0].torsion_friction = 0.
            gym.set_actor_rigid_shape_properties(env, actor_handles[-1], shape_props)
            # 设置视觉属性:把运动箱子染成绿色
            gym.set_rigid_body_color(env, actor_handles[-1], 0, gymapi.MESH_VISUAL, gymapi.Vec3(0., 1., 0.))
            # 设置线速度:让箱子以 2 m/s 沿 Z 轴运动(撞向目标箱)
            gym.set_rigid_linear_velocity(env,
                                          gym.get_rigid_handle(env, name, gym.get_actor_rigid_body_names(env, actor_handles[0])[0]),
                                          gymapi.Vec3(0., 0., 2.))

        # 把目标箱子放进环境(低/中/高密度三种)
        target_box_names = ['target_box_low', 'target_box_med', 'target_box_high']
        target_box_positions = [gymapi.Vec3(0., 0.1, 2.), gymapi.Vec3(1, 0.1, 2.), gymapi.Vec3(2, 0.1, 2.)]
        target_box_color = [gymapi.Vec3(0., 1., 0.), gymapi.Vec3(2, 0.1, 2.), gymapi.Vec3(1., 0., 0.)]
        for dx in range(3):
            actor_handles.append(gym.create_actor(env, box_handles[dx], gymapi.Transform(p=target_box_positions[dx]), target_box_names[dx], i, 0))
            # 读取并设置目标箱子的形状物理属性
            shape_props = gym.get_actor_rigid_shape_properties(env, actor_handles[-1])
            # 摩擦系数设为 0
            shape_props[0].friction = 0.
            shape_props[0].rolling_friction = 0.
            shape_props[0].torsion_friction = 0.
            gym.set_actor_rigid_shape_properties(env, actor_handles[-1], shape_props)
            # 设置目标箱子的颜色
            gym.set_rigid_body_color(env, actor_handles[-1], 0, gymapi.MESH_VISUAL, target_box_color[dx])

    # 场景 2:设置弹性系数的胶囊体落到方块上,演示弹性碰撞效果
    if i == 1:
        # 添加方块 actor(底座固定的静态方块,充当承接平台)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.5, 0.0)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        box_handle = gym.create_actor(env, asset_box, pose, "actor1", i, 0)
        actor_handles.append(box_handle)

        # 设置方块的弹性属性:restitution=1 表示完全弹性碰撞(不损失能量),compliance 为柔顺度
        shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
        shape_props[0].restitution = 1
        shape_props[0].compliance = 0.5
        gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)

        # 添加胶囊体 actor(放在方块上方,让它自由下落)
        pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
        capsule_handle = gym.create_actor(env, asset_capsule, pose, "actor2", i, 0)

        # 设置胶囊体的弹性属性,与方块相同
        shape_props = gym.get_actor_rigid_shape_properties(env, capsule_handle)
        shape_props[0].restitution = 1
        shape_props[0].compliance = 0.5
        gym.set_actor_rigid_shape_properties(env, capsule_handle, shape_props)

    # 场景 3:多个球,有的启用重力、有的禁用重力,对比观察它们的不同表现
    if i == 2:
        # 搭建球塔(把球竖直堆叠起来)
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(0, 0, 0, 1)
        num_balls = 3
        ball_radius = .2
        ball_spacing = 2.5 * ball_radius
        y = 1 * (num_balls - 1) * ball_spacing
        while num_balls > 0:
            pose.p = gymapi.Vec3(num_balls * 0.001, 1. + y, 0)
            # 创建球 actor(加载资产时禁用了重力,所以球默认悬浮)
            ball_handle = gym.create_actor(env, asset_ball, pose, None)
            color_vec = [1, .2, .2]
            if num_balls != 2:
                color_vec = [.3, .8, .3]
                # 给除中间球以外的球重新启用重力(清掉"禁用重力"标志)
                body_props = gym.get_actor_rigid_body_properties(env, ball_handle)
                for b in range(len(body_props)):
                    body_props[b].flags = gymapi.RIGID_BODY_NONE
                gym.set_actor_rigid_body_properties(env, ball_handle, body_props)

            # 设置球的颜色
            color = gymapi.Vec3(color_vec[0], color_vec[1], color_vec[2])
            gym.set_rigid_body_color(env, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            y += ball_spacing
            num_balls -= 1

# 把观察相机对准第一个环境(参数:viewer、环境(None 表示全局坐标)、相机位置、相机看向的目标点)
cam_pos = gymapi.Vec3(6, 4.5, 3)
cam_target = gymapi.Vec3(-0.8, 0.5, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 主循环:只要用户没有关闭 viewer 窗口就一直运行
while not gym.query_viewer_has_closed(viewer):

    # 推进一步物理仿真;fetch_results 等待并取回这一步的计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新渲染:step_graphics 把物理状态同步给图形系统,draw_viewer 绘制画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待真实时间走过一个 dt(时间步长)。
    # 这样可以让物理仿真与渲染帧率同步,按真实时间观看效果。
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
