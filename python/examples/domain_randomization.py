"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


域随机化(Domain Randomization)示例
------------------------------------
域随机化是指在训练时随机改变仿真环境的各种外观和物理参数
(颜色、贴图、灯光、相机视角等),让强化学习策略见过足够多的"随机世界",
从而在真实世界的各种条件下也能正常工作(即模拟到现实的迁移,Sim-to-Real)。
本例演示:
- 随机化颜色
- 随机化纹理(贴图)
- 随机化灯光参数
- 随机化相机位置
"""

import os
import random
from isaacgym import gymapi
from isaacgym import gymutil

# 初始化 gym:获取 Gym 接口实例(一切操作的入口)
gym = gymapi.acquire_gym()

# 解析命令行参数(headless=True 表示默认无窗口运行,--save_images 用于保存渲染图片)
args = gymutil.parse_arguments(
    description="Domain Randomization Example",
    headless=True,
    custom_parameters=[
        {"name": "--save_images", "action": "store_true", "help": "Store Images To Disk"}])

# 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
else:
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

# 添加地面平面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# 创建可视化窗口(仅在非 headless 模式下)
if not args.headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')

# 设置环境(env)网格布局:每个环境是一块独立的仿真空间
num_envs = 1
spacing = 0.75
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 加载蚂蚁机器人资产(asset:模型文件加载到内存后的表示,这里用 MJCF 格式)
asset_root = "../../assets"
asset_file = "mjcf/nv_ant.xml"
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
ant_asset = gym.load_asset(sim, asset_root, asset_file)
if ant_asset is None:
    raise IOError("Failed to load asset")

envs = []
actor_handles = []
camera_handles = []

# 从文件加载纹理(贴图):把目录下所有 .jpg 图片都注册为可用纹理,
# 之后随机化时从中随机挑选贴到机器人身上
texture_files = os.listdir("../../assets/textures/")
loaded_texture_handle_list = []
for file in texture_files:
    if file.endswith(".jpg"):
        loaded_texture_handle_list.append(gym.create_texture_from_file(sim, os.path.join("../../assets/textures/", file)))

# 传感器相机属性(用于生成相机观测图像)
cam_pos = gymapi.Vec3(0.0, 3.0, 3.0)
cam_target = gymapi.Vec3(0.0, 0.0, -1.0)
cam_props = gymapi.CameraProperties()
cam_props.width = 360
cam_props.height = 360

# 创建环境
print('Creating %d environments' % num_envs)
for i in range(num_envs):
    # 创建环境:参数为仿真器、包围盒下界、上界、每行环境数
    env = gym.create_env(sim, env_lower, env_upper, 2)
    envs.append(env)

    # 创建蚂蚁 actor(actor:演员,即资产放入环境后生成的实例)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0.5, 0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    ahandle = gym.create_actor(env, ant_asset, pose, 'ant', i, 1)
    actor_handles.append(ahandle)

    # 配置 DOF(Degree of Freedom,自由度)属性:全部设为无驱动模式,
    # 关节可以自由摆动(不施加任何电机力)
    props = gym.get_actor_dof_properties(env, ahandle)
    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, ahandle, props)

    # 创建传感器相机并挂到蚂蚁的身体上,相机视角会跟着身体一起动
    camera_handle = gym.create_camera_sensor(env, cam_props)
    camera_handles.append(camera_handle)
    body = gym.get_actor_rigid_body_handle(env, ahandle, 0)
    gym.attach_camera_to_body(camera_handle, env, body, gymapi.Transform(p=cam_pos), gymapi.FOLLOW_TRANSFORM)
    gym.set_camera_location(camera_handle, env, cam_pos, cam_target)

# 设置 viewer 观察相机的视角
if not args.headless:
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

sequence_number = 0
# 只有当文件夹不存在时才创建(用于保存随机化后的渲染图)
if not os.path.exists('dr_output_images'):
    os.mkdir("dr_output_images")

while True:
    # 推进一步物理仿真;fetch_results 等待并取回这一步的计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 渲染传感器相机(生成相机观测图像)
    gym.render_all_camera_sensors(sim)

    # 更新图形系统的变换(把物理状态同步给图形)
    gym.step_graphics(sim)

    if not args.headless:
        # 绘制 viewer 窗口画面
        gym.draw_viewer(viewer, sim, True)

        # 等待真实时间走过一个 dt,让 viewer 与仿真速率同步。
        # headless(无窗口)模式下不需要。
        gym.sync_frame_time(sim)

        # 检查退出条件:用户关闭了 viewer 窗口
        if gym.query_viewer_has_closed(viewer):
            break

    # 每 100 帧随机化一次刚体的颜色/贴图和环境灯光 —— 这就是"域随机化"的核心:
    # RL 训练时让策略不断面对外观变化的场景,避免它只认某一种固定外观,
    # 从而把仿真里学到的能力迁移到多姿多彩的真实世界
    if gym.get_frame_count(sim) % 100 == 0:

        for i in range(num_envs):
            env = envs[i]

            # 随机化传感器相机位置(在原位置基础上加随机偏移)
            y_offset = random.uniform(-1.0, 1.0)
            z_offset = random.uniform(-1.0, 1.0)
            cam_pos_new = cam_pos + gymapi.Vec3(0., y_offset, z_offset)
            gym.set_camera_location(camera_handles[i], env, cam_pos_new, cam_target)

            # 随机化每个刚体的颜色和纹理:给蚂蚁的每一段身体随机上色、随机换贴图
            num_bodies = gym.get_actor_rigid_body_count(env, actor_handles[-1])
            for n in range(num_bodies):
                gym.set_rigid_body_color(env, actor_handles[-1], n, gymapi.MESH_VISUAL,
                                         gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                gym.set_rigid_body_texture(env, actor_handles[-1], n, gymapi.MESH_VISUAL,
                                           loaded_texture_handle_list[random.randint(0, len(loaded_texture_handle_list)-1)])

            # 随机化灯光参数:灯光颜色、环境光、光照方向
            l_color = gymapi.Vec3(random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1))
            l_ambient = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            l_direction = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            gym.set_light_parameters(sim, 0, l_color, l_ambient, l_direction)

            # 把当前 RGB 渲染图保存到磁盘
            if args.save_images:
                print("writing dr_output_images/rgb_image_%03d_%03d.png" % (sequence_number,  i))
                rgb_image_filename = "dr_output_images/rgb_image_%03d_%03d.png" % (sequence_number,  i)
                gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, rgb_image_filename)

        sequence_number = sequence_number + 1

print('Done')

if not args.headless:
    gym.destroy_viewer(viewer)

gym.destroy_sim(sim)
