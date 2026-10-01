"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Isaac Gym 图形渲染基础示例
--------------------------
演示 Isaac Gym 的若干图形操作,包括:
- 加载纹理(贴图)/ 从内存缓冲区创建纹理
- 把纹理应用到刚体上
- 创建传感器相机
  * 固定位置的相机
  * 挂在刚体上、跟随刚体一起运动的相机
- 获取不同类型的相机图像(RGB 彩色图、深度图等)

从 Python 写图片文件需要安装 Pillow(以前叫 PIL)库,命令:`pip install pillow`
"""


import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# 获取 Gym 接口实例(一切操作的入口)
gym = gymapi.acquire_gym()

# 解析命令行参数(headless=True 表示默认无窗口运行,--save_images 用于保存图片)
args = gymutil.parse_arguments(description="Graphics Example",
                               headless=True,
                               custom_parameters=[
                                   {"name": "--save_images", "action": "store_true", "help": "Write RGB and Depth Images To Disk"}])


if args.save_images:
    from PIL import Image as im

# 配置仿真参数
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.04
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

# 添加默认地面平面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

if not args.headless:
    # 创建可视化窗口(viewer),使用默认相机属性
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')

# 设置环境(env)网格布局:8 个环境,spacing 决定每个环境的空间大小
num_envs = 8
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 本例使用的资产(asset:模型文件加载到内存后的表示)列表:8 个相同的球
repeat_assets = 8
asset_files = []
asset_root = "../../assets"
for i in range(repeat_assets):
    asset_files.append("urdf/ball.urdf")

# 加载全部资产
assets = []
for i in range(len(asset_files)):
    asset_handle = gym.load_asset(sim, asset_root, asset_files[i])
    assets.append(asset_handle)

# 从文件加载纹理(贴图):把目录下所有 .jpg 图片注册为纹理
texture_files = os.listdir("../../assets/textures/")
texture_handles = []
for file in texture_files:
    if file.endswith(".jpg"):
        h = gym.create_texture_from_file(sim, os.path.join("../../assets/textures/", file))
        if h == gymapi.INVALID_HANDLE:
            print("Couldn't load texture %s" % file)
        else:
            texture_handles.append(h)

# 在 Python 数组里手工生成一张随机彩色噪声图和一张随机灰度噪声图,
# 再通过 create_texture_from_buffer 直接从内存缓冲区创建纹理
tex_dim = 128
noise_texture = 255.0 * np.random.rand(tex_dim, tex_dim*4)
texture_handles.append(gym.create_texture_from_buffer(sim, tex_dim, tex_dim, noise_texture.astype(np.uint8)))

grayscale_noise = np.zeros((tex_dim, tex_dim*4), dtype=np.uint8)
for i in range(tex_dim):
    offset = 0
    for j in range(tex_dim):
        v = np.random.randint(0, 255, dtype=np.uint8)
        grayscale_noise[i, offset] = v  # 红色通道
        grayscale_noise[i, offset + 1] = v  # 绿色通道
        grayscale_noise[i, offset + 2] = v  # 蓝色通道
        grayscale_noise[i, offset + 3] = 255  # 透明度(alpha)
        offset = offset + 4
texture_handles.append(gym.create_texture_from_buffer(sim, tex_dim, tex_dim, grayscale_noise))

# 创建环境
actor_handles = [[]]
envs = []

# 逐个创建环境并在其中摆放 actor
for i in range(num_envs):
    actor_handles.append([])
    env = gym.create_env(sim, env_lower, env_upper, 4)
    envs.append(env)
    num_assets = len(assets)
    for j in range(num_assets):
        # 把资产在环境内摆成网格状。第 0 个 actor 放在 6 米高处,
        # 这样输出图片里能明显看出"跟随相机"的效果(相机挂在这个球上会随之运动)
        grid = np.ceil(np.sqrt(num_assets))
        d = 2 * spacing / grid
        x = d * (0.5 + np.mod(j, grid))
        if j == 0:
            y = 6.0
        else:
            y = 0.25
        z = d * (0.5 + np.floor(j/grid))
        actor_pose = gymapi.Transform()
        actor_pose.p = gymapi.Vec3(x, y, z)
        actor_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        asset_name = "asset_%d" % j
        # create_actor:把资产放入环境生成 actor(演员),返回句柄(handle,引用编号)
        handle = gym.create_actor(env, assets[j], actor_pose, asset_name, i, 1, 0)
        actor_handles[i].append(handle)


# 在每个环境里创建 2 个相机:一个固定在环境中俯瞰原点,
# 另一个挂在第 0 个 actor 的第 0 个刚体上,会跟着该 actor 一起运动
camera_handles = [[]]
for i in range(num_envs):
    camera_handles.append([])
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 360
    camera_properties.height = 240

    # 第一个相机:固定位置和看向目标
    # 位置和目标点都使用环境自身的坐标系
    h1 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(1.5, 1, 1.5)
    camera_target = gymapi.Vec3(0, 0, 0)
    gym.set_camera_location(h1, envs[i], camera_position, camera_target)
    camera_handles[i].append(h1)

    # 第二个相机:挂到环境里第一个 actor(球)的第一个刚体上。
    # 相机偏移量相对 actor 位置;相机旋转则相对全局坐标系(不跟随 actor 的旋转)。
    # FOLLOW_TRANSFORM 表示相机同时跟随刚体的位置和姿态,
    # FOLLOW_POSITION 表示只跟随位置
    h2 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_offset = gymapi.Vec3(1, 0, -1)
    camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(135))
    actor_handle = gym.get_actor_handle(envs[i], 0)
    body_handle = gym.get_actor_rigid_body_handle(envs[i], actor_handle, 0)

    gym.attach_camera_to_body(h2, envs[i], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
    camera_handles[i].append(h2)


# 给每个 actor 贴纹理:遍历所有 actor 的所有刚体,轮流分配不同纹理。
# 因为都是球,每个 actor 只有一个刚体
textures_applied = 0
for i in range(num_envs):
    actor_count = gym.get_actor_count(envs[i])
    for j in range(actor_count):
        actor_handle = gym.get_actor_handle(envs[i], j)
        num_bodies = gym.get_actor_rigid_body_count(envs[i], actor_handle)
        for b in range(num_bodies):
            texture_index = np.mod(textures_applied, len(texture_handles))
            gym.set_rigid_body_texture(envs[i], actor_handle, b, gymapi.MESH_VISUAL_AND_COLLISION, texture_handles[texture_index])
            textures_applied = textures_applied + 1


if not os.path.exists("graphics_images"):
    os.mkdir("graphics_images")
frame_count = 0

# 主仿真循环
while True:
    # 推进一步物理仿真;fetch_results 等待并取回这一步的计算结果
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 把物理状态同步给图形系统
    gym.step_graphics(sim)

    # 渲染所有传感器相机
    gym.render_all_camera_sensors(sim)

    # 每 30 帧保存一次图片(需指定 --save_images)
    if args.save_images and np.mod(frame_count, 30) == 0:
        for i in range(num_envs):
            for j in range(2):
                # gym 自带的写图工具只推荐用于 RGB 彩色图
                rgb_filename = "graphics_images/rgb_env%d_cam%d_frame%d.png" % (i, j, frame_count)
                gym.write_camera_image_to_file(sim, envs[i], camera_handles[i][j], gymapi.IMAGE_COLOR, rgb_filename)

                # 直接取回图像数据。深度图、分割图、光流图请用这种方式获取。
                # 这里取深度图,归一化成人眼可看的亮度图,再用 Pillow 写入磁盘
                depth_image = gym.get_camera_image(sim, envs[i], camera_handles[i][j], gymapi.IMAGE_DEPTH)

                # -inf 表示该像素没有深度值(比如天空),置 0,输出为黑色
                depth_image[depth_image == -np.inf] = 0

                # 把深度截断到 10 米以内,让输出图片更适合人眼观看
                depth_image[depth_image < -10] = -10

                # 翻转方向:近处的物体显示亮,远处的物体显示暗
                normalized_depth = -255.0*(depth_image/np.min(depth_image + 1e-4))

                # 转成 Pillow 图像并保存到磁盘
                normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
                normalized_depth_image.save("graphics_images/depth_env%d_cam%d_frame%d.jpg" % (i, j, frame_count))

    if not args.headless:
        # 绘制 viewer 窗口画面
        gym.draw_viewer(viewer, sim, True)

        # 等待真实时间走过一个 dt,让 viewer 与仿真速率同步。
        # headless(无窗口)模式下不需要。
        gym.sync_frame_time(sim)

        # 检查退出条件:用户关闭了 viewer 窗口
        if gym.query_viewer_has_closed(viewer):
            break

    frame_count = frame_count + 1

print('Done')

# 清理资源:销毁 viewer 窗口和仿真器
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
