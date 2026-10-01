"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


凸分解(Convex Decomposition)示例
----------------------------------
演示凸分解功能:把复杂的非凸网格(mesh)拆分成多个凸包(convex hull),
这样物理引擎就能快速进行碰撞检测。
本例对比了 4 种 YCB 物体分别使用默认参数凸分解、自定义参数凸分解、
以及不使用凸分解时的碰撞体效果。
"""

from isaacgym import gymapi
from isaacgym import gymutil

# 初始化 gym:获取 Gym 接口实例(一切操作的入口)
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(description="Convex decomposition example")

# 配置仿真参数
sim_params = gymapi.SimParams()

# 设置坐标轴:Z 轴向上(机器人仿真常用约定);重力沿 -Z 方向
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
# 每个物理步再细分为 2 个子步;dt 为步长(1/60 秒,即 60Hz)
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

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
    raise Exception("Failed to create sim")

# 创建可视化窗口(viewer),使用默认相机属性
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# 添加地面平面(normal 指定地面朝向的法线,即 +Z 方向朝上)
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 设置环境(env)网格布局:4 个环境、每行 2 个,spacing 决定每个环境的空间大小
num_envs = 4
envs_per_row = 2
spacing = 0.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# actor 的初始根部位姿(root pose):p 是位置,r 是旋转四元数
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.2)

asset_root = "../../assets"
asset_options = gymapi.AssetOptions()

# 从网格加载材质(让渲染效果使用模型自带的贴图/材质)
asset_options.use_mesh_materials = True
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

# 覆盖 YCB 资产文件里不准确的惯性张量和质心属性。
# 打开这两个开关后,惯量等属性会根据几何形状自动重新计算。
asset_options.override_inertia = True
asset_options.override_com = True

# 使用默认参数的凸分解(VHACD 算法):香蕉这种细长弯曲的物体,
# 不拆分成多个凸包就无法准确参与碰撞
asset_options.vhacd_enabled = True
asset0 = gym.load_asset(sim, asset_root, "urdf/ycb/011_banana/011_banana.urdf", asset_options)

# 使用自定义参数的凸分解:resolution(体素分辨率)越大,凸包越精细
asset_options.vhacd_enabled = True
asset_options.vhacd_params = gymapi.VhacdParams()
asset_options.vhacd_params.resolution = 300000
asset1 = gym.load_asset(sim, asset_root, "urdf/ycb/025_mug/025_mug.urdf", asset_options)

# 不使用凸分解:海绵砖形状简单,单个凸包就够用了
asset_options.vhacd_enabled = False
asset2 = gym.load_asset(sim, asset_root, "urdf/ycb/061_foam_brick/061_foam_brick.urdf", asset_options)

# 使用自定义参数的凸分解:罐头盒带凹槽,需要更精细的分辨率
asset_options.vhacd_enabled = True
asset_options.vhacd_params = gymapi.VhacdParams()
asset_options.vhacd_params.resolution = 500000
asset3 = gym.load_asset(sim, asset_root, "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf", asset_options)

# 创建环境并把 4 个资产分别放进 4 个环境
# create_actor 参数:环境、资产、初始位姿、actor 名称、碰撞组、碰撞过滤位掩码
# 同一碰撞组内的物体互相不碰撞;掩码用于更细粒度地控制哪些物体之间可以碰撞
env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
actor = gym.create_actor(env, asset0, initial_pose, 'actor', 0, 1)
env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
actor = gym.create_actor(env, asset1, initial_pose, 'actor', 1, 1)
env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
actor = gym.create_actor(env, asset2, initial_pose, 'actor', 2, 1)
env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
actor = gym.create_actor(env, asset3, initial_pose, 'actor', 3, 1)

# 设置观察相机的位置和看向的目标点
cam_pos = gymapi.Vec3(3, 0, 3)
cam_target = gymapi.Vec3(0, 0, 0)
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
