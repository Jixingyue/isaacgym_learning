"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

演员缩放（Actor Scaling）
------------
- 加载若干 MJCF 和 URDF 资产（asset：从模型文件加载的机器人/物体模板），
  并用运行时缩放 API 对它们进行缩放
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil


# 资产描述小类：记录模型文件路径，以及是否翻转视觉附件（导入 MJCF 模型时的常见选项）
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf", False),
    AssetDesc("urdf/franka_description/robots/franka_panda.urdf", True),
    AssetDesc("urdf/kinova_description/urdf/kinova.urdf", False),
    AssetDesc("urdf/anymal_b_simple_description/urdf/anymal.urdf", True),
    AssetDesc("urdf/kuka_allegro_description/kuka_allegro.urdf", False),
    AssetDesc("mjcf/open_ai_assets/hand/shadow_hand.xml", False),
    AssetDesc("urdf/objects/cube_multicolor.urdf", True),
]

args = gymutil.parse_arguments(
    description="Actor scaling. Demonstrates runtime scaling of actors",
    custom_parameters=[
        {"name": "--min_scale", "type": float, "default": 0.5, "help": "Lower scale value"},
        {"name": "--max_scale", "type": float, "default": 2.0, "help": "Upper scale value"},
        {"name": "--num_columns", "type": int, "default": 4, "help": "Number of actors from the same asset in one row"}
    ]
)

# 获取 Gym 接口实例（一切操作的入口）
gym = gymapi.acquire_gym()

# 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# 创建仿真器：指定计算设备、渲染设备、物理引擎和仿真参数
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# 添加地面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 创建观察者窗口，用于实时观看仿真画面
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 加载资产的根目录（模型文件的相对路径都基于它）
asset_root = "../../assets"


# 逐个加载上面列出的所有资产
assets = []
for asset_desc in asset_descriptors:
    asset_file = asset_desc.file_name

    # 配置资产加载选项
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = asset_desc.flip_visual_attachments
    asset_options.use_mesh_materials = True

    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    assets.append(gym.load_asset(sim, asset_root, asset_file, asset_options))

# 设置环境网格布局（每个环境占的空间范围）
spacing = 1
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 设置观察者相机的位置和看向的目标点
cam_pos = gymapi.Vec3(17.2, 2.0, 16)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 缓存有用的句柄（handle：后续操作这个环境/演员的"凭证"）
envs = []
actor_handles = []

# 计算每行要用的缩放比例：从 min_scale 到 max_scale 均匀取 num_columns 个值
scale_range = args.max_scale - args.min_scale
if args.num_columns == 1:
    scales = [args.min_scale]
else:
    scales = [args.min_scale + scale_range * c / (args.num_columns - 1) for c in range(args.num_columns)]

# 每种资产按不同缩放比例各生成一个演员，方便直观对比大小
for i, asset in enumerate(assets):
    for scale in scales:
        # 创建环境（env：一块独立的仿真空间）
        env = gym.create_env(sim, env_lower, env_upper, args.num_columns)
        envs.append(env)

        # 把演员（actor：放进环境里的具体物体实例）加入环境
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 2, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
        actor_handles.append(actor_handle)

        # 运行时缩放：直接设置该演员的整体大小
        gym.set_actor_scale(env, actor_handle, scale)

# 主循环：只要观察者窗口没被关闭就一直运行
while not gym.query_viewer_has_closed(viewer):
    # 推进物理仿真一步
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新观察者窗口的画面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 等待真实时间走完一个 dt。
    # 这样可以让物理仿真与渲染帧率同步（避免跑得过快）。
    gym.sync_frame_time(sim)

print("Done")

# 销毁观察者窗口
gym.destroy_viewer(viewer)
# 销毁仿真器，释放资源
gym.destroy_sim(sim)
