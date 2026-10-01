"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


资产与环境信息
---------------------------------
演示 Gym API 在资产（asset）和环境（env）层面的"自省"能力：
- 资产一旦加载完成，就可以查询它的各种属性（刚体数、关节数等）
- 环境中的演员也可以查询，还能读取它们的实时状态
"""

import os
from isaacgym import gymapi
from isaacgym import gymutil


# 打印资产的详细信息：刚体、关节、自由度（DOF：关节的可动维度）列表
def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" %
          (num_bodies, num_joints, num_dofs))

    # 遍历所有刚体
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # 遍历所有关节
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # 遍历所有自由度（DOF）
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))


# 打印环境中某个演员的详细信息和实时状态
def print_actor_info(gym, env, actor_handle):

    name = gym.get_actor_name(env, actor_handle)

    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    body_dict = gym.get_actor_rigid_body_dict(env, actor_handle)

    joint_names = gym.get_actor_joint_names(env, actor_handle)
    joint_dict = gym.get_actor_joint_dict(env, actor_handle)

    dof_names = gym.get_actor_dof_names(env, actor_handle)
    dof_dict = gym.get_actor_dof_dict(env, actor_handle)

    print()
    print("===== Actor: %s =======================================" % name)

    print("\nBodies")
    print(body_names)
    print(body_dict)

    print("\nJoints")
    print(joint_names)
    print(joint_dict)

    print("\n Degrees Of Freedom (DOFs)")
    print(dof_names)
    print(dof_dict)
    print()

    # 获取刚体状态信息
    body_states = gym.get_actor_rigid_body_states(
        env, actor_handle, gymapi.STATE_ALL)

    # 打印部分状态切片
    print("Poses from Body State:")
    print(body_states['pose'])          # 只打印位姿

    print("\nVelocities from Body State:")
    print(body_states['vel'])          # 只打印速度
    print()

    # 遍历所有刚体，打印名字和位置
    body_positions = body_states['pose']['p']
    for i in range(len(body_names)):
        print("Body '%s' has position" % body_names[i], body_positions[i])

    print("\nDOF states:")

    # 获取自由度状态
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    # 打印状态切片
    # 打印每个自由度的全部状态（位置、速度等）
    print(dof_states)
    print()

    # 遍历所有自由度，打印名字和位置
    dof_positions = dof_states['pos']
    for i in range(len(dof_names)):
        print("DOF '%s' has position" % dof_names[i], dof_positions[i])


# 获取 Gym 接口实例（一切操作的入口）
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(description="Asset and Environment Information")

# 配置仿真参数
sim_params = gymapi.SimParams()

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# 创建仿真器：指定计算设备、渲染设备、物理引擎和仿真参数
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# 打印当前工作目录
# 有助于确认资产将从哪个相对路径加载
print("Working directory: %s" % os.getcwd())

# 资产的搜索路径（相对于当前工作目录）
asset_root = "../../assets"

# 要加载的资产列表，URDF 和 MJCF 两种格式都支持
asset_files = ["urdf/cartpole.urdf",
               "urdf/franka_description/robots/franka_panda.urdf",
               "mjcf/nv_ant.xml"]
asset_names = ["cartpole", "franka", "ant"]
loaded_assets = []

# 加载资产并确认加载成功
for asset in asset_files:
    print("Loading asset '%s' from '%s'" % (asset, asset_root))

    current_asset = gym.load_asset(sim, asset_root, asset)

    if current_asset is None:
        print("*** Failed to load asset '%s'" % (asset, asset_root))
        quit()
    loaded_assets.append(current_asset)

for i in range(len(loaded_assets)):
    print()
    print_asset_info(loaded_assets[i], asset_names[i])

# 设置环境间距
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

# 创建一个环境（env：一块独立的仿真空间）
env = gym.create_env(sim, lower, upper, 1)

# 把演员（actor：放进环境里的具体物体实例）加入环境
pose = gymapi.Transform()
for i in range(len(loaded_assets)):
    pose.p = gymapi.Vec3(0.0, 0.0, i * 2)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    gym.create_actor(env, loaded_assets[i], pose, asset_names[i], -1, -1)

print("=== Environment info: ================================================")

actor_count = gym.get_actor_count(env)
print("%d actors total" % actor_count)

# 遍历该环境中的所有演员，逐个打印信息
for i in range(actor_count):
    actor_handle = gym.get_actor_handle(env, i)
    print_actor_info(gym, env, actor_handle)

# 清理仿真器，释放资源
gym.destroy_sim(sim)
