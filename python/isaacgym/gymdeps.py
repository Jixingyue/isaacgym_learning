"""Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


导入公共依赖库。该模块仅供内部使用，每个公开模块都应先导入它。"""

import ctypes
import os
import sys


def _import_deps():
    """预加载运行 Isaac Gym 所需的底层动态库。

    必须在加载 Gym 主模块之前，先把 CUDA、PhysX（NVIDIA 物理引擎）、
    USD（Universal Scene Description，通用场景描述格式）等相关库按
    正确顺序加载进进程，否则后续导入 C++ 绑定模块时可能因依赖缺失
    或加载顺序错误而失败。
    """

    # 确保 torch 没有在 isaacgym 之前被导入（导入顺序不对会导致底层库冲突）
    if "torch" in sys.modules:
        raise ImportError("PyTorch was imported before isaacgym modules.  Please import torch after isaacgym modules.")

    # 找到存放 C++ 原生库的目录（即 _bindings 文件夹）的绝对路径
    lib_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "_bindings"))

    if os.name == "nt":
        platform = "windows-x86_64"
    else:
        platform = "linux-x86_64"

    lib_dir = os.path.join(lib_dir, platform)

    # 预加载一些对加载顺序敏感的动态库（必须手动按正确顺序提前加载）
    if os.name == "nt":
        try:
            # PhysX 物理引擎库
            ctypes.WinDLL(os.path.join(lib_dir, "PhysXDevice64.dll"))
            ctypes.WinDLL(os.path.join(lib_dir, "PhysXGpu_64.dll"))
        except OSError:
            print("*** Warning: failed to preload PhysX libs")
    else:
        try:
            # 先加载 CUDA 库，再加载 PhysX（顺序很重要）
            ctypes.CDLL("libcuda.so", ctypes.RTLD_GLOBAL)
        except OSError:
            print("*** Warning: failed to preload CUDA lib")
        try:
            # PhysX 的 GPU 加速库
            ctypes.CDLL(os.path.join(lib_dir, "libPhysXGpu_64.so"))
        except OSError:
            print("*** Warning: failed to preload PhysX libs")
        try:
            # USD 相关依赖库（boost、架构、工具等）
            ctypes.CDLL(os.path.join(lib_dir, "libboost_system.so"))
            ctypes.CDLL(os.path.join(lib_dir, "libboost_thread.so"))
            ctypes.CDLL(os.path.join(lib_dir, "libarch.so"))
            ctypes.CDLL(os.path.join(lib_dir, "libtf.so"))
            ctypes.CDLL(os.path.join(lib_dir, "libmem_filesys.so"))
        except OSError:
            print("*** Warning: failed to preload USD libs")


# 本模块被导入时立即执行：预加载所有底层依赖库
_import_deps()
