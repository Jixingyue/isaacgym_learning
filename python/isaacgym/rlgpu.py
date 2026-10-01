"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

# rlgpu 模块：按 Python 版本动态加载 RL-GPU 相关的 C++ 底层绑定库
# （加载逻辑与 gymapi.py 完全相同，只是模块名前缀换成 rlgpu）

import importlib
import sys
import os

from . import gymdeps


def _format_path(pathstr):
    """把路径统一格式化成"正斜杠 + 全小写"（仅 Windows 需要）。"""
    if os.name == "nt":
        # Windows 下需要把反斜杠换成正斜杠，并转成小写
        return pathstr.replace("\\", "/").lower()
    else:
        return pathstr


def _import_active_version():
    """加载与当前 Python 版本匹配的 rlgpu C++ 底层模块。

    rlgpu 底层同样是 C++ 编译的动态库（例如 Python 3.8 对应 rlgpu_38）。
    流程与 gymapi.py 相同：找到对应版本的库文件 -> 动态导入 ->
    把其中的公开符号搬进当前命名空间。
    """
    major = sys.version_info[0]
    minor = sys.version_info[1]

    # print("Running with Python %d.%d" % (major, minor))  # 调试输出：显示当前 Python 版本（已被注释掉）

    # 找到存放 C++ 原生库的目录（即 _bindings 文件夹）的绝对路径
    lib_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "_bindings"))

    if os.name == "nt":
        # Windows 系统：库放在 windows-x86_64 目录，扩展名为 .pyd
        platform = "windows-x86_64"
        ext = "pyd"
    else:
        # Linux 系统：库放在 linux-x86_64 目录，扩展名为 .so
        platform = "linux-x86_64"
        ext = "so"

    # 根据 Python 版本拼出底层模块名，例如 Python 3.8 对应 rlgpu_38
    module_name = "rlgpu_%d%d" % (major, minor)
    module_dir = os.path.join(lib_dir, platform)
    module_path = os.path.join(module_dir, "%s.%s" % (module_name, ext))
    package_path = "isaacgym._bindings.%s.%s" % (platform, module_name)

    # print(module_name)    # 调试输出：模块名（已被注释掉）
    # print(module_dir)     # 调试输出：模块所在目录（已被注释掉）
    # print(module_path)    # 调试输出：模块文件的完整路径（已被注释掉）
    # print(package_path)   # 调试输出：模块的包导入路径（已被注释掉）

    if os.path.isfile(module_path):
        print("Importing module '%s' (%s)" % (module_name, module_path))
        # 动态导入 C++ 模块，效果相当于 import isaacgym._bindings.<平台>.rlgpu_<版本>
        module = importlib.import_module(package_path)
        # https://stackoverflow.com/questions/28826127/dynamically-load-all-names-from-module-in-python
        # 把 C++ 库里的所有公开函数/类导入到当前模块，效果相当于 from ... import *
        # 如果模块定义了 __all__，就按它列出的符号导入；
        # 否则导入所有不以下划线开头的名字（下划线开头视为私有符号）。
        if hasattr(module, "__all__"):
            attrs = {key: getattr(module, key) for key in module.__all__}
        else:
            attrs = {key: value for key, value in module.__dict__.items() if key[0] != "_"}
        globals().update(attrs)
    else:
        raise RuntimeError("No rlgpu module found for the active version of Python (%d.%d)" % (major, minor))


# 本模块被导入时立即执行：加载与当前 Python 版本对应的 rlgpu 底层库
_import_active_version()
