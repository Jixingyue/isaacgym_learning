"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Gym 绑定库的包装模块（负责按 Python 版本动态加载 C++ 编译的底层库 _bindings）
"""

from __future__ import print_function, division, absolute_import

import importlib
import json
import sys
import os

from . import gymdeps


def _format_path(pathstr):
    """把路径统一格式化成"正斜杠 + 全小写"（仅 Windows 需要）。

    Carbonite（NVIDIA 底层插件框架）等组件要求统一的路径格式，
    Windows 默认的反斜杠和大小写不一致会导致插件查找失败，所以先做标准化。
    """
    if os.name == "nt":
        # Windows 下需要把反斜杠换成正斜杠，并转成小写
        return pathstr.replace("\\", "/").lower()
    else:
        return pathstr


def _import_active_version():
    """加载与当前 Python 版本匹配的 C++ 底层模块。

    Gym API 的核心是用 C++ 编译的动态库，每个 Python 版本各有一份
    编译产物（例如 Python 3.8 对应 gym_38）。本函数流程：
    1. 根据操作系统和 Python 版本拼出底层库文件名及其完整路径；
    2. 动态导入该模块，并把其中所有公开符号搬进当前命名空间；
    3. 初始化 Carbonite（NVIDIA 的底层插件框架）；
    4. 设置 USD（Universal Scene Description，通用场景描述格式）插件的路径。
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

    # 根据 Python 版本拼出底层模块名，例如 Python 3.8 对应 gym_38
    module_name = "gym_%d%d" % (major, minor)
    module_dir = os.path.join(lib_dir, platform)
    module_path = os.path.join(module_dir, "%s.%s" % (module_name, ext))
    package_path = "isaacgym._bindings.%s.%s" % (platform, module_name)

    # print(module_name)    # 调试输出：模块名（已被注释掉）
    # print(module_dir)     # 调试输出：模块所在目录（已被注释掉）
    # print(module_path)    # 调试输出：模块文件的完整路径（已被注释掉）
    # print(package_path)   # 调试输出：模块的包导入路径（已被注释掉）

    if os.path.isfile(module_path):
        # 告诉 Carbonite 框架去哪里找插件（设置应用路径环境变量）
        os.environ["CARB_APP_PATH"] = _format_path(module_dir)

        print("Importing module '%s' (%s)" % (module_name, module_path))
        # 动态导入 C++ 模块，效果相当于 import isaacgym._bindings.<平台>.gym_<版本>
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

        # 初始化 Carbonite 框架（NVIDIA 的底层插件框架，Isaac Gym 的基础）
        # 下面是被注释掉的旧版初始化写法：手动拼一份 JSON 配置再传给 carb_init
        # jcfg = {
        #    "log": {
        #        "level": "info"
        #    }
        # }
        # jcfg["pluginSearchPaths"] = [ _format_path(module_dir) ]
        # cfg_str = json.dumps(jcfg, indent=2)
        # print("Config:\n%s" % cfg_str)
        # carb_init(cfg_str)

        # 改用共享的 carb.config.json 配置文件来初始化 Carbonite
        carb_init()

        # 设置 USD 插件描述文件（plugInfo.json）的自定义路径
        usd_plug_info_path = os.path.join(lib_dir, platform, "usd", "plugInfo.json")
        if os.path.isfile(usd_plug_info_path):
            # 找到了就把路径写入环境变量，供 USD 运行时读取
            os.environ["GYM_USD_PLUG_INFO_PATH"] = usd_plug_info_path
            print("Setting GYM_USD_PLUG_INFO_PATH to %s" % usd_plug_info_path)
        else:
            print("Warning: Failed to find USD plugInfo file (%s)" % usd_plug_info_path)

        # 为特定 Python 版本（3.6）追加 USD 及相关绑定库的模块搜索路径
        if major == 3 and minor == 6:
            sys.path.append(os.path.join(lib_dir, platform, "py36"))

    else:
        raise RuntimeError("No gym module found for the active version of Python (%d.%d)" % (major, minor))


# 本模块被导入时立即执行：加载与当前 Python 版本对应的底层库
_import_active_version()
