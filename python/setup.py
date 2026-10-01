"""isaacgym 的安装配置脚本（执行 pip install / python setup.py 时读取此文件）"""

import sys
import os

from setuptools import setup, find_packages

def collect_files(target_dir):
    """递归收集 target_dir 目录下所有文件的相对路径，返回路径列表。"""
    file_list = []
    for (root, dirs, files) in os.walk(target_dir,followlinks=True):
        for filename in files:
            file_list.append(os.path.join('..', root, filename))
    return file_list

def _do_setup():
    """执行实际的打包配置（包信息、要打包的文件、依赖列表等）。"""
    root_dir = os.path.dirname(os.path.realpath(__file__))

    # 自动发现当前目录下的所有 Python 包
    packages = find_packages(".")
    print(packages)

    #
    # TODO（待办）：想个更聪明的办法，只收集当前 Python 版本对应的绑定库（目前是把整个平台的库都打包进去）
    #

    # 收集要打包进安装包的数据文件（这里是各平台的 C++ 绑定库）
    package_files = []
    if sys.platform.startswith("win"):
        # Windows：收集 windows-x86_64 目录下的全部文件
        package_files = package_files + collect_files("isaacgym/_bindings/windows-x86_64")
    elif sys.platform.startswith("linux"):
        # Linux：收集 linux-x86_64 目录下的全部文件
        package_files = package_files + collect_files("isaacgym/_bindings/linux-x86_64")

    # setuptools 核心配置：包名、版本、描述、作者、依赖等
    setup(name='isaacgym',
          version='1.0.preview4',
          description='GPU-accelerated simulation and reinforcement learning toolkit',
          author='NVIDIA CORPORATION',
          author_email='',
          url='http://developer.nvidia.com/isaac-gym',
          license='Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.',
          packages=packages,
          package_data={
              "isaacgym": package_files
          },
          python_requires='>=3.6,<3.9',  # 仅支持 Python 3.6 ~ 3.8
          install_requires = [
              "torch>=1.8.0",       # PyTorch 深度学习框架
              "torchvision>=0.9.0", # 图像处理库（配套 torch 使用）
              "numpy>=1.16.4",      # 数值计算库
              "scipy>=1.5.0",       # 科学计算库
              "pyyaml>=5.3.1",      # YAML 配置文件解析
              "pillow",             # 图像读写库
              "imageio",            # 图像/视频输入输出库
              "ninja",              # 快速构建工具（编译 C++ 扩展用）
          ],
         )

# 直接执行安装配置
_do_setup()
