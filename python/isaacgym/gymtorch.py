"""Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Gym 张量与 PyTorch 张量的互操作模块（零拷贝共享内存）。

为什么"零拷贝"很重要：物理引擎（Gym）和策略网络（PyTorch）各自管理张量，
如果每次读写都要在 CPU 与 GPU 之间来回搬运数据，会成为 RL 训练的性能瓶颈；
本模块让两侧直接共享同一块内存，一方写入、另一方立即可见，无需任何拷贝，
这是 GPU 流水线（gpu pipeline）训练性能的关键。"""

from . import gymapi

import torch


def _import_gymtorch():
    """模块加载时即时编译（JIT）C++ 扩展 gymtorch，并把其中的符号全部导入本模块的全局命名空间"""
    import os
    import importlib
    import torch.utils.cpp_extension

    global torch
    ver = torch.__version__.split(sep='.')
    ver_major = int(ver[0])
    ver_minor = int(ver[1])

    print("PyTorch version", torch.__version__)

    import torch.cuda
    print("Device count", torch.cuda.device_count())

    thisdir = os.path.dirname(__file__)
    srcdir = os.path.join(thisdir, "_bindings/src/gymtorch")
    print(srcdir)

    sources = [
        os.path.join(srcdir, "gymtorch.cpp")
    ]

    if os.name == "nt":
        cflags = ["/DTORCH_MAJOR=%d" % ver_major, "/DTORCH_MINOR=%d" % ver_minor]
    else:
        cflags = ["-DTORCH_MAJOR=%d" % ver_major, "-DTORCH_MINOR=%d" % ver_minor]

    gt = torch.utils.cpp_extension.load(name="gymtorch", sources=sources, extra_cflags=cflags, verbose=True)
    # print(gt)

    # 把 C++ 扩展模块里的所有符号导入到当前模块
    if hasattr(gt, "__all__"):
        attrs = {key: getattr(gt, key) for key in gt.__all__}
    else:
        attrs = {key: value for key, value in gt.__dict__.items() if key[0] != "_"}
    globals().update(attrs)


def _create_context(device='cuda:0'):
    """强制 PyTorch 在指定设备上创建主 CUDA 上下文（确保 Gym 与 PyTorch 共用同一个 CUDA 上下文）"""
    torch.zeros([1], device=device)


def wrap_tensor(gym_tensor, offsets=None, counts=None):
    """把 Gym 张量包装成 PyTorch 张量——零拷贝，两者共享同一块底层内存。

    物理引擎写入后 PyTorch 侧立即可读（反之亦然），避免 CPU-GPU 之间的数据来回拷贝，
    这是 RL 训练（GPU 流水线）性能的关键。
    offsets/counts 可选，用于只包装张量的某个子区域（按各维偏移和长度取子块）。
    """
    data = gym_tensor.data_ptr
    device = gym_tensor.device
    dtype = int(gym_tensor.dtype)
    shape = gym_tensor.shape
    if offsets is None:
        offsets = tuple([0] * len(shape))
    if counts is None:
        counts = shape
    return wrap_tensor_impl(data, device, dtype, shape, offsets, counts)


def _torch2gym_dtype(torch_dtype):
    """把 PyTorch 数据类型常量转换成对应的 Gym 数据类型常量"""
    if torch_dtype == torch.float32:
        return gymapi.DTYPE_FLOAT32
    elif torch_dtype == torch.uint8:
        return gymapi.DTYPE_UINT8
    elif torch_dtype == torch.int16:
        return gymapi.DTYPE_INT16
    elif torch_dtype == torch.int32:
        return gymapi.DTYPE_UINT32
    elif torch_dtype == torch.int64:
        return gymapi.DTYPE_UINT64
    else:
        raise Exception("Unsupported Gym tensor dtype")


def _torch2gym_device(torch_device):
    """把 PyTorch 设备转换成 Gym 的设备表示（CPU 记为 -1，GPU 记为设备编号）"""
    if torch_device.type == "cpu":
        return -1
    elif torch_device.type == "cuda":
        return torch_device.index
    else:
        raise Exception("Unsupported Gym tensor device")


def unwrap_tensor(torch_tensor):
    """wrap_tensor 的反向操作：把 PyTorch 张量解包成 Gym 张量描述（gymapi.Tensor）。

    同样是零拷贝——只记录数据地址、设备、类型、形状等元信息，
    Gym 侧据此直接访问 PyTorch 张量的底层内存（要求张量必须连续存储）。
    """
    if not torch_tensor.is_contiguous():
        raise Exception("Input tensor must be contiguous")
    gym_tensor = gymapi.Tensor()
    gym_tensor.device = _torch2gym_device(torch_tensor.device)
    gym_tensor.dtype = _torch2gym_dtype(torch_tensor.dtype)
    gym_tensor.shape = list(torch_tensor.shape)
    gym_tensor.data_address = torch_tensor.data_ptr()
    gym_tensor.own_data = False
    return gym_tensor


# 模块被 import 时立即编译并加载 gymtorch C++ 扩展
_import_gymtorch()
