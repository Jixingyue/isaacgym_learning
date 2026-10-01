#!/bin/bash

# 脚本所在目录（即项目根目录）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 环境名称，需要与 YAML 文件里定义的环境名保持一致
ENV_NAME=rlgpu

pushd "${ROOT_DIR}/python"

    # 初始化 conda：找到 conda 的安装位置并加载其 shell 函数
    CONDA_DIR="$(conda info --base)"
    source "${CONDA_DIR}/etc/profile.d/conda.sh"

    # 如果目标环境当前正处于激活状态，先退出它（避免删除正在使用的环境）
    ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
    if [ "${ENV_NAME}" = "${ACTIVE_ENV_NAME}" ]; then
        conda deactivate
    fi

    # 注意！！！这会删除已存在的同名环境（原有环境将被完全移除）
    conda remove -y -n "${ENV_NAME}" --all

    # 根据 YAML 配置文件创建新环境
    conda env create -f ./rlgpu_conda_env.yml
    if [ $? -ne 0 ]; then
        echo "*** Failed to create env"
        exit 1
    fi

    # 激活新创建的环境
    conda activate "${ENV_NAME}"
    if [ $? -ne 0 ]; then
        echo "*** Failed to activate env"
        exit 1
    fi

    # 再次确认当前激活的确实是目标环境（防止激活失败却继续往下执行）
    ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
    if [ "${ENV_NAME}" != "${ACTIVE_ENV_NAME}" ]; then
        echo "*** Env is not active, aborting"
        exit 1
    fi

    # 安装 isaacgym 包（-e 为可编辑模式：源码改动即时生效，无需重复安装）
    pip install -e .

popd

echo "SUCCESS"
