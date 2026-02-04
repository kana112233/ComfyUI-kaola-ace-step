#!/bin/bash
# ComfyUI 启动脚本 - 解决 OpenMP 库冲突导致的 ld.so 崩溃

# 获取 conda 环境路径
CONDA_PREFIX=${CONDA_PREFIX:-"$HOME/miniconda3/envs/comfyenv"}

# 查找 libgomp.so.1
LIBGOMP_PATH="$CONDA_PREFIX/lib/libgomp.so.1"

if [ ! -f "$LIBGOMP_PATH" ]; then
    echo "警告: 未找到 libgomp.so.1，尝试查找..."
    LIBGOMP_PATH=$(find "$CONDA_PREFIX/lib" -name "libgomp.so.*" 2>/dev/null | head -1)
fi

if [ -z "$LIBGOMP_PATH" ]; then
    echo "错误: 无法找到 libgomp，OpenMP 冲突可能导致崩溃"
    exit 1
fi

echo "使用 LD_PRELOAD: $LIBGOMP_PATH"

# 设置 LD_PRELOAD 强制所有库使用同一个 OpenMP
export LD_PRELOAD="$LIBGOMP_PATH"

# 可选：设置 Intel MKL 允许重复库
export KMP_DUPLICATE_LIB_OK=TRUE

# 切换到 ComfyUI 目录
cd ~/work/ai/ComfyUI

# 启动 ComfyUI
exec python main.py "$@"
