#!/bin/bash
# ComfyUI 启动脚本 - 解决 ld.so OpenMP 冲突崩溃

# 强制预加载 libgomp，确保所有库使用同一个 OpenMP
export LD_PRELOAD=~/miniconda3/envs/comfyenv/lib/libgomp.so.1

# 允许重复的 OpenMP 库（备用方案）
export KMP_DUPLICATE_LIB_OK=TRUE

# 启动 ComfyUI
cd ~/work/ai/ComfyUI
exec python main.py "$@"
