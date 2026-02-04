#!/bin/bash
# 诊断 OpenMP 库冲突

echo "==================================="
echo "OpenMP 库诊断"
echo "==================================="

CONDA_PREFIX=${CONDA_PREFIX:-"$HOME/miniconda3/envs/comfyenv"}
LIB_DIR="$CONDA_PREFIX/lib"

echo ""
echo "1. 检查 conda 环境中的 OpenMP 库:"
echo "-----------------------------------"
find "$LIB_DIR" -name "lib*gomp*" -o -name "lib*iomp*" -o -name "lib*omp*" 2>/dev/null | while read f; do
    if [ -f "$f" ]; then
        echo "  $(basename $f)"
        ls -lh "$f" | awk '{print "    大小: " $5}'
        readlink -f "$f" | awk '{print "    路径: " $1}'
    fi
done

echo ""
echo "2. 检查哪些库依赖 OpenMP:"
echo "-----------------------------------"
echo "PyTorch:"
ldd "$LIB_DIR/python3.11/site-packages/torch/lib/libtorch_cpu.so" 2>/dev/null | grep -i omp || echo "  未找到"

echo ""
echo "NumPy:"
np_path=$(python -c "import numpy; import os; print(os.path.dirname(numpy.__file__))" 2>/dev/null)
if [ -n "$np_path" ]; then
    ldd "$np_path"/core/*.so 2>/dev/null | grep -i omp | head -3 || echo "  未找到"
fi

echo ""
echo "3. 建议的解决方案:"
echo "-----------------------------------"
if find "$LIB_DIR" -name "*iomp*" | grep -q .; then
    echo "  ⚠️  检测到 Intel MKL (libiomp5)"
    echo "  建议删除或重命名 libiomp5，只保留 libgomp"
    echo ""
    echo "  命令:"
    echo "    mv $LIB_DIR/libiomp5.so $LIB_DIR/libiomp5.so.bak"
    echo "    mv $LIB_DIR/libiomp5.so.xxx $LIB_DIR/libiomp5.so.xxx.bak"
fi

echo ""
echo "4. 使用启动脚本 (推荐):"
echo "-----------------------------------"
echo "  ./run_comfyui.sh"
