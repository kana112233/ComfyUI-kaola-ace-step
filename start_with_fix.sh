#!/bin/bash
# --------------------------------------------------------------------------------
# ComfyUI Startup Script with TLS Fix
#
# This script automatically finds libgomp.so.1 in your current Conda environment
# (or system path) and preloads it to prevent the "Inconsistency detected by ld.so"
# assertion error.
#
# Usage:
#   1. Copy this script to your ComfyUI root folder: cp custom_nodes/ComfyUI-kaola-ace-step/start_with_fix.sh .
#   2. Give it execution permission: chmod +x start_with_fix.sh
#   3. Run it: ./start_with_fix.sh
# --------------------------------------------------------------------------------

# 1. Determine CONDA_PREFIX (if active) or infer from python path
if [ -z "$CONDA_PREFIX" ]; then
    PYTHON_EXEC=$(which python)
    if [[ "$PYTHON_EXEC" == *"envs"* ]]; then
        # infer prefix, e.g. /path/to/envs/name/bin/python -> /path/to/envs/name
        CONDA_PREFIX=${PYTHON_EXEC%/bin/python}
    fi
fi

LIBGOMP_PATH=""

# 2. Try to find libgomp.so.1 in the conda environment
if [ ! -z "$CONDA_PREFIX" ]; then
    echo "Checking Conda environment: $CONDA_PREFIX"
    FOUND=$(find "$CONDA_PREFIX/lib" -name "libgomp.so.1" | head -n 1)
    if [ ! -z "$FOUND" ]; then
        LIBGOMP_PATH=$FOUND
    fi
fi

# 3. Fallback: try to find it in system paths if not found in conda
if [ -z "$LIBGOMP_PATH" ]; then
    echo "Checking global system paths..."
    # Common locations on Linux
    POSSIBLE_PATHS=(
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1"
        "/usr/lib64/libgomp.so.1"
        "/usr/lib/libgomp.so.1"
    )
    for p in "${POSSIBLE_PATHS[@]}"; do
        if [ -f "$p" ]; then
            LIBGOMP_PATH=$p
            break
        fi
    done
fi

# 4. Use the path if found
if [ ! -z "$LIBGOMP_PATH" ]; then
    echo "✅ Found libgomp: $LIBGOMP_PATH"
    echo "🚀 Starting ComfyUI with LD_PRELOAD fix..."
    echo "--------------------------------------------------------"
    LD_PRELOAD=$LIBGOMP_PATH python main.py "$@"
else
    echo "⚠️  WARNING: Could not find libgomp.so.1 automatically."
    echo "Trying standard start (might fail with ld.so error)..."
    python main.py "$@"
fi
