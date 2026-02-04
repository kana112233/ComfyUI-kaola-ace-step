#!/bin/bash

# Installation script for ACE-Step ComfyUI Nodes

set -e

echo "=== ACE-Step 1.5 ComfyUI Nodes Installation ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Navigate to script directory
cd "$(dirname "$0")"

# Clone ACE-Step if not present
if [ ! -d "acestep_repo" ]; then
    echo ""
    echo "Cloning ACE-Step 1.5 repository..."
    git clone https://github.com/ace-step/ACE-Step-1.5.git acestep_repo
    cd acestep_repo

    echo ""
    echo "Installing ACE-Step dependencies with uv..."
    uv sync

    echo ""
    echo "Downloading ACE-Step models..."
    uv run acestep-download

    cd ..
else
    echo "ACE-Step repository already exists at acestep_repo/"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Make sure you have enough GPU VRAM (6GB+ recommended)"
echo "2. Launch ComfyUI"
echo "3. Look for 'ACE-Step' nodes in the node menu"
echo ""
echo "Note: Set your checkpoint directory in the nodes to point to:"
echo "  $(pwd)/acestep_repo/checkpoints"
