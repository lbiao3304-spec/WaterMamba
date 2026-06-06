#!/bin/bash
set -e
ENV_NAME="vmamba"
PROJECT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
echo "=== WaterMamba Env Setup ==="
echo "Project: $PROJECT_DIR"
echo "Host: $(hostname)"
nvidia-smi --query-gpu-name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first."
    exit 1
fi

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Env $ENV_NAME exists, skip creation."
else
    echo "Creating conda env $ENV_NAME ..."
    conda create -n $ENV_NAME python=3.10 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

echo "Installing dependencies..."
pip install -r "$PROJECT_DIR/deploy/requirements_server.txt"

echo "Verifying..."
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, 'GPUs', torch.cuda.device_count())"
python -c "from basicsr.models.archs.WaterMamba_arch import WaterMamba; print('WaterMamba import OK')"

echo ""
echo "=== Setup Complete ==="
echo "Run: cd $PROJECT_DIR && bash deploy/run.sh WaterMamba_q1.yml"
