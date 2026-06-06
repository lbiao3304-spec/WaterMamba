#!/bin/bash
set -e
CONFIG="${1:-WaterMamba_q1.yml}"
PROJECT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"

echo "=== WaterMamba Training ==="
echo "Config: $CONFIG"
echo "Project: $PROJECT_DIR"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu-name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vmamba

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

python basicsr/train.py -opt "$CONFIG" ${@:2}

echo "=== Training finished ==="
