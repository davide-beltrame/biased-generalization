#!/usr/bin/env bash
# Setup the gpu_env_dl conda environment for biased-generalization.
#
# Usage (HPC):
#   module load miniconda3
#   bash setup_env.sh --gpu          # CUDA 12.1
#
# Usage (local / CPU-only):
#   bash setup_env.sh --cpu
#
# The environment is named gpu_env_dl in both cases so that every script

set -euo pipefail

ENV_NAME="gpu_env_dl"

# ── parse args ────────────────────────────────────────────────────────────────
MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) MODE="gpu"; shift ;;
        --cpu) MODE="cpu"; shift ;;
        *) echo "Unknown option: $1"; echo "Usage: bash setup_env.sh --gpu | --cpu"; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Usage: bash setup_env.sh --gpu | --cpu"
    exit 1
fi

# ── locate repo root (where this script lives) ───────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$MODE" == "gpu" ]]; then
    ENV_FILE="$SCRIPT_DIR/environment_gpu.yml"
else
    ENV_FILE="$SCRIPT_DIR/environment_cpu.yml"
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found."
    exit 1
fi

# ── check conda ───────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. On HPC run 'module load miniconda3' first."
    exit 1
fi
eval "$(conda shell.bash hook)"
echo "conda $(conda --version) — creating $ENV_NAME ($MODE) from $ENV_FILE"

# ── remove stale env if present ───────────────────────────────────────────────
if conda env list | grep -qw "$ENV_NAME"; then
    echo "Removing existing $ENV_NAME environment..."
    conda env remove -n "$ENV_NAME" -y
fi

# ── create env ────────────────────────────────────────────────────────────────
conda env create -f "$ENV_FILE"

# ── install biased-generalization in editable mode ─────────────────────────────────────────
conda activate "$ENV_NAME"
pip install -e "$SCRIPT_DIR" --no-deps

# ── sanity check ──────────────────────────────────────────────────────────────
echo ""
echo "-- sanity check --"
python -c "
import torch, torchvision, numpy, scipy, numba, matplotlib, tqdm, natsort
print(f'torch      {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'torchvision {torchvision.__version__}')
print(f'numpy      {numpy.__version__}')
print(f'scipy      {scipy.__version__}')
print(f'numba      {numba.__version__}')
print(f'matplotlib {matplotlib.__version__}')
"
echo ""
echo "Environment $ENV_NAME is ready."
echo "Activate with:  conda activate $ENV_NAME"
