#!/bin/bash
# One-time HPC environment setup for AgriPay inference
# Run this interactively on the HPC (NOT via SLURM)
#
# Usage: bash setup_hpc_env.sh

set -e

echo "=== AgriPay HPC Environment Setup ==="

# Ensure module system is available (needed for non-interactive SSH)
if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi

# Load Miniforge module
module purge
module load Miniforge/25.3.1
source /opt/miniforge3/etc/profile.d/conda.sh

# Deactivate any active environment
conda deactivate 2>/dev/null || true

# Create the environment
echo "Creating conda environment 'agripay'..."
conda create -n agripay python=3.10 \
  pytorch torchvision pytorch-cuda=12.1 \
  -c pytorch -c nvidia -y

# Activate and install additional deps
conda activate agripay
echo "Installing additional Python packages..."
pip install -r requirements_hpc.txt

# Verify
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'PyTorch CUDA runtime: {torch.version.cuda}')"
python -c "import rasterio; print(f'Rasterio: {rasterio.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo "To activate: conda activate agripay"
echo "Do not run torch.cuda checks interactively on rustam2."
echo "Validate GPU access through SLURM instead: sbatch ~/agripay/inference/run_gpu_smoke_test.sh"

conda deactivate
