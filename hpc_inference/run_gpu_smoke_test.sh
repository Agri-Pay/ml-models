#!/bin/bash
#SBATCH --job-name=agripay_gpu_smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --gres=mps:400
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

export PYTHONUNBUFFERED=1

# Initialize module system (not auto-loaded in SLURM batch jobs on this cluster)
source /etc/profile.d/modules.sh 2>/dev/null || true

module purge
module load Miniforge/25.3.1
module load cuda/13.0
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate agripay

cd ~/agripay/inference
python gpu_smoke_test.py

conda deactivate
