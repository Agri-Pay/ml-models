#!/bin/bash
#SBATCH --job-name=agripay_counting
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

export PYTHONUNBUFFERED=1

source /etc/profile.d/modules.sh 2>/dev/null || true

module purge
module load Miniforge/25.3.1
module load cuda/13.0
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate agripay

CONFIG="/scratch/muhammadtalha/agripay/counting_test/inputs/config.json"

echo "============================================"
echo " AgriPay Counting Test (Counting.png)"
echo " Job ID: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo " Config: ${CONFIG}"
echo "============================================"

cd ~/agripay/inference
python run_inference.py --config "${CONFIG}"

echo "Job complete at $(date)"
conda deactivate
