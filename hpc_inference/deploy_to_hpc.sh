#!/bin/bash
# Deploy/update AgriPay inference package to HPC
# Run from the ml-models/ directory on the AgriPay server
#
# Usage: export SSHPASS='<password>' && bash hpc_inference/deploy_to_hpc.sh <hpc_user> [hpc_host]
# Requires: sshpass installed (apt install sshpass)

set -e

HPC_USER="${1:?Usage: deploy_to_hpc.sh <hpc_user> [hpc_host]}"
HPC_HOST="${2:-203.135.63.78}"
HPC_BASE="~/agripay"

# Require SSHPASS to be set so this script never blocks on interactive password input
if [ -z "${SSHPASS:-}" ]; then
    echo "ERROR: SSHPASS environment variable is not set."
    echo "Export it first:  export SSHPASS='your_password'"
    exit 1
fi

SSH="sshpass -e ssh -o StrictHostKeyChecking=no"
SCP="sshpass -e scp -o StrictHostKeyChecking=no"

echo "=== Deploying to ${HPC_USER}@${HPC_HOST}:${HPC_BASE} ==="

# Create directories on HPC
echo "Creating directories..."
$SSH ${HPC_USER}@${HPC_HOST} "mkdir -p ${HPC_BASE}/models ${HPC_BASE}/inference /scratch/${HPC_USER}/agripay"

# Upload inference package
echo "Uploading inference package..."
$SCP -r hpc_inference/* ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/inference/

# Upload model weights (if they exist locally)
if ls model_weights/checkpoint_*.pth 1>/dev/null 2>&1; then
    echo "Uploading model weights..."
    $SCP model_weights/checkpoint_counting.pth ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/models/
    $SCP model_weights/checkpoint_size_estimation.pth ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/models/
else
    echo "WARNING: No model weights found locally. Skipping weight upload."
    echo "         Place .pth files in model_weights/ and re-run to upload them."
fi

echo ""
echo "=== Deployment Complete ==="
echo "Inference package: ${HPC_HOST}:${HPC_BASE}/inference/"
echo "Model weights:     ${HPC_HOST}:${HPC_BASE}/models/"
echo ""
echo "Next steps:"
echo "  1. SSH to HPC: ssh ${HPC_USER}@${HPC_HOST}"
echo "  2. If first time: bash ${HPC_BASE}/inference/setup_hpc_env.sh"
echo "  3. Test GPU access: cd ${HPC_BASE}/inference && sbatch run_gpu_smoke_test.sh"
echo "  4. Watch queue: squeue -u ${HPC_USER}"
