#!/bin/bash
#SBATCH --job-name=agripay_infer_test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

# Force Python stdout to be unbuffered so progress appears in SLURM output
export PYTHONUNBUFFERED=1

# Initialize module system
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load Miniforge/25.3.1
module load cuda/13.0
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate agripay

# Working directories
TEST_DIR="/scratch/muhammadtalha/agripay/inference_test_${SLURM_JOB_ID}"
WEIGHTS_DIR="/home/muhammadtalha/agripay/models"
INFERENCE_DIR="/home/muhammadtalha/agripay/inference"

mkdir -p "${TEST_DIR}/inputs" "${TEST_DIR}/outputs"

echo "============================================"
echo " AgriPay End-to-End Inference Test"
echo " Job ID: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo " Test dir: ${TEST_DIR}"
echo "============================================"

# Step 1: Create synthetic 3-band GeoTIFF (2000x2000 random RGB, mimics drone imagery)
echo ""
echo "[1/4] Creating synthetic test image (512x512 RGB GeoTIFF — single inference tile)..."
python3 -c "
import numpy as np, rasterio, os
from rasterio.transform import from_bounds
from rasterio.crs import CRS

out = '${TEST_DIR}/inputs/test.tif'
np.random.seed(42)
# Simulate paddy field: mostly green tones to give model realistic input
data = np.zeros((3, 512, 512), dtype=np.uint8)
data[0] = np.random.randint(30, 100, (512, 512))   # R
data[1] = np.random.randint(80, 180, (512, 512))   # G (higher = greener)
data[2] = np.random.randint(20,  80, (512, 512))   # B

tf = from_bounds(73.0, 30.0, 73.01, 30.01, 512, 512)
with rasterio.open(out, 'w', driver='GTiff', height=512, width=512,
                   count=3, dtype='uint8',
                   crs=CRS.from_epsg(4326), transform=tf) as ds:
    ds.write(data)
size_kb = os.path.getsize(out) // 1024
print(f'  Created: {out} ({size_kb} KB, 512x512 = 1 inference tile)')
"

# Step 2: Write config JSON (using absolute paths, NOT tilde paths)
echo ""
echo "[2/4] Writing config.json..."
cat > "${TEST_DIR}/inputs/config.json" << CONFIGEOF
{
    "job_id": "test-${SLURM_JOB_ID}",
    "model_id": "rice_plant_counter_v1",
    "image_path": "${TEST_DIR}/inputs/test.tif",
    "weights_dir": "${WEIGHTS_DIR}",
    "output_dir": "${TEST_DIR}/outputs"
}
CONFIGEOF
echo "  Config:"
cat "${TEST_DIR}/inputs/config.json"

# Step 3: Run inference
echo ""
echo "[3/4] Running inference (run_inference.py)..."
cd "${INFERENCE_DIR}"
python run_inference.py --config "${TEST_DIR}/inputs/config.json"

# Step 4: Print results
echo ""
echo "[4/4] Results:"
echo "--- results.json ---"
cat "${TEST_DIR}/outputs/results.json" | python3 -m json.tool 2>/dev/null || cat "${TEST_DIR}/outputs/results.json"
echo ""
echo "--- status.json ---"
cat "${TEST_DIR}/outputs/status.json"
echo ""

# Summary
echo ""
echo "============================================"
echo " INFERENCE TEST COMPLETE"
echo " Output dir: ${TEST_DIR}/outputs/"
ls -lh "${TEST_DIR}/outputs/"
echo "============================================"

conda deactivate
