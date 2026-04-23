# AgriPay Rustam2 Runbook

Last updated: April 23, 2026

## Status: WORKING ✓

GPU jobs are fully functional. Job 260 confirmed:
- CUDA available: True
- GPU: Tesla V100-SXM3-32GB allocated
- Matrix multiply succeeded
- Exit code: 0

## Server Details

- Host: 203.135.63.78 (rustam2)
- User: muhammadtalha (uid=1020)
- GPUs: 2x Tesla V100-SXM3-32GB
- SLURM: 23.11.4
- NVIDIA Driver: 580.95.05, CUDA module: cuda/13.0
- Partitions: `debug` (default, AllowGroups=ALL, 7-day max), `cpuonly`
- NOTE: `hpcshort` does NOT exist on this server despite being in the manual

## Directory Layout

```
~/agripay/inference/             # Code (persistent, HOME)
~/agripay/models/                # Model weights (persistent, HOME)
/scratch/muhammadtalha/agripay/  # Job I/O (temporary, 90-day purge)
```

HOME is for small persistent files (up to ~10 GB). /scratch is for high-performance temporary job data.

## Working SLURM Script Pattern

Four issues must be addressed for GPU jobs to work on this cluster:

1. **Module system not auto-loaded in batch jobs** — must source `/etc/profile.d/modules.sh` manually
2. **GRES: use `--gres=gpu:1` only** — MPS was turned OFF by admin (April 22, 2026). Single line, no mps line
3. **No `-u` in set flags** — `set -eo pipefail` NOT `set -euo pipefail`; conda MKL activation scripts use unbound variables which `-u` treats as fatal errors
4. **Resources: 8 CPUs, 48G mem** — admin recommended increase (previously 4 CPUs / 16G)

```bash
#!/bin/bash
#SBATCH --job-name=agripay_gpu_smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

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
```

## One-Time Setup (Already Done)

Conda environment `agripay` is already created at `~/.conda/envs/agripay`:
- PyTorch 2.5.1 + CUDA 12.1
- scipy 1.15.3, rasterio 1.4.4, scikit-learn 1.7.2

To recreate from scratch (if needed):
```bash
# From local machine
export SSHPASS='Comp@2026'
sshpass -e scp -r hpc_inference/* muhammadtalha@203.135.63.78:~/agripay/inference/
sshpass -e ssh muhammadtalha@203.135.63.78 "source /etc/profile.d/modules.sh && cd ~/agripay/inference && bash setup_hpc_env.sh"
```

## Running the GPU Smoke Test

```bash
ssh muhammadtalha@203.135.63.78
cd ~/agripay/inference
sbatch run_gpu_smoke_test.sh
squeue -u muhammadtalha
# When complete (usually ~4 seconds):
cat agripay_gpu_smoke_<JOBID>.out
```

Expected output:
```
[TIMESTAMP] Starting AgriPay GPU smoke test
SLURM job id: <N>
SLURM node: rustam2
PyTorch version: 2.5.1
PyTorch CUDA runtime: 12.1
CUDA available: True
Visible GPU count: 1
Using GPU 0: Tesla V100-SXM3-32GB
Matrix multiply succeeded; checksum=<N>
SUCCESS: GPU allocation and CUDA execution work under SLURM
```

## Known HPC Behaviour Notes

- **CUDA first-run is slow**: First forward pass through VGG model takes 3-10 minutes on the V100 due to CUDA kernel compilation. Subsequent tiles are fast. Accept 15-30 min time limit for first-run jobs.
- **GPU contention under MPS**: If other users are running heavy GPU jobs, MPS shares the SM resources. Large images under high contention can be slow. The default 30-minute time limit in production should handle this.
- **COMPLETING state hang**: After SLURM kills a job (timeout or cancel), the GPU cleanup can get stuck in `COMPLETING` state for 30+ minutes. This holds the GPU resource allocation. If your job is stuck `PENDING (Resources)` for more than a few minutes, check `squeue --all` and look for a `COMPLETING` job. Ask admin (noumanz) to clean it up with `scontrol update JobId=<id> EndTime=now`.
- **"Priority" pending reason**: Means the GPU may free up soon but a higher-priority job is ahead of you in the queue. Different from "Resources" (GPU physically unavailable). Nothing wrong with your job — just wait or check if a COMPLETING job is blocking the chain.
- **Python stdout buffering**: SLURM captures stdout to a file, and Python buffers it. Always set `export PYTHONUNBUFFERED=1` in SLURM scripts so output appears immediately.

## What Was Fixed (History)

| Date | Issue | Fix |
|------|-------|-----|
| Apr 17 | Wrong partition `hpcshort` | Changed to `debug` |
| Apr 17 | `pip install torch>=` shell redirection bug | Created `requirements_hpc.txt`, used `pip install -r` |
| Apr 17 | `torch`/`torchvision` in requirements_hpc.txt overwrote CUDA builds | Removed them; conda handles those |
| Apr 17 | `setup_hpc_env.sh` interactive CUDA checks hang under MPS | Removed; GPU validation via SLURM only |
| Apr 17 | `.env` username typo `muhammmadtalha` | Fixed to `muhammadtalha` |
| Apr 17 | `hpc_scratch_dir` was `~/agripay/scratch` | Changed to `/scratch/muhammadtalha/agripay` |
| Apr 17 | gres/gpu not in TRES database | Admin (noumanz) fixed on server side |
| Apr 21 | `module: command not found` in batch jobs | Added `source /etc/profile.d/modules.sh` to all SLURM scripts |
| Apr 21 | `set -u` breaks conda MKL activation | Changed to `set -eo pipefail` (dropped `-u`) |
| Apr 21 | MPS value too low | Updated to 400 per admin recommendation |
| Apr 21 | `from io.image_reader import GeoTiffReader` fails | Renamed `io/` → `image_io/` (stdlib `io` module conflict) |
| Apr 21 | `weights_dir: "~/agripay/models"` not expanded | Added `os.path.expanduser()` in `run_inference.py` |
| Apr 21 | Python stdout buffered in SLURM jobs | Added `export PYTHONUNBUFFERED=1` to all SLURM scripts |
| Apr 21 | `module: command not found` in batch job | Added `source /etc/profile.d/modules.sh` |
| Apr 21 | conda MKL activation fails with `set -u` | Changed to `set -eo pipefail` (removed `-u`) |
| Apr 21 | MPS value too low | Updated to `mps:400` per admin recommendation |

## Diagnostic Commands

```bash
# Check TRES registration
sacctmgr show tres

# Check node GRES
scontrol show node rustam2 | grep -E 'Gres|CfgTRES|AllocTRES'

# Check GPU utilization
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

# Check all jobs
squeue -a -o "%i %u %P %j %t %R %b"

# Recent job history
sacct -u muhammadtalha -o JobID,JobName,State,ExitCode,Elapsed --starttime=today
```
