#!/usr/bin/env python3
"""Minimal GPU smoke test for Rustam2.

This is intended to run inside an allocated SLURM job only.
Do not run it directly in an interactive login shell on Rustam2.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import torch


def main() -> int:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Starting AgriPay GPU smoke test")
    print(f"SLURM job id: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"SLURM node: {os.environ.get('SLURM_JOB_NODELIST', 'N/A')}")
    print(f"Visible devices: {os.environ.get('SLURM_VISIBLE_DEVICES', 'N/A')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available inside this SLURM allocation")
        return 1

    device_count = torch.cuda.device_count()
    print(f"CUDA available: True")
    print(f"Visible GPU count: {device_count}")

    device = torch.device("cuda:0")
    print(f"Using GPU 0: {torch.cuda.get_device_name(device)}")

    left = torch.randn((2048, 2048), device=device)
    right = torch.randn((2048, 2048), device=device)
    product = left @ right
    torch.cuda.synchronize()

    checksum = float(product.sum().item())
    print(f"Matrix multiply succeeded; checksum={checksum:.4f}")
    print("SUCCESS: GPU allocation and CUDA execution work under SLURM")
    return 0


if __name__ == "__main__":
    sys.exit(main())