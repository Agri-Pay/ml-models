#!/usr/bin/env python3
"""HPC inference entry point. Called by SLURM job scripts.

Usage:
    python run_inference.py --config path/to/config.json

Input JSON format:
    {
        "job_id": "uuid-string",
        "model_id": "rice_plant_counter_v1",
        "image_path": "/path/to/image.tif",
        "weights_dir": "/path/to/model/weights",
        "output_dir": "/path/to/output/directory"
    }

Outputs:
    <output_dir>/results.json   - Inference results (points, stats)
    <output_dir>/heatmap.npz    - Compressed numpy heatmap array
    <output_dir>/status.json    - {"status": "completed"} or {"status": "failed", "error": "..."}
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np


# Available models (add new models here as they are created)
MODEL_REGISTRY = {
    "rice_plant_counter_v1": "models.rice_plant_counter.RicePlantCounter",
}


def load_model_class(model_id: str):
    """Dynamically import and return the model class for a given model_id."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    module_path, class_name = MODEL_REGISTRY[model_id].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def write_status(output_dir: str, status: str, error: str = None):
    """Write a status.json file to the output directory."""
    data = {"status": status}
    if error:
        data["error"] = error
    with open(os.path.join(output_dir, "status.json"), "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="AgriPay HPC Inference Runner")
    parser.add_argument("--config", required=True, help="Path to job config JSON")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    job_id = config["job_id"]
    model_id = config["model_id"]
    image_path = os.path.expanduser(config["image_path"])
    weights_dir = os.path.expanduser(config["weights_dir"])
    output_dir = os.path.expanduser(config["output_dir"])

    os.makedirs(output_dir, exist_ok=True)

    print(f"[AgriPay HPC] Job: {job_id}")
    print(f"[AgriPay HPC] Model: {model_id}")
    print(f"[AgriPay HPC] Image: {image_path}")
    print(f"[AgriPay HPC] Weights: {weights_dir}")
    print(f"[AgriPay HPC] Output: {output_dir}")

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[AgriPay HPC] Device: {device}")
        if device == "cuda":
            print(f"[AgriPay HPC] GPU: {torch.cuda.get_device_name(0)}")

        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load model
        print(f"[AgriPay HPC] Loading model...")
        t0 = time.time()
        ModelClass = load_model_class(model_id)
        model = ModelClass(weights_dir=weights_dir)
        model.load(device=device)
        load_time = time.time() - t0
        print(f"[AgriPay HPC] Model loaded in {load_time:.1f}s")

        # Run inference
        print(f"[AgriPay HPC] Running inference...")
        t0 = time.time()

        def progress_cb(current, total, message):
            print(f"[AgriPay HPC] {message}")

        result = model.predict(image_path, progress_cb=progress_cb)
        inference_time = time.time() - t0
        print(f"[AgriPay HPC] Inference complete in {inference_time:.1f}s")

        # Extract heatmap for separate compressed storage
        heatmap = result.pop("heatmap", None)

        # Add timing info
        result["processing_time_seconds"] = round(inference_time, 2)
        result["model_load_time_seconds"] = round(load_time, 2)
        result["device"] = device

        # Convert points to JSON-serializable format
        if "points" in result:
            result["points"] = [
                {"x": float(p["x"]), "y": float(p["y"]), "base_sigma": float(p["base_sigma"])}
                for p in result["points"]
            ]
        if "plant_sizes" in result:
            result["plant_sizes"] = [float(s) for s in result["plant_sizes"]]

        # Save results JSON
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(result, f)
        print(f"[AgriPay HPC] Results saved to {results_path}")

        # Save compressed heatmap
        if heatmap is not None:
            heatmap_path = os.path.join(output_dir, "heatmap.npz")
            np.savez_compressed(heatmap_path, heatmap=heatmap)
            heatmap_size_mb = os.path.getsize(heatmap_path) / (1024 * 1024)
            print(f"[AgriPay HPC] Heatmap saved to {heatmap_path} ({heatmap_size_mb:.1f} MB)")

        # Write success status
        write_status(output_dir, "completed")
        print(f"[AgriPay HPC] Job {job_id} completed successfully")

        # Cleanup
        model.unload()

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"[AgriPay HPC] ERROR: {error_msg}", file=sys.stderr)
        write_status(output_dir, "failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
