"""AgriPay Compute Backend — Modal deployment.

Usage:
    modal serve modal_app.py   # dev (temporary URL)
    modal deploy modal_app.py  # prod (persistent URL)
    modal run modal_app.py     # upload model weights (run once)
"""
import os
import modal

app = modal.App("agripay-compute")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgdal-dev", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "fastapi==0.115.6", "uvicorn[standard]==0.34.0",
        "pydantic-settings==2.7.1", "python-dotenv==1.0.1",
        "python-multipart==0.0.20", "supabase==2.11.0",
        "numpy>=1.24.0", "scipy>=1.11.0",
        "rasterio>=1.3.9", "Pillow>=10.0.0",
        "opencv-python-headless>=4.8.0",
        "pandas>=2.0.0", "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
    )
    .add_local_dir("app", remote_path="/root/app")
)

weights_volume = modal.Volume.from_name(
    "agripay-weights", create_if_missing=True
)


@app.function(
    image=image,
    volumes={"/root/model_weights": weights_volume},
    timeout=600,
    secrets=[modal.Secret.from_name("agripay-env")],
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/root")
    os.environ.setdefault("MODEL_WEIGHTS_DIR", "/root/model_weights")
    os.environ.setdefault("IMAGERY_DIR", "/tmp/agripay_imagery")
    os.makedirs("/tmp/agripay_imagery", exist_ok=True)
    from app.main import app as fastapi_app
    return fastapi_app


@app.local_entrypoint()
def upload_weights():
    """Upload model weight files from local disk to the Modal volume."""
    vol = modal.Volume.from_name("agripay-weights", create_if_missing=True)
    weight_dir = os.path.join(os.path.dirname(__file__), "model_weights")
    with vol.batch_upload() as batch:
        for fn in ["checkpoint_counting.pth", "checkpoint_size_estimation.pth"]:
            local = os.path.join(weight_dir, fn)
            if not os.path.exists(local):
                print(f"SKIP: {local} not found")
                continue
            print(f"Uploading {fn} ...")
            batch.put_file(local, f"/{fn}")
    print("Done.")
