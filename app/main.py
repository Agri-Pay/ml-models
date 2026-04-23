"""AgriPay Compute Backend - FastAPI application."""

import asyncio
import logging
import os
import torch
from contextlib import asynccontextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import settings
from app.api.v1.router import v1_router, upload_router_compat
from app.api.v1.health import router as health_root_router
from app.api.v1 import jobs as jobs_api
from app.api.v1 import analysis as analysis_api
from app.api.v1 import upload as upload_api
from app.models.registry import registry
from app.jobs.models import JobRecord, JobStatus
from app.jobs.in_process_queue import InProcessQueue
from app.storage.temp_results import temp_store
from app.processing.visualization import (
    generate_all_visualizations,
    render_counting_overlay,
    render_heatmap,
    render_size_annotated,
    render_size_color_coded,
)


def run_model_job(job: JobRecord) -> JobRecord:
    """Worker function: runs model inference for a job.

    Called by the InProcessQueue via run_in_executor (runs in a thread).
    This is intentionally synchronous — all operations are CPU/GPU bound.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find and load the model
    model = registry.ensure_loaded(job.model_id, device=device)

    # Determine input
    image_path = job.input_params.get("image_path")
    image_url = job.input_params.get("image_url")
    if image_path and os.path.exists(image_path):
        input_data = image_path
    elif image_url:
        input_data = image_url
    else:
        input_data = image_path  # will fail at model.predict() with clear error
    if not input_data:
        job.status = JobStatus.FAILED
        job.error = "No image_path or image_url provided"
        job.completed_at = datetime.utcnow()
        return job

    # Progress callback
    def on_progress(current, total, message):
        job.progress_current = current
        job.progress_total = total
        job.progress_message = message

    # Run inference
    result = model.predict(input_data, progress_cb=on_progress)

    # Save visualization outputs
    output_files = []
    if "heatmap" in result and "points" in result:
        output_files = generate_all_visualizations(
            job_id=job.id,
            result=result,
            heatmap=result["heatmap"],
            image_path=job.input_params.get("image_path"),
            temp_store=temp_store,
        )

    # Store results (exclude large numpy arrays)
    result_serializable = {
        k: v for k, v in result.items()
        if k != "heatmap"
    }
    # Convert points to serializable format
    if "points" in result_serializable:
        result_serializable["points"] = [
            {"x": p["x"], "y": p["y"], "base_sigma": p["base_sigma"]}
            for p in result_serializable["points"]
        ]

    job.result = result_serializable
    job.output_files = output_files
    job.status = JobStatus.COMPLETED
    job.completed_at = datetime.utcnow()
    return job


def post_process_hpc_job(job: JobRecord, results_dir: str) -> JobRecord:
    """Post-process results downloaded from HPC.

    Reads results.json + heatmap.npz from results_dir, then generates
    visualizations locally using the original image on this server.
    """
    import json
    import numpy as np

    # Load HPC results
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "r") as f:
        result = json.load(f)

    # Load compressed heatmap
    heatmap = None
    heatmap_path = os.path.join(results_dir, "heatmap.npz")
    if os.path.exists(heatmap_path):
        heatmap_data = np.load(heatmap_path)
        heatmap = heatmap_data["heatmap"]

    # Generate visualizations locally
    output_files = []
    if heatmap is not None and result.get("points"):
        output_files = generate_all_visualizations(
            job_id=job.id,
            result=result,
            heatmap=heatmap,
            image_path=job.input_params.get("image_path"),
            temp_store=temp_store,
        )

    # Store results
    job.result = result
    job.output_files = output_files
    job.status = JobStatus.COMPLETED
    job.completed_at = datetime.utcnow()
    return job


# Global dispatcher reference
_dispatcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global _dispatcher

    print(f"Starting AgriPay Compute Backend on port {settings.compute_port}")
    print(f"Dispatch mode: {settings.compute_dispatch_mode}")
    print(f"Imagery dir: {settings.imagery_dir}")
    print(f"Model weights dir: {settings.model_weights_dir}")

    # Clean up stale chunked uploads (abandoned more than 24h ago)
    from app.api.v1.chunked_upload import cleanup_stale_uploads
    cleanup_stale_uploads(max_age_hours=24)

    # Discover models
    print("Discovering models...")
    registry.discover()
    print(f"  Found {len(registry.list_models())} model(s)")

    # Start job dispatcher based on compute_dispatch_mode
    mode = settings.compute_dispatch_mode

    if mode == "hpc":
        from app.jobs.hpc_dispatcher import HPCDispatcher
        _dispatcher = HPCDispatcher(post_process_fn=post_process_hpc_job)
        print("Job dispatcher: HPC (Rustam2)")
    elif mode == "auto":
        from app.jobs.hpc_dispatcher import HPCDispatcher
        from app.jobs.auto_dispatcher import AutoDispatcher
        hpc_disp = HPCDispatcher(post_process_fn=post_process_hpc_job)
        local_disp = InProcessQueue(worker_fn=run_model_job)
        _dispatcher = AutoDispatcher(primary=hpc_disp, fallback=local_disp)
        print("Job dispatcher: Auto (HPC with local fallback)")
    else:  # "local" (default)
        _dispatcher = InProcessQueue(worker_fn=run_model_job)
        print("Job dispatcher: Local (InProcessQueue)")

    await _dispatcher.start()
    print("Job dispatcher started")

    # Wire dispatcher and temp store into API endpoints
    jobs_api.set_dispatcher(_dispatcher)
    jobs_api.set_temp_store(temp_store)
    analysis_api.set_dispatcher(_dispatcher)
    upload_api.set_dispatcher(_dispatcher)

    # Periodic cleanup: remove expired job temp dirs every hour
    async def _periodic_temp_cleanup():
        while True:
            await asyncio.sleep(3600)  # 1 hour
            try:
                removed = temp_store.cleanup_expired()
                if removed:
                    logger.info("Periodic cleanup: removed %d expired job dir(s)", removed)
            except Exception as exc:
                logger.warning("Periodic temp cleanup error: %s", exc)

    cleanup_task = asyncio.create_task(_periodic_temp_cleanup())

    yield

    # Shutdown
    print("Shutting down AgriPay Compute Backend")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    await _dispatcher.stop()
    temp_store.cleanup_expired()


app = FastAPI(
    title="AgriPay Compute Service",
    description="Multi-source AI verification system for agricultural milestone validation",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server and any configured origins
_default_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://agripay.netlify.app",
]
if settings.cors_origins:
    _cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
else:
    _cors_origins = _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https://.*\.(trycloudflare\.com|ts\.net)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(health_root_router, tags=["health"])  # GET /health at root
app.include_router(v1_router)  # All /api/v1/* endpoints
app.include_router(upload_router_compat)  # /upload, /status, /download compat layer
