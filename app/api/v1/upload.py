"""Browser-facing upload/status/download compatibility API.

Provides the same three paths the frontend already uses:
  POST /upload              — receive an image file, start a job
  GET  /status/{job_id}    — poll job progress in the shape the frontend expects
  GET  /download/{job_id}/{file_type} — stream a result image

This is a thin layer over the existing /api/v1/jobs system.
For large files (>50 MB) the frontend should use the chunked upload protocol
(/api/v1/imagery/upload/init → PATCH → /complete?analyze=true) instead.
"""

import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.jobs.models import JobRecord, JobStatus
from app.models.registry import registry
from app.storage.temp_results import temp_store

router = APIRouter()

# Wired in during lifespan (same pattern as jobs.py / analysis.py)
_dispatcher = None


def set_dispatcher(dispatcher):
    global _dispatcher
    _dispatcher = dispatcher


# Maps the frontend's short type key → actual output filename
_FILE_TYPE_MAP = {
    "counting":      "counting_overlay.png",
    "size_annotated": "size_annotated.png",
    "size_colored":  "size_color_coded.png",
    "heatmap":       "density_heatmap.png",
}

# Max upload size: 10 GB
_MAX_FILE_BYTES = 10 * 1024 * 1024 * 1024

# Content-types that are valid images (includes octet-stream for browsers that
# don't know TIFF MIME type, and empty content-type for some file-pickers)
_ALLOWED_IMAGE_TYPES = {"image/", "application/octet-stream"}


def _is_allowed_content_type(ct: Optional[str]) -> bool:
    """Return True if the content-type looks like an image or binary blob."""
    if not ct:
        return True  # no content-type sent — trust the extension
    for allowed in _ALLOWED_IMAGE_TYPES:
        if ct.startswith(allowed):
            return True
    return False


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    model_id: str = Form("rice_plant_counter_v1"),
):
    """Accept an image file upload and start a processing job.

    For large files (>50 MB) prefer the chunked upload protocol to avoid
    Cloudflare 524 timeouts:
      POST /api/v1/imagery/upload/init  (farm_id omitted → analyze mode)
      PATCH /api/v1/imagery/upload/{id}  (50 MB chunks)
      POST /api/v1/imagery/upload/{id}/complete?analyze=true

    Returns:
        {job_id, status, progress, message}
    """
    if _dispatcher is None:
        raise HTTPException(status_code=503, detail="Dispatcher not ready")

    if not _is_allowed_content_type(file.content_type):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Please upload a PNG, TIFF, or JPEG image."
            ),
        )

    job = JobRecord(model_id=registry.resolve_model_id(model_id), input_params={})
    job_dir = temp_store.get_job_dir(job.id)

    ext = os.path.splitext(file.filename or "image.tif")[1].lower() or ".tif"
    upload_path = os.path.join(job_dir, f"input{ext}")
    total = 0
    try:
        with open(upload_path, "wb") as dst:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_FILE_BYTES:
                    dst.close()
                    os.remove(upload_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {_MAX_FILE_BYTES // (1024 * 1024 * 1024)} GB)",
                    )
                dst.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    job.input_params["image_path"] = upload_path
    job.input_params["original_filename"] = file.filename or "upload"

    await _dispatcher.submit(job)
    return {
        "job_id": job.id,
        "status": "pending",
        "progress": 5.0,
        "message": "Upload received, processing started",
    }


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------

@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """Return job status in the shape the frontend polls.

    Returns:
        {job_id, status, progress (0-100), message, result?, error?}
    """
    if _dispatcher is None:
        raise HTTPException(status_code=503, detail="Dispatcher not ready")

    job = await _dispatcher.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Map internal progress to 0-100 percentage
    # HPC jobs use progress_current/progress_total directly (already 0-100 scale)
    if job.dispatch_backend == "hpc":
        if job.status == JobStatus.COMPLETED:
            pct = 100.0
        elif job.progress_current > 0:
            pct = float(job.progress_current)
        elif job.status == JobStatus.RUNNING:
            pct = 10.0
        else:
            pct = 5.0
    else:
        # Local mode: map tile progress (current/total) to 10-95 range
        if job.progress_total and job.progress_total > 0:
            pct = 10.0 + 85.0 * (job.progress_current / job.progress_total)
        elif job.status == JobStatus.COMPLETED:
            pct = 100.0
        elif job.status == JobStatus.RUNNING:
            pct = 10.0
        else:
            pct = 5.0

    response = {
        "job_id": job_id,
        "status": job.status.value,
        "progress": round(pct, 1),
        "message": job.progress_message or _status_message(job.status),
        "dispatch_backend": job.dispatch_backend,
    }

    if job.status == JobStatus.COMPLETED and job.result:
        # Remap keys to what saveResultsToSupabase / pcResult expects:
        #   plant_count  → total_count
        #   avg_size_px  → average_size
        r = job.result
        response["result"] = {
            "total_count":              r.get("plant_count", 0),
            "average_size":             r.get("avg_size_px", 0.0),
            "min_size_px":              r.get("min_size_px", 0.0),
            "max_size_px":              r.get("max_size_px", 0.0),
            "processing_time_seconds":  _processing_seconds(job),
            "tiles_processed":          r.get("tiles_processed", 0),
        }

    if job.status == JobStatus.FAILED:
        response["error"] = job.error or "Processing failed"

    return response


# ---------------------------------------------------------------------------
# GET /download/{job_id}/{file_type}
# ---------------------------------------------------------------------------

@router.get("/download/{job_id}/{file_type}")
async def download_result(job_id: str, file_type: str):
    """Stream a result image file back to the browser.

    file_type is one of: counting | size_annotated | size_colored | heatmap
    """
    if _dispatcher is None:
        raise HTTPException(status_code=503, detail="Dispatcher not ready")

    job = await _dispatcher.get_status(job_id)
    if job is None or job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=404, detail="Results not available yet")

    filename = _FILE_TYPE_MAP.get(file_type)
    if filename is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown file_type '{file_type}'. Valid: {list(_FILE_TYPE_MAP)}",
        )

    if not temp_store.file_exists(job_id, filename):
        raise HTTPException(status_code=404, detail=f"Output file '{filename}' not found")

    path = temp_store.get_output_path(job_id, filename)
    return FileResponse(path, media_type="image/png", filename=f"{job_id}_{file_type}.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_message(status: JobStatus) -> str:
    return {
        JobStatus.PENDING:   "Queued for processing…",
        JobStatus.RUNNING:   "Analysing image…",
        JobStatus.COMPLETED: "Analysis complete",
        JobStatus.FAILED:    "Processing failed",
    }.get(status, "")


def _processing_seconds(job: JobRecord) -> float:
    if job.started_at and job.completed_at:
        return (job.completed_at - job.started_at).total_seconds()
    return 0.0
