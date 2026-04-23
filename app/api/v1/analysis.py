"""Plant counting convenience endpoints."""

import asyncio
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.config import settings
from app.jobs.models import JobRecord

router = APIRouter()

# Set by main.py during lifespan (same pattern as jobs.py)
_dispatcher = None
_temp_store = None


def set_dispatcher(dispatcher):
    global _dispatcher
    _dispatcher = dispatcher


def set_temp_store(store):
    global _temp_store
    _temp_store = store


class PlantCountRequest(BaseModel):
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    filename: Optional[str] = None  # just the filename; backend prepends imagery_dir
    model_id: str  # Required — frontend sends the user-selected model


class PlantCountResponse(BaseModel):
    job_id: str
    status: str
    message: str


class DriveAnalyzeRequest(BaseModel):
    file_id: str   # Google Drive file ID (resolved by /analyze/drive/list-files)
    file_name: str  # Original filename (used for extension + progress display)
    model_id: str = "wheat_plant_counter_v1"


@router.post("/analyze/plant-count", response_model=PlantCountResponse)
async def analyze_plant_count(request: PlantCountRequest):
    """Submit a plant counting analysis job.

    Convenience wrapper around the generic /jobs endpoint
    that pre-selects the plant counting model.
    """
    if _dispatcher is None:
        raise HTTPException(status_code=503, detail="Job dispatcher not initialized")

    # Resolve filename to full path (only when no image_url fallback exists)
    if request.filename and not request.image_path and not request.image_url:
        local_path = os.path.join(settings.imagery_dir, request.filename)
        if os.path.exists(local_path):
            request.image_path = local_path

    if not request.image_path and not request.image_url:
        raise HTTPException(
            status_code=400,
            detail="One of image_path, filename, or image_url must be provided",
        )

    job = JobRecord(
        model_id=request.model_id,
        input_params={
            "image_path": request.image_path,
            "image_url": request.image_url,
        },
    )
    job_id = await _dispatcher.submit(job)
    return PlantCountResponse(
        job_id=job_id,
        status="pending",
        message="Plant counting job submitted. Poll GET /status/{id} for progress.",
    )


@router.post("/analyze/from-drive", response_model=PlantCountResponse)
async def analyze_from_drive(request: DriveAnalyzeRequest):
    """Accept a resolved Drive file_id and queue a plant-counting job.

    The file is downloaded server-side in a background asyncio task so the
    HTTP response returns immediately (no Cloudflare 524 risk).
    The caller should poll GET /status/{job_id}.

    Requires the Drive link to have been pre-resolved via
    POST /api/v1/analyze/drive/list-files so we get a clean file_id.
    """
    if _dispatcher is None or _temp_store is None:
        raise HTTPException(status_code=503, detail="Job dispatcher not initialized")

    from app.config import settings as _settings
    api_key = _settings.google_drive_api_key
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail=(
                "Google Drive API key not configured on the server. "
                "Set GOOGLE_DRIVE_API_KEY in the server environment."
            ),
        )

    # Derive destination extension from the original filename
    ext = os.path.splitext(request.file_name)[1].lower() or ".tif"

    # Create the job record now so we have the job_id to return immediately
    job = JobRecord(model_id=request.model_id, input_params={})
    job_dir = _temp_store.get_job_dir(job.id)
    dest_path = os.path.join(job_dir, f"input{ext}")

    # Store originals so the worker can display them in logs
    job.input_params["original_filename"] = request.file_name
    # image_path will be set by the background task once the download finishes

    # Submit to queue NOW — the worker will block on the file being present,
    # but the background task will place it there before the worker picks it up
    # because the queue is sequential (InProcessQueue processes one at a time).
    # We register the job first so /status/{job_id} works immediately.
    job_id = await _dispatcher.submit(job)

    # Kick off background download
    asyncio.create_task(
        _download_drive_file_for_job(
            job=job,
            file_id=request.file_id,
            dest_path=dest_path,
            api_key=api_key,
        )
    )

    return PlantCountResponse(
        job_id=job_id,
        status="pending",
        message=(
            f"Job queued. Downloading '{request.file_name}' from Google Drive "
            "in background — poll /status/{job_id} for progress."
        ),
    )


async def _download_drive_file_for_job(
    job: JobRecord,
    file_id: str,
    dest_path: str,
    api_key: str,
):
    """Background task: download a Drive file and patch the job's image_path.

    If the download fails the job is marked FAILED so the poller sees it.
    """
    import httpx
    from app.jobs.models import JobStatus
    from datetime import datetime as _dt

    DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
    download_url = f"{DRIVE_API_BASE}/files/{file_id}?alt=media&key={api_key}"

    try:
        job.progress_message = "Downloading from Google Drive…"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30, read=600), follow_redirects=True
        ) as client:
            async with client.stream("GET", download_url) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    raise RuntimeError(
                        f"Drive download failed (HTTP {resp.status_code}): "
                        f"{error_body[:200].decode(errors='replace')}"
                    )
                with open(dest_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)

        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise RuntimeError("Downloaded file is empty or missing.")

        # Patch the job so the worker (already queued) picks up the right path
        job.input_params["image_path"] = dest_path
        job.progress_message = "Download complete — processing…"

    except Exception as exc:
        # Mark job as failed so the frontend poller sees it
        job.status = JobStatus.FAILED
        job.error = f"Drive download failed: {exc}"
        job.completed_at = _dt.utcnow()
