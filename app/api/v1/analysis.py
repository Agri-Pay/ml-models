"""Plant counting convenience endpoint."""

import asyncio
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

import httpx

from app.config import settings
from app.jobs.models import JobRecord
from app.models.registry import registry

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
        model_id=registry.resolve_model_id(request.model_id),
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


# ---------------------------------------------------------------------------
# POST /api/v1/analyze/from-drive
# ---------------------------------------------------------------------------

class FromDriveRequest(BaseModel):
    file_id: str
    file_name: str
    model_id: str = "rice_plant_counter_v1"


@router.post("/analyze/from-drive")
async def analyze_from_drive(req: FromDriveRequest):
    """Start a plant-count analysis from a Google Drive file.

    Returns a job_id immediately.  A background task downloads the file from
    Drive and submits it to the dispatcher — the job can be polled via
    GET /status/{job_id} just like any other analysis job.
    """
    if _dispatcher is None or _temp_store is None:
        raise HTTPException(status_code=503, detail="Job dispatcher not initialized")

    api_key = settings.google_drive_api_key
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Google Drive API key not configured. Set GOOGLE_DRIVE_API_KEY on the server.",
        )

    job = JobRecord(
        model_id=registry.resolve_model_id(req.model_id),
        input_params={"awaiting_download": True},
    )
    # Pre-register the job so /status/{job_id} returns PENDING immediately
    await _dispatcher.submit(job)

    # Kick off background download + re-submit with image_path
    asyncio.create_task(_download_and_resubmit(job, req.file_id, req.file_name, api_key))

    return {
        "job_id": job.id,
        "status": "pending",
        "message": "Downloading image from Google Drive…",
    }


async def _download_and_resubmit(job: JobRecord, file_id: str, file_name: str, api_key: str):
    """Background task: download Drive file then update job with local path.

    Because the job was already submitted to the dispatcher, we can't
    'resubmit' it.  Instead we download the file into the job's temp dir
    and update input_params so the worker can find it when it runs.

    Note: for the HPC dispatcher the job is queued asynchronously, so the
    image_path must be set *before* the HPC submission loop picks it up.
    For the in-process queue the worker thread reads input_params at run time,
    so updating the dict here is safe.
    """
    from app.jobs.models import JobStatus
    from datetime import datetime

    try:
        job_dir = _temp_store.get_job_dir(job.id)
        ext = os.path.splitext(file_name)[1].lower() or ".tif"
        dest_path = os.path.join(job_dir, f"input{ext}")

        dl_url = (
            f"https://www.googleapis.com/drive/v3/files/{file_id}"
            f"?alt=media&key={api_key}"
        )

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, read=3600.0),
        ) as client:
            async with client.stream("GET", dl_url) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Drive API returned HTTP {resp.status_code} for file {file_id}"
                    )
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(1024 * 1024):
                        f.write(chunk)

        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise RuntimeError("Downloaded file was empty")

        # Update job so the worker finds the local file
        job.input_params["image_path"] = dest_path
        job.input_params["original_filename"] = file_name
        job.input_params["awaiting_download"] = False

    except Exception as exc:
        # Mark the job failed so the frontend poll sees it
        job.input_params["awaiting_download"] = False
        job.status = JobStatus.FAILED
        job.error = f"Drive download failed: {exc}"
        job.completed_at = datetime.utcnow()
