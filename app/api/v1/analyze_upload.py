"""Chunked upload endpoints for Plant Counter (any image format).

This is a SEPARATE router from chunked_upload.py (which is GeoTIFF-only for
drone imagery).  This router accepts .tif/.tiff/.png/.jpg/.jpeg, stores the
assembled file in the job temp-store, and immediately submits an ML job on
complete — no DB registration, no imagery-dir move.

Endpoints (all under /api/v1/analyze/upload/...):
  POST   .../init               — start a new analyze-upload session
  PATCH  .../{upload_id}        — send the next chunk
  GET    .../{upload_id}/status  — check resume offset
  POST   .../{upload_id}/complete — finalize and submit ML job → {job_id}
"""

import json
import os
import re
import shutil
import uuid
from datetime import datetime
from typing import Optional

import aiofiles

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.config import settings
from app.jobs.models import JobRecord

router = APIRouter(prefix="/analyze/upload", tags=["analyze-upload"])

# Reuse the same temp dir as chunked_upload so stale-cleanup covers both
UPLOAD_TEMP_DIR = "/data/.uploads_tmp"
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

# Chunk size: 50 MB
CHUNK_SIZE = 50 * 1024 * 1024

# Accepted image extensions for Plant Counter
VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Wired in by main.py lifespan
_dispatcher = None
_temp_store = None


def set_dispatcher(dispatcher):
    global _dispatcher
    _dispatcher = dispatcher


def set_temp_store(store):
    global _temp_store
    _temp_store = store


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnalyzeUploadInitRequest(BaseModel):
    filename: str
    total_size: int
    model_id: str = "wheat_plant_counter_v1"
    content_type: str = "image/tiff"


# ---------------------------------------------------------------------------
# Helpers (mirrors chunked_upload.py's private helpers)
# ---------------------------------------------------------------------------

def _get_upload_dir(upload_id: str) -> str:
    return os.path.join(UPLOAD_TEMP_DIR, upload_id)


def _read_meta(upload_id: str) -> dict:
    meta_path = os.path.join(_get_upload_dir(upload_id), "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(
            status_code=404,
            detail=f"Analyze-upload session '{upload_id}' not found",
        )
    with open(meta_path) as f:
        return json.load(f)


def _write_meta(upload_id: str, meta: dict):
    meta_path = os.path.join(_get_upload_dir(upload_id), "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def _parse_content_range(header: str) -> tuple:
    """Parse 'bytes start-end/total' -> (start, end, total)."""
    m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+)", header)
    if not m:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Content-Range header: '{header}'. Expected: 'bytes start-end/total'",
        )
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ---------------------------------------------------------------------------
# POST /api/v1/analyze/upload/init
# ---------------------------------------------------------------------------

@router.post("/init")
async def init_analyze_upload(req: AnalyzeUploadInitRequest):
    """Initialize a Plant Counter chunked-upload session.

    Accepts any image format (.tif, .tiff, .png, .jpg, .jpeg).
    Returns upload_id and recommended chunk_size.
    """
    ext = os.path.splitext(req.filename)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {sorted(VALID_IMAGE_EXTENSIONS)}"
            ),
        )

    if req.total_size <= 0:
        raise HTTPException(status_code=400, detail="total_size must be positive")
    if req.total_size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({req.total_size} bytes). "
                f"Max: {settings.max_upload_bytes} bytes"
            ),
        )

    upload_id = str(uuid.uuid4())
    upload_dir = _get_upload_dir(upload_id)
    os.makedirs(upload_dir, exist_ok=True)

    meta = {
        "upload_id": upload_id,
        "mode": "analyze",                    # Distinguishes from drone imagery sessions
        "original_filename": req.filename,
        "total_size": req.total_size,
        "model_id": req.model_id,
        "content_type": req.content_type,
        "received_bytes": 0,
        "created_at": datetime.utcnow().isoformat(),
    }
    _write_meta(upload_id, meta)

    # Create empty partial file
    part_path = os.path.join(upload_dir, "file.part")
    open(part_path, "wb").close()

    return {
        "upload_id": upload_id,
        "chunk_size": CHUNK_SIZE,
        "total_size": req.total_size,
    }


# ---------------------------------------------------------------------------
# PATCH /api/v1/analyze/upload/{upload_id}
# ---------------------------------------------------------------------------

@router.patch("/{upload_id}")
async def analyze_upload_chunk(upload_id: str, request: Request):
    """Receive the next chunk for an analyze-upload session.

    Expects:
      Header: Content-Range: bytes {start}-{end}/{total}
      Body: raw binary data
    """
    meta = _read_meta(upload_id)
    if meta.get("mode") != "analyze":
        raise HTTPException(
            status_code=400,
            detail="This upload_id belongs to a drone-imagery session, not an analyze session.",
        )

    range_header = request.headers.get("content-range")
    if not range_header:
        raise HTTPException(status_code=400, detail="Missing Content-Range header")

    start, end, total = _parse_content_range(range_header)

    if total != meta["total_size"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Content-Range total ({total}) doesn't match session total "
                f"({meta['total_size']})"
            ),
        )

    expected_offset = meta["received_bytes"]
    if start != expected_offset:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Expected chunk starting at byte {expected_offset}, got {start}. "
                f"Use GET .../status to check current offset."
            ),
        )

    part_path = os.path.join(_get_upload_dir(upload_id), "file.part")
    chunk_len = 0

    async with aiofiles.open(part_path, "ab") as f:
        async for data in request.stream():
            chunk_len += len(data)
            await f.write(data)

    if chunk_len == 0:
        raise HTTPException(status_code=400, detail="Empty chunk body")

    expected_end = start + chunk_len - 1
    if end != expected_end:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Content-Range end ({end}) doesn't match body length. "
                f"Expected end={expected_end} for {chunk_len} bytes starting at {start}."
            ),
        )

    if start + chunk_len > meta["total_size"]:
        raise HTTPException(status_code=400, detail="Chunk would exceed total file size")

    meta["received_bytes"] = start + chunk_len
    _write_meta(upload_id, meta)

    return {
        "upload_id": upload_id,
        "received_bytes": meta["received_bytes"],
        "total_size": meta["total_size"],
    }


# ---------------------------------------------------------------------------
# GET /api/v1/analyze/upload/{upload_id}/status
# ---------------------------------------------------------------------------

@router.get("/{upload_id}/status")
async def get_analyze_upload_status(upload_id: str):
    """Check how many bytes have been received (for resume)."""
    meta = _read_meta(upload_id)
    return {
        "upload_id": upload_id,
        "received_bytes": meta["received_bytes"],
        "total_size": meta["total_size"],
        "model_id": meta.get("model_id"),
    }


# ---------------------------------------------------------------------------
# POST /api/v1/analyze/upload/{upload_id}/complete
# ---------------------------------------------------------------------------

@router.post("/{upload_id}/complete")
async def complete_analyze_upload(upload_id: str):
    """Finalize an analyze-upload: move file to temp-store and submit ML job.

    Returns {job_id, status, message} — the frontend can start polling
    GET /status/{job_id} immediately.
    """
    if _dispatcher is None or _temp_store is None:
        raise HTTPException(status_code=503, detail="Dispatcher not ready")

    meta = _read_meta(upload_id)
    if meta.get("mode") != "analyze":
        raise HTTPException(
            status_code=400,
            detail="This upload_id belongs to a drone-imagery session.",
        )

    if meta["received_bytes"] != meta["total_size"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Upload incomplete: received {meta['received_bytes']} "
                f"of {meta['total_size']} bytes"
            ),
        )

    # Create a job record — the job dir is where the input file will live
    model_id = meta.get("model_id", "wheat_plant_counter_v1")
    job = JobRecord(model_id=model_id, input_params={})

    job_dir = _temp_store.get_job_dir(job.id)
    ext = os.path.splitext(meta["original_filename"])[1] or ".tif"
    dest_path = os.path.join(job_dir, f"input{ext}")

    part_path = os.path.join(_get_upload_dir(upload_id), "file.part")
    try:
        shutil.move(part_path, dest_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to move uploaded file: {exc}"
        )

    job.input_params["image_path"] = dest_path
    job.input_params["original_filename"] = meta["original_filename"]

    job_id = await _dispatcher.submit(job)

    # Clean up upload temp dir
    upload_dir = _get_upload_dir(upload_id)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir, ignore_errors=True)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Upload complete — plant-counting job queued.",
    }
