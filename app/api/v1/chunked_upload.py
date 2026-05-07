"""Chunked upload endpoints for large GeoTIFF files.

Provides resumable uploads by splitting files into chunks.
Frontend sends chunks sequentially; backend appends them to a partial file.

Endpoints (all under /api/v1/imagery/upload/...):
  POST   .../init               — start a new upload session
  PATCH  .../{upload_id}        — send the next chunk
  GET    .../{upload_id}/status  — check progress (for resume)
  POST   .../{upload_id}/complete — finalize and register in DB
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
from app.api.v1.imagery_helpers import (
    VALID_LAYER_TYPES,
    build_filename,
    register_imagery_in_db,
)
from app.jobs.models import JobRecord
from app.models.registry import registry

router = APIRouter(prefix="/imagery/upload", tags=["chunked-upload"])

# Wired by main.py during lifespan (only needed for analyze=True mode)
_dispatcher = None
_temp_store = None


def set_dispatcher(dispatcher):
    global _dispatcher
    _dispatcher = dispatcher


def set_temp_store(store):
    global _temp_store
    _temp_store = store

# Where partial uploads are stored — on the mounted volume so they are
# visible on the host and survive container restarts.  The leading dot
# hides the directory from casual `ls` and prevents TiTiler from scanning it.
UPLOAD_TEMP_DIR = "/data/.uploads_tmp"
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

# Recommended chunk size: 50 MB
CHUNK_SIZE = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UploadInitRequest(BaseModel):
    filename: str
    total_size: int
    farm_id: Optional[str] = None   # None signals analyze=True (Plant Counter) mode
    flight_date: Optional[str] = None  # YYYY-MM-DD; None allowed in analyze mode
    layer_type: str = "rgb"
    content_type: str = "image/tiff"


class UploadCompleteRequest(BaseModel):
    band_mapping: Optional[str] = None  # JSON string
    pilot_name: Optional[str] = None
    drone_model: Optional[str] = None
    altitude: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_upload_dir(upload_id: str) -> str:
    return os.path.join(UPLOAD_TEMP_DIR, upload_id)


def _read_meta(upload_id: str) -> dict:
    meta_path = os.path.join(_get_upload_dir(upload_id), "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Upload session '{upload_id}' not found")
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
# POST /api/v1/imagery/upload/init
# ---------------------------------------------------------------------------

@router.post("/init")
async def init_upload(req: UploadInitRequest):
    """Initialize a new chunked upload session.

    Returns an upload_id and recommended chunk_size.
    """
    analyze_mode = req.farm_id is None  # True = Plant Counter upload, skip DB fields

    # Validate filename extension
    ext = os.path.splitext(req.filename)[1].lower()
    if analyze_mode:
        # Plant Counter accepts PNG, JPEG, TIFF
        if ext not in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Accepted: .tif, .tiff, .png, .jpg, .jpeg",
            )
    else:
        if ext not in (".tif", ".tiff"):
            raise HTTPException(
                status_code=400,
                detail=f"Only GeoTIFF files (.tif/.tiff) are accepted, got '{ext}'",
            )

    lt = req.layer_type.lower()

    if not analyze_mode:
        # Validate layer_type only for imagery uploads
        if lt not in VALID_LAYER_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid layer_type '{lt}'. Must be one of: {sorted(VALID_LAYER_TYPES)}",
            )

        # Validate date format
        try:
            datetime.strptime(req.flight_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="flight_date must be in YYYY-MM-DD format")

    # Validate total_size
    if req.total_size <= 0:
        raise HTTPException(status_code=400, detail="total_size must be positive")
    if req.total_size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({req.total_size} bytes). Max: {settings.max_upload_bytes} bytes",
        )

    upload_id = str(uuid.uuid4())
    upload_dir = _get_upload_dir(upload_id)
    os.makedirs(upload_dir, exist_ok=True)

    meta = {
        "upload_id": upload_id,
        "original_filename": req.filename,
        "total_size": req.total_size,
        "farm_id": req.farm_id,
        "flight_date": req.flight_date,
        "layer_type": lt,
        "content_type": req.content_type,
        "received_bytes": 0,
        "created_at": datetime.utcnow().isoformat(),
        "analyze_mode": analyze_mode,
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
# PATCH /api/v1/imagery/upload/{upload_id}
# ---------------------------------------------------------------------------

@router.patch("/{upload_id}")
async def upload_chunk(upload_id: str, request: Request):
    """Receive the next chunk of data for an ongoing upload.

    Expects:
      Header: Content-Range: bytes {start}-{end}/{total}
      Body: raw binary data
    """
    meta = _read_meta(upload_id)

    # Parse Content-Range
    range_header = request.headers.get("content-range")
    if not range_header:
        raise HTTPException(status_code=400, detail="Missing Content-Range header")

    start, end, total = _parse_content_range(range_header)

    # Validate consistency
    if total != meta["total_size"]:
        raise HTTPException(
            status_code=400,
            detail=f"Content-Range total ({total}) doesn't match session total ({meta['total_size']})",
        )

    expected_offset = meta["received_bytes"]
    if start != expected_offset:
        raise HTTPException(
            status_code=409,
            detail=f"Expected chunk starting at byte {expected_offset}, got {start}. "
                   f"Use GET .../status to check current offset.",
        )

    # Stream chunk body directly to disk (no full buffering in memory)
    part_path = os.path.join(_get_upload_dir(upload_id), "file.part")
    chunk_len = 0

    async with aiofiles.open(part_path, "ab") as f:
        async for data in request.stream():
            chunk_len += len(data)
            await f.write(data)

    if chunk_len == 0:
        raise HTTPException(status_code=400, detail="Empty chunk body")

    # Validate end matches body length
    expected_end = start + chunk_len - 1
    if end != expected_end:
        raise HTTPException(
            status_code=400,
            detail=f"Content-Range end ({end}) doesn't match body length. "
                   f"Expected end={expected_end} for {chunk_len} bytes starting at {start}.",
        )

    # Check we don't exceed total
    if start + chunk_len > meta["total_size"]:
        raise HTTPException(
            status_code=400,
            detail="Chunk would exceed total file size",
        )

    # Update metadata
    meta["received_bytes"] = start + chunk_len
    _write_meta(upload_id, meta)

    return {
        "upload_id": upload_id,
        "received_bytes": meta["received_bytes"],
        "total_size": meta["total_size"],
    }


# ---------------------------------------------------------------------------
# GET /api/v1/imagery/upload/{upload_id}/status
# ---------------------------------------------------------------------------

@router.get("/{upload_id}/status")
async def get_upload_status(upload_id: str):
    """Check how many bytes have been received for a chunked upload.

    Used by the frontend to resume uploads after a failure.
    """
    meta = _read_meta(upload_id)
    return {
        "upload_id": upload_id,
        "received_bytes": meta["received_bytes"],
        "total_size": meta["total_size"],
        "farm_id": meta["farm_id"],
        "layer_type": meta["layer_type"],
    }


# ---------------------------------------------------------------------------
# POST /api/v1/imagery/upload/{upload_id}/complete
# ---------------------------------------------------------------------------

@router.post("/{upload_id}/complete")
async def complete_upload(
    upload_id: str,
    req: Optional[UploadCompleteRequest] = None,
    analyze: bool = False,
    model_id: str = "rice_plant_counter_v1",
):
    """Finalize a chunked upload.

    Normal mode: move file to imagery dir and register in DB.
    Analyze mode (analyze=True or farm_id was None at init): submit ML job and
    return a job_id for polling without touching the imagery database.
    """
    meta = _read_meta(upload_id)

    # Validate all bytes received
    if meta["received_bytes"] != meta["total_size"]:
        raise HTTPException(
            status_code=400,
            detail=f"Upload incomplete: received {meta['received_bytes']} of {meta['total_size']} bytes",
        )

    part_path = os.path.join(_get_upload_dir(upload_id), "file.part")

    # ── Analyze mode: submit ML job, skip DB registration ──────────────────
    if analyze or meta.get("analyze_mode"):
        if _dispatcher is None or _temp_store is None:
            raise HTTPException(status_code=503, detail="Dispatcher not ready")

        job = JobRecord(model_id=registry.resolve_model_id(model_id), input_params={})
        job_dir = _temp_store.get_job_dir(job.id)

        # Move uploaded file into the job's temp directory
        ext = os.path.splitext(meta["original_filename"])[1].lower() or ".tif"
        dest_path = os.path.join(job_dir, f"input{ext}")
        try:
            shutil.move(part_path, dest_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to stage file for analysis: {exc}")

        job.input_params["image_path"] = dest_path
        job.input_params["original_filename"] = meta["original_filename"]

        # Clean up the upload temp dir (file already moved)
        upload_dir = _get_upload_dir(upload_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir, ignore_errors=True)

        await _dispatcher.submit(job)
        return {
            "job_id": job.id,
            "status": "pending",
            "message": "File received, analysis job queued.",
        }

    # ── Normal imagery mode ─────────────────────────────────────────────────
    # Build destination filename
    filename = build_filename(meta["farm_id"], meta["flight_date"], meta["layer_type"])
    dest_path = os.path.join(settings.imagery_dir, filename)

    # Move partial file to final destination
    try:
        shutil.move(part_path, dest_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to move file to storage: {exc}")

    # Register in database
    band_mapping = req.band_mapping if req else None
    pilot_name = req.pilot_name if req else None
    drone_model = req.drone_model if req else None
    altitude = req.altitude if req else None

    result = await register_imagery_in_db(
        filename=filename,
        total_bytes=meta["total_size"],
        farm_id=meta["farm_id"],
        flight_date=meta["flight_date"],
        layer_type=meta["layer_type"],
        band_mapping=band_mapping,
        pilot_name=pilot_name,
        drone_model=drone_model,
        altitude=altitude,
    )

    # Clean up temp directory
    upload_dir = _get_upload_dir(upload_id)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# Cleanup utility (called from main.py lifespan)
# ---------------------------------------------------------------------------

def cleanup_stale_uploads(max_age_hours: int = 24):
    """Remove upload temp directories older than max_age_hours."""
    if not os.path.exists(UPLOAD_TEMP_DIR):
        return

    now = datetime.utcnow()
    removed = 0

    for entry in os.listdir(UPLOAD_TEMP_DIR):
        upload_dir = os.path.join(UPLOAD_TEMP_DIR, entry)
        if not os.path.isdir(upload_dir):
            continue

        meta_path = os.path.join(upload_dir, "meta.json")
        try:
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                created = datetime.fromisoformat(meta.get("created_at", ""))
                age_hours = (now - created).total_seconds() / 3600
                if age_hours < max_age_hours:
                    continue
        except Exception:
            pass  # If we can't read meta, treat as stale

        shutil.rmtree(upload_dir, ignore_errors=True)
        removed += 1

    if removed:
        print(f"Cleaned up {removed} stale upload session(s)")
