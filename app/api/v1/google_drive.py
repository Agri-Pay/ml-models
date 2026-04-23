"""Google Drive integration endpoints.

Provides server-side listing and importing of GeoTIFF files from Google Drive.
Files are downloaded directly from Drive to the server (no browser relay).

Endpoints (all under /api/v1/imagery/drive/...):
  POST .../list-files  — list GeoTIFF files from a Drive link/folder/ID
  POST .../import      — download a file from Drive to server + register in DB
"""

import json
import os
import re
import shutil
from datetime import datetime
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.api.v1.imagery_helpers import (
    VALID_LAYER_TYPES,
    build_filename,
    register_imagery_in_db,
)

router = APIRouter(prefix="/imagery/drive", tags=["google-drive"])

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ListFilesRequest(BaseModel):
    input: str  # Any Google Drive URL, folder URL, file URL, or raw ID


class ImportRequest(BaseModel):
    file_id: str
    file_name: str
    farm_id: str
    flight_date: str  # YYYY-MM-DD
    layer_type: str = "rgb"
    band_mapping: Optional[str] = None  # JSON string
    pilot_name: Optional[str] = None
    drone_model: Optional[str] = None
    altitude: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Return the configured Google Drive API key or raise."""
    key = settings.google_drive_api_key
    if not key:
        raise HTTPException(
            status_code=500,
            detail="Google Drive API key not configured on the server. "
                   "Set GOOGLE_DRIVE_API_KEY in the server environment.",
        )
    return key


def _parse_drive_input(raw: str) -> tuple[str, str]:
    """Parse any Google Drive URL or raw ID into (type, id).

    Supported formats:
      - https://drive.google.com/drive/folders/FOLDER_ID
      - https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing
      - https://drive.google.com/drive/u/0/folders/FOLDER_ID
      - https://drive.google.com/file/d/FILE_ID/view
      - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
      - https://drive.google.com/open?id=ID
      - Raw ID string (25+ alphanumeric / hyphens / underscores)

    Returns ("folder", id), ("file", id), or ("unknown", id).
    """
    raw = raw.strip()

    # URL-based detection
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        path = parsed.path

        # /drive/folders/FOLDER_ID  or  /drive/u/0/folders/FOLDER_ID
        m = re.search(r"/folders/([a-zA-Z0-9_-]+)", path)
        if m:
            return ("folder", m.group(1))

        # /file/d/FILE_ID/...
        m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", path)
        if m:
            return ("file", m.group(1))

        # /open?id=ID
        if "/open" in path:
            qs = parse_qs(parsed.query)
            if "id" in qs:
                return ("unknown", qs["id"][0])

        # Fallback: try to find an ID-looking segment in the path
        m = re.search(r"([a-zA-Z0-9_-]{20,})", path)
        if m:
            return ("unknown", m.group(1))

        raise HTTPException(
            status_code=400,
            detail="Could not extract a file or folder ID from the provided URL. "
                   "Please paste a Google Drive file link, folder link, or raw ID.",
        )

    # Raw ID (no URL scheme)
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw):
        return ("unknown", raw)

    raise HTTPException(
        status_code=400,
        detail="Invalid input. Please provide a Google Drive URL or ID.",
    )


def _map_drive_error(status: int, body: dict) -> str:
    """Return a user-friendly error message for a Drive API error."""
    if status == 401:
        return "Google Drive API key is invalid or expired. Contact the administrator."
    if status == 403:
        reason = ""
        errors = body.get("error", {}).get("errors", [])
        if errors:
            reason = errors[0].get("reason", "")
        if reason == "notFound" or reason == "fileNotFound":
            return (
                "File or folder not found, or it is not shared publicly. "
                "Make sure the file/folder is shared with 'Anyone with the link'."
            )
        return (
            "Access denied. Make sure the file or folder is shared "
            "with 'Anyone with the link'."
        )
    if status == 404:
        return "Not found on Google Drive. Check the URL or ID."
    return f"Google Drive error (HTTP {status})."


async def _drive_get_file_info(file_id: str, api_key: str) -> Optional[dict]:
    """Get metadata for a single Drive file. Returns None on 404."""
    url = f"{DRIVE_API_BASE}/files/{file_id}"
    params = {"key": api_key, "fields": "id,name,mimeType,size"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code == 404:
        return None
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    raise HTTPException(status_code=resp.status_code, detail=_map_drive_error(resp.status_code, body))


async def _drive_list_folder(folder_id: str, api_key: str) -> list[dict]:
    """List all files in a Drive folder (auto-paginates)."""
    all_files: list[dict] = []
    page_token: Optional[str] = None

    async with httpx.AsyncClient(timeout=15) as client:
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "key": api_key,
                "fields": "nextPageToken,files(id,name,mimeType,size)",
                "pageSize": "100",
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(f"{DRIVE_API_BASE}/files", params=params)

            if resp.status_code != 200:
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=_map_drive_error(resp.status_code, body),
                )

            data = resp.json()
            all_files.extend(data.get("files", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break

    return all_files


VALID_IMAGE_EXTENSIONS_FOR_ANALYSIS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _is_geotiff(f: dict) -> bool:
    name = f.get("name", "").lower()
    return name.endswith(".tif") or name.endswith(".tiff") or f.get("mimeType") == "image/tiff"


def _is_supported_image(f: dict) -> bool:
    """Accept any image format supported by the Plant Counter model."""
    name = f.get("name", "").lower()
    return any(name.endswith(ext) for ext in VALID_IMAGE_EXTENSIONS_FOR_ANALYSIS)


# ---------------------------------------------------------------------------
# Separate router for Plant Counter drive helpers (no GeoTIFF-only filter)
# ---------------------------------------------------------------------------

analyze_router = APIRouter(prefix="/analyze/drive", tags=["analyze-drive"])


class AnalyzeDriveListRequest(BaseModel):
    input: str  # Any Google Drive URL, folder URL, file URL, or raw ID


@analyze_router.post("/list-files")
async def list_files_for_analysis(req: AnalyzeDriveListRequest):
    """Resolve a Drive URL/ID to an image file list for Plant Counter.

    Unlike /imagery/drive/list-files this does NOT filter to GeoTIFF only —
    it accepts .tif, .tiff, .png, .jpg, .jpeg so that Plant Counter can work
    on regular aerial/field photos in addition to GeoTIFFs.

    The drone imagery list-files endpoint is unchanged and unaffected.
    """
    api_key = _get_api_key()
    kind, drive_id = _parse_drive_input(req.input)

    files: list[dict] = []
    resolved_type = kind

    if kind == "file":
        info = await _drive_get_file_info(drive_id, api_key)
        if info is None:
            raise HTTPException(
                status_code=404,
                detail="File not found on Google Drive. Check the URL or ID.",
            )
        files = [info]
        resolved_type = "file"

    elif kind == "folder":
        files = await _drive_list_folder(drive_id, api_key)
        resolved_type = "folder"

    else:
        # Unknown — try file first, then folder
        info = await _drive_get_file_info(drive_id, api_key)
        if info is not None:
            files = [info]
            resolved_type = "file"
        else:
            try:
                files = await _drive_list_folder(drive_id, api_key)
                resolved_type = "folder"
            except HTTPException:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Could not find a file or folder with this ID on Google Drive. "
                        "Make sure the link is correct and the content is shared publicly."
                    ),
                )

    image_files = [f for f in files if _is_supported_image(f)]
    skipped = len(files) - len(image_files)

    if not image_files:
        raise HTTPException(
            status_code=404,
            detail=(
                "No compatible image files found at that Drive link. "
                "Supported formats: .tif, .tiff, .png, .jpg, .jpeg"
            ),
        )

    return {
        "type": resolved_type,
        "files": [
            {
                "id": f["id"],
                "name": f["name"],
                "size": int(f.get("size", 0)),
                "mimeType": f.get("mimeType", ""),
            }
            for f in image_files
        ],
        "skipped_count": skipped,
    }


# ---------------------------------------------------------------------------
# Endpoints (drone imagery — GeoTIFF only, unchanged)
# ---------------------------------------------------------------------------

@router.post("/list-files")
async def list_files(req: ListFilesRequest):
    """List GeoTIFF files available at the given Drive link/ID.

    Accepts any Google Drive URL (file link, folder link, share link) or raw ID.
    Returns the list of GeoTIFF files found, along with count of skipped non-GeoTIFF files.
    """
    api_key = _get_api_key()
    kind, drive_id = _parse_drive_input(req.input)

    files: list[dict] = []
    resolved_type = kind

    if kind == "file":
        info = await _drive_get_file_info(drive_id, api_key)
        if info is None:
            raise HTTPException(status_code=404, detail="File not found on Google Drive. Check the URL or ID.")
        files = [info]
        resolved_type = "file"

    elif kind == "folder":
        files = await _drive_list_folder(drive_id, api_key)
        resolved_type = "folder"

    else:
        # Unknown — try file first, then folder
        info = await _drive_get_file_info(drive_id, api_key)
        if info is not None:
            files = [info]
            resolved_type = "file"
        else:
            try:
                files = await _drive_list_folder(drive_id, api_key)
                resolved_type = "folder"
            except HTTPException:
                raise HTTPException(
                    status_code=404,
                    detail="Could not find a file or folder with this ID on Google Drive. "
                           "Make sure the link is correct and the content is shared publicly.",
                )

    geotiff_files = [f for f in files if _is_geotiff(f)]
    skipped = len(files) - len(geotiff_files)

    return {
        "type": resolved_type,
        "files": [
            {
                "id": f["id"],
                "name": f["name"],
                "size": int(f.get("size", 0)),
                "mimeType": f.get("mimeType", ""),
            }
            for f in geotiff_files
        ],
        "skipped_count": skipped,
    }


@router.post("/import")
async def import_from_drive(req: ImportRequest):
    """Download a file from Google Drive directly to the server and register it.

    Returns a streaming NDJSON response with real-time progress updates:
      {"phase": "downloading", "progress": 45}
      {"phase": "registering", "progress": 100}
      {"phase": "complete", "result": {...}}
    Or on error:
      {"phase": "error", "message": "..."}
    """
    api_key = _get_api_key()

    # Validate inputs
    lt = req.layer_type.lower()
    if lt not in VALID_LAYER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer_type '{lt}'. Must be one of: {sorted(VALID_LAYER_TYPES)}",
        )
    try:
        datetime.strptime(req.flight_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="flight_date must be in YYYY-MM-DD format")

    async def _stream():
        tmp_path = None
        try:
            # Phase 1: Download from Google Drive
            download_url = f"{DRIVE_API_BASE}/files/{req.file_id}?alt=media&key={api_key}"

            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=600)) as client:
                async with client.stream("GET", download_url) as resp:
                    if resp.status_code != 200:
                        # Try to read error body
                        error_body = b""
                        async for chunk in resp.aiter_bytes():
                            error_body += chunk
                            if len(error_body) > 4096:
                                break
                        try:
                            body = json.loads(error_body)
                        except Exception:
                            body = {}
                        msg = _map_drive_error(resp.status_code, body)
                        yield json.dumps({"phase": "error", "message": msg}) + "\n"
                        return

                    total_size = int(resp.headers.get("content-length", 0))
                    received = 0

                    # Stream to a temp file first
                    filename = build_filename(req.farm_id, req.flight_date, lt)
                    tmp_path = os.path.join(settings.imagery_dir, f".tmp_{filename}")
                    dest_path = os.path.join(settings.imagery_dir, filename)

                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                            received += len(chunk)
                            if total_size > 0:
                                pct = min(99, int(received * 100 / total_size))
                            else:
                                pct = min(99, received // (1024 * 1024))  # fallback: MB count
                            yield json.dumps({"phase": "downloading", "progress": pct}) + "\n"

            # Validate file is non-empty
            file_size = os.path.getsize(tmp_path)
            if file_size == 0:
                yield json.dumps({"phase": "error", "message": "Downloaded file is empty."}) + "\n"
                return

            # Move temp file to final destination
            shutil.move(tmp_path, dest_path)
            tmp_path = None  # prevent cleanup

            # Phase 2: Register in DB
            yield json.dumps({"phase": "registering", "progress": 100}) + "\n"

            result = await register_imagery_in_db(
                filename=filename,
                total_bytes=file_size,
                farm_id=req.farm_id,
                flight_date=req.flight_date,
                layer_type=lt,
                band_mapping=req.band_mapping,
                pilot_name=req.pilot_name,
                drone_model=req.drone_model,
                altitude=req.altitude,
            )

            yield json.dumps({"phase": "complete", "result": result}) + "\n"

        except HTTPException as e:
            yield json.dumps({"phase": "error", "message": e.detail}) + "\n"
        except Exception as e:
            yield json.dumps({"phase": "error", "message": str(e)}) + "\n"
        finally:
            # Clean up temp file on failure
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )
