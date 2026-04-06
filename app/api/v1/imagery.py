"""Server-side drone imagery storage API.

Replaces Supabase Storage — files are stored on the server's disk
and metadata is written to Supabase Postgres tables.

Endpoints:
  POST   /api/v1/imagery/upload            — upload a GeoTIFF with metadata
  GET    /api/v1/imagery/{farm_id}         — list imagery for a farm
  DELETE /api/v1/imagery/{farm_id}/{filename} — delete an image file + DB records
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.api.v1.imagery_helpers import (
    VALID_LAYER_TYPES,
    build_filename,
    register_imagery_in_db,
)

router = APIRouter(prefix="/imagery", tags=["imagery"])


def _get_supabase():
    """Late import to avoid import-time failure when Supabase is not configured."""
    from app.db.supabase_client import get_supabase
    return get_supabase()


# ---------------------------------------------------------------------------
# POST /api/v1/imagery/upload
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_imagery(
    file: UploadFile = File(...),
    farm_id: str = Form(...),
    flight_date: str = Form(...),          # YYYY-MM-DD
    layer_type: str = Form("rgb"),
    band_mapping: Optional[str] = Form(None),
    pilot_name: Optional[str] = Form(None),
    drone_model: Optional[str] = Form(None),
    altitude: Optional[float] = Form(None),
):
    """Upload a GeoTIFF drone image and register it in the database.

    The file is stored at {imagery_dir}/{filename} where both TiTiler
    and the compute backend can access it via the shared /data volume.

    For multispectral uploads, pass layer_type="multispectral" and optionally
    a band_mapping JSON string: [{"band_number": 1, "band_name": "Blue", "wavelength": "475nm"}, ...]
    """
    # --- Validation ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".tif", ".tiff"):
        raise HTTPException(
            status_code=400,
            detail=f"Only GeoTIFF files (.tif/.tiff) are accepted, got '{ext}'",
        )

    layer_type = layer_type.lower()
    if layer_type not in VALID_LAYER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer_type '{layer_type}'. Must be one of: {sorted(VALID_LAYER_TYPES)}",
        )

    # Validate date format
    try:
        datetime.strptime(flight_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="flight_date must be in YYYY-MM-DD format",
        )

    # --- Save file to disk ---
    filename = build_filename(farm_id, flight_date, layer_type)
    dest_path = os.path.join(settings.imagery_dir, filename)

    total_bytes = 0
    try:
        with open(dest_path, "wb") as dst:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    dst.close()
                    os.remove(dest_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {settings.max_upload_bytes // (1024*1024)} MB)",
                    )
                dst.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        # Clean up partial file
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # --- Register in database ---
    return await register_imagery_in_db(
        filename=filename,
        total_bytes=total_bytes,
        farm_id=farm_id,
        flight_date=flight_date,
        layer_type=layer_type,
        band_mapping=band_mapping,
        pilot_name=pilot_name,
        drone_model=drone_model,
        altitude=altitude,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/imagery/{farm_id}
# ---------------------------------------------------------------------------

@router.get("/{farm_id}")
async def list_imagery(farm_id: str):
    """List all drone imagery for a farm.

    Returns flights with their associated imagery layers,
    and whether each file actually exists on disk.
    """
    try:
        supabase = _get_supabase()
        result = (
            supabase.table("drone_flights")
            .select("*, drone_imagery_layers(*)")
            .eq("farm_id", farm_id)
            .order("flight_date", desc=True)
            .execute()
        )

        flights = []
        for flight in result.data or []:
            layers = []
            for layer in flight.get("drone_imagery_layers", []):
                fname = layer.get("filename", "")
                file_path = os.path.join(settings.imagery_dir, fname)
                layers.append({
                    **layer,
                    "file_exists": os.path.exists(file_path),
                })
            flights.append({
                **{k: v for k, v in flight.items() if k != "drone_imagery_layers"},
                "layers": layers,
            })

        return {"farm_id": farm_id, "flights": flights}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# DELETE /api/v1/imagery/{farm_id}/{filename}
# ---------------------------------------------------------------------------

@router.delete("/{farm_id}/{filename}")
async def delete_imagery(farm_id: str, filename: str):
    """Delete a drone image file and its database records."""
    file_path = os.path.join(settings.imagery_dir, filename)

    # Delete from disk
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete from database
    try:
        supabase = _get_supabase()

        # Find all layers for this filename (may be multiple for multispectral)
        layers = (
            supabase.table("drone_imagery_layers")
            .select("id, flight_id")
            .eq("filename", filename)
            .execute()
        )

        flight_ids = set()
        for layer in (layers.data or []):
            flight_ids.add(layer["flight_id"])
            supabase.table("drone_imagery_layers").delete().eq("id", layer["id"]).execute()

        # If any flight now has no layers, delete it too
        for fid in flight_ids:
            remaining = (
                supabase.table("drone_imagery_layers")
                .select("id")
                .eq("flight_id", fid)
                .execute()
            )
            if not remaining.data:
                supabase.table("drone_flights").delete().eq("id", fid).execute()

    except Exception as exc:
        print(f"Warning: File deleted but DB cleanup failed: {exc}")
        return {"success": True, "warning": f"File deleted but DB cleanup failed: {str(exc)}"}

    return {"success": True}
