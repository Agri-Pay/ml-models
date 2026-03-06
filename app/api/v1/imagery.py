"""Server-side drone imagery storage API.

Replaces Supabase Storage — files are stored on the server's disk
and metadata is written to Supabase Postgres tables.

Endpoints:
  POST   /api/v1/imagery/upload            — upload a GeoTIFF with metadata
  GET    /api/v1/imagery/{farm_id}         — list imagery for a farm
  DELETE /api/v1/imagery/{farm_id}/{filename} — delete an image file + DB records
"""

import os
import shutil
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings

router = APIRouter(prefix="/imagery", tags=["imagery"])

# Valid layer types
VALID_LAYER_TYPES = {
    "rgb", "ndvi", "ndre", "moisture", "thermal",
    "lai", "gndvi", "savi", "nir", "red_edge",
}


def _get_supabase():
    """Late import to avoid import-time failure when Supabase is not configured."""
    from app.db.supabase_client import get_supabase
    return get_supabase()


def _build_filename(farm_id: str, flight_date: str, layer_type: str) -> str:
    """Build a canonical filename: {farmId}_{YYYYMMDD}_{layerType}.tif"""
    date_str = flight_date.replace("-", "")
    return f"{farm_id}_{date_str}_{layer_type}.tif"


# ---------------------------------------------------------------------------
# POST /api/v1/imagery/upload
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_imagery(
    file: UploadFile = File(...),
    farm_id: str = Form(...),
    flight_date: str = Form(...),          # YYYY-MM-DD
    layer_type: str = Form("rgb"),
    pilot_name: Optional[str] = Form(None),
    drone_model: Optional[str] = Form(None),
    altitude: Optional[float] = Form(None),
):
    """Upload a GeoTIFF drone image and register it in the database.

    The file is stored at {imagery_dir}/{filename} where both TiTiler
    and the compute backend can access it via the shared /data volume.
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
    filename = _build_filename(farm_id, flight_date, layer_type)
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

    # --- Create/update DB records ---
    flight_id = None
    layer_data = None

    try:
        supabase = _get_supabase()

        # 1. Get or create drone_flight record
        existing = (
            supabase.table("drone_flights")
            .select("id")
            .eq("farm_id", farm_id)
            .eq("flight_date", flight_date)
            .execute()
        )

        if existing.data:
            flight_id = existing.data[0]["id"]
            # Update optional fields if provided
            updates = {"storage_location": "local"}
            if pilot_name:
                updates["pilot_name"] = pilot_name
            if drone_model:
                updates["drone_model"] = drone_model
            if altitude is not None:
                updates["altitude_meters"] = altitude
            supabase.table("drone_flights").update(updates).eq("id", flight_id).execute()
        else:
            insert_data = {
                "farm_id": farm_id,
                "flight_date": flight_date,
                "storage_location": "local",
            }
            if pilot_name:
                insert_data["pilot_name"] = pilot_name
            if drone_model:
                insert_data["drone_model"] = drone_model
            if altitude is not None:
                insert_data["altitude_meters"] = altitude

            result = (
                supabase.table("drone_flights")
                .insert(insert_data)
                .execute()
            )
            flight_id = result.data[0]["id"]

        # 2. Upsert drone_imagery_layers record
        layer_record = {
            "flight_id": flight_id,
            "layer_type": layer_type,
            "filename": filename,
            "file_size_bytes": total_bytes,
        }

        # Try to get metadata from TiTiler (sibling container at localhost:8000)
        try:
            import httpx
            titiler_url = os.environ.get("TITILER_INTERNAL_URL", "http://titiler:8000")
            file_url = f"file:///data/{filename}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                info_resp = await client.get(
                    f"{titiler_url}/cog/info",
                    params={"url": file_url},
                )
                if info_resp.status_code == 200:
                    info = info_resp.json()
                    layer_record["crs"] = info.get("crs")
                    layer_record["bounds"] = info.get("bounds")

                stats_resp = await client.get(
                    f"{titiler_url}/cog/statistics",
                    params={"url": file_url},
                )
                if stats_resp.status_code == 200:
                    layer_record["statistics"] = stats_resp.json()
        except Exception as e:
            print(f"TiTiler metadata extraction skipped: {e}")

        # Check if layer already exists for this flight+type
        existing_layer = (
            supabase.table("drone_imagery_layers")
            .select("id")
            .eq("flight_id", flight_id)
            .eq("layer_type", layer_type)
            .execute()
        )

        if existing_layer.data:
            # Update existing
            layer_result = (
                supabase.table("drone_imagery_layers")
                .update(layer_record)
                .eq("id", existing_layer.data[0]["id"])
                .execute()
            )
            layer_data = layer_result.data[0] if layer_result.data else None
        else:
            # Insert new
            layer_result = (
                supabase.table("drone_imagery_layers")
                .insert(layer_record)
                .execute()
            )
            layer_data = layer_result.data[0] if layer_result.data else None

    except HTTPException:
        raise
    except Exception as exc:
        # File saved successfully but DB failed — log but don't delete the file
        print(f"Warning: File saved to {dest_path} but DB insert failed: {exc}")
        return JSONResponse(
            status_code=207,  # Multi-Status
            content={
                "success": True,
                "warning": f"File saved but database update failed: {str(exc)}",
                "filename": filename,
                "file_size_bytes": total_bytes,
                "flight_id": flight_id,
            },
        )

    return {
        "success": True,
        "flight": {
            "id": flight_id,
            "date": flight_date,
        },
        "layer": layer_data,
        "filename": filename,
        "file_size_bytes": total_bytes,
    }


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

        # Find the layer
        layer = (
            supabase.table("drone_imagery_layers")
            .select("id, flight_id")
            .eq("filename", filename)
            .execute()
        )

        if layer.data:
            layer_id = layer.data[0]["id"]
            flight_id = layer.data[0]["flight_id"]

            # Delete the layer
            supabase.table("drone_imagery_layers").delete().eq("id", layer_id).execute()

            # If this was the last layer for the flight, delete the flight too
            remaining = (
                supabase.table("drone_imagery_layers")
                .select("id")
                .eq("flight_id", flight_id)
                .execute()
            )
            if not remaining.data:
                supabase.table("drone_flights").delete().eq("id", flight_id).execute()

    except Exception as exc:
        print(f"Warning: File deleted but DB cleanup failed: {exc}")
        return {"success": True, "warning": f"File deleted but DB cleanup failed: {str(exc)}"}

    return {"success": True}
