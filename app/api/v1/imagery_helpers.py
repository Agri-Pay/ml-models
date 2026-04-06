"""Shared helpers for drone imagery DB registration.

Used by both the single-POST upload endpoint (imagery.py) and the
chunked upload endpoint (chunked_upload.py).
"""

import json
import os
from typing import Optional

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.config import settings


# Valid layer types (single-band layers + multispectral)
VALID_LAYER_TYPES = {
    "rgb", "ndvi", "ndre", "moisture", "thermal",
    "lai", "gndvi", "savi", "nir", "red_edge",
    "multispectral",
}


def _get_supabase():
    """Late import to avoid import-time failure when Supabase is not configured."""
    from app.db.supabase_client import get_supabase
    return get_supabase()


def build_filename(farm_id: str, flight_date: str, layer_type: str) -> str:
    """Build a canonical filename: {farmId}_{YYYYMMDD}_{layerType}.tif"""
    date_str = flight_date.replace("-", "")
    return f"{farm_id}_{date_str}_{layer_type}.tif"


async def _get_titiler_info(filename: str) -> dict:
    """Query TiTiler for raster metadata (CRS, bounds, statistics, band count).

    Returns a dict with keys: crs, bounds, statistics, band_count.
    Any key may be None if TiTiler is unreachable or the query fails.
    """
    result = {"crs": None, "bounds": None, "statistics": None, "band_count": None}
    try:
        titiler_url = os.environ.get("TITILER_INTERNAL_URL", "http://titiler:8000")
        file_url = f"file:///data/{filename}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            info_resp = await client.get(
                f"{titiler_url}/cog/info",
                params={"url": file_url},
            )
            if info_resp.status_code == 200:
                info = info_resp.json()
                result["crs"] = info.get("crs")
                result["bounds"] = info.get("bounds")
                result["band_count"] = info.get("count")

            stats_resp = await client.get(
                f"{titiler_url}/cog/statistics",
                params={"url": file_url},
            )
            if stats_resp.status_code == 200:
                result["statistics"] = stats_resp.json()
    except Exception as e:
        print(f"TiTiler metadata extraction skipped: {e}")

    return result


async def register_imagery_in_db(
    filename: str,
    total_bytes: int,
    farm_id: str,
    flight_date: str,
    layer_type: str,
    band_mapping: Optional[str] = None,
    pilot_name: Optional[str] = None,
    drone_model: Optional[str] = None,
    altitude: Optional[float] = None,
) -> dict:
    """Register an uploaded imagery file in the database.

    Creates/updates drone_flights and drone_imagery_layers records.
    For multispectral uploads, creates one layer record per band.

    Returns the response dict on success, or a JSONResponse on partial failure.
    """
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

        # 2. Get TiTiler metadata
        titiler_info = await _get_titiler_info(filename)

        # 3. Create imagery layer records
        if layer_type == "multispectral":
            layer_data = await _register_multispectral_layers(
                supabase, flight_id, filename, total_bytes,
                titiler_info, band_mapping,
            )
        else:
            layer_data = await _register_single_layer(
                supabase, flight_id, filename, total_bytes,
                layer_type, titiler_info,
            )

    except HTTPException:
        raise
    except Exception as exc:
        dest_path = os.path.join(settings.imagery_dir, filename)
        print(f"Warning: File saved to {dest_path} but DB insert failed: {exc}")
        return JSONResponse(
            status_code=207,
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


async def _register_single_layer(
    supabase, flight_id, filename, total_bytes, layer_type, titiler_info,
):
    """Create or update a single drone_imagery_layers record."""
    layer_record = {
        "flight_id": flight_id,
        "layer_type": layer_type,
        "filename": filename,
        "file_size_bytes": total_bytes,
    }

    if titiler_info["crs"]:
        layer_record["crs"] = titiler_info["crs"]
    if titiler_info["bounds"]:
        layer_record["bounds"] = titiler_info["bounds"]
    if titiler_info["statistics"]:
        layer_record["statistics"] = titiler_info["statistics"]

    existing_layer = (
        supabase.table("drone_imagery_layers")
        .select("id")
        .eq("flight_id", flight_id)
        .eq("layer_type", layer_type)
        .execute()
    )

    if existing_layer.data:
        layer_result = (
            supabase.table("drone_imagery_layers")
            .update(layer_record)
            .eq("id", existing_layer.data[0]["id"])
            .execute()
        )
        return layer_result.data[0] if layer_result.data else None
    else:
        layer_result = (
            supabase.table("drone_imagery_layers")
            .insert(layer_record)
            .execute()
        )
        return layer_result.data[0] if layer_result.data else None


async def _register_multispectral_layers(
    supabase, flight_id, filename, total_bytes, titiler_info, band_mapping_json,
):
    """Create one drone_imagery_layers record per band for a multispectral upload.

    If band_mapping_json is provided, uses the user-supplied band names/wavelengths.
    Otherwise, auto-creates generic band records based on TiTiler band count.
    """
    bands = []

    if band_mapping_json:
        try:
            bands = json.loads(band_mapping_json)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(
                status_code=400,
                detail="Invalid band_mapping JSON format. Expected a JSON array of "
                       '{"band_number": int, "band_name": str, "wavelength": str}',
            )

        if not isinstance(bands, list) or len(bands) == 0:
            raise HTTPException(
                status_code=400,
                detail="band_mapping must be a non-empty JSON array",
            )

        for b in bands:
            if "band_number" not in b or "band_name" not in b:
                raise HTTPException(
                    status_code=400,
                    detail="Each band must have at least 'band_number' and 'band_name'",
                )
    else:
        # Auto-detect band count from TiTiler
        band_count = titiler_info.get("band_count")
        if band_count and band_count > 0:
            bands = [
                {
                    "band_number": i + 1,
                    "band_name": f"Band {i + 1}",
                    "wavelength": None,
                }
                for i in range(band_count)
            ]
        else:
            # Fallback: create a single generic record
            bands = [{"band_number": 1, "band_name": "Band 1", "wavelength": None}]

    # Delete existing band layers for this flight+filename to avoid duplicates
    existing_layers = (
        supabase.table("drone_imagery_layers")
        .select("id")
        .eq("flight_id", flight_id)
        .eq("filename", filename)
        .execute()
    )
    for existing in (existing_layers.data or []):
        supabase.table("drone_imagery_layers").delete().eq("id", existing["id"]).execute()

    # Insert one record per band
    inserted_layers = []
    for band in bands:
        band_name = band["band_name"]
        # Derive layer_type from band name: lowercase, replace spaces with underscore
        derived_layer_type = band_name.lower().replace(" ", "_").replace("-", "_")

        layer_record = {
            "flight_id": flight_id,
            "layer_type": derived_layer_type,
            "filename": filename,
            "file_size_bytes": total_bytes,
            "is_band": True,
            "band_number": band["band_number"],
            "band_name": band_name,
            "band_wavelength": band.get("wavelength"),
        }

        if titiler_info["crs"]:
            layer_record["crs"] = titiler_info["crs"]
        if titiler_info["bounds"]:
            layer_record["bounds"] = titiler_info["bounds"]
        if titiler_info["statistics"]:
            layer_record["statistics"] = titiler_info["statistics"]

        layer_result = (
            supabase.table("drone_imagery_layers")
            .insert(layer_record)
            .execute()
        )
        if layer_result.data:
            inserted_layers.append(layer_result.data[0])

    return inserted_layers
