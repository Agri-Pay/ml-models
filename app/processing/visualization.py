"""Visualization utilities for plant counting results."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.cm


VIZ_MAX_DIM = 2048


def generate_all_visualizations(
    job_id: str,
    result: Dict,
    heatmap: np.ndarray,
    image_path: Optional[str],
    temp_store,
) -> List[str]:
    """Generate all visualization PNGs for a completed inference job.

    Args:
        job_id: The job ID (for temp_store directory)
        result: Inference result dict (must have 'points', 'plant_sizes',
                'image_width', 'image_height')
        heatmap: The density heatmap as a numpy array
        image_path: Path to the original image for overlays (may be None)
        temp_store: TempResultStore instance

    Returns:
        List of output filenames
    """
    output_files = []
    points_raw = result.get("points", [])
    if not points_raw:
        return output_files

    job_dir = temp_store.get_job_dir(job_id)

    # Disable PIL decompression bomb check — drone imagery is trusted and large
    Image.MAX_IMAGE_PIXELS = None

    W = result.get("image_width", 1024)
    H = result.get("image_height", 1024)

    # Load original image for overlays, capped at VIZ_MAX_DIM
    scale = 1.0
    if image_path and os.path.exists(image_path):
        try:
            base_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = base_img.size
            scale = min(1.0, VIZ_MAX_DIM / max(orig_w, orig_h))
            if scale < 1.0:
                viz_w = max(1, int(orig_w * scale))
                viz_h = max(1, int(orig_h * scale))
                base_img = base_img.resize((viz_w, viz_h), Image.LANCZOS)
                print(f"  Viz: downsampled {orig_w}x{orig_h} -> {viz_w}x{viz_h} (scale={scale:.3f})")
        except Exception as exc:
            print(f"  Viz: could not open image ({exc}), using blank canvas")
            scale = min(1.0, VIZ_MAX_DIM / max(W, H))
            viz_w = max(1, int(W * scale))
            viz_h = max(1, int(H * scale))
            base_img = Image.new("RGB", (viz_w, viz_h), (40, 40, 40))
    else:
        scale = min(1.0, VIZ_MAX_DIM / max(W, H))
        viz_w = max(1, int(W * scale))
        viz_h = max(1, int(H * scale))
        base_img = Image.new("RGB", (viz_w, viz_h), (40, 40, 40))

    # Scale point coordinates and sizes to match visualization resolution
    points = [
        {
            "x": int(p["x"] * scale),
            "y": int(p["y"] * scale),
            "base_sigma": p.get("base_sigma", 0) * scale,
        }
        for p in points_raw
    ]
    plant_sizes = [s * scale for s in result.get("plant_sizes", [])]

    # Counting overlay
    overlay = render_counting_overlay(base_img, points)
    overlay.save(f"{job_dir}/counting_overlay.png")
    output_files.append("counting_overlay.png")

    # Heatmap
    heatmap_img = render_heatmap(heatmap)
    heatmap_img.save(f"{job_dir}/density_heatmap.png")
    output_files.append("density_heatmap.png")

    if plant_sizes:
        # Size annotated
        annot = render_size_annotated(base_img, points, plant_sizes)
        annot.save(f"{job_dir}/size_annotated.png")
        output_files.append("size_annotated.png")

        # Color coded
        color = render_size_color_coded(base_img, points, plant_sizes)
        color.save(f"{job_dir}/size_color_coded.png")
        output_files.append("size_color_coded.png")

    return output_files


def render_counting_overlay(
    base_image: Image.Image,
    points: List[Dict],
    dot_radius: int = 5,
    dot_color: Tuple[int, ...] = (255, 0, 0, 128),
) -> Image.Image:
    """Draw detected plant locations as dots on the image."""
    overlay = base_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    for p in points:
        x, y = p["x"], p["y"]
        draw.ellipse(
            (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            fill=dot_color,
        )
    # Add count text
    font = _get_font(32)
    count_text = f"Total Count: {len(points)}"
    draw.text((20, 20), count_text, fill=(255, 255, 255, 255), font=font)
    return overlay.convert("RGB")


def render_size_annotated(
    base_image: Image.Image,
    points: List[Dict],
    plant_sizes: List[float],
) -> Image.Image:
    """Draw plant bounding boxes with size annotations."""
    annot = base_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annot, "RGBA")
    font_small = _get_font(16)
    font_large = _get_font(32)

    for p, size in zip(points, plant_sizes):
        x, y = p["x"], p["y"]
        half = size / 2
        draw.rectangle(
            [x - half, y - half, x + half, y + half],
            outline=(255, 0, 0, 255),
            width=1,
        )
        draw.text(
            (x, y), f"{size:.1f}",
            fill=(255, 255, 0, 255), font=font_small, anchor="mm",
        )

    if plant_sizes:
        avg = np.mean(plant_sizes)
        draw.text((20, 20), f"Average Size: {avg:.2f}", fill=(255, 255, 255, 255), font=font_large)

    return annot.convert("RGB")


def render_size_color_coded(
    base_image: Image.Image,
    points: List[Dict],
    plant_sizes: List[float],
) -> Image.Image:
    """Draw plants color-coded by size: red=small, orange=medium, yellow=large."""
    color_img = base_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(color_img, "RGBA")

    if len(plant_sizes) > 1:
        lower_q = float(np.percentile(plant_sizes, 25))
        upper_q = float(np.percentile(plant_sizes, 75))
    elif len(plant_sizes) == 1:
        lower_q = upper_q = plant_sizes[0]
    else:
        return color_img.convert("RGB")

    for p, size in zip(points, plant_sizes):
        x, y = p["x"], p["y"]
        if size < lower_q:
            color = (255, 0, 0, 255)
        elif size > upper_q:
            color = (255, 255, 0, 255)
        else:
            color = (255, 165, 0, 255)
        half = size / 2
        draw.rectangle([x - half, y - half, x + half, y + half], outline=color, width=1)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    return color_img.convert("RGB")


def render_heatmap(heatmap: np.ndarray) -> Image.Image:
    """Render a density heatmap as a jet-colored image."""
    h_min = np.nanmin(heatmap)
    h_max = np.nanmax(heatmap)
    denom = h_max - h_min
    if denom < 1e-8:
        denom = 1.0
    norm_heat = (heatmap - h_min) / denom
    cmap = matplotlib.colormaps["jet"]
    heat_rgba = (cmap(norm_heat) * 255).astype(np.uint8)
    return Image.fromarray(heat_rgba).convert("RGB")


def _get_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """Try to load a TrueType font, falling back to default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return None
