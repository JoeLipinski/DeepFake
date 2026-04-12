"""Depth processing worker — orchestrates the full pipeline for a single job.

Called as a coroutine factory by the job queue consumer.
Sequence:
  1. Load image from disk (saved by upload endpoint)
  2. Optional SVG rasterization
  3. Optional background removal
  4. Depth estimation (GPU) → saves raw_depth.npy
  5. Variant generation (CPU, parallel)
  6. Save variant PNGs
  7. Update job status → complete
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.core import job_queue, storage
from app.schemas.job import JobStep

logger = logging.getLogger(__name__)


async def run_depth_job(
    job_id: str,
    image_bytes: bytes,
    mime_type: str,
    remove_background: bool,
) -> None:
    loop = asyncio.get_event_loop()
    start = time.perf_counter()

    job_queue.update_job(job_id, status="running", step=JobStep.queued, progress=0.05)

    # --- Step 1: Decode image ---
    image = await loop.run_in_executor(None, _decode_image, image_bytes, mime_type)
    storage.ensure_job_dir(job_id)

    # Save original preview (JPEG, smaller)
    preview_path = storage.original_preview_path(job_id)
    await loop.run_in_executor(None, _save_preview, image, preview_path)

    # --- Step 2: Background removal (optional) ---
    if remove_background:
        job_queue.update_job(
            job_id, step=JobStep.removing_background, progress=0.15
        )
        from app.pipeline.background_remover import remove_background as do_rembg
        image = await loop.run_in_executor(None, do_rembg, image)

    # --- Step 3: Depth estimation ---
    job_queue.update_job(job_id, step=JobStep.estimating_depth, progress=0.25)
    from app.pipeline.depth_estimator import estimate_depth
    raw_depth: np.ndarray = await loop.run_in_executor(
        None, estimate_depth, job_id, image
    )
    job_queue.update_job(job_id, progress=0.70)

    # --- Step 4: Generate style variants ---
    job_queue.update_job(job_id, step=JobStep.generating_variants, progress=0.75)
    from app.pipeline.style_variants import generate_all_variants
    variants = await loop.run_in_executor(None, generate_all_variants, raw_depth)

    # --- Step 5: Save variant PNGs ---
    for name, img in variants.items():
        path = storage.variant_path(job_id, name)
        await loop.run_in_executor(None, img.save, str(path), "PNG")

    elapsed = time.perf_counter() - start
    w, h = image.size

    result = {
        "variants": {
            name: f"/api/export/{job_id}/{name}"
            for name in variants
        },
        "original_preview": f"/api/export/{job_id}/original_preview",
        "metadata": {
            "width": w,
            "height": h,
            "processing_time_seconds": round(elapsed, 1),
        },
    }

    job_queue.update_job(
        job_id,
        status="complete",
        step=JobStep.done,
        progress=1.0,
        result=result,
    )
    logger.info("Job %s complete in %.1fs", job_id, elapsed)


def _decode_image(image_bytes: bytes, mime_type: str) -> Image.Image:
    if mime_type == "image/svg+xml":
        from app.pipeline.svg_rasterizer import rasterize_svg
        return rasterize_svg(image_bytes)

    image = Image.open(io.BytesIO(image_bytes))
    # Normalise to RGB (handles RGBA PNGs, palette images, etc.)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def _save_preview(image: Image.Image, path: Path) -> None:
    # Max 1024px on longest side, JPEG for small file size
    w, h = image.size
    max_side = 1024
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    image.convert("RGB").save(str(path), "JPEG", quality=85)
