"""GET /api/export/{job_id}/{variant} — stream a depth map PNG.

Optional query params for on-the-fly re-rendering (uses cached raw depth):
  depth_intensity, blur_radius, contrast, edge_enhancement, invert, upscale
"""

from __future__ import annotations

import asyncio
import io

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core import storage

router = APIRouter()

_VARIANT_NAMES = {"soft", "standard", "detailed", "sharp", "original_preview"}


@router.get("/export/{job_id}/{variant}")
async def export_variant(
    job_id: str,
    variant: str,
    upscale: bool = Query(default=False),
    # Optional live re-render params (None = use saved file as-is)
    depth_intensity: float | None = Query(default=None, ge=0.0, le=2.0),
    blur_radius: float | None = Query(default=None, ge=0.0, le=10.0),
    contrast: float | None = Query(default=None, ge=0.0, le=3.0),
    edge_enhancement: float | None = Query(default=None, ge=0.0, le=1.0),
    invert: bool = Query(default=False),
):
    loop = asyncio.get_event_loop()

    # Allow custom variant names (e.g. "standard_custom")
    if variant == "original_preview":
        path = storage.original_preview_path(job_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Preview not found.")
        return _stream_file(path, "original_preview.jpg", "image/jpeg")

    # Check if any live re-render params were supplied
    wants_rerender = any(
        p is not None for p in [depth_intensity, blur_radius, contrast, edge_enhancement]
    )

    if wants_rerender:
        img = await _rerender(
            job_id,
            variant,
            loop,
            depth_intensity=depth_intensity or 1.0,
            blur_radius=blur_radius or 1.0,
            contrast=contrast or 1.1,
            edge_enhancement=edge_enhancement or 0.3,
            invert=invert,
        )
    else:
        path = storage.variant_path(job_id, variant)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Variant '{variant}' not found.")
        from PIL import Image
        img = await loop.run_in_executor(None, Image.open, str(path))

    if upscale:
        from app.pipeline.upscaler import upscale as do_upscale
        img = await loop.run_in_executor(None, do_upscale, img)

    buf = io.BytesIO()
    await loop.run_in_executor(None, lambda: img.save(buf, "PNG"))
    buf.seek(0)

    filename = f"depthmap_{variant}.png"
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _rerender(job_id, variant, loop, **params):
    from app.pipeline import depth_estimator, style_variants

    raw_depth = await loop.run_in_executor(
        None, depth_estimator.load_raw_depth, job_id
    )
    img = await loop.run_in_executor(
        None,
        style_variants.apply_custom_variant,
        raw_depth,
        params["depth_intensity"],
        params["blur_radius"],
        params["contrast"],
        params["edge_enhancement"],
        params["invert"],
    )
    return img


def _stream_file(path, filename: str, media_type: str):
    import aiofiles

    async def iterfile():
        async with aiofiles.open(str(path), "rb") as f:
            while chunk := await f.read(65536):
                yield chunk

    return StreamingResponse(
        iterfile(),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
