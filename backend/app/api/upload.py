"""POST /api/upload — accept an image, enqueue a depth processing job."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.core import job_queue
from app.schemas.upload import UploadResponse
from app.workers.depth_worker import run_depth_job

router = APIRouter()

_ALLOWED_MIME = {"image/jpeg", "image/png", "image/svg+xml", "image/webp"}


_ALLOWED_IMAGE_TYPES = {"photo", "illustration"}


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_image(
    file: UploadFile = File(...),
    remove_background: bool = Form(default=False),
    image_type: str = Form(default="photo"),
    use_marigold: bool = Form(default=False),
):
    settings = get_settings()

    # Validate MIME type
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Accepted: JPEG, PNG, SVG, WebP.",
        )

    if image_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid image_type '{image_type}'. Must be 'photo' or 'illustration'.",
        )

    # Read + validate size
    image_bytes = await file.read()
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {settings.max_upload_mb} MB.",
        )

    job_id = str(uuid.uuid4())

    # Capture variables in closure for the coroutine factory
    _bytes = image_bytes
    _mime = content_type
    _rembg = remove_background
    _image_type = image_type
    _marigold = use_marigold

    async def coro_factory():
        await run_depth_job(job_id, _bytes, _mime, _rembg, _image_type, _marigold)

    await job_queue.enqueue(job_id, coro_factory)

    estimated = 60 if use_marigold else 20
    return UploadResponse(job_id=job_id, status="queued", estimated_seconds=estimated)
