"""GET /api/jobs/{job_id} — poll status.
POST /api/jobs/{job_id}/reprocess — re-apply custom params without re-running inference.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core import job_queue, storage
from app.pipeline import depth_estimator, style_variants
from app.schemas.job import JobStatus
from app.schemas.processing import ReprocessParams

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "complete":
        raise HTTPException(status_code=409, detail=f"Job not complete (status: {job.status}).")
    return job.result


@router.post("/jobs/{job_id}/reprocess")
async def reprocess(job_id: str, params: ReprocessParams):
    """Re-run post-processing on cached raw depth tensor with new user params.

    Fast (~200ms) — skips GPU inference entirely.
    """
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "complete":
        raise HTTPException(
            status_code=409,
            detail="Job must be complete before reprocessing.",
        )

    raw_depth_path = storage.raw_depth_path(job_id)
    if not raw_depth_path.exists():
        raise HTTPException(status_code=404, detail="Raw depth data not found.")

    loop = asyncio.get_event_loop()

    raw_depth = await loop.run_in_executor(
        None, depth_estimator.load_raw_depth, job_id
    )

    # apply_custom_variant uses keyword-only args; wrap in lambda for run_in_executor
    processed = await loop.run_in_executor(
        None,
        lambda: style_variants.apply_custom_variant(
            raw_depth,
            depth_intensity=params.depth_intensity,
            blur_radius=params.blur_radius,
            contrast=params.contrast,
            edge_enhancement=params.edge_enhancement,
            invert=params.invert,
        ),
    )

    variant_name = params.variant
    out_path = storage.variant_path(job_id, f"{variant_name}_custom")
    await loop.run_in_executor(None, processed.save, str(out_path), "PNG")

    return {
        "preview_url": f"/api/export/{job_id}/{variant_name}_custom",
    }
