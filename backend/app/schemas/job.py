from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class JobStep(str, Enum):
    queued = "queued"
    removing_background = "removing_background"
    estimating_depth = "estimating_depth"
    generating_variants = "generating_variants"
    done = "done"


VARIANT_NAMES = ["soft", "standard", "detailed", "sharp"]


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | complete | failed
    step: JobStep = JobStep.queued
    progress: float = 0.0
    error: str | None = None
    result: Any | None = None
    created_at: datetime
    completed_at: datetime | None = None


class JobResult(BaseModel):
    job_id: str
    variants: dict[str, str]  # variant_name → URL
    original_preview: str
    metadata: dict[str, Any]
