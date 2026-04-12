"""Local filesystem storage helpers for job artifacts."""

import os
import shutil
from pathlib import Path

from app.config import get_settings


def jobs_root() -> Path:
    return Path(get_settings().jobs_dir)


def job_dir(job_id: str) -> Path:
    return jobs_root() / job_id


def ensure_job_dir(job_id: str) -> Path:
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_depth_path(job_id: str) -> Path:
    return job_dir(job_id) / "raw_depth.npy"


def variant_path(job_id: str, variant: str) -> Path:
    return job_dir(job_id) / f"{variant}.png"


def original_preview_path(job_id: str) -> Path:
    return job_dir(job_id) / "original_preview.jpg"


def delete_job(job_id: str) -> None:
    d = job_dir(job_id)
    if d.exists():
        shutil.rmtree(d)


def list_job_ids() -> list[str]:
    root = jobs_root()
    if not root.exists():
        return []
    return [p.name for p in root.iterdir() if p.is_dir()]
