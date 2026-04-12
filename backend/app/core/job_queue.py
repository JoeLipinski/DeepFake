"""In-memory async job queue.

One consumer task processes jobs serially (GPU concurrency > 1 → OOM risk).
JobStatus objects live in a module-level dict for the lifetime of the process.
If you need persistence across restarts, swap this for Celery + Redis.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from app.schemas.job import JobStatus, JobStep

logger = logging.getLogger(__name__)

_queue: asyncio.Queue[tuple[str, Callable[[], Coroutine]]] = asyncio.Queue()
_jobs: dict[str, JobStatus] = {}
_consumer_task: asyncio.Task | None = None


def get_job(job_id: str) -> JobStatus | None:
    return _jobs.get(job_id)


def all_jobs() -> dict[str, JobStatus]:
    return _jobs


async def enqueue(job_id: str, coro_factory: Callable[[], Coroutine]) -> None:
    """Add a job to the queue. coro_factory is called by the consumer."""
    _jobs[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        step=JobStep.queued,
        progress=0.0,
        created_at=datetime.now(timezone.utc),
    )
    await _queue.put((job_id, coro_factory))
    logger.info("Job %s enqueued (queue depth: %d)", job_id, _queue.qsize())


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    step: JobStep | None = None,
    progress: float | None = None,
    error: str | None = None,
    result: Any | None = None,
) -> None:
    job = _jobs.get(job_id)
    if job is None:
        return
    if status is not None:
        job.status = status
    if step is not None:
        job.step = step
    if progress is not None:
        job.progress = progress
    if error is not None:
        job.error = error
    if result is not None:
        job.result = result
    if status == "complete":
        job.completed_at = datetime.now(timezone.utc)


def queue_depth() -> int:
    return _queue.qsize()


async def _consumer() -> None:
    logger.info("Job queue consumer started.")
    while True:
        job_id, coro_factory = await _queue.get()
        logger.info("Processing job %s", job_id)
        try:
            await coro_factory()
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            update_job(job_id, status="failed", error=str(exc))
        finally:
            _queue.task_done()


def start_consumer() -> None:
    global _consumer_task
    if _consumer_task is None or _consumer_task.done():
        _consumer_task = asyncio.create_task(_consumer())
        logger.info("Job consumer task created.")


def stop_consumer() -> None:
    global _consumer_task
    if _consumer_task and not _consumer_task.done():
        _consumer_task.cancel()
