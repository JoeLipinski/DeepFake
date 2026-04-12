"""FastAPI application factory with lifespan model loading."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core import job_queue, model_manager
from app.api.router import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting up — loading models (this may take 60-120s)...")

    # Run blocking model loads in a thread so the event loop stays responsive
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        model_manager.load_all_models,
        settings.depth_model_id,
        settings.realesrgan_model_url,
    )

    job_queue.start_consumer()
    cleanup_task = asyncio.create_task(_cleanup_loop(settings.job_ttl_hours))
    logger.info("Application ready.")

    yield

    logger.info("Shutting down...")
    cleanup_task.cancel()
    job_queue.stop_consumer()


async def _cleanup_loop(ttl_hours: int) -> None:
    """Hourly sweep: delete job directories older than ttl_hours."""
    from datetime import datetime, timezone, timedelta
    from app.core import storage

    while True:
        await asyncio.sleep(3600)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        for job_id in storage.list_job_ids():
            d = storage.job_dir(job_id)
            mtime = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                storage.delete_job(job_id)
                logger.info("Cleaned up expired job %s", job_id)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="DepthForge — 2.5D Depth Map Generator",
        description="Converts 2D images to grayscale depth maps optimized for laser engraving.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    return app


app = create_app()
