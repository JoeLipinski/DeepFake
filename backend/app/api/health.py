import torch
from fastapi import APIRouter
from app.core import job_queue, model_manager

router = APIRouter()


@router.get("/health")
async def health():
    gpu_available = torch.cuda.is_available()
    vram_used_gb = None
    if gpu_available:
        try:
            vram_used_gb = round(
                torch.cuda.memory_allocated(0) / (1024 ** 3), 2
            )
        except Exception:
            pass

    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "device": model_manager.get_device(),
        "models_loaded": model_manager.get_loaded_models(),
        "queue_depth": job_queue.queue_depth(),
        "vram_used_gb": vram_used_gb,
    }
