"""Singleton model manager — loads all ML models once at startup.

Loaded at FastAPI lifespan start. All inference code accesses models through
these module-level getters; never re-load per request.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Module-level singletons
_depth_pipe: Any = None
_marigold_pipe: Any = None
_rembg_session: Any = None
_realesrgan_model: Any = None
_device: str = "cpu"
_loaded_models: list[str] = []


def get_device() -> str:
    return _device


def get_loaded_models() -> list[str]:
    return list(_loaded_models)


def get_depth_pipe() -> Any:
    if _depth_pipe is None:
        raise RuntimeError("Depth model not loaded. Was load_all_models() called?")
    return _depth_pipe


def get_marigold_pipe() -> Any:
    """Lazy-load Marigold on first use — only downloaded when Ultra mode is requested."""
    global _marigold_pipe
    if _marigold_pipe is None:
        _load_marigold()
    return _marigold_pipe


def get_rembg_session() -> Any:
    if _rembg_session is None:
        raise RuntimeError("rembg session not loaded.")
    return _rembg_session


def get_realesrgan_model() -> Any:
    # Real-ESRGAN not loaded; upscaler.py uses PIL Lanczos instead.
    raise RuntimeError("Real-ESRGAN is not available. Use PIL Lanczos upscaler.")


def load_all_models(depth_model_id: str, realesrgan_model_url: str) -> None:
    """Called once at application startup inside FastAPI lifespan."""
    global _depth_pipe, _rembg_session, _realesrgan_model, _device, _loaded_models

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", _device)

    _load_depth_model(depth_model_id)
    _load_rembg()
    # Real-ESRGAN skipped: basicsr incompatible with torchvision>=0.17.
    # Upscaling falls back to PIL Lanczos (see pipeline/upscaler.py).
    _loaded_models.append("upscaler_lanczos")

    logger.info("All models loaded: %s", _loaded_models)


def _load_depth_model(model_id: str) -> None:
    global _depth_pipe, _loaded_models
    logger.info("Loading depth model: %s", model_id)
    try:
        from transformers import pipeline as hf_pipeline

        dtype = torch.float16 if _device == "cuda" else torch.float32
        _depth_pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if _device == "cuda" else -1,
            torch_dtype=dtype,
        )
        _loaded_models.append("depth_anything_v2")
        logger.info("Depth model loaded.")
    except Exception:
        logger.exception("Failed to load depth model")
        raise


def _load_marigold() -> None:
    global _marigold_pipe, _loaded_models
    logger.info("Lazy-loading Marigold LCM depth pipeline (first Ultra request)...")
    try:
        from diffusers import MarigoldDepthPipeline

        dtype = torch.float16 if _device == "cuda" else torch.float32
        variant = "fp16" if dtype == torch.float16 else None
        _marigold_pipe = MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-lcm-v1-0",
            variant=variant,
            torch_dtype=dtype,
        ).to(_device)
        _loaded_models.append("marigold_lcm")
        logger.info("Marigold LCM loaded.")
    except Exception:
        logger.exception("Failed to load Marigold")
        raise


def _load_rembg() -> None:
    global _rembg_session, _loaded_models
    logger.info("Loading rembg (u2net)...")
    try:
        from rembg import new_session

        _rembg_session = new_session("u2net")
        _loaded_models.append("rembg")
        logger.info("rembg loaded.")
    except Exception:
        logger.exception("Failed to load rembg")
        raise


def _load_realesrgan(model_url: str) -> None:
    global _realesrgan_model, _loaded_models
    logger.info("Loading Real-ESRGAN...")
    try:
        import urllib.request
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        cache_dir = Path(os.environ.get("HOME", "/root")) / ".cache" / "realesrgan"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "RealESRGAN_x4plus.pth"

        if not model_path.exists():
            logger.info("Downloading Real-ESRGAN weights...")
            urllib.request.urlretrieve(model_url, str(model_path))

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        half = _device == "cuda"
        _realesrgan_model = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=_device,
        )
        _loaded_models.append("realesrgan")
        logger.info("Real-ESRGAN loaded.")
    except Exception:
        logger.exception("Failed to load Real-ESRGAN")
        raise
