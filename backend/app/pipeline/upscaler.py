"""HD upscaling via PIL Lanczos (4×).

Real-ESRGAN is blocked by a basicsr/torchvision>=0.17 incompatibility.
PIL Lanczos is a high-quality sinc filter that works well for grayscale depth
maps — there are no photo textures to hallucinate, so the quality difference
vs. a learned upscaler is minimal for laser engraving use.
"""

from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)

_SCALE = 4


def upscale(image: Image.Image) -> Image.Image:
    """Upscale a grayscale PIL image 4× using Lanczos resampling."""
    w, h = image.size
    new_size = (w * _SCALE, h * _SCALE)
    logger.info("Upscaling %dx%d → %dx%d (Lanczos)", w, h, *new_size)
    result = image.resize(new_size, Image.LANCZOS)
    logger.info("Upscale complete.")
    return result
