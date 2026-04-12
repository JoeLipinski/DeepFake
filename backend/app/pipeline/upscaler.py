"""HD upscaling via Real-ESRGAN (×4).

Applied lazily at export time — not precomputed for all variants.
Works on grayscale depth maps by briefly converting to RGB for the model,
then converting back to grayscale.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from app.core import model_manager

logger = logging.getLogger(__name__)


def upscale(image: Image.Image) -> Image.Image:
    """Upscale a grayscale PIL image 4× using Real-ESRGAN.

    Input must be a grayscale ('L') PIL Image.
    Returns a grayscale ('L') PIL Image at 4× resolution.
    """
    logger.info("Upscaling %dx%d image via Real-ESRGAN...", *image.size)

    # Real-ESRGAN expects BGR uint8 ndarray
    arr_rgb = np.array(image.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

    esrgan = model_manager.get_realesrgan_model()
    output_bgr, _ = esrgan.enhance(arr_bgr, outscale=4)

    # Convert back to grayscale
    output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
    result = Image.fromarray(output_gray, mode="L")
    logger.info("Upscaled to %dx%d.", *result.size)
    return result
