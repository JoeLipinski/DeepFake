"""Depth estimation using Depth Anything V2 Large.

Saves the raw float32 depth tensor as .npy for fast re-processing.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from app.core import model_manager, storage

logger = logging.getLogger(__name__)

# Depth Anything V2 was trained at 518px — inference at larger sizes wastes VRAM
# without improving quality. We resize to fit within this square, run inference,
# then bicubic-upsample the depth map back to the original image resolution.
_INFERENCE_SIZE = 518


def estimate_depth(job_id: str, image: Image.Image) -> np.ndarray:
    """Run Depth Anything V2 on `image`, save raw tensor, return float32 array [0,1].

    The returned array matches the spatial dimensions of `image`.
    """
    original_size = image.size  # (W, H)
    logger.info("Job %s: estimating depth for %dx%d image", job_id, *original_size)

    # Resize for inference
    inference_img = _resize_for_inference(image)

    pipe = model_manager.get_depth_pipe()
    result = pipe(inference_img)
    depth_pil: Image.Image = result["depth"]

    # depth_pil is a grayscale PIL image; convert to float32 [0,1]
    depth_arr = np.array(depth_pil).astype(np.float32)
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_arr = np.zeros_like(depth_arr)

    # Upsample back to original resolution
    if depth_pil.size != original_size:
        depth_full = Image.fromarray((depth_arr * 255).astype(np.uint8)).resize(
            original_size, Image.BICUBIC
        )
        depth_arr = np.array(depth_full).astype(np.float32) / 255.0

    # Persist raw depth tensor
    npy_path = storage.raw_depth_path(job_id)
    np.save(str(npy_path), depth_arr)
    logger.info("Job %s: raw depth saved to %s", job_id, npy_path)

    return depth_arr


def load_raw_depth(job_id: str) -> np.ndarray:
    """Load previously computed raw depth tensor for re-processing."""
    path = storage.raw_depth_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"Raw depth not found for job {job_id}")
    return np.load(str(path))


def _resize_for_inference(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = _INFERENCE_SIZE / max(w, h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image
