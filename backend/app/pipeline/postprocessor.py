"""Post-processing pipeline applied to raw float32 depth tensors.

Steps (in order):
  1. Depth intensity / gamma scaling
  2. CLAHE (adaptive histogram equalization) — critical for laser engraving tonal range
  3. Gaussian blur — smooths carving paths
  4. Contrast enhancement (PIL ImageEnhance)
  5. Unsharp mask — accentuates ridge edges for crisp relief transitions
  6. Invert (optional)
  7. Final normalization to uint8 [0, 255]
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_postprocessing(
    raw_depth: np.ndarray,
    *,
    depth_intensity: float = 1.0,
    blur_radius: float = 1.0,
    contrast: float = 1.1,
    edge_enhancement: float = 0.3,
    clahe_clip_limit: float = 2.0,
    unsharp_radius: float = 1.0,
    unsharp_percent: int = 100,
    invert: bool = False,
) -> Image.Image:
    """Apply post-processing to a raw float32 depth array [0,1].

    Returns an 8-bit grayscale PIL Image ready for PNG export.
    """
    # 1. Depth intensity: power-law scaling (>1 = more contrast at extremes)
    arr = np.clip(raw_depth, 0.0, 1.0)
    if depth_intensity != 1.0:
        # Apply gamma: values > 1 compress midtones toward white (higher relief);
        # values < 1 expand midtones
        gamma = 1.0 / max(depth_intensity, 0.01)
        arr = np.power(arr, gamma)

    # Convert to uint8 for CV2 operations
    arr_u8 = (arr * 255).clip(0, 255).astype(np.uint8)

    # 2. CLAHE — distributes tonal range to maximize dynamic range for engraving
    if clahe_clip_limit > 0:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(8, 8),
        )
        arr_u8 = clahe.apply(arr_u8)

    # 3. Gaussian blur — smooths depth transitions for cleaner carving paths
    if blur_radius > 0:
        ksize = _odd_kernel(blur_radius)
        arr_u8 = cv2.GaussianBlur(arr_u8, (ksize, ksize), blur_radius)

    # Convert to PIL for contrast + unsharp mask
    pil = Image.fromarray(arr_u8, mode="L")

    # 4. Contrast enhancement
    if contrast != 1.0:
        pil = ImageEnhance.Contrast(pil).enhance(contrast)

    # 5. Unsharp mask — accentuates feature edges for crisp relief transitions
    if edge_enhancement > 0 and unsharp_percent > 0:
        pil = pil.filter(
            ImageFilter.UnsharpMask(
                radius=unsharp_radius,
                percent=int(unsharp_percent * edge_enhancement),
                threshold=3,
            )
        )

    # 6. Invert
    if invert:
        arr_inv = 255 - np.array(pil)
        pil = Image.fromarray(arr_inv, mode="L")

    # 7. Ensure final range is uint8
    final = np.array(pil).clip(0, 255).astype(np.uint8)
    return Image.fromarray(final, mode="L")


def _odd_kernel(sigma: float) -> int:
    """Return the nearest odd integer kernel size for a given sigma."""
    k = int(sigma * 6 + 1)
    return k if k % 2 == 1 else k + 1
