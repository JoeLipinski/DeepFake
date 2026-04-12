"""Four style variant configurations for laser engraving depth maps.

All variants are derived from the same single depth inference pass (raw_depth.npy).
Each config is tuned for a different engraving use-case:

  soft     — smooth, gentle relief. Good for portraits, fabric textures.
  standard — balanced default. Works for most subjects.
  detailed — high tonal range. Good for architectural details, coins.
  sharp    — maximum edge definition. Best for logos, text, crisp geometry.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.pipeline.postprocessor import apply_postprocessing


@dataclass(frozen=True)
class PostProcessingConfig:
    name: str
    depth_intensity: float
    blur_radius: float
    contrast: float
    edge_enhancement: float
    clahe_clip_limit: float
    unsharp_radius: float
    unsharp_percent: int
    invert: bool = False


STYLE_VARIANTS: dict[str, PostProcessingConfig] = {
    "soft": PostProcessingConfig(
        name="Soft",
        depth_intensity=0.75,
        blur_radius=2.5,
        contrast=0.9,
        edge_enhancement=0.0,
        clahe_clip_limit=0.0,
        unsharp_radius=0.0,
        unsharp_percent=0,
    ),
    "standard": PostProcessingConfig(
        name="Standard",
        depth_intensity=1.0,
        blur_radius=1.0,
        contrast=1.1,
        edge_enhancement=0.3,
        clahe_clip_limit=2.0,
        unsharp_radius=1.0,
        unsharp_percent=100,
    ),
    "detailed": PostProcessingConfig(
        name="Detailed",
        depth_intensity=1.15,
        blur_radius=0.5,
        contrast=1.3,
        edge_enhancement=0.6,
        clahe_clip_limit=3.5,
        unsharp_radius=2.0,
        unsharp_percent=150,
    ),
    "sharp": PostProcessingConfig(
        name="Sharp",
        depth_intensity=1.3,
        blur_radius=0.0,
        contrast=1.5,
        edge_enhancement=1.0,
        clahe_clip_limit=5.0,
        unsharp_radius=3.0,
        unsharp_percent=200,
    ),
}


def generate_all_variants(raw_depth: np.ndarray) -> dict[str, Image.Image]:
    """Generate all 4 variants in parallel using a thread pool (CPU-bound)."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            name: pool.submit(_apply_variant, raw_depth, cfg)
            for name, cfg in STYLE_VARIANTS.items()
        }
    return {name: fut.result() for name, fut in futures.items()}


def apply_custom_variant(
    raw_depth: np.ndarray,
    *,
    depth_intensity: float,
    blur_radius: float,
    contrast: float,
    edge_enhancement: float,
    invert: bool = False,
) -> Image.Image:
    """Apply user-defined params to a raw depth tensor (for re-processing endpoint)."""
    return apply_postprocessing(
        raw_depth,
        depth_intensity=depth_intensity,
        blur_radius=blur_radius,
        contrast=contrast,
        edge_enhancement=edge_enhancement,
        clahe_clip_limit=2.0,  # fixed for custom params
        unsharp_radius=2.0,
        unsharp_percent=int(edge_enhancement * 150),
        invert=invert,
    )


def _apply_variant(raw_depth: np.ndarray, cfg: PostProcessingConfig) -> Image.Image:
    return apply_postprocessing(
        raw_depth,
        depth_intensity=cfg.depth_intensity,
        blur_radius=cfg.blur_radius,
        contrast=cfg.contrast,
        edge_enhancement=cfg.edge_enhancement,
        clahe_clip_limit=cfg.clahe_clip_limit,
        unsharp_radius=cfg.unsharp_radius,
        unsharp_percent=cfg.unsharp_percent,
        invert=cfg.invert,
    )
