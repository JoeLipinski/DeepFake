"""Depth estimation pipeline.

For raster images (JPEG/PNG/WebP) — DAv2 path:
  1. If image > _INFERENCE_SIZE: tiled inference at full resolution.
       - Overlapping 1008px tiles, 25% cosine-feathered overlap.
       - Per-tile depth is globally aligned via linear regression against
         a low-res reference run (preserves correct relative scale).
       - Tiles are stitched with weighted accumulation.
     Else: single inference with guided bilateral upsample.
  2. Source detail blend — high-frequency surface texture (pores, grain,
     illustration outlines) blended into depth at mode-appropriate strength.

Ultra mode (Marigold LCM):
  prs-eth/marigold-lcm-v1-0, 10 steps, ensemble of 10 — metric depth.
  Detail blend applied after, same as DAv2.

SVG / flat vector art:
  Luminance-inversion depth — dark shapes become raised relief.

Illustration mode — per-region refinement (separate worker step):
  Felzenszwalb segmentation → per-region smooth + boundary sharpening.

Saves raw float32 depth tensor as .npy for fast re-processing.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from app.core import model_manager, storage

logger = logging.getLogger(__name__)

# Tile / inference size. 1008 = 72 × 14 (ViT patch multiple).
_INFERENCE_SIZE = 1008

# Overlap between adjacent tiles (25%). Cosine feathering blends the seams.
_TILE_OVERLAP = _INFERENCE_SIZE // 4  # 252px

# Detail blend strength per image type.
_DETAIL_BLEND_STRENGTH_PHOTO = 0.35
_DETAIL_BLEND_STRENGTH_ILLUSTRATION = 0.25


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_depth(
    job_id: str,
    image: Image.Image,
    use_luminance: bool = False,
    image_type: str = "photo",
    use_marigold: bool = False,
) -> np.ndarray:
    """Run depth estimation, save raw tensor, return float32 array [0,1].

    Args:
        job_id:        Job identifier for storage paths.
        image:         Source PIL image (RGB or L).
        use_luminance: Use luminance-inversion depth (SVG/flat vector art).
        image_type:    "photo" or "illustration" — detail blend strength.
        use_marigold:  Use Marigold LCM diffusion depth (Ultra mode).
    """
    original_size = image.size  # (W, H)
    mode = (
        "luminance" if use_luminance
        else "marigold" if use_marigold
        else f"ai/{image_type}"
    )
    logger.info("Job %s: depth estimation %dx%d (mode=%s)", job_id, *original_size, mode)

    blend_strength = (
        _DETAIL_BLEND_STRENGTH_ILLUSTRATION
        if image_type == "illustration"
        else _DETAIL_BLEND_STRENGTH_PHOTO
    )

    if use_luminance:
        depth_arr = _luminance_depth(image)
    elif use_marigold:
        depth_arr = _marigold_depth(image, original_size, image_type)
        depth_arr = _blend_source_detail(depth_arr, image, strength=blend_strength)
    else:
        depth_arr = _ai_depth(image, original_size, image_type)
        depth_arr = _blend_source_detail(depth_arr, image, strength=blend_strength)

    npy_path = storage.raw_depth_path(job_id)
    np.save(str(npy_path), depth_arr)
    logger.info("Job %s: raw depth saved (%dx%d)", job_id, *original_size)
    return depth_arr


def apply_sam_refinement(
    job_id: str,
    depth: np.ndarray,
    image: Image.Image,
) -> np.ndarray:
    """Per-region depth refinement via felzenszwalb segmentation.

    Called by the worker as a separate step after estimate_depth() so the
    job queue can emit a distinct progress update. Re-saves raw_depth.npy.
    """
    logger.info("Job %s: per-region refinement start", job_id)
    refined = _apply_region_refinement(depth, image)
    npy_path = storage.raw_depth_path(job_id)
    np.save(str(npy_path), refined)
    logger.info("Job %s: refined depth saved", job_id)
    return refined


def load_raw_depth(job_id: str) -> np.ndarray:
    path = storage.raw_depth_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"Raw depth not found for job {job_id}")
    return np.load(str(path))


# ---------------------------------------------------------------------------
# DAv2 depth — single or tiled
# ---------------------------------------------------------------------------

def _ai_depth(
    image: Image.Image,
    original_size: tuple[int, int],
    image_type: str = "photo",
) -> np.ndarray:
    """Dispatch to tiled or single inference based on image size."""
    w, h = original_size
    if max(w, h) > _INFERENCE_SIZE:
        return _tiled_ai_depth(image, original_size, image_type)
    return _single_ai_depth(image, original_size, image_type)


def _single_ai_depth(
    image: Image.Image,
    original_size: tuple[int, int],
    image_type: str = "photo",
) -> np.ndarray:
    """DAv2 on the full image (downscaled to _INFERENCE_SIZE), guided upsample back."""
    inference_img = _resize_for_inference(image)

    pipe = model_manager.get_depth_pipe()
    result = pipe(inference_img)
    depth_pil: Image.Image = result["depth"]

    depth_arr = np.array(depth_pil).astype(np.float32)
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_arr = np.zeros_like(depth_arr)

    if depth_pil.size != original_size:
        depth_arr = _guided_upsample(depth_arr, image, original_size, image_type)

    return depth_arr


def _tiled_ai_depth(
    image: Image.Image,
    original_size: tuple[int, int],
    image_type: str = "photo",
) -> np.ndarray:
    """Run DAv2 on overlapping tiles at original resolution, then stitch.

    Each tile is 1008×1008px (full inference resolution). Tiles overlap by
    25% (_TILE_OVERLAP) and are blended with a cosine-feathered weight map
    so seams are invisible.

    Per-tile scale problem: DAv2 normalises each tile's depth independently
    to [0,1], so raw tile values are incompatible. We fix this by running a
    single low-res reference pass first, then computing a linear (scale +
    offset) correction per tile so it aligns with the reference in that region.
    """
    w, h = original_size
    tile_size = _INFERENCE_SIZE
    stride = tile_size - _TILE_OVERLAP

    # Step 1: low-res reference for per-tile scale normalisation
    logger.info("Job tiled: computing low-res reference depth for scale alignment")
    reference_depth = _single_ai_depth(image, original_size, image_type)

    # Step 2: tile coordinates
    tiles = _get_tile_coords(w, h, tile_size, stride)
    logger.info("Job tiled: %d tiles for %dx%d image (tile=%dpx, stride=%dpx)",
                len(tiles), w, h, tile_size, stride)

    # Step 3: accumulate weighted tile contributions
    accumulated = np.zeros((h, w), dtype=np.float64)
    weight_map = np.zeros((h, w), dtype=np.float64)

    for idx, (x0, y0, x1, y1) in enumerate(tiles):
        th, tw = y1 - y0, x1 - x0
        logger.debug("Tile %d/%d (%dx%d) at (%d,%d)", idx + 1, len(tiles), tw, th, x0, y0)

        tile_img = image.crop((x0, y0, x1, y1))
        tile_depth = _infer_tile(tile_img, tw, th)

        # Align tile scale/offset to the reference depth in this region
        ref_region = reference_depth[y0:y1, x0:x1]
        tile_depth = _align_tile_to_reference(tile_depth, ref_region)

        # Cosine-feathered weight — high in centre, zero at edges
        weight = _make_tile_weight(th, tw, _TILE_OVERLAP)

        accumulated[y0:y1, x0:x1] += tile_depth.astype(np.float64) * weight
        weight_map[y0:y1, x0:x1] += weight

    stitched = (accumulated / np.maximum(weight_map, 1e-8)).astype(np.float32)
    return np.clip(stitched, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _get_tile_coords(
    w: int, h: int, tile_size: int, stride: int
) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) for every tile covering the image.

    The last tile in each axis is always flush with the image edge so no
    pixels are missed regardless of stride alignment.
    """
    def axis_starts(dim: int) -> list[int]:
        if dim <= tile_size:
            return [0]
        starts = list(range(0, dim - tile_size, stride))
        starts.append(dim - tile_size)          # flush end tile
        return sorted(set(starts))

    xs = axis_starts(w)
    ys = axis_starts(h)
    return [
        (x0, y0, min(x0 + tile_size, w), min(y0 + tile_size, h))
        for y0 in ys
        for x0 in xs
    ]


def _make_tile_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """2-D cosine feather weight: 1.0 in the centre, tapers to 0 at edges."""
    def fade_1d(n: int, margin: int) -> np.ndarray:
        r = np.ones(n, dtype=np.float64)
        margin = min(margin, n // 2)
        if margin > 0:
            ramp = (1.0 - np.cos(np.linspace(0.0, np.pi / 2, margin))) / 1.0
            r[:margin] = np.sin(np.linspace(0.0, np.pi / 2, margin))
            r[n - margin:] = r[:margin][::-1]
        return r

    wy = fade_1d(h, overlap)
    wx = fade_1d(w, overlap)
    return np.outer(wy, wx)


def _infer_tile(tile_img: Image.Image, tw: int, th: int) -> np.ndarray:
    """Run DAv2 on one tile, return float32 [0,1] at (th, tw)."""
    pipe = model_manager.get_depth_pipe()
    result = pipe(tile_img)
    depth_pil: Image.Image = result["depth"]

    depth_arr = np.array(depth_pil).astype(np.float32)
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_arr = np.zeros_like(depth_arr)

    # Pipeline may return slightly different dimensions — resize to exact tile size
    if depth_arr.shape != (th, tw):
        depth_arr = cv2.resize(depth_arr, (tw, th), interpolation=cv2.INTER_LINEAR)

    return depth_arr


def _align_tile_to_reference(
    tile: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """Linear scale + offset so tile matches ref in this region.

    Fits:  ref ≈ a * tile + b  using least squares, then applies the
    correction.  Skipped if either is too flat (no variation to fit on).
    """
    t = tile.ravel().astype(np.float64)
    r = ref.ravel().astype(np.float64)

    if t.std() < 1e-4 or r.std() < 1e-4:
        return tile  # flat region — no meaningful alignment possible

    A = np.stack([t, np.ones_like(t)], axis=1)
    result, *_ = np.linalg.lstsq(A, r, rcond=None)
    a, b = result

    return np.clip(a * tile + b, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Marigold (Ultra mode)
# ---------------------------------------------------------------------------

def _marigold_depth(
    image: Image.Image,
    original_size: tuple[int, int],
    image_type: str = "photo",
) -> np.ndarray:
    """Marigold LCM diffusion depth — 10 steps, ensemble of 10.

    Lazy-loaded on first call (~1.7 GB download).
    """
    pipe = model_manager.get_marigold_pipe()
    rgb_image = image.convert("RGB")

    output = pipe(
        rgb_image,
        num_inference_steps=10,
        ensemble_size=10,
    )

    # prediction shape varies by diffusers version; squeeze to (H, W)
    depth_arr = output.prediction.squeeze().astype(np.float32)

    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_arr = np.zeros_like(depth_arr)

    w, h = original_size
    if depth_arr.shape != (h, w):
        depth_arr = _guided_upsample(depth_arr, image, original_size, image_type)

    return depth_arr


# ---------------------------------------------------------------------------
# Guided upsample
# ---------------------------------------------------------------------------

def _guided_upsample(
    depth_low: np.ndarray,
    source: Image.Image,
    target_size: tuple[int, int],  # (W, H)
    image_type: str = "photo",
) -> np.ndarray:
    """Edge-aware bicubic upsample guided by source image Scharr gradients."""
    w, h = target_size

    depth_u8 = (depth_low * 255).clip(0, 255).astype(np.uint8)
    depth_up = cv2.resize(depth_u8, (w, h), interpolation=cv2.INTER_CUBIC)

    guide = np.array(source.resize((w, h), Image.LANCZOS).convert("L"))
    sx = cv2.Scharr(guide, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(guide, cv2.CV_32F, 0, 1)
    edge_mag = np.sqrt(sx ** 2 + sy ** 2)
    edge_mag = (edge_mag / (edge_mag.max() + 1e-8)).astype(np.float32)
    edge_mask = cv2.GaussianBlur(edge_mag, (0, 0), sigmaX=2.0)

    sigma = 20 if image_type == "illustration" else 40
    depth_bilateral = cv2.bilateralFilter(
        depth_up.astype(np.float32), d=9, sigmaColor=sigma, sigmaSpace=sigma
    )

    depth_blended = (
        edge_mask * depth_up.astype(np.float32)
        + (1.0 - edge_mask) * depth_bilateral
    )
    return depth_blended.clip(0, 255).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Source detail blend
# ---------------------------------------------------------------------------

def _blend_source_detail(
    depth: np.ndarray,
    source: Image.Image,
    strength: float = _DETAIL_BLEND_STRENGTH_PHOTO,
) -> np.ndarray:
    """Blend high-frequency surface texture from source into depth.

    Multi-scale detail (σ=1 fine + σ=3 medium) extracted from source
    luminance and additively blended in. Transfers pores, grain, and
    illustration line work that the depth model misses.
    """
    h, w = depth.shape
    guide = np.array(
        source.resize((w, h), Image.LANCZOS).convert("L")
    ).astype(np.float32) / 255.0

    blur_fine = cv2.GaussianBlur(guide, (0, 0), sigmaX=1.0)
    blur_med = cv2.GaussianBlur(guide, (0, 0), sigmaX=3.0)
    detail = 0.6 * (guide - blur_fine) + 0.4 * (guide - blur_med)

    d_abs_max = np.abs(detail).max()
    if d_abs_max > 1e-6:
        detail = detail / (d_abs_max * 2.0)

    return np.clip(depth + strength * detail, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Illustration mode — per-region refinement
# ---------------------------------------------------------------------------

def _apply_region_refinement(depth: np.ndarray, image: Image.Image) -> np.ndarray:
    """Felzenszwalb segmentation → per-region smooth + boundary sharpening."""
    from skimage.segmentation import felzenszwalb

    h, w = depth.shape
    image_np = np.array(image.convert("RGB").resize((w, h), Image.LANCZOS))

    labels = felzenszwalb(image_np, scale=150, sigma=0.8, min_size=200)
    n_regions = labels.max() + 1
    logger.debug("Felzenszwalb: %d regions at %dx%d", n_regions, w, h)

    refined = depth.copy()
    boundary_map = np.zeros((h, w), dtype=np.float32)
    depth_smooth = cv2.GaussianBlur(depth, (0, 0), sigmaX=2.0)

    for label_id in range(n_regions):
        seg = (labels == label_id)
        area = int(seg.sum())
        if area < 50:
            continue

        refined[seg] = depth_smooth[seg]

        radius = 4 if area > 10_000 else 2
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        eroded = cv2.erode(seg.astype(np.uint8), k)
        boundary_map = np.maximum(
            boundary_map, (seg.astype(np.uint8) - eroded).astype(np.float32)
        )

    sharpened = _unsharp_depth(refined, sigma=1.0, strength=2.5)
    boundary_weight = cv2.GaussianBlur(boundary_map, (0, 0), sigmaX=1.5).clip(0.0, 1.0)
    refined = (1.0 - boundary_weight) * refined + boundary_weight * sharpened

    return np.clip(refined, 0.0, 1.0)


def _unsharp_depth(depth: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(depth, (0, 0), sigmaX=sigma)
    return depth + strength * (depth - blurred)


# ---------------------------------------------------------------------------
# SVG / luminance path
# ---------------------------------------------------------------------------

def _luminance_depth(image: Image.Image) -> np.ndarray:
    """Luminance-inversion depth for SVG / flat vector art."""
    gray = np.array(image.convert("L")).astype(np.float32) / 255.0
    depth = 1.0 - gray
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=1.0)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


# ---------------------------------------------------------------------------
# Resize helper
# ---------------------------------------------------------------------------

def _resize_for_inference(image: Image.Image) -> Image.Image:
    """Downscale to _INFERENCE_SIZE on the long edge, snapped to ViT patch grid (×14)."""
    w, h = image.size
    scale = _INFERENCE_SIZE / max(w, h)
    if scale < 1.0:
        new_w = int(round(w * scale / 14) * 14)
        new_h = int(round(h * scale / 14) * 14)
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image
