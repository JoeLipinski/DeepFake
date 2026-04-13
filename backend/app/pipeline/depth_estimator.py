"""Depth estimation pipeline.

For raster images (JPEG/PNG/WebP) — DAv2 path:
  1. Source CLAHE pre-processing (LAB L-channel only) — better model gradient
     cues on dark / low-contrast source images.
  2. Tiled inference if image > _INFERENCE_SIZE, else single inference.
       Tiles: 1008px, 33% cosine-feathered overlap.
       Per-tile depth aligned to low-res reference via linear regression.
  3. Multi-scale source detail blend (σ=0.5, 1, 3) — injects surface texture
     the depth model misses.
  4. Normal-map slope sharpening — amplifies surface gradients via
     Frankot-Chellappa integration, making ridges/valleys crisper for
     engraving without unsharp-mask halos.

Ultra mode (Marigold LCM):
  prs-eth/marigold-lcm-v1-0, 10 steps, ensemble 10.
  Detail blend + normal sharpening applied identically.

SVG / flat vector art:
  Luminance-inversion depth.

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

# Tile size (ViT patch multiple: 1008 = 72 × 14).
_INFERENCE_SIZE = 1008

# 33% overlap between adjacent tiles — smoother stitching than 25%.
_TILE_OVERLAP = _INFERENCE_SIZE // 3  # 336 px

# Detail blend strengths per image type.
_DETAIL_BLEND_STRENGTH_PHOTO = 0.35
_DETAIL_BLEND_STRENGTH_ILLUSTRATION = 0.25

# Normal-map slope amplification factor (>1 = steeper slopes).
_NORMAL_SHARPEN_STRENGTH = 1.3


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
    """Run depth estimation, save raw tensor, return float32 array [0,1]."""
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
        depth_arr = _normal_sharpen(depth_arr)
    else:
        # Pre-normalise source luminance so the model sees better contrast cues.
        # Keep original image for detail blend (real texture, not CLAHE version).
        preprocessed = _clahe_preprocess(image)
        depth_arr = _ai_depth(preprocessed, original_size, image_type)
        depth_arr = _blend_source_detail(depth_arr, image, strength=blend_strength)
        depth_arr = _normal_sharpen(depth_arr)

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
# Source pre-processing
# ---------------------------------------------------------------------------

def _clahe_preprocess(image: Image.Image) -> Image.Image:
    """Apply CLAHE to the L channel (LAB space) of the source image.

    Improves depth model gradient cues on dark, under-exposed, or flat-contrast
    images without altering hue or saturation.  The original image is kept for
    the detail-blend pass so real texture (not the CLAHE version) is injected.
    """
    img_np = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)

    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_eq)


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
    """Run DAv2 on overlapping tiles at full resolution, then stitch.

    Tiles: _INFERENCE_SIZE × _INFERENCE_SIZE, _TILE_OVERLAP cosine feathering.
    Per-tile normalisation: linear regression against a low-res reference pass.
    """
    w, h = original_size
    tile_size = _INFERENCE_SIZE
    stride = tile_size - _TILE_OVERLAP

    logger.info("Tiled inference: computing low-res reference for scale alignment")
    reference_depth = _single_ai_depth(image, original_size, image_type)

    tiles = _get_tile_coords(w, h, tile_size, stride)
    logger.info(
        "Tiled inference: %d tiles for %dx%d (tile=%dpx overlap=%dpx)",
        len(tiles), w, h, tile_size, _TILE_OVERLAP,
    )

    accumulated = np.zeros((h, w), dtype=np.float64)
    weight_map = np.zeros((h, w), dtype=np.float64)

    for idx, (x0, y0, x1, y1) in enumerate(tiles):
        th, tw = y1 - y0, x1 - x0
        logger.debug("Tile %d/%d (%dx%d) at (%d,%d)", idx + 1, len(tiles), tw, th, x0, y0)

        tile_img = image.crop((x0, y0, x1, y1))
        tile_depth = _infer_tile(tile_img, tw, th)

        ref_region = reference_depth[y0:y1, x0:x1]
        tile_depth = _align_tile_to_reference(tile_depth, ref_region)

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
    def axis_starts(dim: int) -> list[int]:
        if dim <= tile_size:
            return [0]
        starts = list(range(0, dim - tile_size, stride))
        starts.append(dim - tile_size)
        return sorted(set(starts))

    return [
        (x0, y0, min(x0 + tile_size, w), min(y0 + tile_size, h))
        for y0 in axis_starts(h)
        for x0 in axis_starts(w)
    ]


def _make_tile_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """Cosine feather weight: 1.0 in centre, tapers to 0 at edges."""
    def fade_1d(n: int, margin: int) -> np.ndarray:
        r = np.ones(n, dtype=np.float64)
        margin = min(margin, n // 2)
        if margin > 0:
            ramp = np.sin(np.linspace(0.0, np.pi / 2, margin))
            r[:margin] = ramp
            r[n - margin:] = ramp[::-1]
        return r

    return np.outer(fade_1d(h, overlap), fade_1d(w, overlap))


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

    if depth_arr.shape != (th, tw):
        depth_arr = cv2.resize(depth_arr, (tw, th), interpolation=cv2.INTER_LINEAR)

    return depth_arr


def _align_tile_to_reference(tile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Linear scale + offset so tile matches ref (least-squares fit)."""
    t = tile.ravel().astype(np.float64)
    r = ref.ravel().astype(np.float64)

    if t.std() < 1e-4 or r.std() < 1e-4:
        return tile

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
    """Marigold LCM — 10 denoising steps, ensemble of 10."""
    pipe = model_manager.get_marigold_pipe()
    rgb_image = image.convert("RGB")

    output = pipe(rgb_image, num_inference_steps=10, ensemble_size=10)

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
# Normal-map slope sharpening
# ---------------------------------------------------------------------------

def _normal_sharpen(
    depth: np.ndarray,
    strength: float = _NORMAL_SHARPEN_STRENGTH,
) -> np.ndarray:
    """Amplify surface slopes via Frankot-Chellappa gradient integration.

    Unlike unsharp masking (which sharpens depth *values*), this amplifies the
    depth gradient field — making ridges higher and valleys deeper everywhere
    in proportion to local slope.  The result has crisper engraving relief
    without the halo artifacts that come from value-domain sharpening.

    strength > 1.0 = steeper slopes (1.3 ≈ 30% more pronounced relief).
    """
    # Surface gradients
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    # Amplify
    gx_amp = gx * strength
    gy_amp = gy * strength

    # Reconstruct depth from amplified gradients
    enhanced = _frankot_chellappa(gx_amp, gy_amp)

    # Normalise to [0, 1] preserving relative structure
    e_min, e_max = enhanced.min(), enhanced.max()
    if e_max > e_min:
        enhanced = (enhanced - e_min) / (e_max - e_min)
    else:
        return depth

    return enhanced.astype(np.float32)


def _frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Reconstruct a depth surface from its gradients p=dz/dx, q=dz/dy.

    Least-squares solution in the frequency domain (Frankot & Chellappa 1988).
    Produces an integrable surface consistent with the given gradient field.
    """
    rows, cols = p.shape
    wx = np.fft.fftfreq(cols) * 2.0 * np.pi
    wy = np.fft.fftfreq(rows) * 2.0 * np.pi
    wx, wy = np.meshgrid(wx, wy)

    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)

    denom = wx ** 2 + wy ** 2
    denom[0, 0] = 1.0  # avoid DC singularity

    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0.0  # zero mean depth

    return np.real(np.fft.ifft2(Z)).astype(np.float32)


# ---------------------------------------------------------------------------
# Guided upsample
# ---------------------------------------------------------------------------

def _guided_upsample(
    depth_low: np.ndarray,
    source: Image.Image,
    target_size: tuple[int, int],
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
    """Blend multi-scale high-frequency texture from source into depth.

    Three scales:
      σ=0.5  ultra-fine — sub-pixel texture, hair strands, micro line-work
      σ=1.0  fine       — pores, grain, fine illustration detail
      σ=3.0  medium     — broader surface texture, skin, fabric weave
    """
    h, w = depth.shape
    guide = np.array(
        source.resize((w, h), Image.LANCZOS).convert("L")
    ).astype(np.float32) / 255.0

    blur_uf  = cv2.GaussianBlur(guide, (0, 0), sigmaX=0.5)
    blur_fine = cv2.GaussianBlur(guide, (0, 0), sigmaX=1.0)
    blur_med  = cv2.GaussianBlur(guide, (0, 0), sigmaX=3.0)

    detail = (
        0.40 * (guide - blur_uf)
        + 0.35 * (guide - blur_fine)
        + 0.25 * (guide - blur_med)
    )

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
