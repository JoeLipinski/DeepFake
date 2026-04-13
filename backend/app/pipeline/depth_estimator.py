"""Depth estimation pipeline.

For raster images (JPEG/PNG/WebP):
  1. Depth Anything V2 Large inference at 756px (was 518 — larger = more detail)
  2. Guided joint bilateral upsample back to original resolution using source image
     as the guide — source edges transfer into the depth map
  3. Source detail blend — high-frequency surface texture (pores, grain, fabric)
     extracted from source luminance and additively blended into the depth tensor
     (photo mode only; skipped for illustration/CGI to avoid blending flat-fill noise)

For SVG-derived inputs (flat vector art):
  Luminance-inversion depth — dark shapes become raised relief.
  AI inference on flat vector fills produces near-uniform depth.

Ultra mode (Marigold LCM):
  Diffusion-based depth via prs-eth/marigold-lcm-v1-0 (~45s on GPU).
  Produces sharper boundaries and better tonal separation than DAv2.
  Lazy-loaded on first use.

Saves raw float32 depth tensor as .npy for fast re-processing.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from app.core import model_manager, storage

logger = logging.getLogger(__name__)

# 756px = 54 × 14 (ViT patch size) — larger than default 518px, captures more
# fine detail without meaningful VRAM increase on GB10.
_INFERENCE_SIZE = 756

# How strongly source image high-frequency detail is blended into depth.
# Applied in photo mode only; skipped for illustration/CGI.
_DETAIL_BLEND_STRENGTH = 0.35


def estimate_depth(
    job_id: str,
    image: Image.Image,
    use_luminance: bool = False,
    image_type: str = "photo",
    use_marigold: bool = False,
) -> np.ndarray:
    """Run depth estimation, save raw tensor, return float32 array [0,1].

    Args:
        job_id:       Job identifier for storage paths.
        image:        Source PIL image (RGB or L).
        use_luminance: Use luminance-inversion depth (SVG/flat vector art).
        image_type:   "photo" or "illustration" — controls detail blending
                      and bilateral filter tightness.
        use_marigold: Use Marigold LCM diffusion depth (Ultra mode).
    """
    original_size = image.size  # (W, H)
    mode = "luminance" if use_luminance else ("marigold" if use_marigold else f"ai/{image_type}")
    logger.info(
        "Job %s: depth estimation %dx%d (mode=%s)",
        job_id, *original_size, mode,
    )

    if use_luminance:
        depth_arr = _luminance_depth(image)
    elif use_marigold:
        depth_arr = _marigold_depth(image, original_size, image_type)
    else:
        depth_arr = _ai_depth(image, original_size, image_type)
        if image_type == "photo":
            depth_arr = _blend_source_detail(depth_arr, image)

    npy_path = storage.raw_depth_path(job_id)
    np.save(str(npy_path), depth_arr)
    logger.info("Job %s: raw depth saved (%dx%d)", job_id, *original_size)
    return depth_arr


def apply_sam_refinement(
    job_id: str,
    depth: np.ndarray,
    image: Image.Image,
) -> np.ndarray:
    """Public entry point: per-region depth refinement (felzenszwalb segmentation).

    Called by the worker as a separate step after estimate_depth() so the
    job queue can emit a distinct progress update.  Re-saves raw_depth.npy
    with the refined tensor so all downstream variant generation uses it.

    Args:
        job_id: Job identifier for storage paths.
        depth:  Float32 depth array [0,1] from estimate_depth().
        image:  Original source image (used to guide segmentation).
    """
    logger.info("Job %s: per-region refinement start", job_id)
    refined = _apply_sam_refinement(depth, image)
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
# Internal helpers
# ---------------------------------------------------------------------------

def _ai_depth(
    image: Image.Image,
    original_size: tuple[int, int],
    image_type: str = "photo",
) -> np.ndarray:
    """Depth Anything V2 at _INFERENCE_SIZE, with guided bilateral upsample."""
    inference_img = _resize_for_inference(image)

    pipe = model_manager.get_depth_pipe()
    result = pipe(inference_img)
    depth_pil: Image.Image = result["depth"]

    # Normalise to [0, 1]
    depth_arr = np.array(depth_pil).astype(np.float32)
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_arr = np.zeros_like(depth_arr)

    # Guided bilateral upsample: source image guides edge-aware upscaling
    if depth_pil.size != original_size:
        depth_arr = _guided_upsample(depth_arr, image, original_size, image_type)

    return depth_arr


def _marigold_depth(
    image: Image.Image,
    original_size: tuple[int, int],
    image_type: str = "photo",
) -> np.ndarray:
    """Marigold LCM diffusion depth (Ultra mode).

    Uses prs-eth/marigold-lcm-v1-0 (4 denoising steps, ensemble of 5 runs).
    Lazy-loaded on first call — ~1.7GB download on first use.
    """
    pipe = model_manager.get_marigold_pipe()

    # Marigold handles its own internal resize; we pass the original image.
    # match_input_res=False returns depth at processing_res, then we apply
    # our guided bilateral upsample for consistency with the rest of the pipeline.
    # Marigold requires RGB; image may be RGBA after background removal
    rgb_image = image.convert("RGB")

    output = pipe(
        rgb_image,
        num_inference_steps=4,  # LCM fast mode
        ensemble_size=5,        # Average 5 runs → stable, low noise
    )

    # prediction shape varies by diffusers version: (1,1,H,W) or (1,H,W) or (H,W).
    # squeeze() collapses all leading size-1 dims safely.
    depth_arr = output.prediction.squeeze().astype(np.float32)

    # Normalise (should already be [0,1] but enforce it)
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_arr = np.zeros_like(depth_arr)

    # Guided upsample back to original resolution
    w, h = original_size
    if depth_arr.shape != (h, w):
        depth_arr = _guided_upsample(depth_arr, image, original_size, image_type)

    return depth_arr


def _guided_upsample(
    depth_low: np.ndarray,
    source: Image.Image,
    target_size: tuple[int, int],  # (W, H)
    image_type: str = "photo",
) -> np.ndarray:
    """Upsample depth map guided by the high-res source image.

    Steps:
      1. Bicubic upsample to target size
      2. Build an edge-weight map from source luminance (Scharr gradient)
      3. Use edge weights to blend a bilateral-filtered version with the
         bicubic version — sharp at source edges, smooth in flat areas

    illustration mode: tighter bilateral (sigmaColor/Space 20 vs 40) so flat
    fills stay clean and hard outline boundaries stay crisp.
    """
    w, h = target_size

    # 1. Bicubic base
    depth_u8 = (depth_low * 255).clip(0, 255).astype(np.uint8)
    depth_up = cv2.resize(depth_u8, (w, h), interpolation=cv2.INTER_CUBIC)

    # 2. Source edge map (guides where to preserve vs. smooth)
    guide = np.array(source.resize((w, h), Image.LANCZOS).convert("L"))
    sx = cv2.Scharr(guide, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(guide, cv2.CV_32F, 0, 1)
    edge_mag = np.sqrt(sx ** 2 + sy ** 2)
    edge_mag = (edge_mag / (edge_mag.max() + 1e-8)).astype(np.float32)
    # Smooth the edge mask slightly so blending isn't jagged
    edge_mask = cv2.GaussianBlur(edge_mag, (0, 0), sigmaX=2.0)

    # 3. Bilateral-filtered version (smoothed, good for flat areas)
    # Illustration: tighter sigma keeps flat fills clean; photo: softer blending
    sigma = 20 if image_type == "illustration" else 40
    depth_bilateral = cv2.bilateralFilter(
        depth_up.astype(np.float32), d=9, sigmaColor=sigma, sigmaSpace=sigma
    )

    # Blend: strong edges keep bicubic sharpness; flat areas use smooth bilateral
    depth_blended = (
        edge_mask * depth_up.astype(np.float32)
        + (1.0 - edge_mask) * depth_bilateral
    )

    return depth_blended.clip(0, 255).astype(np.float32) / 255.0


def _blend_source_detail(
    depth: np.ndarray,
    source: Image.Image,
    strength: float = _DETAIL_BLEND_STRENGTH,
) -> np.ndarray:
    """Additively blend high-frequency surface detail from source into depth.

    Photo mode only. For illustration/CGI, flat fills produce noise rather than
    real texture — skip this step via the image_type check in estimate_depth().

    Extracts the detail layer (source - gaussian_blur(source)) and adds it
    to the depth map. This transfers surface texture — pores, grain, fabric
    weave — that the depth model misses because it focuses on geometry.
    """
    h, w = depth.shape
    guide = np.array(
        source.resize((w, h), Image.LANCZOS).convert("L")
    ).astype(np.float32) / 255.0

    # Multi-scale detail: fine (σ=1) + medium (σ=3)
    blur_fine = cv2.GaussianBlur(guide, (0, 0), sigmaX=1.0)
    blur_med = cv2.GaussianBlur(guide, (0, 0), sigmaX=3.0)
    detail = 0.6 * (guide - blur_fine) + 0.4 * (guide - blur_med)

    # Normalise detail to [-0.5, 0.5] so strength is consistent across images
    d_abs_max = np.abs(detail).max()
    if d_abs_max > 1e-6:
        detail = detail / (d_abs_max * 2.0)

    enhanced = depth + strength * detail
    return np.clip(enhanced, 0.0, 1.0)


def _apply_sam_refinement(depth: np.ndarray, image: Image.Image) -> np.ndarray:
    """Felzenszwalb graph-based segmentation → per-region smoothing + boundary sharpening.

    Felzenszwalb segments along colour/intensity boundaries — exactly the hard
    edges that illustrations, CGI, and AI art have between fills.  The result is
    equivalent to SAM for static flat-colour content without requiring a GPU
    model download.

    Algorithm:
      1. Segment the source image into regions with felzenszwalb.
      2. For each region: replace depth values with a lightly Gaussian-smoothed
         version (σ=2) — kills flat-fill noise while keeping broad gradients.
      3. Accumulate a boundary ring (erode + XOR) for every region edge.
      4. Apply unsharp mask to the refined depth and blend it at boundaries —
         illustration outlines become crisp ridges in the depth map.
    """
    from skimage.segmentation import felzenszwalb

    h, w = depth.shape

    # Resize to depth map dimensions for pixel-accurate mask alignment
    image_np = np.array(image.convert("RGB").resize((w, h), Image.LANCZOS))

    # scale=150: merges small colour patches into broader regions (good for flat fills)
    # sigma=0.8: light pre-blur handles anti-aliased edges without blurring hard lines
    # min_size=200: ignore dust/noise regions smaller than ~200px
    labels = felzenszwalb(image_np, scale=150, sigma=0.8, min_size=200)
    n_regions = labels.max() + 1
    logger.debug("Felzenszwalb: %d regions at %dx%d", n_regions, w, h)

    refined = depth.copy()
    boundary_map = np.zeros((h, w), dtype=np.float32)

    # Light smooth: σ=2 kills pixel-level noise, keeps broad depth gradients
    depth_smooth = cv2.GaussianBlur(depth, (0, 0), sigmaX=2.0)

    for label_id in range(n_regions):
        seg = (labels == label_id)
        area = int(seg.sum())
        if area < 50:
            continue

        # Per-region smoothing
        refined[seg] = depth_smooth[seg]

        # Boundary ring — wider for larger regions
        radius = 4 if area > 10_000 else 2
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        eroded = cv2.erode(seg.astype(np.uint8), k)
        boundary_map = np.maximum(
            boundary_map, (seg.astype(np.uint8) - eroded).astype(np.float32)
        )

    # Boundary sharpening — unsharp mask blended at region edges
    sharpened = _unsharp_depth(refined, sigma=1.0, strength=2.5)
    boundary_weight = cv2.GaussianBlur(boundary_map, (0, 0), sigmaX=1.5).clip(0.0, 1.0)
    refined = (1.0 - boundary_weight) * refined + boundary_weight * sharpened

    return np.clip(refined, 0.0, 1.0)


def _unsharp_depth(depth: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """Unsharp mask on a depth map — amplifies edges and transitions."""
    blurred = cv2.GaussianBlur(depth, (0, 0), sigmaX=sigma)
    return depth + strength * (depth - blurred)


def _luminance_depth(image: Image.Image) -> np.ndarray:
    """Luminance-inversion depth for SVG / flat vector art.

    Dark areas → high depth (deep cut) — standard for engraving logos.
    """
    gray = np.array(image.convert("L")).astype(np.float32) / 255.0
    depth = 1.0 - gray
    # Smooth aliased SVG raster edges
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=1.0)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


def _resize_for_inference(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = _INFERENCE_SIZE / max(w, h)
    if scale < 1.0:
        new_w = int(round(w * scale / 14) * 14)  # snap to ViT patch grid
        new_h = int(round(h * scale / 14) * 14)
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image
