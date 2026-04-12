"""SVG → PIL Image rasterization via cairosvg.

Rasterizes at 300 DPI, capped at 4096px on the longest side.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)

_MAX_PX = 4096
_DPI = 300


def rasterize_svg(svg_bytes: bytes) -> Image.Image:
    """Convert raw SVG bytes to an RGB PIL Image."""
    try:
        import cairosvg
    except ImportError:
        raise RuntimeError("cairosvg is not installed. Cannot process SVG files.")

    logger.info("Rasterizing SVG...")
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, dpi=_DPI)
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    # Cap at max size
    w, h = image.size
    if max(w, h) > _MAX_PX:
        scale = _MAX_PX / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    logger.info("SVG rasterized to %dx%d.", *image.size)
    return image
