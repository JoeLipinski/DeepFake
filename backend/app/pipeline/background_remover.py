"""Background removal via rembg (u2net).

Fills the removed background with white (255) so the depth estimator sees
a clean subject with no ambiguous background depth.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from app.core import model_manager

logger = logging.getLogger(__name__)


def remove_background(image: Image.Image) -> Image.Image:
    """Return an RGB image with background replaced by white."""
    logger.info("Removing background...")
    session = model_manager.get_rembg_session()

    from rembg import remove as rembg_remove

    rgba: Image.Image = rembg_remove(image, session=session)

    # Composite onto white background
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[3])  # alpha channel as mask
    logger.info("Background removal complete.")
    return background
