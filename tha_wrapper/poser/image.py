"""Character image loading shared by the poser backends."""

from __future__ import annotations

import numpy as np


def load_character_rgba(path: str, size: int) -> np.ndarray:
    """Load a character image as (size, size, 4) uint8 RGBA.

    The image is resized (preserving aspect ratio) and centered on a
    transparent square canvas. THA3 expects a 512x512 RGBA image of a
    forward-facing anime-style character occupying roughly the upper half
    of the frame — see the THA3 project docs for the exact framing spec.
    """
    from PIL import Image

    image = Image.open(path).convert("RGBA")
    scale = size / max(image.width, image.height)
    if scale != 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return np.asarray(canvas, dtype=np.uint8).copy()
