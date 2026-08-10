from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

BACKGROUND_COLORS = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
}


class FrameOutput(ABC):
    """A sink for rendered RGB frames."""

    def start(self) -> None:
        pass

    @abstractmethod
    def send(self, frame_rgb: np.ndarray) -> None:
        """Consume an (H, W, 3) uint8 RGB frame."""

    def close(self) -> None:
        pass


def composite_background(frame_rgba: np.ndarray, background) -> np.ndarray:
    """Alpha-composite an RGBA frame onto a background; returns RGB uint8.

    `background` is a color name from BACKGROUND_COLORS, an (r, g, b) tuple,
    or an (H, W, 3) uint8 image of the same size. Green/magenta are meant
    for chroma-keying in OBS (virtual cameras carry no alpha channel).
    """
    rgb = frame_rgba[:, :, :3].astype(np.float32)
    alpha = frame_rgba[:, :, 3:4].astype(np.float32) / 255.0
    if isinstance(background, str):
        background = BACKGROUND_COLORS[background]
    if isinstance(background, np.ndarray):
        bg = background.astype(np.float32)
    else:
        bg = np.empty_like(rgb)
        bg[:] = background
    out = rgb * alpha + bg * (1.0 - alpha)
    return out.astype(np.uint8)
