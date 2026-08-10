from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tha_wrapper.pose import AvatarPose


class PoserBackend(ABC):
    """Renders the character image under a pose."""

    #: Output image side length in pixels (frames are square RGBA).
    size: int = 512

    @abstractmethod
    def pose(self, pose: AvatarPose) -> np.ndarray:
        """Render one frame.

        Returns an (size, size, 4) uint8 RGBA numpy array with straight
        (non-premultiplied) alpha.
        """

    def close(self) -> None:  # pragma: no cover - trivial default
        pass
