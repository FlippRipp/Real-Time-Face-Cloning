from __future__ import annotations

from abc import ABC, abstractmethod

from tha_wrapper.pose import AvatarPose


class PoseDriver(ABC):
    """Produces the avatar's pose, advanced once per rendered frame."""

    @abstractmethod
    def update(self, dt: float) -> AvatarPose:
        """Advance the driver by `dt` seconds and return the current pose."""

    def start(self) -> None:
        """Acquire resources (cameras, threads). Called once before the loop."""

    def close(self) -> None:
        """Release resources. Called once after the loop."""
