"""OBS virtual camera output via pyvirtualcam.

Backends (auto-detected by pyvirtualcam):
* Windows: the OBS Virtual Camera driver (install OBS >= 26) or Unity Capture
* macOS:   OBS Virtual Camera
* Linux:   v4l2loopback (`sudo modprobe v4l2loopback devices=1`)

The virtual camera then shows up as a webcam in OBS, Discord, Zoom, etc.
"""

from __future__ import annotations

import logging

import numpy as np

from tha_wrapper.output.base import FrameOutput

logger = logging.getLogger(__name__)


class VirtualCamOutput(FrameOutput):
    def __init__(self, width: int, height: int, fps: int,
                 device: str | None = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self._cam = None

    def start(self) -> None:
        import pyvirtualcam

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        self._cam = pyvirtualcam.Camera(
            width=self.width,
            height=self.height,
            fps=self.fps,
            fmt=pyvirtualcam.PixelFormat.RGB,
            **kwargs,
        )
        logger.info(
            "virtual camera started: %s (%dx%d @ %d fps)",
            self._cam.device, self.width, self.height, self.fps,
        )

    def send(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.shape[0] != self.height or frame_rgb.shape[1] != self.width:
            import cv2

            frame_rgb = cv2.resize(
                frame_rgb, (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
        self._cam.send(frame_rgb)

    def close(self) -> None:
        if self._cam is not None:
            self._cam.close()
