"""OpenCV preview window with a small HUD and hotkeys.

Hotkeys (window must be focused):
  q / ESC  quit
  c        calibrate the webcam driver (face the camera neutrally)
  1-9      emotion presets (procedural/hybrid mode)
  0        reset to neutral
  space    blink
  t        toggle random talking
"""

from __future__ import annotations

import numpy as np

from tha_wrapper.output.base import FrameOutput


class WindowOutput(FrameOutput):
    def __init__(self, title: str = "tha_wrapper"):
        self.title = title
        self.last_key: int = -1
        self._fps_text = ""

    def start(self) -> None:
        import cv2

        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)

    def set_hud(self, text: str) -> None:
        self._fps_text = text

    def send(self, frame_rgb: np.ndarray) -> None:
        import cv2

        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self._fps_text:
            cv2.putText(
                bgr, self._fps_text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                bgr, self._fps_text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.imshow(self.title, bgr)
        self.last_key = cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(self.title)
