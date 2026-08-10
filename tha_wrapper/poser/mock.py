"""A dependency-light poser for testing the pipeline without THA3.

It does NOT produce real animation — it just applies a cheap 2D affine
transform (head translation/rotation) and a blink darkening so you can see
the drivers, control channel, and outputs working end to end before wiring
up the real THA3 models. Requires only numpy (+ Pillow for image loading);
uses OpenCV for the affine warp when available.
"""

from __future__ import annotations

import math

import numpy as np

from tha_wrapper.pose import AvatarPose
from tha_wrapper.poser.base import PoserBackend
from tha_wrapper.poser.image import load_character_rgba


class MockPoser(PoserBackend):
    def __init__(self, char_image_path: str | None = None, size: int = 512):
        self.size = size
        if char_image_path:
            self._base = load_character_rgba(char_image_path, size)
        else:
            self._base = self._placeholder(size)

    @staticmethod
    def _placeholder(size: int) -> np.ndarray:
        """A flat disc 'character' so the app runs with no image at all."""
        img = np.zeros((size, size, 4), dtype=np.uint8)
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy, r = size // 2, int(size * 0.38), int(size * 0.28)
        face = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[face] = (255, 220, 200, 255)
        for ex in (cx - r // 2, cx + r // 2):
            eye = (xx - ex) ** 2 + (yy - (cy - r // 5)) ** 2 <= (r // 6) ** 2
            img[eye] = (40, 40, 60, 255)
        return img

    def pose(self, pose: AvatarPose) -> np.ndarray:
        frame = self._base
        angle = pose.get("neck_z") * 15.0 + pose.get("body_z") * 5.0
        tx = -(pose.get("head_y") * 0.04 + pose.get("body_y") * 0.02) * self.size
        ty = pose.get("head_x") * 0.03 * self.size
        ty += (0.5 - pose.get("breathing")) * 0.004 * self.size

        try:
            import cv2

            matrix = cv2.getRotationMatrix2D(
                (self.size / 2, self.size * 0.4), angle, 1.0
            )
            matrix[0, 2] += tx
            matrix[1, 2] += ty
            frame = cv2.warpAffine(
                frame,
                matrix,
                (self.size, self.size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
        except ImportError:
            # numpy-only fallback: integer translation, no rotation.
            frame = np.roll(frame, (round(ty), round(tx)), axis=(0, 1))

        frame = frame.copy()
        blink = max(pose.get("eye_wink_left"), pose.get("eye_wink_right"))
        mouth = max(
            pose.get("mouth_aaa"),
            pose.get("mouth_ooo"),
            pose.get("mouth_eee"),
            pose.get("mouth_iii"),
            pose.get("mouth_uuu"),
            pose.get("mouth_delta"),
        )
        # Visualize blink / mouth openness as brightness bands so movement
        # is visible even on the placeholder disc.
        if blink > 0.0:
            y0, y1 = int(self.size * 0.30), int(self.size * 0.38)
            band = frame[y0:y1, :, :3].astype(np.float32)
            frame[y0:y1, :, :3] = (band * (1.0 - 0.6 * blink)).astype(np.uint8)
        if mouth > 0.0:
            y0, y1 = int(self.size * 0.44), int(self.size * 0.50)
            band = frame[y0:y1, :, :3].astype(np.float32)
            frame[y0:y1, :, :3] = np.clip(
                band * (1.0 - 0.5 * math.sin(math.pi * min(mouth, 1.0))), 0, 255
            ).astype(np.uint8)
        return frame
