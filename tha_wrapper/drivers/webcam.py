"""Webcam pose driver: MediaPipe FaceLandmarker -> THA3 pose parameters.

Same role as EasyVtuber's webcam/OpenSeeFace input path, but self-contained:
only OpenCV + MediaPipe. Uses the mediapipe `tasks` FaceLandmarker API (the
legacy `solutions.face_mesh` API was dropped in mediapipe 1.0 and never
shipped for Python 3.13); it outputs the same 478 landmarks, so the mapping
below is unchanged from the FaceMesh version. The ~4 MB `.task` model file
is downloaded on first use (or by setup) into `models/`.

A capture thread keeps grabbing frames so the render loop never blocks on
the camera; `update()` just processes the latest frame.

Mapping summary:
* eye openness  -> eye_wink_left/right (eye aspect ratio, calibrated)
* mouth open    -> mouth_aaa, mouth wide/narrow -> mouth_iii / mouth_uuu mix
* brow height   -> eyebrow_raised / eyebrow_lowered
* iris position -> iris_rotation_x/y (iris landmarks 468..477)
* head pose     -> head_x (pitch), head_y (yaw), neck_z (roll) via solvePnP

Press-to-calibrate: call `calibrate()` while facing the camera neutrally; the
current measurements become the neutral baselines.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import urllib.request
from typing import Dict, Optional

import numpy as np

from tha_wrapper.drivers.base import PoseDriver
from tha_wrapper.pose import AvatarPose

logger = logging.getLogger(__name__)

# FaceLandmarker model, pinned (google-hosted, ~4 MB).
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_FACE_LANDMARKER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "face_landmarker.task",
)


def ensure_face_landmarker_model(path: str = DEFAULT_FACE_LANDMARKER_PATH) -> str:
    """Download the FaceLandmarker .task model to `path` if not present."""
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info("downloading %s to %s", FACE_LANDMARKER_MODEL_URL, path)
        partial = path + ".part"
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, partial)
        os.replace(partial, path)
    return path


# Face landmark indices (FaceLandmarker, same topology as FaceMesh with
# refine_landmarks: 468 mesh points + 10 iris points).
_L_EYE_TOP, _L_EYE_BOTTOM, _L_EYE_OUTER, _L_EYE_INNER = 159, 145, 33, 133
_R_EYE_TOP, _R_EYE_BOTTOM, _R_EYE_OUTER, _R_EYE_INNER = 386, 374, 263, 362
_L_BROW, _R_BROW = 105, 334
_LIP_TOP, _LIP_BOTTOM = 13, 14
_MOUTH_L, _MOUTH_R = 61, 291
_NOSE_TIP, _CHIN = 1, 152
_L_IRIS = (468, 469, 470, 471, 472)
_R_IRIS = (473, 474, 475, 476, 477)

# Generic 3D face model points (mm) for solvePnP head pose:
# nose tip, chin, left eye outer, right eye outer, left mouth, right mouth.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)
_PNP_LANDMARKS = (_NOSE_TIP, _CHIN, _L_EYE_OUTER, _R_EYE_OUTER, _MOUTH_L, _MOUTH_R)


class _Baseline:
    """Neutral-face measurements, refined by calibration."""

    eye_open = 0.28        # eye height / eye width when open
    mouth_open = 0.02      # lip gap / mouth width when closed
    mouth_wide = 0.42      # mouth width / face width, relaxed
    brow = 0.20            # brow-to-eye distance / face height, relaxed


class WebcamDriver(PoseDriver):
    def __init__(
        self,
        camera_index: int = 0,
        smoothing: float = 0.35,
        mirror: bool = True,
        model_path: str = DEFAULT_FACE_LANDMARKER_PATH,
    ):
        """`smoothing` is EMA weight of the previous value (0 = raw)."""
        self.camera_index = camera_index
        self.smoothing = smoothing
        self.mirror = mirror
        self.model_path = model_path
        self.baseline = _Baseline()

        self._capture = None
        self._mp = None
        self._landmarker = None
        self._start_time = 0.0
        self._last_timestamp_ms = -1
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._calibrate_request = False

        self._smoothed: Dict[str, float] = {}
        self._face_seen = False

    # ------------------------------------------------------------------

    def start(self) -> None:
        import cv2
        import mediapipe as mp

        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open webcam index {self.camera_index}")
        self._mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=ensure_face_landmarker_model(self.model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
        if self._landmarker is not None:
            self._landmarker.close()

    def calibrate(self) -> None:
        """Treat the current face as neutral (call while relaxed, facing camera)."""
        self._calibrate_request = True

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._frame_lock:
                self._latest_frame = frame

    # ------------------------------------------------------------------

    def update(self, dt: float) -> AvatarPose:
        import cv2

        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode requires strictly increasing timestamps.
            timestamp_ms = max(
                int((time.monotonic() - self._start_time) * 1000.0),
                self._last_timestamp_ms + 1,
            )
            self._last_timestamp_ms = timestamp_ms
            results = self._landmarker.detect_for_video(image, timestamp_ms)
            landmarks = results.face_landmarks[0] if results.face_landmarks else None
            if landmarks is not None:
                self._face_seen = True
                raw = self._extract(landmarks, frame.shape[1], frame.shape[0])
                self._smooth_toward(raw)
            else:
                self._face_seen = False

        if not self._face_seen:
            # Ease back to neutral when tracking is lost.
            self._smooth_toward({n: 0.0 for n in self._smoothed})

        return AvatarPose(self._smoothed)

    def _smooth_toward(self, raw: Dict[str, float]) -> None:
        alpha = min(0.95, max(0.0, self.smoothing))
        for name, value in raw.items():
            prev = self._smoothed.get(name, 0.0)
            self._smoothed[name] = prev * alpha + value * (1.0 - alpha)

    # ------------------------------------------------------------------

    def _extract(self, lm, width: int, height: int) -> Dict[str, float]:
        def point(i) -> np.ndarray:
            return np.array([lm[i].x * width, lm[i].y * height])

        def dist(a, b) -> float:
            return float(np.linalg.norm(point(a) - point(b)))

        face_width = dist(_L_EYE_OUTER, _R_EYE_OUTER)
        face_height = dist(_NOSE_TIP, _CHIN) * 2.0
        if face_width < 1e-3 or face_height < 1e-3:
            return {}
        base = self.baseline

        # --- Eyes -----------------------------------------------------
        l_eye_w = max(dist(_L_EYE_OUTER, _L_EYE_INNER), 1e-3)
        r_eye_w = max(dist(_R_EYE_OUTER, _R_EYE_INNER), 1e-3)
        l_open = dist(_L_EYE_TOP, _L_EYE_BOTTOM) / l_eye_w
        r_open = dist(_R_EYE_TOP, _R_EYE_BOTTOM) / r_eye_w

        # --- Mouth ----------------------------------------------------
        mouth_w = max(dist(_MOUTH_L, _MOUTH_R), 1e-3)
        mouth_open = dist(_LIP_TOP, _LIP_BOTTOM) / mouth_w
        mouth_wide = mouth_w / face_width

        # --- Brows ----------------------------------------------------
        brow = (
            dist(_L_BROW, _L_EYE_TOP) + dist(_R_BROW, _R_EYE_TOP)
        ) / (2.0 * face_height)

        if self._calibrate_request:
            self._calibrate_request = False
            base.eye_open = (l_open + r_open) / 2.0
            base.mouth_open = mouth_open
            base.mouth_wide = mouth_wide
            base.brow = brow
            logger.info(
                "calibrated: eye=%.3f mouth=%.3f wide=%.3f brow=%.3f",
                base.eye_open, base.mouth_open, base.mouth_wide, base.brow,
            )

        raw: Dict[str, float] = {}

        def clamp01(x: float) -> float:
            return min(1.0, max(0.0, x))

        # Eye openness -> wink. Fully open at baseline, closed at ~35% of it.
        closed = base.eye_open * 0.35
        span = max(base.eye_open - closed, 1e-3)
        wink_l = clamp01(1.0 - (l_open - closed) / span)
        wink_r = clamp01(1.0 - (r_open - closed) / span)
        # MediaPipe's left eye (33/133...) is the *image* left, i.e. the
        # subject's right eye. With mirroring on, map straight across.
        if self.mirror:
            raw["eye_wink_left"], raw["eye_wink_right"] = wink_l, wink_r
        else:
            raw["eye_wink_left"], raw["eye_wink_right"] = wink_r, wink_l

        # Mouth: openness drives aaa; width modulates iii (wide) / uuu (pursed).
        raw["mouth_aaa"] = clamp01((mouth_open - base.mouth_open) / 0.55)
        wide_delta = (mouth_wide - base.mouth_wide) / 0.12
        if wide_delta >= 0:
            raw["mouth_iii"] = clamp01(wide_delta) * (1.0 - raw["mouth_aaa"])
        else:
            raw["mouth_uuu"] = clamp01(-wide_delta) * (1.0 - raw["mouth_aaa"])

        # Brows.
        brow_delta = (brow - base.brow) / (base.brow * 0.6 + 1e-3)
        if brow_delta >= 0:
            for side in ("left", "right"):
                raw[f"eyebrow_raised_{side}"] = clamp01(brow_delta)
        else:
            for side in ("left", "right"):
                raw[f"eyebrow_lowered_{side}"] = clamp01(-brow_delta)

        # Iris (FaceLandmarker always includes iris points 468..477).
        if len(lm) > _R_IRIS[-1]:
            l_iris = np.mean([point(i) for i in _L_IRIS], axis=0)
            r_iris = np.mean([point(i) for i in _R_IRIS], axis=0)
            def iris_offset(iris, inner, outer, top, bottom):
                center = (point(inner) + point(outer)) / 2.0
                w = max(dist(inner, outer), 1e-3)
                h = max(dist(top, bottom), 1e-3)
                dx = (iris[0] - center[0]) / (w * 0.5)
                dy = (iris[1] - center[1]) / (h * 0.9)
                return dx, dy
            ldx, ldy = iris_offset(l_iris, _L_EYE_INNER, _L_EYE_OUTER,
                                   _L_EYE_TOP, _L_EYE_BOTTOM)
            rdx, rdy = iris_offset(r_iris, _R_EYE_INNER, _R_EYE_OUTER,
                                   _R_EYE_TOP, _R_EYE_BOTTOM)
            dx = (ldx + rdx) / 2.0
            dy = (ldy + rdy) / 2.0
            sign = -1.0 if self.mirror else 1.0
            raw["iris_rotation_y"] = float(np.clip(sign * dx, -1.0, 1.0))
            raw["iris_rotation_x"] = float(np.clip(-dy, -1.0, 1.0))

        # Head pose.
        pitch, yaw, roll = self._head_pose(lm, width, height)
        sign = -1.0 if self.mirror else 1.0
        raw["head_x"] = float(np.clip(pitch / 15.0, -1.0, 1.0))
        raw["head_y"] = float(np.clip(sign * yaw / 15.0, -1.0, 1.0))
        raw["neck_z"] = float(np.clip(sign * roll / 15.0, -1.0, 1.0))
        raw["body_y"] = raw["head_y"] * 0.3
        raw["body_z"] = raw["neck_z"] * 0.3

        return raw

    def _head_pose(self, lm, width: int, height: int):
        """Return (pitch, yaw, roll) in degrees via solvePnP."""
        import cv2

        image_points = np.array(
            [(lm[i].x * width, lm[i].y * height) for i in _PNP_LANDMARKS],
            dtype=np.float64,
        )
        focal = width
        camera_matrix = np.array(
            [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        ok, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0
        rot, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
            yaw = math.degrees(math.atan2(-rot[2, 0], sy))
            roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rot[1, 2], rot[1, 1]))
            yaw = math.degrees(math.atan2(-rot[2, 0], sy))
            roll = 0.0
        # The model has +y up in face space; solvePnP pitch comes out offset
        # by 180 degrees. Normalize to small angles around neutral.
        pitch = pitch - 180.0 if pitch > 90 else pitch + 180.0 if pitch < -90 else pitch
        return -pitch, yaw, roll
