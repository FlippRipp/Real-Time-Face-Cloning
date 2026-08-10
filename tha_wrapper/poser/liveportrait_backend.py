"""The LivePortrait poser backend (FasterLivePortrait, out-of-process).

Wraps the FasterLivePortrait TensorRT pipeline
(https://github.com/warmshao/FasterLivePortrait — the extract-and-run
Windows package) as a `PoserBackend`. LivePortrait is driven by a motion
descriptor (head angles, translation, scale, implicit expression deltas),
not by pixels, so an `AvatarPose` can be mapped onto it directly; no webcam
or driving video is involved.

The pipeline runs **out of process**: its TensorRT engines were built by the
package's bundled venv (Python 3.10, torch 2.4 cu121, TensorRT 9.0.1 dev
build, custom onnxruntime-gpu) and engines only load under that exact
TensorRT version, none of which can be installed into this repo's Python
3.13 venv. This backend therefore spawns `liveportrait_worker.py` with the
package's own `venv\\python.exe` and exchanges motion parameters / frames
over a localhost socket (~1 ms per 512px frame, negligible next to
inference).

Pose mapping (step 2 of docs/liveportrait-integration.md):

  head_x/head_y/neck_z -> pitch/yaw/roll degrees (~15 deg per THA3 unit)
  body_y/body_z        -> horizontal translation (+ a little extra roll)
  breathing            -> subtle scale oscillation
  eye_wink_* etc.      -> retarget_eye ratio (one ratio for both eyes —
                          the retarget net takes a single target, so
                          asymmetric winks close both; exp offsets can
                          refine this later)
  mouth_aaa etc.       -> retarget_lip ratio
  everything else      -> ignored for now; `exp_offset` is the hook where
                          the recorded expression bank (step 3) plugs in.

All mapping constants below are tuned blind (no live testing yet) — like
the webcam driver baselines, treat them as the first suspects when motion
looks wrong, including the signs of the angle/translation mappings.

LivePortrait outputs RGB with no alpha. The character's own alpha channel
(if any) is reused as a static matte: the source image is flattened onto
gray for the pipeline, and the output frame gets the original alpha back.
That is exact for everything outside the face crop and a close
approximation near the silhouette while the head moves.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import subprocess
import tempfile
from typing import Optional

import numpy as np

from tha_wrapper.pose import AvatarPose
from tha_wrapper.poser.base import PoserBackend

logger = logging.getLogger(__name__)


class LivePortraitPoser(PoserBackend):
    # ---- mapping constants (untested-by-live-use; tune freely) ----
    HEAD_DEGREES = 15.0        # degrees per THA3 head unit
    BODY_ROLL_DEGREES = 4.0    # extra roll per body_z unit
    BODY_SWAY_TX = 0.02        # x-translation per body_y unit (crop units)
    BODY_LEAN_TX = 0.012       # x-translation per body_z unit
    BREATH_SCALE = 0.012       # peak-to-peak scale swing over a breath cycle
    EYE_WIDEN_GAIN = 0.25      # eye-open ratio boost at full eye_surprised
    EYE_RATIO_MAX = 0.8
    LIP_OPEN_SPAN = 0.35       # lip ratio added at a full-open viseme
    MATTE_RGB = (128, 128, 128)  # under-alpha fill fed to the pipeline

    ACCEPT_TIMEOUT = 60.0      # worker process startup
    PREPARE_TIMEOUT = 600.0    # TRT engine load + prepare_source + warmup
    POSE_TIMEOUT = 30.0

    def __init__(
        self,
        char_image_path: str,
        lp_path: str,
        cfg: str = "configs/trt_infer.yaml",
        python: Optional[str] = None,
        animal: bool = False,
    ):
        lp_path = os.path.abspath(os.path.expanduser(lp_path))
        if not os.path.isfile(
            os.path.join(lp_path, "src", "pipelines",
                         "faster_live_portrait_pipeline.py")
        ):
            raise FileNotFoundError(
                f"{lp_path} does not look like a FasterLivePortrait checkout "
                "(no src/pipelines/faster_live_portrait_pipeline.py). Pass "
                "--lp-path pointing at the extract-and-run package."
            )
        if python is None:
            python = os.path.join(lp_path, "venv", "python.exe")
        if not os.path.isfile(python):
            raise FileNotFoundError(
                f"worker python not found at {python}; pass --lp-python "
                "(the FasterLivePortrait package normally bundles "
                "venv\\python.exe)"
            )

        self._proc = None
        self._sock = None
        self._temp_image = None

        #: Hook for the future expression bank: a (21, 3) float array added
        #: to the neutral expression every frame (None = no offset).
        self.exp_offset: Optional[np.ndarray] = None

        source_path = self._flatten_character(char_image_path)
        self._spawn_worker(python, lp_path, cfg, animal)

        reply, _ = self._request(
            {"cmd": "prepare", "image": source_path},
            timeout=self.PREPARE_TIMEOUT,
        )
        self._height, self._width = reply["height"], reply["width"]
        self._eye_open = reply.get("eye_open_ratio")
        self._lip_neutral = reply.get("lip_ratio")
        if self._eye_open is None:
            logger.warning(
                "no facial-landmark ratios from the worker (animal mode?); "
                "eye and mouth posing disabled"
            )
        self._build_canvas()
        logger.info(
            "LivePortrait ready: %dx%d source, %d px square output",
            self._width, self._height, self.size,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _flatten_character(self, char_image_path: str) -> str:
        """Load the character, keep its alpha, hand the pipeline opaque RGB."""
        from PIL import Image

        char_image_path = os.path.abspath(os.path.expanduser(char_image_path))
        rgba = np.asarray(Image.open(char_image_path).convert("RGBA"))
        self._char_alpha = rgba[..., 3]
        if not (self._char_alpha < 255).any():
            self._char_alpha = None  # fully opaque; no matte needed
            return char_image_path

        alpha = self._char_alpha[..., None].astype(np.float32) / 255.0
        matte = np.float32(self.MATTE_RGB)
        flat = (rgba[..., :3] * alpha + matte * (1.0 - alpha)).astype(np.uint8)
        fd, path = tempfile.mkstemp(suffix=".png", prefix="lp_char_")
        os.close(fd)
        Image.fromarray(flat).save(path)
        self._temp_image = path
        return path

    def _spawn_worker(self, python, lp_path, cfg, animal):
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "liveportrait_worker.py")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        cmd = [python, worker, "--root", lp_path, "--cfg", cfg,
               "--port", str(port)]
        if animal:
            cmd.append("--animal")
        logger.info("starting LivePortrait worker: %s", " ".join(cmd))
        # stdout/stderr inherit: the pipeline's own logging stays visible.
        self._proc = subprocess.Popen(cmd, cwd=lp_path)
        listener.settimeout(self.ACCEPT_TIMEOUT)
        try:
            self._sock, _ = listener.accept()
        except socket.timeout:
            raise RuntimeError(
                "LivePortrait worker did not connect within "
                f"{self.ACCEPT_TIMEOUT:.0f}s — check the worker console "
                "output above"
            ) from None
        finally:
            listener.close()

    def _build_canvas(self):
        """Square RGBA canvas; per frame only the RGB region is rewritten."""
        from PIL import Image

        h, w = self._height, self._width
        self.size = side = max(h, w)
        self._y0 = (side - h) // 2
        self._x0 = (side - w) // 2
        if self._char_alpha is not None:
            alpha = np.asarray(
                Image.fromarray(self._char_alpha).resize(
                    (w, h), Image.BILINEAR
                )
            )
        else:
            alpha = np.full((h, w), 255, np.uint8)
        self._canvas = np.zeros((side, side, 4), dtype=np.uint8)
        self._canvas[self._y0:self._y0 + h, self._x0:self._x0 + w, 3] = alpha

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def _request(self, msg: dict, timeout: float):
        try:
            self._sock.settimeout(timeout)
            data = json.dumps(msg).encode("utf-8")
            self._sock.sendall(struct.pack(">I", len(data)) + data)
            (length,) = struct.unpack(">I", self._recv_exact(4))
            reply = json.loads(self._recv_exact(length).decode("utf-8"))
            raw = self._recv_exact(reply["raw_len"]) if reply.get("raw_len") \
                else b""
        except (OSError, ConnectionError) as exc:
            code = self._proc.poll() if self._proc else None
            raise RuntimeError(
                f"LivePortrait worker communication failed ({exc}); "
                + (f"worker exited with code {code}" if code is not None
                   else "worker still running")
                + " — see the worker console output"
            ) from exc
        if not reply.get("ok"):
            error = reply.get("error", "unknown worker error")
            if "no face" in error:
                error += (
                    "\nLivePortrait's face detector needs human-like facial "
                    "features. Try a tighter head-and-shoulders crop "
                    "(python -m tha_wrapper.prepare can make one); landmark "
                    "injection for stylized characters is a planned follow-up."
                )
            raise RuntimeError(f"LivePortrait worker error:\n{error}")
        return reply, raw

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("worker closed the connection")
            buf += chunk
        return buf

    # ------------------------------------------------------------------
    # AvatarPose -> motion parameters (the adapter mapping)
    # ------------------------------------------------------------------

    def _motion_params(self, pose: AvatarPose) -> dict:
        g = pose.get
        params = {
            # Head rotation. Sign conventions are a guess until live-tested.
            "pitch": g("head_x") * self.HEAD_DEGREES,
            "yaw": g("head_y") * self.HEAD_DEGREES,
            "roll": g("neck_z") * self.HEAD_DEGREES
                    + g("body_z") * self.BODY_ROLL_DEGREES,
            # Body sway reads as the face translating within the frame.
            "tx": -(g("body_y") * self.BODY_SWAY_TX
                    + g("body_z") * self.BODY_LEAN_TX),
            "ty": 0.0,
            # Breathing: gentle in-out zoom around the neutral scale.
            "scale": 1.0 + (g("breathing") - 0.5) * self.BREATH_SCALE,
        }

        if self._eye_open is not None:
            # All the eye morphs that close the lids, weighted by how far
            # each one closes them in THA3; one shared ratio for both eyes.
            closure = max(
                g("eye_wink_left"), g("eye_wink_right"),
                0.8 * max(g("eye_happy_wink_left"), g("eye_happy_wink_right")),
                0.6 * max(g("eye_relaxed_left"), g("eye_relaxed_right")),
                0.35 * max(g("eye_unimpressed_left"),
                           g("eye_unimpressed_right")),
            )
            widen = max(g("eye_surprised_left"), g("eye_surprised_right"))
            ratio = self._eye_open * (1.0 - min(1.0, closure)) \
                * (1.0 + self.EYE_WIDEN_GAIN * widen)
            params["eye_ratio"] = min(self.EYE_RATIO_MAX, max(0.0, ratio))

        if self._lip_neutral is not None:
            # Viseme weights approximate how wide each vowel opens the jaw.
            openness = max(
                g("mouth_aaa"), 0.9 * g("mouth_ooo"), 0.8 * g("mouth_delta"),
                0.6 * g("mouth_eee"), 0.5 * g("mouth_iii"),
                0.45 * g("mouth_uuu"),
            )
            params["lip_ratio"] = (
                self._lip_neutral + min(1.0, openness) * self.LIP_OPEN_SPAN
            )

        if self.exp_offset is not None:
            params["exp"] = (
                np.asarray(self.exp_offset, np.float32).reshape(-1).tolist()
            )
        return params

    # ------------------------------------------------------------------
    # PoserBackend interface
    # ------------------------------------------------------------------

    def pose(self, pose: AvatarPose) -> np.ndarray:
        reply, raw = self._request(
            {"cmd": "pose", **self._motion_params(pose)},
            timeout=self.POSE_TIMEOUT,
        )
        frame = np.frombuffer(raw, np.uint8).reshape(
            reply["height"], reply["width"], 3
        )
        self._canvas[self._y0:self._y0 + self._height,
                     self._x0:self._x0 + self._width, :3] = frame
        return self._canvas

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._request({"cmd": "close"}, timeout=5.0)
            except RuntimeError:
                pass
            self._sock.close()
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
            self._proc = None
        if self._temp_image is not None:
            try:
                os.unlink(self._temp_image)
            except OSError:
                pass
            self._temp_image = None
