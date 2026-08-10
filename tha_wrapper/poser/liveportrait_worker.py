"""LivePortrait render worker: runs inside the FasterLivePortrait venv.

This script is NOT imported by tha_wrapper. `LivePortraitPoser` launches it
as a subprocess using the FasterLivePortrait package's own Python
(`venv\\python.exe`, Python 3.10 / torch 2.4 cu121 / TensorRT 9.0.1), because
the package's TensorRT engines and custom onnxruntime build cannot be loaded
from this repo's venv. Keep it self-contained: stdlib + numpy at module
level, everything heavy imported lazily once the first command arrives.

Protocol (single TCP connection back to the parent):

    message   = 4-byte big-endian length + JSON (utf-8)
                if the JSON has "raw_len": N, exactly N raw bytes follow.

    parent -> worker
      {"cmd": "prepare", "image": "<abs path>"}
      {"cmd": "pose", "pitch": deg, "yaw": deg, "roll": deg,   # offsets from
       "tx": f, "ty": f,          # translation offset, crop-relative units
       "scale": f,                # scale multiplier (1.0 = neutral)
       "eye_ratio": f | absent,   # absolute eye-open ratio (retarget_eye)
       "lip_ratio": f | absent,   # absolute lip-open ratio (retarget_lip)
       "exp": [63 floats] | absent}  # 21x3 expression offset (emotion bank)
      {"cmd": "close"}

    worker -> parent
      prepare reply: {"ok": true, "height": h, "width": w,
                      "eye_open_ratio": f|null, "lip_ratio": f|null}
      pose reply:    {"ok": true, "height": h, "width": w, "raw_len": h*w*3}
                     + raw RGB bytes (the pasted-back full source image)
      any error:     {"ok": false, "error": "<traceback>"}

How posing works: `prepare` extracts the source image's own motion
descriptor (the neutral) and renders it once with first_frame=True, so the
pipeline's relative-motion baseline x_d_0 IS the neutral. Every `pose` then
submits neutral+offsets, which makes the pipeline's relative algebra
(R_d @ R_d_0^T @ R_s, exp - exp_0, scale/scale_0, t - t_0) collapse into
direct absolute control of head rotation, expression, scale, and
translation. Eye/lip retargeting cannot go through the pipeline's cfg flags
(in relative-motion mode those replace the whole pose with x_s), so the
retarget networks are called directly and their keypoint-space deltas are
folded into the descriptor's exp term divided by the effective scale —
algebraically identical to adding them after the keypoint transform, which
is what the pipeline's own retargeting branch does before stitching.
"""

import argparse
import json
import os
import socket
import struct
import sys
import traceback

import numpy as np


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("parent closed the connection")
        buf += chunk
    return buf


def recv_msg(sock):
    (length,) = struct.unpack(">I", _recv_exact(sock, 4))
    msg = json.loads(_recv_exact(sock, length).decode("utf-8"))
    raw = _recv_exact(sock, msg["raw_len"]) if msg.get("raw_len") else b""
    return msg, raw


def send_msg(sock, msg, raw=b""):
    if raw:
        msg = dict(msg, raw_len=len(raw))
    data = json.dumps(msg).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data + raw)


class Renderer:
    def __init__(self, cfg_path, animal):
        from omegaconf import OmegaConf
        from src.pipelines.faster_live_portrait_pipeline import (
            FasterLivePortraitPipeline,
        )
        from src.utils import utils as lp_utils

        self._utils = lp_utils
        cfg = OmegaConf.load(cfg_path)
        # Force the flag combination the descriptor math above relies on.
        ip = cfg.infer_params
        ip.flag_relative_motion = True
        ip.flag_eye_retargeting = False   # done manually via exp injection
        ip.flag_lip_retargeting = False   # done manually via exp injection
        ip.flag_stitching = True
        ip.flag_pasteback = True
        ip.flag_do_crop = True
        ip.flag_normalize_lip = False     # lip openness is fully ours
        ip.flag_crop_driving_video = False
        ip.animation_region = "all"
        self.pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=animal)
        self._first = True

    def prepare(self, image_path):
        if not self.pipe.prepare_source(image_path, realtime=False):
            return {
                "ok": False,
                "error": "LivePortrait found no face in the source image",
            }
        # src_infos[image][face] = [x_s_info, source_lmk, R_s, f_s, x_s,
        #                           x_c_s, lip_delta, flag_lip_zero, mask, M]
        info = self.pipe.src_infos[0][0]
        self.x_s_info, self.source_lmk, self.x_s = info[0], info[1], info[4]

        eye_open = lip = None
        if not self.pipe.is_animal:
            # The source's own open-eye / at-rest-lip ratios; the parent maps
            # pose parameters to absolute targets relative to these.
            self.c_s_eyes = self._utils.calc_eye_close_ratio(
                self.source_lmk[None]
            ).astype(np.float32)
            self.c_s_lip = self._utils.calc_lip_close_ratio(
                self.source_lmk[None]
            ).astype(np.float32)
            eye_open = float(self.c_s_eyes.mean())
            lip = float(self.c_s_lip[0, 0])

        # Warmup + baseline: the neutral descriptor becomes frame 0 (x_d_0).
        frame = self.render({})
        h, w = frame.shape[:2]
        return {
            "ok": True,
            "height": h,
            "width": w,
            "eye_open_ratio": eye_open,
            "lip_ratio": lip,
        }

    def _descriptor(self, p):
        n = self.x_s_info
        pitch = n["pitch"] + float(p.get("pitch", 0.0))
        yaw = n["yaw"] + float(p.get("yaw", 0.0))
        roll = n["roll"] + float(p.get("roll", 0.0))
        t = n["t"].copy()
        t[..., 0] += float(p.get("tx", 0.0))
        t[..., 1] += float(p.get("ty", 0.0))
        scale = n["scale"] * float(p.get("scale", 1.0))

        exp = n["exp"].copy()
        if p.get("exp") is not None:
            exp = exp + np.asarray(p["exp"], np.float32).reshape(exp.shape)

        if not self.pipe.is_animal:
            # Retarget deltas live in post-transform keypoint space; the
            # descriptor's exp is multiplied by scale inside
            # x_d = scale * (x_c @ R + exp) + t, so divide to compensate.
            deltas = None
            if p.get("eye_ratio") is not None:
                combined = np.concatenate(
                    [self.c_s_eyes, np.float32([[p["eye_ratio"]]])], axis=1
                )
                deltas = self.pipe.retarget_eye(self.x_s, combined)
            if p.get("lip_ratio") is not None:
                combined = np.concatenate(
                    [self.c_s_lip, np.float32([[p["lip_ratio"]]])], axis=1
                )
                lip_delta = self.pipe.retarget_lip(self.x_s, combined)
                deltas = lip_delta if deltas is None else deltas + lip_delta
            if deltas is not None:
                exp = exp + deltas.reshape(exp.shape) / float(scale.reshape(-1)[0])

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "t": t,
            "exp": exp,
            "scale": scale,
            "kp": n["kp"],
            "R": self._utils.get_rotation_matrix(pitch, yaw, roll),
        }

    def render(self, p):
        desc = self._descriptor(p)
        # NB: realtime must stay at its default (False) — run_with_pkl would
        # forward it twice into _run — and False is what enables paste_back.
        out_crop, out_org = self.pipe.run_with_pkl(
            [desc, None, None],
            self.pipe.src_imgs[0],
            self.pipe.src_infos[0],
            first_frame=self._first,
        )
        self._first = False
        return np.ascontiguousarray(out_org)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True,
                        help="FasterLivePortrait checkout directory")
    parser.add_argument("--cfg", default="configs/trt_infer.yaml",
                        help="inference config, relative to --root")
    parser.add_argument("--port", required=True, type=int,
                        help="parent's listening port on 127.0.0.1")
    parser.add_argument("--animal", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)  # the package resolves model/mask paths relative to CWD
    sys.path.insert(0, root)

    sock = socket.create_connection(("127.0.0.1", args.port), timeout=30)
    sock.settimeout(None)
    renderer = None
    try:
        while True:
            msg, _ = recv_msg(sock)
            cmd = msg.get("cmd")
            if cmd == "close":
                send_msg(sock, {"ok": True})
                break
            try:
                if cmd == "prepare":
                    if renderer is None:
                        renderer = Renderer(args.cfg, args.animal)
                    send_msg(sock, renderer.prepare(msg["image"]))
                elif cmd == "pose":
                    frame = renderer.render(msg)
                    h, w = frame.shape[:2]
                    send_msg(sock, {"ok": True, "height": h, "width": w},
                             raw=frame.tobytes())
                else:
                    send_msg(sock, {"ok": False,
                                    "error": f"unknown command {cmd!r}"})
            except Exception:
                send_msg(sock, {"ok": False, "error": traceback.format_exc()})
    finally:
        sock.close()


if __name__ == "__main__":
    main()
