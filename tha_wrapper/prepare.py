"""Character image preparation wizard for the THA3 input constraints.

    python -m tha_wrapper.prepare INPUT [-o OUTPUT] [--auto]

Takes arbitrary character art and produces a THA3-ready 512x512 RGBA PNG in
`characters/`: transparent background (AI matting via rembg's anime model),
the head placed in the 128x128 box in the middle of the top half, and a
report on the constraints that can only be checked, not fixed.

Every step is supervised: the wizard runs in three stages, each shown in a
preview window and only applied on explicit accept ([a]); [b] goes back a
stage and [q] aborts without writing anything.

  1. Cutout   - background removal shown over a checkerboard, with a
                difference view highlighting removed pixels.
  2. Framing  - auto-placement from face detection, adjustable with the
                arrow keys / +/- against the THA3 framing guides.
  3. Review   - the final 512x512 result against green/checker/etc.
                backgrounds plus the constraint report; saves on [s].

`--auto` skips the windows and applies the automatic cutout + framing
directly (used for scripting/testing; the report still prints).

Heavy dependencies (cv2, mediapipe, rembg) are imported lazily; the module
itself needs only numpy + Pillow. rembg's models (~170 MB for the anime one)
download to the user cache on first use.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CANVAS = 512
# THA3 spec: head roughly inside the 128x128 box in the middle of the top
# half of the image -> x 192..320, y 64..192.
HEAD_BOX = (192, 64, 320, 192)
HEAD_TARGET_CENTER = (256.0, 128.0)
HEAD_TARGET_HEIGHT = 118.0  # leave a little margin inside the 128 box

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARACTERS_DIR = os.path.join(REPO_ROOT, "characters")

REMBG_MODELS = {"anime": "isnet-anime", "general": "isnet-general-use"}
_REMBG_SESSIONS: Dict[str, object] = {}

Placement = Tuple[float, float, float]  # scale, tx, ty (top-left offset)
HeadBox = Tuple[float, float, float, float]  # cx, cy, w, h


# ----------------------------------------------------------------------
# Pure image math (numpy + Pillow only; unit-tested)
# ----------------------------------------------------------------------

def load_rgba(path: str) -> Tuple[np.ndarray, bool]:
    """Load any image as (H, W, 4) uint8; also report if it had real alpha."""
    from PIL import Image

    image = Image.open(path)
    had_alpha = (
        image.mode in ("RGBA", "LA", "PA")
        or (image.mode == "P" and "transparency" in image.info)
    )
    return np.asarray(image.convert("RGBA"), dtype=np.uint8).copy(), had_alpha


def compose(char: np.ndarray, placement: Placement, canvas: int = CANVAS) -> np.ndarray:
    """Scale `char` and paste it at (tx, ty) on a transparent square canvas."""
    from PIL import Image

    scale, tx, ty = placement
    image = Image.fromarray(char, "RGBA")
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    image = image.resize((width, height), Image.LANCZOS)
    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    out.paste(image, (round(tx), round(ty)), image)
    return np.asarray(out, dtype=np.uint8).copy()


def placement_for_head(head: HeadBox) -> Placement:
    """Placement that puts a detected head (source pixels) in the THA3 box."""
    cx, cy, _w, h = head
    scale = HEAD_TARGET_HEIGHT / max(h, 1.0)
    tx = HEAD_TARGET_CENTER[0] - cx * scale
    ty = HEAD_TARGET_CENTER[1] - cy * scale
    return scale, tx, ty


def fallback_placement(width: int, height: int) -> Placement:
    """No face found: fit the image, centered horizontally, top-aligned."""
    scale = CANVAS / max(width, height, 1)
    return scale, (CANVAS - width * scale) / 2.0, 0.0


def transform_head(head: HeadBox, placement: Placement) -> HeadBox:
    """Map a head box from source pixels to canvas pixels."""
    scale, tx, ty = placement
    cx, cy, w, h = head
    return cx * scale + tx, cy * scale + ty, w * scale, h * scale


def checkerboard(height: int, width: int, cell: int = 16) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    light = ((ys // cell + xs // cell) % 2 == 0)
    board = np.where(light[..., None], 200, 160).astype(np.uint8)
    return np.repeat(board, 3, axis=2) if board.shape[2] == 1 else board


def over_background(rgba: np.ndarray, background) -> np.ndarray:
    """Alpha-blend onto a color tuple or (H, W, 3) array; returns RGB."""
    if isinstance(background, tuple):
        bg = np.full((*rgba.shape[:2], 3), background, dtype=np.uint8)
    else:
        bg = background
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    return (rgb * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def unique_output_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def head_box_verdict(head: Optional[HeadBox]) -> Tuple[bool, str]:
    """Judge a canvas-space head box against the THA3 128x128 target."""
    if head is None:
        return False, "head position unknown (no face detected)"
    cx, cy, _w, h = head
    x0, y0, x1, y1 = HEAD_BOX
    if not (x0 <= cx <= x1 and y0 <= cy <= y1):
        return False, f"head center ({cx:.0f}, {cy:.0f}) outside the target box"
    if not 85.0 <= h <= 150.0:
        return False, f"head height {h:.0f}px (target roughly 100-130px)"
    return True, f"head at ({cx:.0f}, {cy:.0f}), height {h:.0f}px"


# ----------------------------------------------------------------------
# AI helpers (lazy heavy imports)
# ----------------------------------------------------------------------

def run_rembg(rgba: np.ndarray, model: str) -> np.ndarray:
    """Background removal; `model` is a key of REMBG_MODELS. May raise
    ImportError if rembg/onnxruntime are not installed."""
    from PIL import Image
    from rembg import new_session, remove

    name = REMBG_MODELS[model]
    if name not in _REMBG_SESSIONS:
        logger.info("loading rembg model %s (downloads on first use)", name)
        _REMBG_SESSIONS[name] = new_session(name)
    result = remove(Image.fromarray(rgba, "RGBA"), session=_REMBG_SESSIONS[name])
    return np.asarray(result.convert("RGBA"), dtype=np.uint8).copy()


def head_search_crops(alpha: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Crop candidates (x0, y0, x1, y1) to search for a face, best-guess last.

    The FaceLandmarker misses faces that are small relative to the frame, so
    after the full image we retry on the character's alpha bounding box and
    then on its top portion (an upright character's head is there). For a
    fully opaque image the bounding box is the whole frame, so the last crop
    degrades to the image's top half.
    """
    height, width = alpha.shape
    crops = [(0, 0, width, height)]
    # AI mattes leave stray faint pixels; a bounding box from raw alpha>0
    # can span the whole frame. Only count rows/columns with a meaningful
    # amount of solid alpha.
    mask = alpha >= 32
    min_px = max(2, round(0.004 * max(width, height)))
    rows = np.where(mask.sum(axis=1) >= min_px)[0]
    cols = np.where(mask.sum(axis=0) >= min_px)[0]
    if len(rows) == 0 or len(cols) == 0:
        return crops
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    if (x1 - x0) * (y1 - y0) < 0.95 * width * height:
        crops.append((x0, y0, x1, y1))
    top = max(int((y1 - y0) * 0.45), min(64, y1 - y0))
    crops.append((x0, y0, x1, min(y0 + top, y1)))
    return crops


def detect_head(rgba: np.ndarray) -> Tuple[Optional[HeadBox], int]:
    """(head box in source pixels or None, number of faces found).

    Returns (None, -1) when mediapipe is unavailable. The face mesh covers
    brows to chin; the box is expanded to approximate the full head with
    hair, which the framing stage lets the user correct anyway.
    """
    try:
        import mediapipe as mp
        from tha_wrapper.drivers.webcam import ensure_face_landmarker_model
    except ImportError:
        return None, -1

    rgb = over_background(rgba, (128, 128, 128))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=ensure_face_landmarker_model()
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=4,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for cx0, cy0, cx1, cy1 in head_search_crops(rgba[..., 3]):
            crop = np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1])
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            )
            if result.face_landmarks:
                break
        else:
            return None, 0
    lm = result.face_landmarks[0]
    xs = [p.x * (cx1 - cx0) + cx0 for p in lm]
    ys = [p.y * (cy1 - cy0) + cy0 for p in lm]
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0 - face_h * 0.35  # hair extends upward
    return (cx, cy, face_w * 1.9, face_h * 2.1), len(result.face_landmarks)


# ----------------------------------------------------------------------
# Constraint report
# ----------------------------------------------------------------------

def constraint_report(
    final: np.ndarray, n_faces: int, head: Optional[HeadBox]
) -> List[Tuple[str, str]]:
    """[(status, message)] with status OK / WARN / CHECK."""
    lines: List[Tuple[str, str]] = []
    lines.append(("OK", "512x512 with alpha channel"))

    if n_faces == 1:
        lines.append(("OK", "exactly one face detected"))
    elif n_faces == 0:
        lines.append(("WARN", "no face detected (framing was manual/fallback)"))
    elif n_faces < 0:
        lines.append(("WARN", "mediapipe unavailable - face count not checked"))
    else:
        lines.append(("WARN", f"{n_faces} faces detected - THA3 needs exactly one character"))

    ok, detail = head_box_verdict(head)
    lines.append(("OK" if ok else "WARN", f"head box: {detail}"))

    alpha = final[..., 3]
    transparent = float((alpha == 0).mean())
    lines.append(("OK" if transparent > 0.05 else "WARN",
                  f"{transparent:.0%} of the canvas is fully transparent"))

    # Contact with the top/side edges means the character got cropped;
    # touching the bottom edge is normal for half/full-body art.
    edges = {"top": alpha[0, :], "left": alpha[:, 0], "right": alpha[:, -1]}
    touching = [name for name, line in edges.items() if (line > 0).any()]
    if touching:
        lines.append(("WARN", f"character touches the {'/'.join(touching)} edge (cropped?)"))
    else:
        lines.append(("OK", "character clear of the top/side edges"))

    lines.append(("CHECK", "upright, facing forward, hands below and away from the head"))
    return lines


def print_report(lines: List[Tuple[str, str]]) -> None:
    for status, message in lines:
        print(f"  [{status:5s}] {message}")


# ----------------------------------------------------------------------
# Interactive wizard (cv2)
# ----------------------------------------------------------------------

_WINDOW = "character wizard"
_PANEL_W = 380
_VIEW = 560  # max preview side length
_BACKGROUNDS = ("checker", "green", "magenta", "white", "black")
_BACKGROUND_COLORS = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

# cv2.waitKeyEx arrow codes on Windows.
_KEY_LEFT, _KEY_UP, _KEY_RIGHT, _KEY_DOWN = 2424832, 2490368, 2555904, 2621440


class PrepareWizard:
    def __init__(self, source_path: str, output_path: Optional[str] = None):
        self.source_path = source_path
        self.source, self.source_had_alpha = load_rgba(source_path)
        stem = os.path.splitext(os.path.basename(source_path))[0]
        self.output_path = output_path or unique_output_path(
            os.path.join(CHARACTERS_DIR, f"{stem}.png")
        )

        self.cutout = self.source
        self.cutout_label = "original alpha" if self.source_had_alpha else "original (no alpha!)"
        self.show_diff = False

        self.placement: Placement = fallback_placement(*self.source.shape[1::-1])
        self.head: Optional[HeadBox] = None  # in cutout source pixels
        self.n_faces = -1

        self.background = 0
        self.saved_to: Optional[str] = None

    # -- shared -------------------------------------------------------

    def _cutout_with(self, model: str) -> None:
        try:
            self.cutout = run_rembg(self.source, model)
            self.cutout_label = f"rembg {REMBG_MODELS[model]}"
        except ImportError:
            print("rembg is not installed - run: pip install rembg[cpu]")

    def _autodetect(self) -> None:
        self.head, self.n_faces = detect_head(self.cutout)
        if self.head is not None:
            self.placement = placement_for_head(self.head)
        else:
            self.placement = fallback_placement(*self.cutout.shape[1::-1])

    def _final(self) -> np.ndarray:
        return compose(self.cutout, self.placement)

    def _canvas_head(self) -> Optional[HeadBox]:
        return transform_head(self.head, self.placement) if self.head else None

    # -- rendering helpers --------------------------------------------

    def _show(self, view_rgb: np.ndarray, lines: List[str]) -> int:
        """Compose preview + text panel, display, return waitKeyEx code."""
        import cv2

        height = max(view_rgb.shape[0], 24 * len(lines) + 16)
        frame = np.full((height, view_rgb.shape[1] + _PANEL_W, 3), 34, np.uint8)
        frame[: view_rgb.shape[0], : view_rgb.shape[1]] = view_rgb[..., ::-1]  # RGB->BGR
        for i, text in enumerate(lines):
            cv2.putText(
                frame, text, (view_rgb.shape[1] + 14, 26 + 24 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA,
            )
        cv2.imshow(_WINDOW, frame)
        return cv2.waitKeyEx(30)

    def _fit(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        scale = _VIEW / max(rgb.shape[:2])
        if scale >= 1.0:
            return rgb
        return cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # -- stage 1: cutout ----------------------------------------------

    def _stage_cutout(self) -> int:
        view = over_background(self.cutout, checkerboard(*self.cutout.shape[:2]))
        if self.show_diff:
            removed = (self.source[..., 3] > 0) & (self.cutout[..., 3] < 64)
            view = view.copy()
            view[removed] = (view[removed] * 0.3 + np.array([255, 40, 40]) * 0.7).astype(np.uint8)
        lines = [
            "STAGE 1/3: cutout",
            f"current: {self.cutout_label}",
            "",
            "[1] AI cutout, anime model",
            "[2] AI cutout, general model",
            "[k] keep image's own alpha",
            "[d] toggle removed-pixels view" + (" (ON)" if self.show_diff else ""),
            "",
            "[a] accept    [q] quit",
        ]
        key = self._show(self._fit(view), lines)
        if key == ord("1"):
            self._cutout_with("anime")
        elif key == ord("2"):
            self._cutout_with("general")
        elif key == ord("k"):
            self.cutout = self.source
            self.cutout_label = "original alpha" if self.source_had_alpha else "original (no alpha!)"
        elif key == ord("d"):
            self.show_diff = not self.show_diff
        elif key == ord("a"):
            self._autodetect()
            return 1
        elif key in (ord("q"), 27):
            return 99
        return 0

    # -- stage 2: framing ---------------------------------------------

    def _stage_framing(self) -> int:
        import cv2

        final = self._final()
        view = over_background(final, checkerboard(CANVAS, CANVAS))[..., ::-1].copy()  # BGR
        x0, y0, x1, y1 = HEAD_BOX
        cv2.line(view, (0, CANVAS // 2), (CANVAS, CANVAS // 2), (90, 90, 90), 1)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 200, 0), 1)
        head = self._canvas_head()
        if head is not None:
            cx, cy, w, h = head
            cv2.rectangle(
                view,
                (round(cx - w / 2), round(cy - h / 2)),
                (round(cx + w / 2), round(cy + h / 2)),
                (0, 200, 255), 1,
            )
        scale, _, _ = self.placement
        lines = [
            "STAGE 2/3: framing",
            "green: THA3 head target box",
            "yellow: detected head" if head is not None else "no face detected",
            f"scale: {scale:.3f}",
            "",
            "[arrows] move    [+]/[-] scale",
            "[r] re-run auto framing",
            "",
            "[a] accept    [b] back    [q] quit",
        ]
        key = self._show(view[..., ::-1], lines)
        step = 4.0
        s, tx, ty = self.placement
        if key == _KEY_LEFT:
            self.placement = (s, tx - step, ty)
        elif key == _KEY_RIGHT:
            self.placement = (s, tx + step, ty)
        elif key == _KEY_UP:
            self.placement = (s, tx, ty - step)
        elif key == _KEY_DOWN:
            self.placement = (s, tx, ty + step)
        elif key in (ord("+"), ord("=")) or key in (ord("-"), ord("_")):
            factor = 1.03 if key in (ord("+"), ord("=")) else 1 / 1.03
            ax, ay = HEAD_TARGET_CENTER
            self.placement = (
                s * factor, ax - (ax - tx) * factor, ay - (ay - ty) * factor,
            )
        elif key == ord("r"):
            self._autodetect()
        elif key == ord("a"):
            return 1
        elif key == ord("b"):
            return -1
        elif key in (ord("q"), 27):
            return 99
        return 0

    # -- stage 3: review ----------------------------------------------

    def _stage_review(self) -> int:
        final = self._final()
        name = _BACKGROUNDS[self.background]
        if name == "checker":
            view = over_background(final, checkerboard(CANVAS, CANVAS))
        else:
            view = over_background(final, _BACKGROUND_COLORS[name])
        report = constraint_report(final, self.n_faces, self._canvas_head())
        lines = [
            "STAGE 3/3: review",
            f"background: {name}  ([g] cycle)",
            "",
            *[f"[{status}] {msg}" for status, msg in report],
            "",
            f"save to: {os.path.relpath(self.output_path, REPO_ROOT)}",
            "[s] save    [b] back    [q] quit",
        ]
        key = self._show(view, lines)
        if key == ord("g"):
            self.background = (self.background + 1) % len(_BACKGROUNDS)
        elif key in (ord("s"), 13):
            self._save(final)
            return 1
        elif key == ord("b"):
            return -1
        elif key in (ord("q"), 27):
            return 99
        return 0

    # -- driver -------------------------------------------------------

    def _save(self, final: np.ndarray) -> None:
        from PIL import Image

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        Image.fromarray(final, "RGBA").save(self.output_path)
        self.saved_to = self.output_path

    def run(self) -> bool:
        import cv2

        stages = (self._stage_cutout, self._stage_framing, self._stage_review)
        stage = 0
        cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
        try:
            while 0 <= stage < len(stages):
                delta = stages[stage]()
                if delta == 99:
                    return False
                stage = max(0, stage + delta)
        finally:
            cv2.destroyWindow(_WINDOW)
        return True

    def run_auto(self, model: str = "anime") -> bool:
        """Non-interactive: AI cutout + auto framing + save + report."""
        self._cutout_with(model)
        self._autodetect()
        final = self._final()
        self._save(final)
        print(f"cutout: {self.cutout_label}")
        print(f"saved: {self.saved_to}")
        print_report(constraint_report(final, self.n_faces, self._canvas_head()))
        return True


# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a character image for THA3 (see module docstring)."
    )
    parser.add_argument("input", help="source character image (any size/format)")
    parser.add_argument("-o", "--output", default=None,
                        help="output PNG (default: characters/<name>.png, never overwrites)")
    parser.add_argument("--auto", action="store_true",
                        help="no wizard: AI cutout + auto framing, then save")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    wizard = PrepareWizard(args.input, args.output)
    done = wizard.run_auto() if args.auto else wizard.run()
    if done and wizard.saved_to:
        print(f"\ncharacter ready: {wizard.saved_to}")
        print("it will show up in run.bat's character menu")
        return 0
    print("aborted - nothing was written")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
