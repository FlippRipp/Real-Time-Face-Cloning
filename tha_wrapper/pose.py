"""The avatar pose model.

THA3 poses a character with a 45-dimensional pose vector. This module gives
those 45 parameters names, ranges, and a small `AvatarPose` value type that
every driver produces and every poser backend consumes.

The canonical parameter order below matches the THA3 demo
(`tha3/poser/modes/pose_parameters.py`). When the real THA3 backend is used,
the name -> index mapping is rebuilt from the loaded poser itself (see
`tha_wrapper.poser.tha3_backend`), so this table only has to be correct for
the mock backend and for documentation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

# name -> (min, max), in canonical THA3 order.
POSE_PARAMETER_RANGES: Dict[str, Tuple[float, float]] = {
    # Eyebrows: six morphs, one per side.
    "eyebrow_troubled_left": (0.0, 1.0),
    "eyebrow_troubled_right": (0.0, 1.0),
    "eyebrow_angry_left": (0.0, 1.0),
    "eyebrow_angry_right": (0.0, 1.0),
    "eyebrow_lowered_left": (0.0, 1.0),
    "eyebrow_lowered_right": (0.0, 1.0),
    "eyebrow_raised_left": (0.0, 1.0),
    "eyebrow_raised_right": (0.0, 1.0),
    "eyebrow_happy_left": (0.0, 1.0),
    "eyebrow_happy_right": (0.0, 1.0),
    "eyebrow_serious_left": (0.0, 1.0),
    "eyebrow_serious_right": (0.0, 1.0),
    # Eyes: six morphs, one per side. `eye_wink_*` is the plain blink.
    "eye_wink_left": (0.0, 1.0),
    "eye_wink_right": (0.0, 1.0),
    "eye_happy_wink_left": (0.0, 1.0),
    "eye_happy_wink_right": (0.0, 1.0),
    "eye_surprised_left": (0.0, 1.0),
    "eye_surprised_right": (0.0, 1.0),
    "eye_relaxed_left": (0.0, 1.0),
    "eye_relaxed_right": (0.0, 1.0),
    "eye_unimpressed_left": (0.0, 1.0),
    "eye_unimpressed_right": (0.0, 1.0),
    "eye_raised_lower_eyelid_left": (0.0, 1.0),
    "eye_raised_lower_eyelid_right": (0.0, 1.0),
    "iris_small_left": (0.0, 1.0),
    "iris_small_right": (0.0, 1.0),
    # Mouth: vowel shapes (visemes) and corner/smirk morphs.
    "mouth_aaa": (0.0, 1.0),
    "mouth_iii": (0.0, 1.0),
    "mouth_uuu": (0.0, 1.0),
    "mouth_eee": (0.0, 1.0),
    "mouth_ooo": (0.0, 1.0),
    "mouth_delta": (0.0, 1.0),
    "mouth_lowered_corner_left": (0.0, 1.0),
    "mouth_lowered_corner_right": (0.0, 1.0),
    "mouth_raised_corner_left": (0.0, 1.0),
    "mouth_raised_corner_right": (0.0, 1.0),
    "mouth_smirk": (0.0, 1.0),
    # Gaze direction. x = up/down, y = left/right.
    "iris_rotation_x": (-1.0, 1.0),
    "iris_rotation_y": (-1.0, 1.0),
    # Head rotation, each unit ~= 15 degrees.
    # head_x = pitch (nod), head_y = yaw (turn), neck_z = roll (tilt).
    "head_x": (-1.0, 1.0),
    "head_y": (-1.0, 1.0),
    "neck_z": (-1.0, 1.0),
    # Body rotation/lean.
    "body_y": (-1.0, 1.0),
    "body_z": (-1.0, 1.0),
    # Breathing cycle, 0 = exhaled, 1 = inhaled.
    "breathing": (0.0, 1.0),
}

POSE_PARAMETER_NAMES: Tuple[str, ...] = tuple(POSE_PARAMETER_RANGES)

assert len(POSE_PARAMETER_NAMES) == 45, "THA3 expects a 45-dim pose vector"

# Convenience groups, useful for blending/override logic.
EYEBROW_PARAMS = tuple(n for n in POSE_PARAMETER_NAMES if n.startswith("eyebrow_"))
EYE_PARAMS = tuple(n for n in POSE_PARAMETER_NAMES if n.startswith("eye_"))
MOUTH_PARAMS = tuple(n for n in POSE_PARAMETER_NAMES if n.startswith("mouth_"))
IRIS_PARAMS = ("iris_rotation_x", "iris_rotation_y")
HEAD_PARAMS = ("head_x", "head_y", "neck_z")
BODY_PARAMS = ("body_y", "body_z")


def clamp_param(name: str, value: float) -> float:
    lo, hi = POSE_PARAMETER_RANGES[name]
    return min(hi, max(lo, float(value)))


class AvatarPose:
    """A named, sparse view of the THA3 pose vector.

    Unset parameters read as 0.0 (the neutral value for every THA3 parameter).
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, float] | None = None):
        self._values: Dict[str, float] = {}
        if values:
            for name, value in values.items():
                self.set(name, value)

    def get(self, name: str) -> float:
        if name not in POSE_PARAMETER_RANGES:
            raise KeyError(f"unknown pose parameter: {name!r}")
        return self._values.get(name, 0.0)

    def set(self, name: str, value: float) -> None:
        if name not in POSE_PARAMETER_RANGES:
            raise KeyError(f"unknown pose parameter: {name!r}")
        self._values[name] = clamp_param(name, value)

    def update(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self.set(name, value)

    def as_dict(self, sparse: bool = False) -> Dict[str, float]:
        if sparse:
            return {n: v for n, v in self._values.items() if v != 0.0}
        return {n: self.get(n) for n in POSE_PARAMETER_NAMES}

    def as_list(self, order: Iterable[str] = POSE_PARAMETER_NAMES) -> list:
        return [self.get(n) for n in order]

    def copy(self) -> "AvatarPose":
        return AvatarPose(self._values)

    def lerp(self, other: "AvatarPose", t: float) -> "AvatarPose":
        """Linear interpolation: t=0 -> self, t=1 -> other."""
        t = min(1.0, max(0.0, t))
        names = set(self._values) | set(other._values)
        return AvatarPose(
            {n: self.get(n) + (other.get(n) - self.get(n)) * t for n in names}
        )

    def merged_over(self, base: "AvatarPose", names: Iterable[str]) -> "AvatarPose":
        """Return `base` with this pose's values written over it for `names`."""
        out = base.copy()
        for name in names:
            out.set(name, self.get(name))
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AvatarPose):
            return NotImplemented
        return all(self.get(n) == other.get(n) for n in POSE_PARAMETER_NAMES)

    def __repr__(self) -> str:
        active = ", ".join(f"{n}={v:.2f}" for n, v in sorted(self._values.items()) if v)
        return f"AvatarPose({active})"
