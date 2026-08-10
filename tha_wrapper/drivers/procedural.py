"""Procedural pose driver: animate the avatar from code instead of a webcam.

This is the piece that lets an AI agent puppet the avatar. All public
methods are thread-safe, so they can be called from a control server, an
asyncio app, or any other thread while the render loop calls `update()`.

Layering model (applied in order every frame):

1. **Tweens** — explicit parameter targets (from `set_params`, `set_emotion`,
   `set_head`, `look_at`, ...) eased toward over a duration.
2. **Idle layers** — breathing, subtle head/body sway, auto-blink. These are
   additive/max-combined so they never fight the explicit targets badly.
3. **Speech layer** — viseme timeline, live audio energy, or random
   "talking" flapping, written over the mouth vowel parameters.
4. **Blink envelope** — manual or automatic blinks, max-combined onto
   `eye_wink_*` so a blink always closes the eyes fully.

`touched()` reports which parameters this driver is actively claiming; the
hybrid driver uses that to decide procedural-vs-webcam per parameter.

On top of the layers sits `perform()`: a timed script of actions (emotion
changes, speech, gaze, blinks...) that the driver schedules and fires on
its own clock, so an AI can hand over a whole little screenplay in one
non-blocking call instead of timing individual commands itself.
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from tha_wrapper.drivers.base import PoseDriver
from tha_wrapper.pose import (
    MOUTH_PARAMS,
    POSE_PARAMETER_RANGES,
    AvatarPose,
    clamp_param,
)

# Emotion presets, expressed as sparse pose targets. Intensity scales them.
EMOTIONS: Dict[str, Dict[str, float]] = {
    "neutral": {},
    "happy": {
        "eyebrow_happy_left": 1.0,
        "eyebrow_happy_right": 1.0,
        "eye_happy_wink_left": 0.35,
        "eye_happy_wink_right": 0.35,
        "mouth_raised_corner_left": 0.8,
        "mouth_raised_corner_right": 0.8,
    },
    "sad": {
        "eyebrow_troubled_left": 1.0,
        "eyebrow_troubled_right": 1.0,
        "eye_relaxed_left": 0.4,
        "eye_relaxed_right": 0.4,
        "mouth_lowered_corner_left": 0.7,
        "mouth_lowered_corner_right": 0.7,
    },
    "angry": {
        "eyebrow_angry_left": 1.0,
        "eyebrow_angry_right": 1.0,
        "eye_raised_lower_eyelid_left": 0.5,
        "eye_raised_lower_eyelid_right": 0.5,
        "mouth_lowered_corner_left": 0.4,
        "mouth_lowered_corner_right": 0.4,
    },
    "surprised": {
        "eyebrow_raised_left": 1.0,
        "eyebrow_raised_right": 1.0,
        "eye_surprised_left": 0.9,
        "eye_surprised_right": 0.9,
        "mouth_ooo": 0.5,
    },
    "amused": {
        "eyebrow_happy_left": 0.6,
        "eyebrow_happy_right": 0.6,
        "eye_wink_left": 0.15,
        "eye_wink_right": 0.15,
        "mouth_smirk": 0.8,
    },
    "smug": {
        "eyebrow_serious_left": 0.8,
        "eyebrow_serious_right": 0.8,
        "eye_unimpressed_left": 0.5,
        "eye_unimpressed_right": 0.5,
        "mouth_smirk": 1.0,
    },
    "tired": {
        "eyebrow_lowered_left": 0.6,
        "eyebrow_lowered_right": 0.6,
        "eye_relaxed_left": 0.8,
        "eye_relaxed_right": 0.8,
        "mouth_uuu": 0.2,
    },
    "confused": {
        "eyebrow_troubled_left": 0.9,
        "eyebrow_serious_right": 0.7,
        "eye_wink_left": 0.0,
        "eye_unimpressed_right": 0.45,
        "mouth_uuu": 0.35,
        "neck_z": 0.25,
    },
}

# Viseme -> mouth shape. Keys accepted by speak()/speak_visemes().
VISEMES: Dict[str, Dict[str, float]] = {
    "sil": {},
    "aa": {"mouth_aaa": 1.0},
    "ih": {"mouth_iii": 1.0},
    "ou": {"mouth_uuu": 1.0},
    "eh": {"mouth_eee": 1.0},
    "oh": {"mouth_ooo": 1.0},
    "dd": {"mouth_delta": 1.0},
}

_VOWEL_TO_VISEME = {
    "a": "aa", "i": "ih", "u": "ou", "e": "eh", "o": "oh", "y": "ih",
}

MOUTH_VOWEL_PARAMS = ("mouth_aaa", "mouth_iii", "mouth_uuu", "mouth_eee",
                      "mouth_ooo", "mouth_delta")

# perform() script actions: action key -> modifier keys it accepts.
SCRIPT_ACTIONS: Dict[str, Tuple[str, ...]] = {
    "emotion": ("intensity", "duration"),
    "say": ("seconds_per_syllable",),
    "visemes": (),
    "look": ("duration", "head_follow"),
    "head": ("duration",),
    "body": ("duration",),
    "params": ("duration",),
    "blink": (),
    "talking": (),
}

# Script action kind -> driver method it fires.
_SCRIPT_DISPATCH: Dict[str, str] = {
    "emotion": "set_emotion",
    "say": "speak_visemes",
    "visemes": "speak_visemes",
    "look": "look_at",
    "head": "set_head",
    "body": "set_body",
    "params": "set_params",
    "blink": "blink",
    "talking": "set_talking",
}


def _text_to_viseme_events(
    text: str, seconds_per_syllable: float
) -> List[Tuple[str, float]]:
    events: List[Tuple[str, float]] = []
    for ch in text.lower():
        if ch in _VOWEL_TO_VISEME:
            events.append((_VOWEL_TO_VISEME[ch], seconds_per_syllable))
        elif ch in " ,.!?;:\n":
            events.append(("sil", seconds_per_syllable * 0.6))
    events.append(("sil", 0.1))
    return events


def _smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class _Tween:
    start: float
    target: float
    duration: float
    elapsed: float = 0.0

    def value(self) -> float:
        if self.duration <= 0.0:
            return self.target
        return self.start + (self.target - self.start) * _smoothstep(
            self.elapsed / self.duration
        )

    def done(self) -> bool:
        return self.elapsed >= self.duration


@dataclass
class _VisemeEvent:
    viseme: str
    duration: float
    weight: float = 1.0


@dataclass
class _ScriptAction:
    at: float                 # offset from script start, seconds
    kind: str                 # key into _SCRIPT_DISPATCH
    kwargs: Dict[str, object] # arguments for the dispatched method
    duration: float           # intrinsic duration, for the script-length estimate


@dataclass
class _SpeechState:
    # Timeline playback
    timeline: List[_VisemeEvent] = field(default_factory=list)
    timeline_pos: float = 0.0
    # Live modes
    talking: bool = False           # random flapping
    energy: float = 0.0             # live audio amplitude 0..1
    energy_decay: float = 8.0       # per-second decay of pushed energy
    # Current blended mouth output
    current: Dict[str, float] = field(default_factory=dict)
    _flap_timer: float = 0.0
    _flap_target: Tuple[str, float] = ("mouth_aaa", 0.0)


class ProceduralDriver(PoseDriver):
    def __init__(
        self,
        auto_blink: bool = True,
        breathing: bool = True,
        idle_sway: bool = True,
        rng: random.Random | None = None,
    ):
        self._lock = threading.RLock()
        self._rng = rng or random.Random()
        self._values: Dict[str, float] = {}   # currently held values (post-tween)
        self._tweens: Dict[str, _Tween] = {}
        self._time = 0.0

        self.auto_blink = auto_blink
        self.breathing = breathing
        self.idle_sway = idle_sway
        self.idle_sway_amount = 1.0

        self._blink_phase: float | None = None  # None = not blinking
        self._blink_close = 0.08                # seconds to close
        self._blink_open = 0.14                 # seconds to open
        self._next_auto_blink = self._roll_blink_interval()

        self._speech = _SpeechState()
        self._emotion = ("neutral", 0.0)
        self._script: List[_ScriptAction] = []  # pending, sorted by `at`
        self._script_pos = 0.0

    # ------------------------------------------------------------------
    # Public control API (thread-safe)
    # ------------------------------------------------------------------

    def set_param(self, name: str, value: float, duration: float = 0.15) -> None:
        """Ease one raw THA3 parameter to `value` over `duration` seconds."""
        self.set_params({name: value}, duration)

    def set_params(self, params: Dict[str, float], duration: float = 0.15) -> None:
        with self._lock:
            for name, value in params.items():
                if name not in POSE_PARAMETER_RANGES:
                    raise KeyError(f"unknown pose parameter: {name!r}")
                value = clamp_param(name, value)
                current = self._current_value(name)
                if duration <= 0.0:
                    self._tweens.pop(name, None)
                    self._values[name] = value
                else:
                    self._tweens[name] = _Tween(current, value, duration)

    def set_emotion(
        self, name: str, intensity: float = 1.0, duration: float = 0.4
    ) -> None:
        """Blend to an emotion preset. `neutral` clears the expression."""
        if name not in EMOTIONS:
            raise KeyError(
                f"unknown emotion {name!r}; available: {sorted(EMOTIONS)}"
            )
        intensity = min(1.0, max(0.0, intensity))
        with self._lock:
            prev_params = {
                p: 0.0
                for p in EMOTIONS[self._emotion[0]]
            }
            target = dict(prev_params)  # zero out the old emotion...
            for p, v in EMOTIONS[name].items():
                target[p] = v * intensity  # ...then apply the new one
            self._emotion = (name, intensity)
            self.set_params(target, duration)

    def set_head(
        self,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        duration: float = 0.3,
    ) -> None:
        """Head rotation in [-1, 1] per axis (~15 degrees per unit)."""
        params = {}
        if pitch is not None:
            params["head_x"] = pitch
        if yaw is not None:
            params["head_y"] = yaw
        if roll is not None:
            params["neck_z"] = roll
        self.set_params(params, duration)

    def set_body(self, y: float | None = None, z: float | None = None,
                 duration: float = 0.4) -> None:
        params = {}
        if y is not None:
            params["body_y"] = y
        if z is not None:
            params["body_z"] = z
        self.set_params(params, duration)

    def look_at(self, x: float, y: float, duration: float = 0.15,
                head_follow: float = 0.0) -> None:
        """Point the gaze at (x, y), each in [-1, 1]. x = right, y = up.

        `head_follow` > 0 also turns the head partway toward the target.
        """
        params = {"iris_rotation_y": x, "iris_rotation_x": y}
        if head_follow > 0.0:
            params["head_y"] = x * head_follow
            params["head_x"] = y * head_follow
        self.set_params(params, duration)

    def blink(self, double: bool = False) -> None:
        with self._lock:
            self._blink_phase = 0.0
            if double:
                # Schedule the second blink by pulling the auto-blink timer in.
                self._next_auto_blink = self._blink_close + self._blink_open + 0.06

    def speak_visemes(
        self, events: Sequence[Tuple[str, float] | Tuple[str, float, float]]
    ) -> None:
        """Queue a viseme timeline: [(viseme, duration_sec[, weight]), ...].

        Viseme names: sil, aa, ih, ou, eh, oh, dd. This is the precise
        lipsync path — a TTS engine's phoneme timings map directly onto it.
        """
        timeline = []
        for event in events:
            viseme, duration = event[0], float(event[1])
            weight = float(event[2]) if len(event) > 2 else 1.0
            if viseme not in VISEMES:
                raise KeyError(
                    f"unknown viseme {viseme!r}; available: {sorted(VISEMES)}"
                )
            timeline.append(_VisemeEvent(viseme, max(0.0, duration), weight))
        with self._lock:
            self._speech.timeline = timeline
            self._speech.timeline_pos = 0.0

    def speak_text(self, text: str, seconds_per_syllable: float = 0.18) -> float:
        """Naive text -> viseme timeline (vowels drive the mouth).

        Good enough for placeholder lipsync when no TTS timings exist.
        Returns the total duration of the generated timeline.
        """
        events = _text_to_viseme_events(text, seconds_per_syllable)
        self.speak_visemes(events)
        return sum(d for _, d in events)

    def perform(self, script: Sequence[Dict[str, object]]) -> float:
        """Play a timed action script; replaces any script already playing.

        Each action is a dict with an optional `at` offset (seconds from
        script start, default 0) plus exactly one action key:

            {"at": 0.0, "emotion": "happy", "intensity": 1.0, "duration": 0.4}
            {"at": 0.2, "say": "Hello!", "seconds_per_syllable": 0.18}
            {"at": 0.2, "visemes": [["oh", 0.18], ["aa", 0.22]]}
            {"at": 1.5, "look": [0.4, -0.1], "duration": 0.15, "head_follow": 0.3}
            {"at": 2.0, "head": {"pitch": 0.1, "yaw": -0.2, "roll": 0.1}}
            {"at": 2.0, "body": {"y": 0.1, "z": 0.0}}
            {"at": 2.2, "params": {"iris_small_left": 0.5}, "duration": 0.2}
            {"at": 2.5, "blink": true}      # or "double"
            {"at": 3.0, "talking": false}

        Actions fire on the driver's clock as `update()` advances, so this
        returns immediately; the whole script is validated up front and
        rejected atomically on any error. Returns the estimated total
        duration. An empty script cancels the current performance.
        """
        actions: List[_ScriptAction] = []
        total = 0.0
        for index, entry in enumerate(script):
            action = self._parse_script_action(index, entry)
            actions.append(action)
            total = max(total, action.at + action.duration)
        actions.sort(key=lambda a: a.at)  # stable: ties keep script order
        with self._lock:
            self._script = actions
            self._script_pos = 0.0
        return total

    def stop_performance(self) -> None:
        """Cancel pending script actions and stop speech; holds the pose."""
        with self._lock:
            self._script = []
            self._script_pos = 0.0
            self.stop_speaking()

    def set_talking(self, talking: bool) -> None:
        """Random mouth flapping — crude lipsync while audio plays."""
        with self._lock:
            self._speech.talking = talking

    def push_audio_energy(self, level: float) -> None:
        """Feed a live loudness sample (0..1); the mouth follows it."""
        with self._lock:
            self._speech.energy = max(self._speech.energy, min(1.0, max(0.0, level)))

    def stop_speaking(self) -> None:
        with self._lock:
            self._speech.timeline = []
            self._speech.talking = False
            self._speech.energy = 0.0

    def reset(self, duration: float = 0.3) -> None:
        """Ease everything back to neutral."""
        with self._lock:
            self._script = []
            self._script_pos = 0.0
            self.stop_speaking()
            self._emotion = ("neutral", 0.0)
            targets = {n: 0.0 for n in set(self._values) | set(self._tweens)}
            self.set_params(targets, duration)

    def state(self) -> Dict[str, object]:
        with self._lock:
            return {
                "emotion": self._emotion[0],
                "emotion_intensity": self._emotion[1],
                "talking": self._speech.talking or bool(self._speech.timeline),
                "performing": bool(self._script),
                "auto_blink": self.auto_blink,
                "breathing": self.breathing,
                "idle_sway": self.idle_sway,
                "pose": {n: round(v, 4) for n, v in self._values.items() if v},
            }

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def update(self, dt: float) -> AvatarPose:
        with self._lock:
            self._time += dt
            self._advance_script(dt)
            self._advance_tweens(dt)
            pose = AvatarPose(self._values)
            touched = set(self._values) | set(self._tweens)

            touched |= self._apply_idle(pose, dt)
            touched |= self._apply_speech(pose, dt)
            touched |= self._apply_blink(pose, dt)

            self._touched = touched
            return pose

    def touched(self) -> Set[str]:
        """Parameters actively claimed by this driver (for hybrid blending)."""
        with self._lock:
            return set(getattr(self, "_touched", set()))

    # ------------------------------------------------------------------
    # Internals (call with lock held)
    # ------------------------------------------------------------------

    def _parse_script_action(self, index: int, entry: object) -> _ScriptAction:
        if not isinstance(entry, dict):
            raise ValueError(f"script action #{index} must be an object")
        keys = set(entry) - {"at"}
        kinds = keys & set(SCRIPT_ACTIONS)
        if len(kinds) != 1:
            raise ValueError(
                f"script action #{index} needs exactly one action key "
                f"({', '.join(sorted(SCRIPT_ACTIONS))}); "
                f"got {sorted(keys) or 'none'}"
            )
        kind = kinds.pop()
        extra = keys - {kind} - set(SCRIPT_ACTIONS[kind])
        if extra:
            raise ValueError(
                f"script action #{index} ({kind}): unknown keys {sorted(extra)}"
            )
        at = float(entry.get("at", 0.0))
        if at < 0.0:
            raise ValueError(f"script action #{index}: 'at' must be >= 0")

        if kind == "emotion":
            name = entry["emotion"]
            if name not in EMOTIONS:
                raise KeyError(
                    f"script action #{index}: unknown emotion {name!r}; "
                    f"available: {sorted(EMOTIONS)}"
                )
            duration = float(entry.get("duration", 0.4))
            kwargs = {"name": name,
                      "intensity": float(entry.get("intensity", 1.0)),
                      "duration": duration}
        elif kind == "say":
            events = _text_to_viseme_events(
                str(entry["say"]),
                float(entry.get("seconds_per_syllable", 0.18)),
            )
            duration = sum(d for _, d in events)
            kwargs = {"events": events}
        elif kind == "visemes":
            events = []
            for event in entry["visemes"]:
                event = tuple(event)
                if not 2 <= len(event) <= 3 or event[0] not in VISEMES:
                    raise ValueError(
                        f"script action #{index}: visemes must be "
                        f"[viseme, duration(, weight)] with viseme in "
                        f"{sorted(VISEMES)}; got {event!r}"
                    )
                events.append(event)
            duration = sum(float(e[1]) for e in events)
            kwargs = {"events": events}
        elif kind == "look":
            target = entry["look"]
            if not isinstance(target, (list, tuple)) or len(target) != 2:
                raise ValueError(
                    f"script action #{index}: 'look' must be [x, y]"
                )
            duration = float(entry.get("duration", 0.15))
            kwargs = {"x": float(target[0]), "y": float(target[1]),
                      "duration": duration,
                      "head_follow": float(entry.get("head_follow", 0.0))}
        elif kind in ("head", "body"):
            axes = {"head": ("pitch", "yaw", "roll"), "body": ("y", "z")}[kind]
            rotation = entry[kind]
            if not isinstance(rotation, dict) or not rotation or \
                    not set(rotation) <= set(axes):
                raise ValueError(
                    f"script action #{index}: {kind!r} must be an object "
                    f"with keys from {axes}"
                )
            duration = float(entry.get("duration", 0.3 if kind == "head" else 0.4))
            kwargs = {axis: float(value) for axis, value in rotation.items()}
            kwargs["duration"] = duration
        elif kind == "params":
            params = entry["params"]
            if not isinstance(params, dict):
                raise ValueError(
                    f"script action #{index}: 'params' must be an object "
                    "of {param: value}"
                )
            for name in params:
                if name not in POSE_PARAMETER_RANGES:
                    raise KeyError(
                        f"script action #{index}: unknown pose parameter "
                        f"{name!r}"
                    )
            duration = float(entry.get("duration", 0.15))
            kwargs = {"params": {k: float(v) for k, v in params.items()},
                      "duration": duration}
        elif kind == "blink":
            flag = entry["blink"]
            if not flag:
                raise ValueError(
                    f"script action #{index}: 'blink' must be true or "
                    f"\"double\""
                )
            duration = self._blink_close + self._blink_open
            kwargs = {"double": flag == "double"}
        else:  # talking
            duration = 0.0
            kwargs = {"talking": bool(entry["talking"])}
        return _ScriptAction(at, kind, kwargs, duration)

    def _advance_script(self, dt: float) -> None:
        if not self._script:
            return
        self._script_pos += dt
        while self._script and self._script[0].at <= self._script_pos:
            action = self._script.pop(0)
            getattr(self, _SCRIPT_DISPATCH[action.kind])(**action.kwargs)

    def _current_value(self, name: str) -> float:
        tween = self._tweens.get(name)
        if tween is not None:
            return tween.value()
        return self._values.get(name, 0.0)

    def _advance_tweens(self, dt: float) -> None:
        finished = []
        for name, tween in self._tweens.items():
            tween.elapsed += dt
            self._values[name] = clamp_param(name, tween.value())
            if tween.done():
                finished.append(name)
        for name in finished:
            del self._tweens[name]
        # Drop zeros that aren't animating so touched() shrinks to what's real.
        for name in [n for n, v in self._values.items()
                     if v == 0.0 and n not in self._tweens]:
            del self._values[name]

    def _apply_idle(self, pose: AvatarPose, dt: float) -> Set[str]:
        touched: Set[str] = set()
        t = self._time
        if self.breathing:
            pose.set("breathing", 0.5 + 0.5 * math.sin(t * (2 * math.pi / 4.2)))
            touched.add("breathing")
        if self.idle_sway:
            amount = self.idle_sway_amount
            # Two incommensurate sines per axis ≈ organic drift.
            sway_y = 0.045 * math.sin(t * 0.61) + 0.03 * math.sin(t * 0.187 + 1.7)
            sway_z = 0.035 * math.sin(t * 0.43 + 0.9) + 0.02 * math.sin(t * 0.113)
            sway_x = 0.03 * math.sin(t * 0.53 + 2.1)
            body_y = 0.05 * math.sin(t * 0.31 + 0.4)
            for name, value in (
                ("head_y", sway_y), ("neck_z", sway_z),
                ("head_x", sway_x), ("body_y", body_y),
            ):
                pose.set(name, pose.get(name) + value * amount)
                touched.add(name)
        return touched

    def _apply_speech(self, pose: AvatarPose, dt: float) -> Set[str]:
        speech = self._speech
        target: Dict[str, float] = {}
        active = False

        if speech.timeline:
            active = True
            speech.timeline_pos += dt
            pos = speech.timeline_pos
            for event in speech.timeline:
                if pos <= event.duration or event is speech.timeline[-1]:
                    for name, value in VISEMES[event.viseme].items():
                        target[name] = value * event.weight
                    break
                pos -= event.duration
            else:  # pragma: no cover
                pass
            total = sum(e.duration for e in speech.timeline)
            if speech.timeline_pos >= total:
                speech.timeline = []
                speech.timeline_pos = 0.0

        if speech.talking:
            active = True
            speech._flap_timer -= dt
            if speech._flap_timer <= 0.0:
                speech._flap_timer = self._rng.uniform(0.06, 0.14)
                name = self._rng.choice(
                    ("mouth_aaa", "mouth_aaa", "mouth_eee", "mouth_ooo", "mouth_iii")
                )
                speech._flap_target = (name, self._rng.uniform(0.15, 0.95))
            name, value = speech._flap_target
            target[name] = max(target.get(name, 0.0), value)

        if speech.energy > 0.001:
            active = True
            target["mouth_aaa"] = max(target.get("mouth_aaa", 0.0), speech.energy)
            speech.energy = max(0.0, speech.energy - speech.energy_decay * dt)

        # Smooth the mouth toward the target so shapes don't pop.
        if active or speech.current:
            rate = 1.0 - math.exp(-dt * 22.0)
            for name in MOUTH_VOWEL_PARAMS:
                current = speech.current.get(name, 0.0)
                current += (target.get(name, 0.0) - current) * rate
                if current < 0.005 and name not in target:
                    speech.current.pop(name, None)
                    continue
                speech.current[name] = current
                pose.set(name, max(pose.get(name), current))
            return set(MOUTH_VOWEL_PARAMS)
        return set()

    def _roll_blink_interval(self) -> float:
        return self._rng.uniform(2.0, 6.0)

    def _apply_blink(self, pose: AvatarPose, dt: float) -> Set[str]:
        if self.auto_blink and self._blink_phase is None:
            self._next_auto_blink -= dt
            if self._next_auto_blink <= 0.0:
                self._blink_phase = 0.0
                self._next_auto_blink = self._roll_blink_interval()

        if self._blink_phase is None:
            return set()

        self._blink_phase += dt
        phase = self._blink_phase
        if phase < self._blink_close:
            envelope = phase / self._blink_close
        elif phase < self._blink_close + self._blink_open:
            envelope = 1.0 - (phase - self._blink_close) / self._blink_open
        else:
            self._blink_phase = None
            envelope = 0.0
        envelope = _smoothstep(envelope)
        pose.set("eye_wink_left", max(pose.get("eye_wink_left"), envelope))
        pose.set("eye_wink_right", max(pose.get("eye_wink_right"), envelope))
        return {"eye_wink_left", "eye_wink_right"}
