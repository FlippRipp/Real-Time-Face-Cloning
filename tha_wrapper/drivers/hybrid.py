"""Hybrid driver: webcam tracking with procedural overrides.

The webcam provides the base pose (your real face). Any parameter the
procedural driver is actively animating — an emotion, a scripted look, a
viseme timeline — overrides the webcam value for as long as it is active,
then control eases back to the webcam.

This lets an AI (or hotkeys) trigger expressions on top of live tracking,
similar to expression hotkeys in VTuber apps but fully programmable.
"""

from __future__ import annotations

from tha_wrapper.drivers.base import PoseDriver
from tha_wrapper.drivers.procedural import ProceduralDriver
from tha_wrapper.pose import AvatarPose


class HybridDriver(PoseDriver):
    def __init__(self, webcam: PoseDriver, procedural: ProceduralDriver,
                 fade: float = 8.0):
        self.webcam = webcam
        self.procedural = procedural
        self._fade = fade  # per-second rate at which overrides blend in/out
        self._weights: dict = {}  # param -> current override weight 0..1
        # Idle layers would fight the live tracking, so default them off;
        # the control API can turn them back on for AFK mode.
        self.procedural.idle_sway = False
        self.procedural.breathing = True  # webcam does not produce breathing

    def start(self) -> None:
        self.webcam.start()
        self.procedural.start()

    def close(self) -> None:
        self.webcam.close()
        self.procedural.close()

    def update(self, dt: float) -> AvatarPose:
        base = self.webcam.update(dt)
        overlay = self.procedural.update(dt)
        claimed = self.procedural.touched()

        # Smoothly raise weights for claimed params, decay the rest.
        step = min(1.0, self._fade * dt)
        for name in claimed:
            self._weights[name] = min(1.0, self._weights.get(name, 0.0) + step)
        for name in [n for n in self._weights if n not in claimed]:
            self._weights[name] -= step
            if self._weights[name] <= 0.0:
                del self._weights[name]

        pose = base.copy()
        for name, weight in self._weights.items():
            blended = base.get(name) + (overlay.get(name) - base.get(name)) * weight
            pose.set(name, blended)
        return pose
