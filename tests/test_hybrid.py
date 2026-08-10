import random

import pytest

from tha_wrapper.drivers.base import PoseDriver
from tha_wrapper.drivers.hybrid import HybridDriver
from tha_wrapper.drivers.procedural import ProceduralDriver
from tha_wrapper.pose import AvatarPose


class FakeWebcam(PoseDriver):
    """Stands in for the webcam driver with a fixed tracked pose."""

    def __init__(self, values):
        self.values = values

    def update(self, dt):
        return AvatarPose(self.values)


def make_hybrid(webcam_values):
    procedural = ProceduralDriver(auto_blink=False, breathing=False,
                                  idle_sway=False, rng=random.Random(7))
    hybrid = HybridDriver(FakeWebcam(webcam_values), procedural)
    hybrid.procedural.breathing = False
    return hybrid, procedural


def run(driver, seconds, dt=1 / 60):
    pose = driver.update(dt)
    for _ in range(int(seconds / dt)):
        pose = driver.update(dt)
    return pose


def test_webcam_passes_through_when_procedural_idle():
    hybrid, _ = make_hybrid({"head_y": 0.5, "mouth_aaa": 0.3})
    pose = run(hybrid, 0.5)
    assert pose.get("head_y") == pytest.approx(0.5)
    assert pose.get("mouth_aaa") == pytest.approx(0.3)


def test_procedural_overrides_claimed_params_only():
    hybrid, procedural = make_hybrid({"head_y": 0.5, "mouth_aaa": 0.3})
    procedural.set_emotion("happy", duration=0.1)
    pose = run(hybrid, 1.0)
    # Emotion params overridden...
    assert pose.get("mouth_raised_corner_left") == pytest.approx(0.8, abs=0.02)
    # ...but unclaimed tracking passes through.
    assert pose.get("head_y") == pytest.approx(0.5, abs=0.02)
    assert pose.get("mouth_aaa") == pytest.approx(0.3, abs=0.02)


def test_control_returns_to_webcam_after_reset():
    hybrid, procedural = make_hybrid({"mouth_raised_corner_left": 0.1})
    procedural.set_emotion("happy", duration=0.05)
    run(hybrid, 0.5)
    procedural.reset(duration=0.05)
    pose = run(hybrid, 2.0)
    assert pose.get("mouth_raised_corner_left") == pytest.approx(0.1, abs=0.02)
