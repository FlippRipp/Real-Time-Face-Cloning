import random

import pytest

from tha_wrapper.drivers.procedural import EMOTIONS, VISEMES, ProceduralDriver


def make_driver(**kwargs) -> ProceduralDriver:
    defaults = dict(auto_blink=False, breathing=False, idle_sway=False,
                    rng=random.Random(42))
    defaults.update(kwargs)
    return ProceduralDriver(**defaults)


def run(driver, seconds, dt=1 / 60):
    pose = driver.update(dt)
    steps = max(1, int(seconds / dt))
    for _ in range(steps):
        pose = driver.update(dt)
    return pose


def test_set_param_eases_to_target():
    driver = make_driver()
    driver.set_param("mouth_aaa", 0.8, duration=0.2)
    early = run(driver, 0.05)
    assert 0.0 < early.get("mouth_aaa") < 0.8
    settled = run(driver, 0.5)
    assert settled.get("mouth_aaa") == pytest.approx(0.8, abs=1e-6)


def test_instant_set():
    driver = make_driver()
    driver.set_param("head_y", -0.5, duration=0.0)
    assert driver.update(1 / 60).get("head_y") == pytest.approx(-0.5)


def test_emotion_applies_and_switches_cleanly():
    driver = make_driver()
    driver.set_emotion("happy", duration=0.1)
    pose = run(driver, 0.5)
    for name, value in EMOTIONS["happy"].items():
        assert pose.get(name) == pytest.approx(value, abs=1e-5)

    driver.set_emotion("angry", duration=0.1)
    pose = run(driver, 0.5)
    for name in EMOTIONS["happy"]:
        if name not in EMOTIONS["angry"]:
            assert pose.get(name) == pytest.approx(0.0, abs=1e-5)
    for name, value in EMOTIONS["angry"].items():
        assert pose.get(name) == pytest.approx(value, abs=1e-5)


def test_emotion_intensity_scales():
    driver = make_driver()
    driver.set_emotion("happy", intensity=0.5, duration=0.05)
    pose = run(driver, 0.5)
    assert pose.get("eyebrow_happy_left") == pytest.approx(0.5, abs=1e-5)


def test_unknown_emotion_raises():
    with pytest.raises(KeyError):
        make_driver().set_emotion("melancholy")


def test_blink_closes_and_reopens():
    driver = make_driver()
    driver.blink()
    peak = 0.0
    for _ in range(60):
        pose = driver.update(1 / 120)
        peak = max(peak, pose.get("eye_wink_left"))
    assert peak > 0.9
    settled = run(driver, 0.5)
    assert settled.get("eye_wink_left") == 0.0


def test_auto_blink_fires():
    driver = make_driver(auto_blink=True)
    peak = 0.0
    for _ in range(int(8.0 * 60)):
        pose = driver.update(1 / 60)
        peak = max(peak, pose.get("eye_wink_left"))
    assert peak > 0.9


def test_viseme_timeline_plays_and_ends():
    driver = make_driver()
    driver.speak_visemes([("aa", 0.3), ("oh", 0.3), ("sil", 0.1)])
    early = run(driver, 0.15)
    assert early.get("mouth_aaa") > 0.3
    mid = run(driver, 0.3)
    assert mid.get("mouth_ooo") > 0.3
    done = run(driver, 1.0)
    assert done.get("mouth_aaa") == pytest.approx(0.0, abs=0.01)
    assert done.get("mouth_ooo") == pytest.approx(0.0, abs=0.01)


def test_speak_text_generates_mouth_motion():
    driver = make_driver()
    duration = driver.speak_text("hello ai avatar")
    assert duration > 0.3
    moved = 0.0
    for _ in range(int(duration * 60) + 10):
        pose = driver.update(1 / 60)
        moved = max(
            moved,
            *(pose.get(n) for n in
              ("mouth_aaa", "mouth_eee", "mouth_ooo", "mouth_iii", "mouth_uuu")),
        )
    assert moved > 0.4


def test_talking_flaps_mouth():
    driver = make_driver()
    driver.set_talking(True)
    peak = 0.0
    for _ in range(120):
        pose = driver.update(1 / 60)
        peak = max(peak, *(pose.get(n) for n in
                           ("mouth_aaa", "mouth_eee", "mouth_ooo", "mouth_iii")))
    assert peak > 0.1
    driver.set_talking(False)
    settled = run(driver, 1.0)
    assert settled.get("mouth_aaa") == pytest.approx(0.0, abs=0.01)


def test_audio_energy_moves_mouth_and_decays():
    driver = make_driver()
    driver.push_audio_energy(0.9)
    pose = driver.update(1 / 60)
    assert pose.get("mouth_aaa") > 0.1
    settled = run(driver, 1.0)
    assert settled.get("mouth_aaa") == pytest.approx(0.0, abs=0.01)


def test_idle_layers():
    driver = ProceduralDriver(auto_blink=False, breathing=True, idle_sway=True,
                              rng=random.Random(1))
    pose = run(driver, 1.0)
    assert 0.0 <= pose.get("breathing") <= 1.0
    swayed = any(
        abs(run(driver, 0.5).get("head_y")) > 1e-4 for _ in range(4)
    )
    assert swayed


def test_look_at_with_head_follow():
    driver = make_driver()
    driver.look_at(0.6, -0.4, duration=0.05, head_follow=0.5)
    pose = run(driver, 0.5)
    assert pose.get("iris_rotation_y") == pytest.approx(0.6, abs=1e-5)
    assert pose.get("iris_rotation_x") == pytest.approx(-0.4, abs=1e-5)
    assert pose.get("head_y") == pytest.approx(0.3, abs=1e-5)


def test_reset_returns_to_neutral():
    driver = make_driver()
    driver.set_emotion("surprised", duration=0.05)
    driver.set_talking(True)
    run(driver, 0.3)
    driver.reset(duration=0.1)
    pose = run(driver, 0.5)
    assert pose == type(pose)()  # fully neutral


def test_touched_reflects_activity():
    driver = make_driver()
    run(driver, 0.1)
    assert driver.touched() == set()
    driver.set_emotion("happy", duration=0.1)
    run(driver, 0.05)
    assert "mouth_raised_corner_left" in driver.touched()
    driver.reset(duration=0.05)
    run(driver, 1.0)
    assert driver.touched() == set()


def test_perform_fires_actions_at_their_offsets():
    driver = make_driver()
    driver.perform([
        {"emotion": "happy", "duration": 0.1},
        {"at": 1.0, "look": [0.6, -0.2], "duration": 0.05},
    ])
    early = run(driver, 0.5)
    assert early.get("eyebrow_happy_left") == pytest.approx(1.0, abs=1e-4)
    assert early.get("iris_rotation_y") == 0.0  # look not fired yet
    assert driver.state()["performing"] is True
    late = run(driver, 1.0)
    assert late.get("iris_rotation_y") == pytest.approx(0.6, abs=1e-4)
    assert driver.state()["performing"] is False


def test_perform_say_returns_duration_and_moves_mouth():
    driver = make_driver()
    duration = driver.perform([{"at": 0.2, "say": "hello avatar"}])
    assert duration > 0.5  # 0.2 offset + speech
    moved = 0.0
    for _ in range(int(duration * 60) + 10):
        pose = driver.update(1 / 60)
        moved = max(moved, *(pose.get(n) for n in
                             ("mouth_aaa", "mouth_eee", "mouth_ooo",
                              "mouth_iii", "mouth_uuu")))
    assert moved > 0.4


def test_perform_new_script_replaces_pending():
    driver = make_driver()
    driver.perform([{"at": 0.5, "emotion": "angry", "duration": 0.05}])
    run(driver, 0.1)
    driver.perform([{"at": 0.1, "emotion": "happy", "duration": 0.05}])
    pose = run(driver, 1.0)
    assert pose.get("eyebrow_angry_left") == pytest.approx(0.0, abs=1e-4)
    assert pose.get("eyebrow_happy_left") == pytest.approx(1.0, abs=1e-4)


def test_perform_empty_script_cancels():
    driver = make_driver()
    driver.perform([{"at": 0.5, "emotion": "angry"}])
    assert driver.perform([]) == 0.0
    pose = run(driver, 1.0)
    assert pose.get("eyebrow_angry_left") == 0.0


def test_perform_rejects_bad_scripts_atomically():
    driver = make_driver()
    with pytest.raises(ValueError, match="exactly one action key"):
        driver.perform([{"emotion": "happy"}, {"at": 0.1}])
    with pytest.raises(ValueError, match="unknown keys"):
        driver.perform([{"emotion": "happy", "head_follow": 0.3}])
    with pytest.raises(KeyError, match="unknown emotion"):
        driver.perform([{"emotion": "melancholy"}])
    with pytest.raises(KeyError, match="unknown pose parameter"):
        driver.perform([{"params": {"tail_wag": 1.0}}])
    # Nothing from the rejected scripts was scheduled.
    pose = run(driver, 0.5)
    assert pose == type(pose)()
    assert driver.state()["performing"] is False


def test_perform_blink_and_talking_actions():
    driver = make_driver()
    driver.perform([
        {"blink": True},
        {"at": 0.05, "talking": True},
        {"at": 0.6, "talking": False},
    ])
    peak = 0.0
    for _ in range(60):
        pose = driver.update(1 / 60)
        peak = max(peak, pose.get("eye_wink_left"))
    assert peak > 0.9
    settled = run(driver, 1.5)
    assert settled.get("mouth_aaa") == pytest.approx(0.0, abs=0.01)


def test_stop_performance_cancels_but_holds_pose():
    driver = make_driver()
    driver.perform([
        {"emotion": "happy", "duration": 0.05},
        {"say": "hello hello hello"},
        {"at": 5.0, "emotion": "angry", "duration": 0.05},
    ])
    run(driver, 0.3)
    driver.stop_performance()
    state = driver.state()
    assert state["performing"] is False
    assert state["talking"] is False
    pose = run(driver, 1.0)
    assert pose.get("eyebrow_happy_left") == pytest.approx(1.0, abs=1e-4)
    assert pose.get("eyebrow_angry_left") == 0.0


def test_reset_cancels_performance():
    driver = make_driver()
    driver.perform([{"at": 2.0, "emotion": "angry"}])
    driver.reset(duration=0.05)
    pose = run(driver, 3.0)
    assert pose == type(pose)()
    assert driver.state()["performing"] is False


def test_visemes_registry_valid():
    from tha_wrapper.pose import POSE_PARAMETER_RANGES

    for shape in VISEMES.values():
        for name in shape:
            assert name in POSE_PARAMETER_RANGES
    for shape in EMOTIONS.values():
        for name in shape:
            assert name in POSE_PARAMETER_RANGES
