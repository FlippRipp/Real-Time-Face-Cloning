import pytest

from tha_wrapper.pose import (
    POSE_PARAMETER_NAMES,
    POSE_PARAMETER_RANGES,
    AvatarPose,
)


def test_45_parameters():
    assert len(POSE_PARAMETER_NAMES) == 45


def test_default_is_neutral():
    pose = AvatarPose()
    assert all(pose.get(n) == 0.0 for n in POSE_PARAMETER_NAMES)


def test_set_get_and_clamp():
    pose = AvatarPose()
    pose.set("mouth_aaa", 0.5)
    assert pose.get("mouth_aaa") == 0.5
    pose.set("mouth_aaa", 2.0)
    assert pose.get("mouth_aaa") == 1.0
    pose.set("head_y", -3.0)
    assert pose.get("head_y") == -1.0


def test_unknown_parameter_rejected():
    pose = AvatarPose()
    with pytest.raises(KeyError):
        pose.set("nonexistent", 1.0)
    with pytest.raises(KeyError):
        pose.get("nonexistent")


def test_as_list_order_matches_names():
    pose = AvatarPose({"eyebrow_troubled_left": 0.25, "breathing": 1.0})
    values = pose.as_list()
    assert values[0] == 0.25
    assert values[-1] == 1.0
    assert len(values) == 45


def test_lerp():
    a = AvatarPose({"mouth_aaa": 0.0, "head_x": -1.0})
    b = AvatarPose({"mouth_aaa": 1.0, "head_x": 1.0})
    mid = a.lerp(b, 0.5)
    assert mid.get("mouth_aaa") == pytest.approx(0.5)
    assert mid.get("head_x") == pytest.approx(0.0)


def test_merged_over():
    base = AvatarPose({"mouth_aaa": 0.2, "head_y": 0.4})
    over = AvatarPose({"mouth_aaa": 0.9})
    merged = over.merged_over(base, ["mouth_aaa"])
    assert merged.get("mouth_aaa") == 0.9
    assert merged.get("head_y") == 0.4


def test_ranges_sane():
    for name, (lo, hi) in POSE_PARAMETER_RANGES.items():
        assert lo < hi
        assert lo in (-1.0, 0.0)
        assert hi == 1.0
