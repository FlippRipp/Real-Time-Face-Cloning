import numpy as np

from tha_wrapper.pose import AvatarPose
from tha_wrapper.poser.mock import MockPoser
from tha_wrapper.output.base import composite_background


def test_mock_poser_renders_rgba():
    poser = MockPoser(size=128)
    frame = poser.pose(AvatarPose())
    assert frame.shape == (128, 128, 4)
    assert frame.dtype == np.uint8


def test_mock_poser_responds_to_pose():
    poser = MockPoser(size=128)
    neutral = poser.pose(AvatarPose())
    moved = poser.pose(AvatarPose({"head_y": 1.0, "eye_wink_left": 1.0}))
    assert not np.array_equal(neutral, moved)


def test_composite_background_named_color():
    poser = MockPoser(size=64)
    frame = poser.pose(AvatarPose())
    rgb = composite_background(frame, "green")
    assert rgb.shape == (64, 64, 3)
    # A transparent corner must be pure background.
    assert tuple(rgb[0, 0]) == (0, 255, 0)


def test_composite_background_tuple_and_image():
    frame = np.zeros((8, 8, 4), dtype=np.uint8)
    frame[:, :, 3] = 0
    rgb = composite_background(frame, (10, 20, 30))
    assert tuple(rgb[3, 3]) == (10, 20, 30)
    bg_image = np.full((8, 8, 3), 77, dtype=np.uint8)
    rgb = composite_background(frame, bg_image)
    assert tuple(rgb[3, 3]) == (77, 77, 77)
