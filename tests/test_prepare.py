"""Tests for the pure image math in tha_wrapper.prepare (no cv2/mediapipe/rembg)."""

import numpy as np
import pytest

from tha_wrapper.prepare import (
    CANVAS,
    HEAD_BOX,
    HEAD_TARGET_CENTER,
    HEAD_TARGET_HEIGHT,
    checkerboard,
    compose,
    constraint_report,
    fallback_placement,
    head_box_verdict,
    head_search_crops,
    load_rgba,
    over_background,
    placement_for_head,
    transform_head,
    unique_output_path,
)


def solid_char(width=100, height=200, color=(255, 0, 0, 255)):
    char = np.zeros((height, width, 4), dtype=np.uint8)
    char[...] = color
    return char


def test_compose_identity_size_and_transparency():
    final = compose(solid_char(), (1.0, 10.0, 20.0))
    assert final.shape == (CANVAS, CANVAS, 4)
    assert final[19, 10, 3] == 0          # above the paste position
    assert final[20, 10, 3] == 255        # top-left of the character
    assert final[219, 109, 3] == 255      # bottom-right of the character
    assert final[220, 110, 3] == 0


def test_compose_scales():
    final = compose(solid_char(100, 100), (2.0, 0.0, 0.0))
    assert final[199, 199, 3] == 255
    assert final[201, 201, 3] == 0


def test_compose_offcanvas_clips_without_error():
    final = compose(solid_char(), (1.0, -50.0, -50.0))
    assert final[0, 0, 3] == 255
    assert final[..., 3].any()


def test_placement_for_head_centers_head():
    head = (300.0, 400.0, 100.0, 120.0)
    placement = placement_for_head(head)
    cx, cy, _w, h = transform_head(head, placement)
    assert cx == pytest.approx(HEAD_TARGET_CENTER[0])
    assert cy == pytest.approx(HEAD_TARGET_CENTER[1])
    assert h == pytest.approx(HEAD_TARGET_HEIGHT)


def test_placement_for_head_lands_in_target_box():
    placement = placement_for_head((123.0, 456.0, 80.0, 90.0))
    ok, _detail = head_box_verdict(transform_head((123.0, 456.0, 80.0, 90.0), placement))
    assert ok


def test_fallback_placement_fits_and_centers():
    scale, tx, ty = fallback_placement(1000, 2000)
    assert scale == pytest.approx(CANVAS / 2000)
    assert tx == pytest.approx((CANVAS - 1000 * scale) / 2)
    assert ty == 0.0


def test_head_box_verdict_rejects_outside_and_missing():
    assert not head_box_verdict(None)[0]
    assert not head_box_verdict((0.0, 0.0, 100.0, 118.0))[0]        # off-box center
    assert not head_box_verdict((256.0, 128.0, 100.0, 300.0))[0]    # oversized
    x0, y0, x1, y1 = HEAD_BOX
    assert head_box_verdict(((x0 + x1) / 2, (y0 + y1) / 2, 100.0, 118.0))[0]


def test_checkerboard_and_over_background():
    board = checkerboard(32, 48, cell=8)
    assert board.shape == (32, 48, 3)
    assert {160, 200} == set(np.unique(board))

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[0, 0] = (10, 20, 30, 255)
    rgb = over_background(rgba, (0, 255, 0))
    assert tuple(rgb[0, 0]) == (10, 20, 30)
    assert tuple(rgb[1, 1]) == (0, 255, 0)


def test_unique_output_path(tmp_path):
    target = tmp_path / "char.png"
    assert unique_output_path(str(target)) == str(target)
    target.write_bytes(b"")
    assert unique_output_path(str(target)) == str(tmp_path / "char_1.png")
    (tmp_path / "char_1.png").write_bytes(b"")
    assert unique_output_path(str(target)) == str(tmp_path / "char_2.png")


def test_load_rgba_reports_alpha(tmp_path):
    from PIL import Image

    rgb_path = tmp_path / "flat.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(rgb_path)
    array, had_alpha = load_rgba(str(rgb_path))
    assert array.shape == (8, 8, 4) and not had_alpha
    assert (array[..., 3] == 255).all()

    rgba_path = tmp_path / "alpha.png"
    Image.new("RGBA", (8, 8), (1, 2, 3, 0)).save(rgba_path)
    _array, had_alpha = load_rgba(str(rgba_path))
    assert had_alpha


def test_head_search_crops_alpha_guided():
    alpha = np.zeros((400, 300), dtype=np.uint8)
    alpha[100:380, 50:250] = 255  # character occupies a sub-region
    crops = head_search_crops(alpha)
    assert crops[0] == (0, 0, 300, 400)          # full frame first
    assert crops[1] == (50, 100, 250, 380)       # alpha bounding box
    x0, y0, x1, y1 = crops[2]                    # top portion of the bbox
    assert (x0, y0, x1) == (50, 100, 250)
    assert y1 == 100 + int(280 * 0.45)


def test_head_search_crops_opaque_and_empty():
    opaque = np.full((200, 100), 255, dtype=np.uint8)
    crops = head_search_crops(opaque)
    assert crops[0] == (0, 0, 100, 200)
    assert crops[-1] == (0, 0, 100, int(200 * 0.45))  # degrades to top of frame
    assert (0, 0, 100, 200) not in crops[1:]          # bbox==frame not duplicated

    empty = np.zeros((50, 50), dtype=np.uint8)
    assert head_search_crops(empty) == [(0, 0, 50, 50)]


def test_constraint_report_flags():
    final = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
    final[100:200, 200:300, 3] = 255
    report = dict.fromkeys(s for s, _ in constraint_report(final, 1, (256.0, 128.0, 100.0, 118.0)))
    assert "WARN" not in report

    # Character touching the top edge and two faces -> warnings.
    final[0, :, 3] = 255
    statuses = [s for s, _ in constraint_report(final, 2, None)]
    assert statuses.count("WARN") >= 3
