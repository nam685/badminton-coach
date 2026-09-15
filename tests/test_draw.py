from __future__ import annotations

import numpy as np

from badminton_coach.measure.keypoints import BODY_NUM_KPTS, RACKET_NUM_KPTS
from badminton_coach.render import draw


def _blank(w=200, h=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_text_vietnamese_and_german_no_crash() -> None:
    frame = _blank()
    out = draw.draw_text(frame, "Cú đánh cao sâu - Überkopf-Clear", (5, 5))
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)  # something was actually drawn


def test_draw_text_does_not_mutate_input() -> None:
    frame = _blank()
    original = frame.copy()
    draw.draw_text(frame, "test", (5, 5))
    np.testing.assert_array_equal(frame, original)


def test_draw_body_draws_something_when_valid() -> None:
    frame = _blank()
    kpts = np.zeros((BODY_NUM_KPTS, 3))
    kpts[:, 0] = 100
    kpts[:, 1] = 100
    kpts[:, 2] = 0.9
    out = draw.draw_body(frame, kpts)
    assert not np.array_equal(out, frame)


def test_draw_body_skips_low_confidence() -> None:
    frame = _blank()
    kpts = np.zeros((BODY_NUM_KPTS, 3))
    kpts[:, 0] = 100
    kpts[:, 1] = 100
    kpts[:, 2] = 0.0  # all below threshold
    out = draw.draw_body(frame, kpts, kpt_thr=0.3)
    np.testing.assert_array_equal(out, frame)


def test_draw_body_highlight_uses_red() -> None:
    frame = _blank()
    kpts = np.full((BODY_NUM_KPTS, 3), np.nan)
    from badminton_coach.measure.keypoints import LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST

    kpts[LEFT_SHOULDER] = [50, 50, 0.9]
    kpts[LEFT_ELBOW] = [60, 60, 0.9]
    kpts[LEFT_WRIST] = [70, 70, 0.9]
    out = draw.draw_body(frame, kpts, highlight_joints={LEFT_ELBOW})
    # the red channel should be nonzero somewhere near the highlighted joint
    region = out[55:65, 55:65]
    assert region[:, :, 2].max() > 0


def test_draw_racket_basic() -> None:
    frame = _blank()
    kpts = np.zeros((RACKET_NUM_KPTS, 3))
    kpts[:, 0] = np.array([100, 120, 100, 90, 110])
    kpts[:, 1] = np.array([50, 100, 150, 100, 100])
    kpts[:, 2] = 0.9
    out = draw.draw_racket(frame, kpts)
    assert not np.array_equal(out, frame)


def test_draw_shuttle_trail_skips_nan_points() -> None:
    frame = _blank()
    points = [(np.nan, np.nan), (50.0, 50.0), (60.0, 60.0)]
    out = draw.draw_shuttle_trail(frame, points)
    assert not np.array_equal(out, frame)


def test_draw_angle_arc_missing_point_noop() -> None:
    frame = _blank()
    a = np.array([np.nan, np.nan])
    b = np.array([50.0, 50.0])
    c = np.array([60.0, 60.0])
    out = draw.draw_angle_arc(frame, a, b, c)
    np.testing.assert_array_equal(out, frame)


def test_draw_angle_arc_with_label() -> None:
    frame = _blank()
    a = np.array([50.0, 100.0])
    b = np.array([50.0, 50.0])
    c = np.array([100.0, 50.0])
    out = draw.draw_angle_arc(frame, a, b, c, label="90deg")
    assert not np.array_equal(out, frame)


def test_draw_hud_multiple_lines() -> None:
    frame = _blank()
    out = draw.draw_hud(frame, ["swing 1", "contact: shuttle"])
    assert not np.array_equal(out, frame)


def test_draw_banner_spans_width() -> None:
    frame = _blank()
    out = draw.draw_banner(frame, "Elbow bent at contact")
    assert not np.array_equal(out, frame)
    assert out.shape == frame.shape
