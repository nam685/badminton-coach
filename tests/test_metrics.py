from __future__ import annotations

import numpy as np
import pytest

from badminton_coach import metrics as m
from badminton_coach.measure.keypoints import (
    BODY_NUM_KPTS,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RACKET_BOTTOM,
    RACKET_HANDLE,
    RACKET_LEFT,
    RACKET_NUM_KPTS,
    RACKET_RIGHT,
    RACKET_TOP,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


def _player(scale: float = 1.0) -> np.ndarray:
    """A simple right-handed standing pose, arm straight out to the side (180 deg elbow), scaled by
    `scale` (used to test normalization invariance)."""
    p = np.full((BODY_NUM_KPTS, 2), np.nan)
    s = scale
    p[NOSE] = [0, -100 * s]
    p[LEFT_SHOULDER] = [-20 * s, -80 * s]
    p[RIGHT_SHOULDER] = [20 * s, -80 * s]
    p[LEFT_HIP] = [-15 * s, 0]
    p[RIGHT_HIP] = [15 * s, 0]
    p[LEFT_ELBOW] = [-20 * s, -40 * s]
    p[LEFT_WRIST] = [-20 * s, 0]
    p[RIGHT_ELBOW] = [60 * s, -80 * s]  # straight out to the side => 180 deg elbow
    p[RIGHT_WRIST] = [100 * s, -80 * s]
    p[LEFT_ANKLE] = [-15 * s, 80 * s]
    p[RIGHT_ANKLE] = [15 * s, 80 * s]
    return p


def _racket(top: tuple[float, float], bottom: tuple[float, float], left=None, right=None) -> np.ndarray:
    r = np.full((RACKET_NUM_KPTS, 2), np.nan)
    r[RACKET_TOP] = top
    r[RACKET_BOTTOM] = bottom
    r[RACKET_HANDLE] = bottom
    if left is not None:
        r[RACKET_LEFT] = left
    if right is not None:
        r[RACKET_RIGHT] = right
    return r


TORSO = 80.0  # |mid_shoulder - mid_hip| for scale=1.0 in _player()


def test_resolve_sides_right() -> None:
    sides = m.resolve_sides("right")
    assert sides.shoulder == RIGHT_SHOULDER
    assert sides.wrist == RIGHT_WRIST
    assert sides.other_wrist == LEFT_WRIST


def test_resolve_sides_left() -> None:
    sides = m.resolve_sides("left")
    assert sides.shoulder == LEFT_SHOULDER
    assert sides.other_shoulder == RIGHT_SHOULDER


def test_net_dir_sign() -> None:
    assert m.net_dir_sign("right") == 1.0
    assert m.net_dir_sign("left") == -1.0


def test_angle_at_straight_line_is_180() -> None:
    a, b, c = np.array([0, 0]), np.array([1, 0]), np.array([2, 0])
    assert m.angle_at(a, b, c) == pytest.approx(180.0)


def test_angle_at_right_angle_is_90() -> None:
    a, b, c = np.array([1, 0]), np.array([0, 0]), np.array([0, 1])
    assert m.angle_at(a, b, c) == pytest.approx(90.0)


def test_angle_at_degenerate_returns_nan() -> None:
    a, b, c = np.array([0, 0]), np.array([0, 0]), np.array([1, 0])
    assert math_isnan(m.angle_at(a, b, c))


def math_isnan(x: float) -> bool:
    return x != x


def test_elbow_angle_straight_arm_is_180() -> None:
    sides = m.resolve_sides("right")
    assert m.elbow_angle(_player(), sides) == pytest.approx(180.0, abs=1e-6)


def test_elbow_angle_missing_keypoint_is_none() -> None:
    p = _player()
    p[RIGHT_ELBOW] = [np.nan, np.nan]
    sides = m.resolve_sides("right")
    assert m.elbow_angle(p, sides) is None


def test_contact_height_positive_when_racket_above_nose() -> None:
    p = _player()
    r = _racket(top=(0, -150), bottom=(0, -100))  # well above nose (nose y=-100)
    result = m.contact_height(p, r, TORSO)
    assert result is not None
    assert result["vs_nose"] > 0


def test_contact_height_negative_when_racket_below_nose() -> None:
    p = _player()
    r = _racket(top=(0, 0), bottom=(0, 50))
    result = m.contact_height(p, r, TORSO)
    assert result["vs_nose"] < 0


def test_contact_forward_sign_flips_with_net_side() -> None:
    p = _player()
    sides = m.resolve_sides("right")
    # racket top to the right (+x) of the right shoulder
    r = _racket(top=(100, -80), bottom=(60, -80))
    forward_net_right = m.contact_forward(p, r, TORSO, sides, m.net_dir_sign("right"))
    forward_net_left = m.contact_forward(p, r, TORSO, sides, m.net_dir_sign("left"))
    assert forward_net_right > 0
    assert forward_net_left < 0
    assert forward_net_right == pytest.approx(-forward_net_left)


def test_contact_forward_left_handed_mirrors() -> None:
    """A left-handed player's racket-side shoulder is the left shoulder; same net direction should
    still give a positive contact_forward when the racket is toward the net."""
    p = _player()
    sides_left = m.resolve_sides("left")
    r = _racket(top=(-100, -80), bottom=(-60, -80))  # out to the left, "in front" if net is on the left
    forward = m.contact_forward(p, r, TORSO, sides_left, m.net_dir_sign("left"))
    assert forward > 0


def test_racket_shaft_angle_vertical_is_zero() -> None:
    r = _racket(top=(0, -50), bottom=(0, 0))
    assert m.racket_shaft_angle(r, 1.0) == pytest.approx(0.0, abs=1e-6)


def test_racket_shaft_angle_tilted_toward_net_is_positive() -> None:
    r = _racket(top=(50, -50), bottom=(0, 0))
    assert m.racket_shaft_angle(r, 1.0) > 0
    assert m.racket_shaft_angle(r, -1.0) < 0


def test_racket_face_proxy() -> None:
    r = _racket(top=(0, -50), bottom=(0, 0), left=(-10, -25), right=(10, -25))
    val = m.racket_face_proxy(r)
    assert val == pytest.approx(20 / 50)


def test_trunk_lean_upright_is_zero() -> None:
    p = _player()
    assert m.trunk_lean(p, 1.0) == pytest.approx(0.0, abs=1e-6)


def test_shoulder_and_hip_width_ratio_scale_invariant() -> None:
    for scale in (0.5, 1.0, 2.0):
        p = _player(scale=scale)
        torso = scale * TORSO
        sw = m.shoulder_width_ratio(p, torso)
        hw = m.hip_width_ratio(p, torso)
        assert sw == pytest.approx(40 / 80, abs=1e-6)
        assert hw == pytest.approx(30 / 80, abs=1e-6)


def test_non_racket_wrist_height_positive_when_raised() -> None:
    p = _player()
    sides = m.resolve_sides("right")  # non-racket = left
    # left wrist already above the left shoulder in _player()? shoulder y=-80, wrist y=0 -> below
    val = m.non_racket_wrist_height(p, sides, TORSO)
    assert val < 0  # wrist below shoulder in the default fixture
    p[LEFT_WRIST] = [-20, -150]  # raise it above the shoulder
    val2 = m.non_racket_wrist_height(p, sides, TORSO)
    assert val2 > 0


def test_stance_width() -> None:
    p = _player()
    assert m.stance_width(p, TORSO) == pytest.approx(30 / 80, abs=1e-6)


def test_front_foot_matches_net_side() -> None:
    p = _player()
    p[LEFT_ANKLE] = [-50, 80]
    p[RIGHT_ANKLE] = [10, 80]
    assert m.front_foot(p, m.net_dir_sign("right")) == "right"
    assert m.front_foot(p, m.net_dir_sign("left")) == "left"


def test_jump_height_zero_when_standing() -> None:
    p = _player()
    standing_y = min(p[LEFT_ANKLE][1], p[RIGHT_ANKLE][1])
    assert m.jump_height(p, standing_y, TORSO) == pytest.approx(0.0)


def test_jump_height_positive_when_airborne() -> None:
    p = _player()
    standing_y = 80.0
    p[LEFT_ANKLE][1] = 40.0
    p[RIGHT_ANKLE][1] = 45.0
    val = m.jump_height(p, standing_y, TORSO)
    assert val > 0


def test_racket_speed_series_basic() -> None:
    n = 5
    racket = np.full((n, RACKET_NUM_KPTS, 2), np.nan)
    for t in range(n):
        racket[t, RACKET_TOP] = [t * 10.0, 0.0]
    speed = m.racket_speed_series(racket, fps=10.0, torso_len=10.0)
    assert np.isnan(speed[0])
    assert speed[1] == pytest.approx(10.0)  # 10px moved * 10fps / 10 torso = 10


def test_racket_speed_series_nan_on_gap() -> None:
    n = 5
    racket = np.full((n, RACKET_NUM_KPTS, 2), np.nan)
    racket[0, RACKET_TOP] = [0, 0]
    racket[2, RACKET_TOP] = [20, 0]
    speed = m.racket_speed_series(racket, fps=10.0, torso_len=10.0)
    assert np.isnan(speed[1])
    assert np.isnan(speed[2])  # frame 1 missing breaks the t-1 pairing for frame 2 too
