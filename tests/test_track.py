from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from badminton_coach.measure.keypoints import (
    BODY_NUM_KPTS,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RACKET_HANDLE,
    RACKET_NUM_KPTS,
    RACKET_TOP,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from badminton_coach.track import (
    _fill_small_gaps,
    _iou,
    _majority_handedness,
    _smooth_with_gaps,
    associate_racket,
    determine_net_direction,
    select_player,
)


def _make_person(cx: float, cy: float, scale: float, score: float = 0.9) -> np.ndarray:
    """A plausible standing-person Halpe-26 skeleton centered at (cx, cy)."""
    kpts = np.full((BODY_NUM_KPTS, 3), np.nan, dtype=np.float32)
    offsets = {
        LEFT_SHOULDER: (-0.3, -0.8),
        RIGHT_SHOULDER: (0.3, -0.8),
        LEFT_HIP: (-0.2, 0.0),
        RIGHT_HIP: (0.2, 0.0),
        LEFT_WRIST: (-0.5, -0.5),
        RIGHT_WRIST: (0.5, -0.5),
    }
    for idx, (ox, oy) in offsets.items():
        kpts[idx] = [cx + ox * scale, cy + oy * scale, score]
    return kpts


def test_select_player_prefers_foreground_over_background_far_from_racket() -> None:
    n_frames = 5
    # person 0: small (far/background), person 1: large (foreground) and near the racket
    body = np.full((n_frames, 2, BODY_NUM_KPTS, 3), np.nan, dtype=np.float32)
    racket = np.full((n_frames, 1, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    for t in range(n_frames):
        body[t, 0] = _make_person(cx=600, cy=100, scale=40)  # small, background, far away
        body[t, 1] = _make_person(cx=300, cy=300, scale=150)  # large, foreground
        racket[t, 0, RACKET_HANDLE] = [320, 350, 0.9]  # near person 1's wrist

    result = select_player(body, racket, kpt_conf_thr=0.3)
    assert (result.chosen == 1).all()


def test_select_player_background_person_wins_when_holding_the_racket() -> None:
    """Even a smaller person should win if the racket handle is right at their wrist and clearly far
    (several torso-lengths) from the larger person — proximity to the racket matters, not just size."""
    n_frames = 3
    body = np.full((n_frames, 2, BODY_NUM_KPTS, 3), np.nan, dtype=np.float32)
    racket = np.full((n_frames, 1, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    for t in range(n_frames):
        body[t, 0] = _make_person(cx=1000, cy=100, scale=40)  # small, but has the racket
        body[t, 1] = _make_person(cx=300, cy=300, scale=150)  # large, racket clearly not theirs
        racket[t, 0, RACKET_HANDLE] = [1000 + 0.5 * 40, 100 - 0.5 * 40, 0.9]  # at person 0's right wrist

    result = select_player(body, racket, kpt_conf_thr=0.3)
    assert (result.chosen == 0).all()


def test_select_player_no_candidates_returns_minus_one() -> None:
    body = np.full((2, 1, BODY_NUM_KPTS, 3), np.nan, dtype=np.float32)
    racket = np.full((2, 1, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    result = select_player(body, racket, kpt_conf_thr=0.3)
    assert (result.chosen == -1).all()


def test_iou_basic() -> None:
    a = np.array([0, 0, 10, 10])
    b = np.array([5, 5, 15, 15])
    assert 0.0 < _iou(a, b) < 1.0
    assert _iou(a, a) == 1.0
    c = np.array([100, 100, 110, 110])
    assert _iou(a, c) == 0.0


def test_associate_racket_and_handedness_right() -> None:
    n_frames = 10
    player = np.full((n_frames, BODY_NUM_KPTS, 3), np.nan, dtype=np.float32)
    racket = np.full((n_frames, 1, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    for t in range(n_frames):
        player[t] = _make_person(cx=300, cy=300, scale=150)
        # racket handle right at the right wrist
        rw = player[t, RIGHT_WRIST, 0:2]
        racket[t, 0, RACKET_HANDLE] = [rw[0], rw[1], 0.9]
        racket[t, 0, RACKET_TOP] = [rw[0], rw[1] - 50, 0.9]

    track_arr, sides = associate_racket(player, racket, kpt_conf_thr=0.3)
    assert all(s == "right" for s in sides)
    assert _majority_handedness(sides) == "right"
    assert not np.isnan(track_arr[0, RACKET_HANDLE, 0])


def test_associate_racket_too_far_is_not_associated() -> None:
    n_frames = 3
    player = np.full((n_frames, BODY_NUM_KPTS, 3), np.nan, dtype=np.float32)
    racket = np.full((n_frames, 1, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    for t in range(n_frames):
        player[t] = _make_person(cx=300, cy=300, scale=150)
        racket[t, 0, RACKET_HANDLE] = [3000, 3000, 0.9]  # absurdly far

    _track_arr, sides = associate_racket(player, racket, kpt_conf_thr=0.3, max_dist_torso=3.0)
    assert all(s is None for s in sides)


def test_majority_handedness_no_votes_returns_none() -> None:
    assert _majority_handedness([None, None]) is None


def test_majority_handedness_tie_prefers_right() -> None:
    assert _majority_handedness(["left", "right"]) == "right"


def test_fill_small_gaps_interpolates_short_gap() -> None:
    x = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    out = _fill_small_gaps(x, max_gap=2)
    assert not np.isnan(out).any()
    assert out[2] == pytest.approx(3.0)
    assert out[3] == pytest.approx(4.0)


def test_fill_small_gaps_leaves_long_gap() -> None:
    x = np.array([1.0, 2.0] + [np.nan] * 10 + [20.0, 21.0])
    out = _fill_small_gaps(x, max_gap=3)
    assert np.isnan(out[2:12]).all()


def test_fill_small_gaps_edge_gap_not_filled() -> None:
    # a gap touching the start/end has no anchor on one side and is intentionally left as NaN
    x = np.array([np.nan, np.nan, 3.0, 4.0])
    out = _fill_small_gaps(x, max_gap=3)
    assert np.isnan(out[0:2]).all()


def test_smooth_with_gaps_preserves_peak_location() -> None:
    t = np.linspace(0, 1, 60)
    x = np.sin(2 * np.pi * t) + np.random.default_rng(0).normal(0, 0.02, size=60)
    smoothed = _smooth_with_gaps(x.copy(), window=7, polyorder=2)
    assert np.argmax(smoothed) in range(np.argmax(x) - 2, np.argmax(x) + 3)
    # smoothing should reduce noise (lower local variance) without destroying the signal
    assert np.std(np.diff(smoothed)) < np.std(np.diff(x))


def test_smooth_with_gaps_short_run_left_unsmoothed_but_no_crash() -> None:
    x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 5.0, 6.0])
    out = _smooth_with_gaps(x, window=7, polyorder=2)
    assert not np.isnan(out[0:2]).any()
    assert np.isnan(out[2:5]).all()


def test_determine_net_direction_from_shuttle() -> None:
    n = 20
    racket = np.full((n, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    for t in range(n):
        racket[t, RACKET_TOP] = [100 + t * (20 if t < 10 else 0), 100, 0.9]
    shuttle_df = pd.DataFrame(
        {
            "Frame": list(range(n)),
            "X": [100 + t * 5 for t in range(n)],
            "Y": [100] * n,
            "Visibility": [1] * n,
            "Confidence": [0.9] * n,
        }
    )
    net_side, source = determine_net_direction(racket, shuttle_df, fps=30.0)
    assert net_side == "right"
    assert source == "shuttle"


def test_determine_net_direction_falls_back_to_racket_travel_without_shuttle() -> None:
    n = 20
    racket = np.full((n, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    for t in range(n):
        x = 500 - t * (20 if t < 10 else 0) - (t - 10) * 15 if t >= 10 else 500 - t * 20
        racket[t, RACKET_TOP] = [x, 100, 0.9]
    net_side, source = determine_net_direction(racket, None, fps=30.0)
    assert net_side == "left"
    assert source == "racket_travel"


def test_determine_net_direction_override_wins() -> None:
    racket = np.full((5, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    net_side, source = determine_net_direction(racket, None, fps=30.0, net_side_override="left")
    assert net_side == "left"
    assert source == "override"


def test_determine_net_direction_insufficient_data_is_unknown() -> None:
    racket = np.full((5, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    net_side, source = determine_net_direction(racket, None, fps=30.0)
    assert source == "unknown"
