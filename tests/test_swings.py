from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from badminton_coach.config import Config
from badminton_coach.measure.keypoints import BODY_NUM_KPTS, RACKET_NUM_KPTS, RACKET_TOP
from badminton_coach.run import RunDir
from badminton_coach.swings import find_swings, segment_swings

FPS = 30.0
TORSO = 100.0


def _racket_with_peaks(n_frames: int, peak_frames: list[int], amplitude: float = 200.0) -> np.ndarray:
    """A racket track whose RACKET_TOP traces a smooth 1D path with a sharp burst of motion (a "swing")
    around each frame in `peak_frames`, and is otherwise nearly static."""
    racket = np.full((n_frames, RACKET_NUM_KPTS, 2), np.nan)
    x = 0.0
    xs = np.zeros(n_frames)
    for t in range(n_frames):
        near_peak = any(abs(t - p) <= 3 for p in peak_frames)
        x += amplitude if near_peak else 0.0
        xs[t] = x
    racket[:, RACKET_TOP, 0] = xs
    racket[:, RACKET_TOP, 1] = 100.0
    racket[:, 1:, :] = 100.0  # fill other keypoints so downstream code doesn't choke on all-NaN rows
    racket[:, RACKET_TOP, :] = np.stack([xs, np.full(n_frames, 100.0)], axis=1)
    return racket


def _player_track(n_frames: int) -> np.ndarray:
    player = np.full((n_frames, BODY_NUM_KPTS, 2), 100.0)
    return player


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def test_find_swings_three_peaks_detected(cfg: Config) -> None:
    n_frames = 150  # 5s at 30fps
    peaks = [30, 75, 120]  # >= 0.8s (24 frames) apart
    racket = _racket_with_peaks(n_frames, peaks)
    player = _player_track(n_frames)

    swings, speed, source = find_swings(racket, player, None, FPS, TORSO, "right", cfg)
    assert len(swings) == 3
    for swing, expected_peak in zip(swings, peaks, strict=True):
        assert abs(swing.contact_frame - expected_peak) <= 3
        assert swing.mode == "shadow"
        assert swing.contact_source == "speed"


def test_find_swings_uses_shuttle_proximity_when_available(cfg: Config) -> None:
    n_frames = 60
    peak = 30
    racket = _racket_with_peaks(n_frames, [peak])
    player = _player_track(n_frames)

    # shuttle very close to the racket top exactly at frame `peak - 2` (a plausible true contact a
    # couple frames before the racket-speed signal peaks, as speed keeps rising just past contact)
    true_contact = peak - 2
    racket_x_at_contact = racket[true_contact, RACKET_TOP, 0]
    shuttle_df = pd.DataFrame(
        {
            "Frame": list(range(n_frames)),
            "X": [racket_x_at_contact if t == true_contact else 5000.0 for t in range(n_frames)],
            "Y": [100.0] * n_frames,
            "Visibility": [1 if t == true_contact else 0 for t in range(n_frames)],
            "Confidence": [0.9] * n_frames,
        }
    )

    swings, _, _ = find_swings(racket, player, shuttle_df, FPS, TORSO, "right", cfg)
    assert len(swings) == 1
    assert swings[0].contact_frame == true_contact
    assert swings[0].contact_source == "shuttle"
    assert swings[0].mode == "live"


def test_find_swings_shadow_when_shuttle_far_from_racket(cfg: Config) -> None:
    n_frames = 60
    peak = 30
    racket = _racket_with_peaks(n_frames, [peak])
    player = _player_track(n_frames)

    shuttle_df = pd.DataFrame(
        {
            "Frame": list(range(n_frames)),
            "X": [9000.0] * n_frames,  # nowhere near the racket, ever
            "Y": [9000.0] * n_frames,
            "Visibility": [1] * n_frames,
            "Confidence": [0.9] * n_frames,
        }
    )
    swings, _, _ = find_swings(racket, player, shuttle_df, FPS, TORSO, "right", cfg)
    assert len(swings) == 1
    assert swings[0].mode == "shadow"
    assert swings[0].contact_source == "speed"


def test_find_swings_falls_back_to_wrist_speed_when_racket_sparse(cfg: Config) -> None:
    from badminton_coach.measure.keypoints import RIGHT_WRIST

    n_frames = 60
    racket = np.full((n_frames, RACKET_NUM_KPTS, 2), np.nan)  # entirely missing
    player = _player_track(n_frames)
    xs = np.zeros(n_frames)
    x = 0.0
    for t in range(n_frames):
        x += 200.0 if abs(t - 30) <= 3 else 0.0
        xs[t] = x
    player[:, RIGHT_WRIST, 0] = xs

    swings, speed, source = find_swings(racket, player, None, FPS, TORSO, "right", cfg)
    assert source == "racket_arm_wrist"
    assert len(swings) == 1
    assert abs(swings[0].contact_frame - 30) <= 3


def test_find_swings_prep_and_follow_windows(cfg: Config) -> None:
    n_frames = 90
    peak = 45
    racket = _racket_with_peaks(n_frames, [peak])
    # give the racket top a clear low-point (large y) just before contact for prep_end detection
    racket[max(0, peak - 10), RACKET_TOP, 1] = 500.0
    player = _player_track(n_frames)

    swings, _, _ = find_swings(racket, player, None, FPS, TORSO, "right", cfg)
    swing = swings[0]
    assert swing.prep_start_frame < swing.contact_frame
    assert swing.follow_end_frame > swing.contact_frame
    assert swing.prep_end_frame is not None
    assert swing.prep_start_frame <= swing.prep_end_frame < swing.contact_frame


def test_segment_swings_stage_writes_and_caches(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    (run.root / "track").mkdir(parents=True)
    (run.root / "measure").mkdir(parents=True)

    n_frames = 90
    racket = _racket_with_peaks(n_frames, [45])
    player = _player_track(n_frames)
    np.savez_compressed(
        run.root / "track" / "player.npz", raw=player, smooth=player, conf=np.ones((n_frames, BODY_NUM_KPTS))
    )
    np.savez_compressed(
        run.root / "track" / "racket.npz", raw=racket, smooth=racket, conf=np.ones((n_frames, RACKET_NUM_KPTS))
    )
    (run.root / "track" / "track.json").write_text(
        '{"handedness": "right", "net_side": "right", "net_side_source": "unknown", "torso_len_median_px": 100.0}'
    )

    result1 = segment_swings(run, cfg=cfg, fps=FPS)
    assert result1["n_swings"] == 1
    assert (run.root / "swings" / "swings.json").exists()
    assert (run.root / "plots" / "swing_0.png").exists()

    status_after_first = run.stage_status("swings")
    result2 = segment_swings(run, cfg=cfg, fps=FPS)
    assert result2 == result1
    assert run.stage_status("swings") == status_after_first
