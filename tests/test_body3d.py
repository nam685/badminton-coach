from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from badminton_coach.config import Config
from badminton_coach.measure.body3d import (
    _bbox_for_frame,
    _consistency_check,
    _interpolate_and_smooth,
    _windows_from_swings,
    estimate_3d,
)
from badminton_coach.run import RunDir
from badminton_coach.schema import Swing


def _swing(index: int, prep_start: int, contact: int, follow_end: int) -> Swing:
    return Swing(
        index=index,
        mode="live",
        contact_frame=contact,
        contact_time_s=contact / 30,
        contact_source="shuttle",
        prep_start_frame=prep_start,
        prep_end_frame=(prep_start + contact) // 2,
        follow_end_frame=follow_end,
    )


def test_windows_from_swings_basic() -> None:
    swings = [_swing(0, 10, 20, 30)]
    windows = _windows_from_swings(swings, n_frames=100, margin_s=0.1, fps=30.0)
    assert len(windows) == 1
    lo, hi = windows[0]
    assert lo <= 10 and hi >= 30


def test_windows_from_swings_merges_overlapping() -> None:
    swings = [_swing(0, 10, 20, 30), _swing(1, 32, 40, 50)]
    windows = _windows_from_swings(swings, n_frames=100, margin_s=0.1, fps=30.0)
    assert len(windows) == 1  # 30+margin overlaps with 32-margin


def test_windows_from_swings_keeps_separate_when_far_apart() -> None:
    swings = [_swing(0, 10, 20, 30), _swing(1, 200, 210, 220)]
    windows = _windows_from_swings(swings, n_frames=300, margin_s=0.1, fps=30.0)
    assert len(windows) == 2


def test_windows_from_swings_clamped_to_frame_range() -> None:
    swings = [_swing(0, 0, 5, 10)]
    windows = _windows_from_swings(swings, n_frames=15, margin_s=1.0, fps=30.0)
    lo, hi = windows[0]
    assert lo >= 0
    assert hi <= 14


def test_bbox_for_frame_basic() -> None:
    player = np.full((26, 2), np.nan)
    player[0] = [100, 100]
    player[1] = [200, 100]
    player[2] = [150, 300]
    bbox = _bbox_for_frame(player, img_w=1000, img_h=1000, margin=0.0)
    assert bbox == pytest.approx([100.0, 100.0, 200.0, 300.0])


def test_bbox_for_frame_with_margin_expands_and_clips() -> None:
    player = np.full((26, 2), np.nan)
    player[0] = [10, 10]
    player[1] = [20, 20]
    player[2] = [15, 25]
    bbox = _bbox_for_frame(player, img_w=100, img_h=100, margin=1.0)
    x1, y1, x2, y2 = bbox
    assert x1 == 0.0  # clipped
    assert y1 == 0.0
    assert x2 > 20 and y2 > 25


def test_bbox_for_frame_too_few_valid_points_is_none() -> None:
    player = np.full((26, 2), np.nan)
    player[0] = [10, 10]
    assert _bbox_for_frame(player, 100, 100) is None


def test_interpolate_and_smooth_short_array_passthrough() -> None:
    frame_indices = np.array([0, 1])
    kpts = np.zeros((2, 70, 3))
    out = _interpolate_and_smooth(frame_indices, kpts, Config())
    np.testing.assert_array_equal(out, kpts)


def test_interpolate_and_smooth_fills_and_smooths() -> None:
    n = 20
    frame_indices = np.arange(n)
    kpts = np.zeros((n, 70, 3), dtype=np.float64)
    t = np.linspace(0, 1, n)
    kpts[:, 5, 0] = np.sin(2 * np.pi * t)
    kpts[3, 5, 0] = np.nan  # a single-frame gap
    cfg = Config()
    out = _interpolate_and_smooth(frame_indices, kpts, cfg)
    assert not np.isnan(out[3, 5, 0])  # gap filled
    assert out.shape == kpts.shape


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def test_consistency_check_masks_disagreeing_frames(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    (run.root / "measure").mkdir(parents=True)
    n = 3
    body_kpts = np.zeros((n, 1, 26, 3), dtype=np.float32)
    for t in range(n):
        body_kpts[t, 0, :15] = 50.0  # rtmlib 2D at inference resolution
        body_kpts[t, 0, :, 2] = 0.9
    np.savez_compressed(run.root / "measure" / "body.npz", kpts=body_kpts, n_persons=np.ones(n), scale_long_side=640)

    frame_indices = np.array([0, 1, 2])
    keypoints_2d = np.zeros((3, 70, 2), dtype=np.float32)
    keypoints_2d[0, :15] = 50.0  # agrees (scale=1.0 since scale_long_side==orig max dim)
    keypoints_2d[1, :15] = 50.0
    keypoints_2d[2, :15] = 5000.0  # wildly disagrees -> should be masked out

    selection_json = [{"frame": t, "chosen": 0} for t in range(n)]

    mask = _consistency_check(
        frame_indices,
        keypoints_2d,
        run,
        body_scale_long_side=640,
        orig_w=640,
        orig_h=360,
        selection_json=selection_json,
        torso_len_px=80.0,
        thr_torso=0.5,
    )
    assert mask.tolist() == [True, True, False]


def test_consistency_check_no_selection_masks_all(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    (run.root / "measure").mkdir(parents=True)
    n = 2
    body_kpts = np.zeros((n, 1, 26, 3), dtype=np.float32)
    np.savez_compressed(run.root / "measure" / "body.npz", kpts=body_kpts, n_persons=np.ones(n), scale_long_side=640)
    frame_indices = np.array([0, 1])
    keypoints_2d = np.zeros((2, 70, 2), dtype=np.float32)
    mask = _consistency_check(frame_indices, keypoints_2d, run, 640, 640, 360, [], 80.0, 0.5)
    assert mask.tolist() == [False, False]


def test_estimate_3d_no_swings_writes_none_backend(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    (run.root / "track").mkdir(parents=True)
    (run.root / "swings").mkdir(parents=True)
    n = 10
    player = np.zeros((n, 26, 2))
    np.savez_compressed(run.root / "track" / "player.npz", smooth=player)
    (run.root / "swings" / "swings.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "handedness": "right",
                "net_side": "right",
                "net_side_source": "unknown",
                "torso_len_median_px": 80.0,
                "speed_source": "racket_top",
                "swings": [],
            }
        )
    )
    run.update_run_json(ingest={"normalized_path": "dummy.mp4", "width": 640, "height": 360})

    result = estimate_3d(run, cfg=cfg, device="cpu")
    assert result["backend"] == "none"
    assert result["n_frames"] == 0
    data = np.load(run.root / "body3d" / "joints.npz", allow_pickle=True)
    assert str(data["backend"]) == "none"
