"""Tests for compute_swing_metrics() and the compute_metrics_stage() orchestration (separate from
tests/test_metrics.py's pure per-frame geometry unit tests)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from badminton_coach.config import Config
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
    RACKET_NUM_KPTS,
    RACKET_TOP,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from badminton_coach.metrics import compute_metrics_stage, compute_swing_metrics
from badminton_coach.run import RunDir
from badminton_coach.schema import Swing


def _player_frame() -> np.ndarray:
    p = np.full((BODY_NUM_KPTS, 2), np.nan)
    p[NOSE] = [0, -100]
    p[LEFT_SHOULDER] = [-20, -80]
    p[RIGHT_SHOULDER] = [20, -80]
    p[LEFT_HIP] = [-15, 0]
    p[RIGHT_HIP] = [15, 0]
    p[LEFT_ELBOW] = [-20, -40]
    p[LEFT_WRIST] = [-20, -130]
    p[RIGHT_ELBOW] = [60, -80]
    p[RIGHT_WRIST] = [100, -80]
    p[LEFT_ANKLE] = [-15, 80]
    p[RIGHT_ANKLE] = [15, 80]
    return p


def _racket_frame(top_y: float = -180) -> np.ndarray:
    r = np.full((RACKET_NUM_KPTS, 2), np.nan)
    r[RACKET_TOP] = [0, top_y]
    r[RACKET_BOTTOM] = [0, -100]
    r[RACKET_HANDLE] = [0, -100]
    return r


def test_compute_swing_metrics_basic_shapes_and_values() -> None:
    n = 20
    player = np.stack([_player_frame() for _ in range(n)])
    racket = np.stack([_racket_frame() for _ in range(n)])
    speed = np.full(n, 5.0)
    swing = Swing(
        index=0,
        mode="live",
        contact_frame=10,
        contact_time_s=10 / 30,
        contact_source="shuttle",
        prep_start_frame=2,
        prep_end_frame=8,
        follow_end_frame=15,
    )
    out = compute_swing_metrics(
        swing, player, racket, speed, torso_len=80.0, handedness="right", net_side="right", fps=30.0
    )

    assert out["elbow_angle_contact"] == pytest.approx(180.0, abs=1e-3)
    assert out["contact_height_vs_nose"] > 0  # racket well above nose
    assert out["peak_racket_speed"] == pytest.approx(5.0)
    assert out["time_prep_to_contact_s"] == pytest.approx((10 - 8) / 30)
    assert out["follow_through_duration_s"] == pytest.approx((15 - 10) / 30)
    assert out["weight_transfer"] in (True, False)
    assert isinstance(out["front_foot_contact"], str)


def test_compute_swing_metrics_missing_instant_is_none() -> None:
    n = 5
    player = np.stack([_player_frame() for _ in range(n)])
    racket = np.stack([_racket_frame() for _ in range(n)])
    speed = np.full(n, 1.0)
    swing = Swing(
        index=0,
        mode="shadow",
        contact_frame=2,
        contact_time_s=2 / 30,
        contact_source="speed",
        prep_start_frame=0,
        prep_end_frame=None,  # no prep instant found
        follow_end_frame=4,
    )
    out = compute_swing_metrics(
        swing, player, racket, speed, torso_len=80.0, handedness="right", net_side="right", fps=30.0
    )
    assert out["elbow_angle_prep_end"] is None
    assert out["time_prep_to_contact_s"] is None
    assert out["elbow_angle_contact"] is not None


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def _write_track_and_swings(run: RunDir, n: int = 20) -> None:
    player = np.stack([_player_frame() for _ in range(n)])
    racket = np.stack([_racket_frame() for _ in range(n)])
    (run.root / "track").mkdir(parents=True, exist_ok=True)
    (run.root / "swings").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run.root / "track" / "player.npz", raw=player, smooth=player, conf=np.ones((n, BODY_NUM_KPTS)))
    np.savez_compressed(
        run.root / "track" / "racket.npz", raw=racket, smooth=racket, conf=np.ones((n, RACKET_NUM_KPTS))
    )
    (run.root / "track" / "track.json").write_text(
        json.dumps({"handedness": "right", "net_side": "right", "torso_len_median_px": 80.0})
    )
    swing = {
        "index": 0,
        "mode": "live",
        "contact_frame": 10,
        "contact_time_s": 10 / 30,
        "contact_source": "shuttle",
        "prep_start_frame": 2,
        "prep_end_frame": 8,
        "follow_end_frame": 15,
    }
    (run.root / "swings" / "swings.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "handedness": "right",
                "net_side": "right",
                "net_side_source": "unknown",
                "torso_len_median_px": 80.0,
                "speed_source": "racket_top",
                "swings": [swing],
            }
        )
    )
    np.save(run.root / "swings" / "speed_series.npy", np.full(n, 5.0))


def test_compute_metrics_stage_writes_file_without_references(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    _write_track_and_swings(run)

    result = compute_metrics_stage(run, cfg=cfg, references=[])
    assert result["n_swings"] == 1
    metrics_path = run.root / "metrics" / "metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert data["swings"][0]["metrics"]["elbow_angle_contact"]["value"] == pytest.approx(180.0, abs=1e-3)
    assert data["swings"][0]["metrics"]["elbow_angle_contact"]["ref_mean"] is None


def test_compute_metrics_stage_is_cached(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    _write_track_and_swings(run)
    r1 = compute_metrics_stage(run, cfg=cfg, references=[])
    status1 = run.stage_status("metrics")
    r2 = compute_metrics_stage(run, cfg=cfg, references=[])
    assert r1 == r2
    assert run.stage_status("metrics") == status1


def test_compute_metrics_stage_with_reference(cfg: Config) -> None:
    ref_run = RunDir(cfg.references_dir / "test-ref")
    _write_track_and_swings(ref_run)
    compute_metrics_stage(ref_run, cfg=cfg, references=[])

    run = RunDir.create("nam", cfg=cfg)
    _write_track_and_swings(run)
    result = compute_metrics_stage(run, cfg=cfg, references=[ref_run])
    assert result["n_swings"] == 1
    data = json.loads((run.root / "metrics" / "metrics.json").read_text())
    mv = data["swings"][0]["metrics"]["elbow_angle_contact"]
    assert mv["ref_mean"] == pytest.approx(180.0, abs=1e-3)
    assert mv["n_ref"] == 1
