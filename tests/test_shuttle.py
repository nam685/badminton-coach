from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from badminton_coach.config import Config
from badminton_coach.measure.shuttle import ShuttleTracker, _remove_ddp_prefix, compute_median, track_shuttle
from badminton_coach.measure.tracknet import TRACKNET_HEIGHT, TRACKNET_WIDTH, TrackNetV3
from badminton_coach.run import RunDir
from tests.fixtures.make_fixture import ensure_fixture

FIXTURE = ensure_fixture()


def _make_clip(
    path: Path, width=64, height=48, duration_s=0.5, fps=10, color: tuple[int, int, int] | None = None
) -> Path:
    if color is None:
        src = f"testsrc=size={width}x{height}:rate={fps}:duration={duration_s}"
    else:
        b, g, r = color
        src = f"color=c=0x{r:02x}{g:02x}{b:02x}:size={width}x{height}:rate={fps}:duration={duration_s}"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", src, "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True
    )
    return path


def test_remove_ddp_prefix() -> None:
    sd = {"module.conv.weight": 1, "module.conv.bias": 2, "other": 3}
    out = _remove_ddp_prefix(sd)
    assert out == {"conv.weight": 1, "conv.bias": 2, "other": 3}


def test_compute_median_on_constant_color_clip(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "clip.mp4", color=(10, 20, 30))
    median = compute_median(clip, n_samples=5)
    assert median.shape == (48, 64, 3)
    # allow encoding tolerance
    assert np.abs(median.astype(int) - np.array([10, 20, 30])).max() < 15


def test_predict_location_finds_bright_blob() -> None:
    tracker = object.__new__(ShuttleTracker)  # skip __init__ (no checkpoint needed)
    tracker.thre = 0.5
    heatmap = np.zeros((TRACKNET_HEIGHT, TRACKNET_WIDTH), dtype=np.float32)
    heatmap[100:110, 200:210] = 0.9
    x, y, w, h, conf = tracker._predict_location(heatmap)
    assert 195 <= x <= 205
    assert 95 <= y <= 105
    assert conf > 0.5


def test_predict_location_empty_heatmap_returns_zeros() -> None:
    tracker = object.__new__(ShuttleTracker)
    tracker.thre = 0.5
    heatmap = np.zeros((TRACKNET_HEIGHT, TRACKNET_WIDTH), dtype=np.float32)
    x, y, w, h, conf = tracker._predict_location(heatmap)
    assert (x, y, w, h, conf) == (0, 0, 0, 0, 0.0)


def test_shuttle_tracker_handles_clip_shorter_than_seq_len(tmp_path: Path) -> None:
    """A 2-frame clip (shorter than SEQ_LEN=4) must not crash: the sliding-window index clamping
    (fids = [max(0, j) for j in range(k - SEQ_LEN, k)]) always yields SEQ_LEN valid indices."""
    clip = _make_clip(tmp_path / "clip.mp4", width=64, height=48, duration_s=0.2, fps=10)

    tracker = object.__new__(ShuttleTracker)
    tracker.thre = 0.5
    tracker.device = torch.device("cpu")
    tracker.model = TrackNetV3()
    tracker.model.eval()

    results = tracker(clip)
    assert len(results) >= 1
    for r in results:
        assert set(r.keys()) == {"Frame", "X", "Y", "Visibility", "Confidence"}


@pytest.mark.models
@pytest.mark.skipif(FIXTURE is None, reason="fixture not generated; run tests/fixtures/make_fixture.py")
def test_track_shuttle_on_fixture(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    # reuse the real checkpoint already downloaded into the main data dir rather than re-downloading
    from badminton_coach.config import CONFIG

    real_checkpoint = CONFIG.data_dir / "models" / "racketvision" / "checkpoints" / "balltrack_best.pth"
    if not real_checkpoint.exists():
        pytest.skip("racketvision checkpoints not downloaded")
    ckpt_dir = cfg.data_dir / "models" / "racketvision" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(real_checkpoint, ckpt_dir / "balltrack_best.pth")

    run = RunDir.create("nam", cfg=cfg)
    # device="cuda" (the production default) here, not "cpu": CPU inference of this model over a
    # ~200-frame clip takes several minutes in this environment (plausibly WSL2 CPU-virtualization
    # overhead) — fine for the rare real fallback case (latency isn't a concern there) but bad for a
    # test that should run in a normal dev loop. The CPU code path itself is covered separately and
    # cheaply by test_shuttle_tracker_handles_clip_shorter_than_seq_len below on a 2-frame clip.
    summary = track_shuttle(run, FIXTURE, cfg=cfg, device="cuda")
    assert summary["n_frames"] > 50
    assert 0.0 <= summary["frac_visible"] <= 1.0

    import pandas as pd

    df = pd.read_csv(run.root / "measure" / "shuttle.csv")
    assert list(df.columns) == ["Frame", "X", "Y", "Visibility", "Confidence"]
