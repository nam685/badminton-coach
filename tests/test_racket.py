from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import badminton_coach.measure.racket as racket_mod
from badminton_coach.config import Config
from badminton_coach.measure.racket import measure_racket
from badminton_coach.run import RunDir
from tests.fixtures.make_fixture import ensure_fixture

FIXTURE = ensure_fixture()


def _fake_subprocess(raw: dict, summary: dict):
    def _run(video, out_json, **kwargs):  # noqa: ARG001
        Path(out_json).write_text(json.dumps(raw))
        return summary

    return _run


def test_measure_racket_writes_padded_npz(tmp_path: Path, monkeypatch) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    run = RunDir.create("nam", cfg=cfg)

    raw = {
        "0": [
            {
                "bbox": [1, 2, 3, 4],
                "bbox_score": 0.9,
                "keypoints": [[i, i] for i in range(5)],
                "keypoint_scores": [0.5] * 5,
            }
        ],
        "2": [
            {
                "bbox": [5, 6, 7, 8],
                "bbox_score": 0.8,
                "keypoints": [[i, i] for i in range(5)],
                "keypoint_scores": [0.4] * 5,
            },
            {
                "bbox": [9, 10, 11, 12],
                "bbox_score": 0.35,
                "keypoints": [[i, i] for i in range(5)],
                "keypoint_scores": [0.3] * 5,
            },
        ],
    }
    summary = {"n_frames_processed": 3, "n_frames_with_racket": 2, "elapsed_s": 1.0, "fps": 3.0}
    monkeypatch.setattr(racket_mod, "run_racket_subprocess", _fake_subprocess(raw, summary))
    # avoid the "script not found" guard for this unit test
    monkeypatch.setattr(racket_mod, "_INFER_SCRIPT", tmp_path / "infer_frames.py")
    (tmp_path / "infer_frames.py").write_text("")

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a real video, only hashed")

    result = measure_racket(run, video, cfg=cfg)
    assert result["n_frames"] == 3
    assert result["max_rackets"] == 2

    data = np.load(run.root / "measure" / "racket.npz")
    assert data["kpts"].shape == (3, 2, 5, 3)
    assert data["bboxes"].shape == (3, 2, 4)
    assert data["bbox_scores"].shape == (3, 2)

    # frame 0: one racket present, second slot NaN
    assert not np.isnan(data["bbox_scores"][0, 0])
    assert np.isnan(data["bbox_scores"][0, 1])
    # frame 1: no detections at all
    assert np.all(np.isnan(data["bbox_scores"][1]))
    # frame 2: two rackets, ordered as given
    assert data["bbox_scores"][2, 0] == pytest.approx(0.8)
    assert data["bbox_scores"][2, 1] == pytest.approx(0.35)

    status = run.stage_status("measure_racket")
    assert status["status"] == "done"
    assert status["frac_frames_with_racket"] == pytest.approx(2 / 3, abs=1e-3)


def test_measure_racket_is_cached(tmp_path: Path, monkeypatch) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    run = RunDir.create("nam", cfg=cfg)
    raw = {"0": [{"bbox": [1, 2, 3, 4], "bbox_score": 0.9, "keypoints": [[0, 0]] * 5, "keypoint_scores": [0.5] * 5}]}
    summary = {"n_frames_processed": 1, "n_frames_with_racket": 1, "elapsed_s": 0.1, "fps": 10.0}
    calls = []

    def _run(video, out_json, **kwargs):  # noqa: ARG001
        calls.append(1)
        Path(out_json).write_text(json.dumps(raw))
        return summary

    monkeypatch.setattr(racket_mod, "run_racket_subprocess", _run)
    monkeypatch.setattr(racket_mod, "_INFER_SCRIPT", tmp_path / "infer_frames.py")
    (tmp_path / "infer_frames.py").write_text("")

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"hash-me")

    measure_racket(run, video, cfg=cfg)
    measure_racket(run, video, cfg=cfg)
    assert len(calls) == 1


@pytest.mark.models
@pytest.mark.skipif(FIXTURE is None, reason="fixture not generated; run tests/fixtures/make_fixture.py")
def test_measure_racket_real_subprocess_short_range(tmp_path: Path) -> None:
    """Slow (real CPU inference via the pinned env subprocess) — kept to a short frame range."""
    from badminton_coach.measure.racket import run_racket_subprocess

    out_json = tmp_path / "racket_raw.json"
    summary = run_racket_subprocess(FIXTURE, out_json, start=100, end=130, device="cpu", timeout=600)
    assert summary["n_frames_processed"] == 30
    assert out_json.exists()
    raw = json.loads(out_json.read_text())
    assert isinstance(raw, dict)
