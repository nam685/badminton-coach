from __future__ import annotations

import json
from pathlib import Path

import pytest

from badminton_coach.config import Config
from badminton_coach.reference import load_reference_metric_values, load_references
from badminton_coach.run import RunDir
from tests.fixtures.make_fixture import ensure_fixture

FIXTURE = ensure_fixture()


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def _make_fake_reference(cfg: Config, name: str, values: list[float]) -> RunDir:
    ref_dir = cfg.references_dir / name
    ref_dir.mkdir(parents=True, exist_ok=True)
    run = RunDir(ref_dir)
    (ref_dir / "metrics").mkdir(exist_ok=True)
    swings = [
        {
            "index": i,
            "mode": "live",
            "contact_frame": i * 10,
            "contact_time_s": i / 3,
            "metrics": {"elbow_angle_contact": {"value": v}},
        }
        for i, v in enumerate(values)
    ]
    (ref_dir / "metrics" / "metrics.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "handedness": "right",
                "net_side": "right",
                "torso_len_median_px": 80.0,
                "swings": swings,
                "cross_attempt_summary": {},
            }
        )
    )
    return run


def test_load_references_only_finished_ones(cfg: Config) -> None:
    _make_fake_reference(cfg, "ref-a", [160.0])
    (cfg.references_dir / "ref-b-unfinished").mkdir(parents=True)  # no metrics.json

    refs = load_references(cfg)
    assert len(refs) == 1
    assert refs[0].root.name == "ref-a"


def test_load_references_empty_when_no_dir(cfg: Config) -> None:
    assert load_references(cfg) == []


def test_load_reference_metric_values_gathers_across_references(cfg: Config) -> None:
    ref_a = _make_fake_reference(cfg, "ref-a", [160.0, 170.0])
    ref_b = _make_fake_reference(cfg, "ref-b", [150.0])

    values = load_reference_metric_values([ref_a, ref_b])
    assert values["elbow_angle_contact"] == [160.0, 170.0, 150.0]


def test_load_reference_metric_values_skips_missing_file(cfg: Config) -> None:
    ref_a = _make_fake_reference(cfg, "ref-a", [160.0])
    ghost = RunDir(cfg.references_dir / "ghost")
    ghost.root.mkdir(parents=True, exist_ok=True)

    values = load_reference_metric_values([ref_a, ghost])
    assert values["elbow_angle_contact"] == [160.0]


def _make_clip(path: Path, duration_s: float = 4.0) -> Path:
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=10:duration={duration_s}", str(path)],
        check=True,
        capture_output=True,
    )
    return path


def test_add_reference_retrims_when_window_changes_even_without_force(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    """Real bug found while picking a reference clip: `clip.mp4` was only regenerated when it didn't
    exist yet or `force=True` was passed -- calling `reference add` again with a *different* --start/--end
    (the normal way to fix a bad trim) silently kept reusing the stale first trim. `add_reference` should
    detect the window changed and retrim on its own."""
    from badminton_coach import metrics as metrics_mod
    from badminton_coach import pipeline as pipeline_mod
    from badminton_coach.reference import add_reference

    calls: list[bool] = []

    def fake_pipeline(*args, **kwargs) -> dict:  # noqa: ARG001
        calls.append(kwargs["force"])
        return {}

    monkeypatch.setattr(pipeline_mod, "run_measurement_pipeline", fake_pipeline)
    monkeypatch.setattr(metrics_mod, "compute_metrics_stage", lambda *args, **kwargs: {})  # noqa: ARG005

    source = _make_clip(tmp_path / "src.mp4")

    run1 = add_reference(str(source), "retrim-test", cfg=cfg, start_s=0.0, end_s=1.0)
    clip_path = run1.root / "clip.mp4"
    assert clip_path.exists()
    mtime1 = clip_path.stat().st_mtime_ns
    assert calls == [True]  # first-ever build: nothing to compare against, so it builds fresh

    run2 = add_reference(str(source), "retrim-test", cfg=cfg, start_s=1.0, end_s=2.0)  # different window
    mtime2 = clip_path.stat().st_mtime_ns
    assert mtime2 != mtime1  # retrimmed, not silently reused
    assert calls == [True, True]  # forced the pipeline rerun despite no explicit --force

    run3 = add_reference(str(source), "retrim-test", cfg=cfg, start_s=1.0, end_s=2.0)  # same window again
    mtime3 = clip_path.stat().st_mtime_ns
    assert mtime3 == mtime2  # unchanged window: cached, not retrimmed
    assert calls == [True, True, False]
    assert run2.root == run3.root == run1.root


@pytest.mark.models
@pytest.mark.skipif(FIXTURE is None, reason="fixture not generated; run tests/fixtures/make_fixture.py")
def test_add_reference_from_local_file_end_to_end(cfg: Config) -> None:
    from badminton_coach.reference import add_reference

    run = add_reference(str(FIXTURE), "local-test-ref", cfg=cfg, device="cuda", start_s=None, end_s=None)
    assert (run.root / "metrics" / "metrics.json").exists()
    refs = load_references(cfg)
    assert len(refs) == 1
    assert refs[0].root.name == "local-test-ref"
