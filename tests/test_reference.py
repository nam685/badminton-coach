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


@pytest.mark.models
@pytest.mark.skipif(FIXTURE is None, reason="fixture not generated; run tests/fixtures/make_fixture.py")
def test_add_reference_from_local_file_end_to_end(cfg: Config) -> None:
    from badminton_coach.reference import add_reference

    run = add_reference(str(FIXTURE), "local-test-ref", cfg=cfg, device="cuda", start_s=None, end_s=None)
    assert (run.root / "metrics" / "metrics.json").exists()
    refs = load_references(cfg)
    assert len(refs) == 1
    assert refs[0].root.name == "local-test-ref"
