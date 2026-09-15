from __future__ import annotations

import json
from pathlib import Path

import pytest

from badminton_coach.config import Config
from badminton_coach.run import RunDir
from badminton_coach.viewer import build_index, write_index


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def test_build_index_empty(cfg: Config) -> None:
    html = build_index(cfg)
    assert "No runs yet" in html


def test_build_index_lists_players_and_runs(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    html = build_index(cfg)
    assert "nam" in html
    assert run.root.name in html


def test_build_index_shows_top_finding(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    (run.root / "judge").mkdir(parents=True)
    (run.root / "judge" / "findings.json").write_text(
        json.dumps({"findings": [{"id": "f1", "title": "Elbow bent"}], "priority": ["f1"]})
    )
    (run.root / "index.html").write_text("<html></html>")
    html = build_index(cfg)
    assert "Elbow bent" in html


def test_write_index_creates_file(cfg: Config) -> None:
    RunDir.create("nam", cfg=cfg)
    write_index(cfg)
    assert (cfg.data_dir / "index.html").exists()
