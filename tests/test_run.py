from __future__ import annotations

from pathlib import Path

import pytest

from badminton_coach.config import Config
from badminton_coach.run import RunDir


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def test_create_run_dir_layout(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    assert run.root.exists()
    assert run.root.parent.name == "nam"
    assert run.root.parent.parent == cfg.players_dir
    assert run.run_json_path.exists()
    data = run.load_run_json()
    assert data["player"] == "nam"
    assert "created_at" in data


def test_create_run_dir_with_slug(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg, slug="clear session")
    assert run.root.name.endswith("-clear-session")


def test_open_existing_run(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    reopened = RunDir.open(run.root)
    assert reopened.root == run.root


def test_open_missing_run_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RunDir.open(tmp_path / "nope")


def test_stage_runs_once_and_caches(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    calls = []

    def do_stage(force=False):
        with run.stage("mystage", version="1", input_hash="abc", force=force) as st:
            if st.skip:
                return "cached"
            calls.append(1)
            return "computed"

    assert do_stage() == "computed"
    assert do_stage() == "cached"
    assert len(calls) == 1
    assert run.stage_status("mystage")["status"] == "done"


def test_stage_reruns_on_input_hash_change(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    calls = []

    def do_stage(input_hash):
        with run.stage("mystage", version="1", input_hash=input_hash) as st:
            if st.skip:
                return "cached"
            calls.append(input_hash)
            return "computed"

    do_stage("hash1")
    do_stage("hash1")
    do_stage("hash2")
    assert calls == ["hash1", "hash2"]


def test_stage_reruns_on_force(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    calls = []

    def do_stage(force):
        with run.stage("mystage", version="1", input_hash="abc", force=force) as st:
            if st.skip:
                return
            calls.append(1)

    do_stage(False)
    do_stage(False)
    do_stage(True)
    assert len(calls) == 2


def test_stage_records_failure_and_reraises(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    with pytest.raises(ValueError):
        with run.stage("mystage", version="1", input_hash="abc"):
            raise ValueError("boom")
    status = run.stage_status("mystage")
    assert status["status"] == "failed"
    assert "boom" in status["error"]


def test_stage_extra_fields_recorded(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    with run.stage("mystage", version="1", input_hash="abc") as st:
        run.mark_extra(st, fps=23.4, backend="cuda")
    status = run.stage_status("mystage")
    assert status["fps"] == 23.4
    assert status["backend"] == "cuda"
