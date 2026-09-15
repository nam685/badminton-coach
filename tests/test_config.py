from __future__ import annotations

import os
from pathlib import Path

from badminton_coach.config import Config


def test_default_data_dir_is_relative_data() -> None:
    cfg = Config()
    assert cfg.data_dir.name == "data"


def test_derived_dirs_are_under_data_dir() -> None:
    cfg = Config()
    assert cfg.models_dir == cfg.data_dir / "models"
    assert cfg.players_dir == cfg.data_dir / "players"
    assert cfg.references_dir == cfg.data_dir / "references"
    assert cfg.fixtures_dir == cfg.data_dir / "fixtures"


def test_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BADMINTON_COACH_DATA", str(tmp_path / "custom"))
    cfg = Config()
    assert cfg.data_dir == (tmp_path / "custom").resolve()


def test_claude_bin_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    assert Config().claude_bin == "claude"
    monkeypatch.setenv("CLAUDE_BIN", "klaude")
    assert Config().claude_bin == "klaude"


def test_supported_langs_includes_vi_de_en() -> None:
    cfg = Config()
    assert set(cfg.supported_langs) == {"vi", "de", "en"}
