"""Shared pytest configuration: auto-skip `@pytest.mark.models` tests when no model weights have been
downloaded yet (`data/models/` empty or absent) — see pyproject.toml's marker description."""

from __future__ import annotations

import pytest

from badminton_coach.config import CONFIG


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    # `config` is required by pytest's hook signature even though this hook doesn't use it.
    models_available = CONFIG.models_dir.exists() and any(CONFIG.models_dir.iterdir())
    if models_available:
        return
    skip_marker = pytest.mark.skip(reason="data/models is empty; run `badminton-coach models download` first")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip_marker)
