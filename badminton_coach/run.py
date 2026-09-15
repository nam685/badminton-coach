"""Run directory: one analysis session's on-disk home, `run.json` bookkeeping, and per-stage caching.

Every pipeline stage writes its artifacts under a `RunDir` and records its status/timing/version in
`run.json` so that (a) `--from-stage` can resume without recomputing earlier stages, and (b) every
intermediate result is inspectable on disk (the explicit dev-convenience requirement in the spec).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from badminton_coach.config import CONFIG, Config

_SLUG_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"


def _slugify(text: str) -> str:
    return "".join(c if c in _SLUG_SAFE else "-" for c in text).strip("-") or "run"


@dataclass
class StageHandle:
    """Yielded by `RunDir.stage()`. `skip` is True when a valid cached result already exists."""

    skip: bool
    name: str
    extra: dict[str, Any] = field(default_factory=dict)


class RunDir:
    """Wraps a single run directory: `data/players/<player>/<timestamp>[-slug]/` (or an arbitrary
    directory for references/tests)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, player: str, cfg: Config = CONFIG, slug: str | None = None, when: datetime | None = None) -> RunDir:
        when = when or datetime.now(timezone.utc)
        ts = when.strftime("%Y%m%d-%H%M%S")
        dirname = f"{ts}-{_slugify(slug)}" if slug else ts
        root = cfg.players_dir / _slugify(player) / dirname
        run = cls(root)
        run._init_run_json(player=player, created_at=when.isoformat())
        return run

    @classmethod
    def open(cls, path: str | Path) -> RunDir:
        root = Path(path)
        if not (root / "run.json").exists():
            raise FileNotFoundError(f"{root} is not a run directory (no run.json)")
        return cls(root)

    # --- run.json ---

    @property
    def run_json_path(self) -> Path:
        return self.root / "run.json"

    def _init_run_json(self, **top_level: Any) -> None:
        if self.run_json_path.exists():
            return
        data = {"stages": {}, **top_level}
        self._save_run_json(data)

    def load_run_json(self) -> dict[str, Any]:
        if not self.run_json_path.exists():
            return {"stages": {}}
        return json.loads(self.run_json_path.read_text())

    def _save_run_json(self, data: dict[str, Any]) -> None:
        self.run_json_path.write_text(json.dumps(data, indent=2, default=str))

    def update_run_json(self, **top_level: Any) -> None:
        data = self.load_run_json()
        data.update(top_level)
        self._save_run_json(data)

    def stage_status(self, name: str) -> dict[str, Any] | None:
        return self.load_run_json().get("stages", {}).get(name)

    # --- stage caching ---

    @contextmanager
    def stage(self, name: str, version: str, input_hash: str, force: bool = False) -> Iterator[StageHandle]:
        """Context manager marking a pipeline stage's execution window in `run.json`.

        Usage:
            with run.stage("measure_body", version="1", input_hash=ingest_sha) as st:
                if st.skip:
                    return load_existing(...)
                ... do the work ...
        `skip` is True when a previous run recorded status "done" for this exact (version, input_hash)
        and `force` is False. Any exception inside the block is recorded as status "failed" and re-raised.
        """
        prior = self.stage_status(name)
        skip = (
            not force
            and prior is not None
            and prior.get("status") == "done"
            and prior.get("version") == version
            and prior.get("input_hash") == input_hash
        )
        handle = StageHandle(skip=skip, name=name)
        if skip:
            yield handle
            return

        started = time.monotonic()
        data = self.load_run_json()
        data.setdefault("stages", {})[name] = {
            "status": "running",
            "version": version,
            "input_hash": input_hash,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_run_json(data)
        try:
            yield handle
        except Exception as exc:
            data = self.load_run_json()
            data["stages"][name] = {
                "status": "failed",
                "version": version,
                "input_hash": input_hash,
                "error": str(exc)[:2000],
                "duration_s": round(time.monotonic() - started, 3),
            }
            self._save_run_json(data)
            raise
        else:
            data = self.load_run_json()
            entry = data["stages"].get(name, {})
            entry.update(
                {
                    "status": "done",
                    "version": version,
                    "input_hash": input_hash,
                    "duration_s": round(time.monotonic() - started, 3),
                    **handle.extra,
                }
            )
            data["stages"][name] = entry
            self._save_run_json(data)

    def mark_extra(self, handle: StageHandle, **kv: Any) -> None:
        """Attach extra fields (e.g. measured fps, backend used) to a stage's run.json entry.

        Call before the `with run.stage(...)` block exits.
        """
        handle.extra.update(kv)
