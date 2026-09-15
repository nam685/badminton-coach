"""Builds the judge's per-run workspace directory (spec §6.1): everything the judge agent needs, and
nothing else (no `CLAUDE.md`, so nothing extra gets pulled into its system prompt).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from badminton_coach.config import CONFIG, Config
from badminton_coach.judge.contact_sheet import render_contact_sheet, render_crops
from badminton_coach.judge.prompt import build_task
from badminton_coach.run import RunDir
from badminton_coach.schema import Swing

_RUBRIC_SRC = Path(__file__).resolve().parent / "rubric.md"


def build_history_md(player: str, cfg: Config = CONFIG, exclude_run: RunDir | None = None) -> str:
    """Previous sessions' top findings + key metric means for this player, oldest first. Empty string
    if there are no prior finished runs (a fresh player's first session)."""
    player_dir = cfg.players_dir / player
    if not player_dir.exists():
        return ""

    entries = []
    for run_dir in sorted(player_dir.iterdir()):
        if exclude_run is not None and run_dir == exclude_run.root:
            continue
        findings_path = run_dir / "judge" / "findings.json"
        if not findings_path.exists():
            continue
        try:
            data = json.loads(findings_path.read_text())
        except json.JSONDecodeError:
            continue
        entries.append((run_dir.name, data))

    if not entries:
        return ""

    lines = [f"# {player}'s previous sessions\n"]
    for date, data in entries:
        lines.append(f"## {date}")
        top = data.get("priority", [])[:2]
        by_id = {f["id"]: f for f in data.get("findings", [])}
        for fid in top:
            f = by_id.get(fid)
            if f:
                lines.append(f"- **{f['title']}** ({f['category']}, {f['pattern']}): {f['explanation']}")
        if data.get("strengths"):
            lines.append(f"- Strengths noted: {'; '.join(data['strengths'])}")
        lines.append("")
    return "\n".join(lines)


def build_workspace(
    run: RunDir,
    cfg: Config = CONFIG,
    references: list[RunDir] | None = None,
    lang: str = "en",
    player: str = "",
) -> Path:
    references = references or []
    workspace = run.root / "judge" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    metrics_src = run.root / "metrics" / "metrics.json"
    shutil.copy(metrics_src, workspace / "metrics.json")

    swings_data = json.loads((run.root / "swings" / "swings.json").read_text())
    fps = swings_data["fps"]
    for s in swings_data["swings"]:
        swing = Swing(**s)
        render_contact_sheet(run, swing, fps, workspace / "swings" / str(swing.index) / "contact_sheet.png")
        render_crops(run, swing, fps, workspace / "swings" / str(swing.index) / "crops")

    plots_src = run.root / "plots"
    if plots_src.exists():
        shutil.copytree(plots_src, workspace / "plots", dirs_exist_ok=True)

    for ref in references:
        ref_dir = workspace / "references" / ref.root.name
        ref_metrics = ref.root / "metrics" / "metrics.json"
        if not ref_metrics.exists():
            continue
        ref_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ref_metrics, ref_dir / "metrics.json")
        ref_swings_path = ref.root / "swings" / "swings.json"
        if ref_swings_path.exists():
            ref_swings_data = json.loads(ref_swings_path.read_text())
            ref_fps = ref_swings_data["fps"]
            for s in ref_swings_data["swings"][:1]:  # one representative swing is enough context
                swing = Swing(**s)
                render_contact_sheet(ref, swing, ref_fps, ref_dir / "contact_sheet.png")

    shutil.copy(_RUBRIC_SRC, workspace / "rubric.md")

    history = build_history_md(player, cfg, exclude_run=run) if player else ""
    (workspace / "history.md").write_text(history)

    (workspace / "TASK.md").write_text(build_task(lang))

    return workspace
