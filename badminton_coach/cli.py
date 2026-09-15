"""`badminton-coach` CLI (spec §4). Built incrementally: `frames` first (Task 8, the judge's own tool
for pulling more frames mid-investigation); `analyze`/`reference`/`judge`/`render`/`view`/`models` land
in Task 10 once render.py exists.
"""

from __future__ import annotations

from pathlib import Path

import click

from badminton_coach.config import CONFIG
from badminton_coach.render.draw import draw_body, draw_racket, draw_text
from badminton_coach.run import RunDir
from badminton_coach.video import read_frame


@click.group()
def main() -> None:
    """badminton-coach: analyze a badminton stroke from phone video."""


@main.command()
@click.argument("workspace", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--swing", type=int, required=True, help="swing index (matches swings/<i>/)")
@click.option("--from", "from_s", type=float, required=True, help="start time, seconds")
@click.option("--to", "to_s", type=float, required=True, help="end time, seconds")
@click.option("--step", type=int, default=1, help="frame step")
@click.option("--crop", type=click.Choice(["none", "upper"]), default="none")
def frames(workspace: Path, swing: int, from_s: float, to_s: float, step: int, crop: str) -> None:
    """Judge tool: write annotated frames from a run into <workspace>/swings/<swing>/frames/ and print
    their paths. `workspace` MUST be a judge workspace directory (has ../../ pointing at the real run —
    resolved by walking up to the run root that owns it); refuses to write anywhere else."""
    run_root = workspace.resolve()
    while run_root != run_root.parent and not (run_root / "swings" / "swings.json").exists():
        run_root = run_root.parent
    if not (run_root / "swings" / "swings.json").exists():
        raise click.ClickException("could not find a run (swings/swings.json) above this workspace")
    run = RunDir(run_root)

    import numpy as np

    swings_data = __import__("json").loads((run.root / "swings" / "swings.json").read_text())
    fps = swings_data["fps"]
    ingest = run.load_run_json().get("ingest", {})
    video = ingest["normalized_path"]
    player_smooth = np.load(run.root / "track" / "player.npz")["smooth"]
    racket_smooth = np.load(run.root / "track" / "racket.npz")["smooth"]

    out_dir = workspace / "swings" / str(swing) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_f, end_f = int(round(from_s * fps)), int(round(to_s * fps))
    written = []
    for t in range(start_f, end_f + 1, max(1, step)):
        if t < 0 or t >= player_smooth.shape[0]:
            continue
        frame = read_frame(video, t)
        frame = draw_body(frame, player_smooth[t])
        frame = draw_racket(frame, racket_smooth[t])
        frame = draw_text(frame, f"f{t} t={t / fps:.2f}s", (8, 8), size=16, bg=(0, 0, 0))
        if crop == "upper":
            frame = frame[: frame.shape[0] // 2]
        import cv2

        p = out_dir / f"f_{t:06d}.png"
        cv2.imwrite(str(p), frame)
        written.append(p)

    for p in written:
        click.echo(str(p))


@main.group()
def reference() -> None:
    """Manage reference (pro) clips."""


@reference.command("add")
@click.argument("url_or_file")
@click.option("--name", required=True)
@click.option("--start", "start_s", type=float, default=None)
@click.option("--end", "end_s", type=float, default=None)
@click.option("--device", default="cuda")
def reference_add(url_or_file: str, name: str, start_s: float | None, end_s: float | None, device: str) -> None:
    from badminton_coach.reference import add_reference

    run = add_reference(url_or_file, name, cfg=CONFIG, start_s=start_s, end_s=end_s, device=device)
    click.echo(f"reference '{name}' built at {run.root}")


@reference.command("list")
def reference_list() -> None:
    from badminton_coach.reference import load_references

    for ref in load_references(CONFIG):
        click.echo(ref.root.name)
