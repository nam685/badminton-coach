"""`badminton-coach` CLI (spec §4)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import click

from badminton_coach.config import CONFIG, Config
from badminton_coach.render.draw import draw_body, draw_racket, draw_text
from badminton_coach.run import RunDir
from badminton_coach.video import read_frame

# Order matches the pipeline; used to interpret --from-stage on `analyze` (force recompute from this
# stage onward) and to validate the flag's choices.
STAGE_ORDER = [
    "ingest",
    "measure_body",
    "measure_racket",
    "measure_shuttle",
    "track",
    "swings",
    "body3d",
    "metrics",
    "judge",
    "render",
]


def _resolve_references(cfg: Config, refs: str) -> list:
    from badminton_coach.reference import load_references

    available = load_references(cfg)
    if refs == "all":
        return available
    names = {n.strip() for n in refs.split(",") if n.strip()}
    return [r for r in available if r.root.name in names]


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
@click.option("--force", is_flag=True, help="re-fetch/re-trim/re-run even if cached output looks up to date")
def reference_add(
    url_or_file: str, name: str, start_s: float | None, end_s: float | None, device: str, force: bool
) -> None:
    from badminton_coach.reference import add_reference

    run = add_reference(url_or_file, name, cfg=CONFIG, start_s=start_s, end_s=end_s, device=device, force=force)
    click.echo(f"reference '{name}' built at {run.root}")


@reference.command("list")
def reference_list() -> None:
    from badminton_coach.reference import load_references

    for ref in load_references(CONFIG):
        click.echo(ref.root.name)


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--player", required=True, help="player name (run dir is created under data/players/<player>/)")
@click.option("--lang", type=click.Choice(CONFIG.supported_langs), default=CONFIG.default_lang)
@click.option(
    "--shot", type=click.Choice(["clear"]), default="clear", help="v1 supports the forehand overhead clear only"
)
@click.option("--net-side", "net_side", type=click.Choice(["auto", "left", "right"]), default="auto")
@click.option("--refs", default="all", help="'all' (default), 'none', or a comma-separated list of reference names")
@click.option("--model", "judge_model", default=None, help="overrides config judge_model (default: opus)")
@click.option("--effort", "judge_effort", default=None, help="overrides config judge_effort (default: high)")
@click.option(
    "--from-stage",
    "from_stage",
    type=click.Choice(STAGE_ORDER),
    default=None,
    help="force recompute from this stage onward",
)
@click.option("--no-judge", is_flag=True, help="stop after metrics; skip judge and rendering")
@click.option("--skip-body3d", is_flag=True, help="skip 3D hip/shoulder rotation (faster)")
@click.option("--skip-shuttle", is_flag=True, help="skip shuttle tracking (e.g. shadow-swing-only footage)")
@click.option("--device", default="cuda")
def analyze(
    inputs: tuple[Path, ...],
    player: str,
    lang: str,
    shot: str,
    net_side: str,
    refs: str,
    judge_model: str | None,
    judge_effort: str | None,
    from_stage: str | None,
    no_judge: bool,
    skip_body3d: bool,
    skip_shuttle: bool,
    device: str,
) -> None:
    """Run the full pipeline on one or more video files. Multiple inputs become one run (attempts
    concatenated, per-attempt frame offsets kept in run.json's "ingest" key)."""
    from badminton_coach.judge.run import run_judge
    from badminton_coach.judge.workspace import build_workspace
    from badminton_coach.measure.body3d import estimate_3d
    from badminton_coach.metrics import compute_metrics_stage
    from badminton_coach.pipeline import run_measurement_pipeline
    from badminton_coach.render.report import write_report
    from badminton_coach.render.video import render_annotated_video

    cfg = CONFIG
    if judge_model:
        cfg = replace(cfg, judge_model=judge_model)
    if judge_effort:
        cfg = replace(cfg, judge_effort=judge_effort)

    def force_from(stage: str) -> bool:
        return from_stage is not None and STAGE_ORDER.index(stage) >= STAGE_ORDER.index(from_stage)

    run = RunDir.create(player, cfg=cfg)
    run.update_run_json(lang=lang, shot=shot)
    click.echo(f"run: {run.root}")

    net_side_override = None if net_side == "auto" else net_side
    pipeline_force = from_stage is not None and STAGE_ORDER.index(from_stage) <= STAGE_ORDER.index("swings")

    summary = run_measurement_pipeline(
        list(inputs),
        run,
        cfg=cfg,
        device=device,
        net_side_override=net_side_override,
        force=pipeline_force,
        skip_shuttle=skip_shuttle,
    )

    if not skip_body3d:
        summary["body3d"] = estimate_3d(run, cfg=cfg, device=device, force=pipeline_force or force_from("body3d"))

    references = [] if refs == "none" else _resolve_references(cfg, refs)
    summary["metrics"] = compute_metrics_stage(
        run, cfg=cfg, references=references, force=pipeline_force or force_from("metrics")
    )

    findings = None
    if not no_judge:
        workspace = build_workspace(run, cfg=cfg, references=references, lang=lang, player=player)
        summary["judge"] = run_judge(run, workspace, lang=lang, cfg=cfg, force=pipeline_force or force_from("judge"))
        findings_path = run.root / "judge" / "findings.json"
        if findings_path.exists():
            findings = json.loads(findings_path.read_text())

        video_out = run.root / "annotated.mp4"
        summary["render_video"] = render_annotated_video(run, video_out, findings=findings, cfg=cfg, lang=lang)
        summary["report"] = write_report(run, cfg=cfg, lang=lang)
        click.echo(f"report: {run.root / 'index.html'}")

    click.echo(json.dumps(summary, indent=2, default=str))


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--lang", default=None, help="default: whatever was used for this run's `analyze` call")
@click.option("--refs", default="all", help="'all' (default), 'none', or a comma-separated list of reference names")
@click.option("--model", "judge_model", default=None)
@click.option("--effort", "judge_effort", default=None)
def judge(run_dir: Path, lang: str | None, refs: str, judge_model: str | None, judge_effort: str | None) -> None:
    """Re-run the judge only (prompt iteration) against an existing run. Always re-runs the judge model
    (the point of calling this directly), rebuilding the workspace fresh from current metrics/rubric."""
    from badminton_coach.judge.run import run_judge
    from badminton_coach.judge.workspace import build_workspace

    run = RunDir.open(run_dir)
    run_json = run.load_run_json()
    lang = lang or run_json.get("lang", CONFIG.default_lang)
    player = run_json.get("player", "")

    cfg = CONFIG
    if judge_model:
        cfg = replace(cfg, judge_model=judge_model)
    if judge_effort:
        cfg = replace(cfg, judge_effort=judge_effort)

    references = [] if refs == "none" else _resolve_references(cfg, refs)
    workspace = build_workspace(run, cfg=cfg, references=references, lang=lang, player=player)
    result = run_judge(run, workspace, lang=lang, cfg=cfg, force=True)
    click.echo(json.dumps(result, indent=2, default=str))


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--lang", default=None, help="default: whatever was used for this run's `analyze` call")
def render(run_dir: Path, lang: str | None) -> None:
    """Re-render the annotated video + report only, from the run's existing metrics/judge output."""
    from badminton_coach.render.report import write_report
    from badminton_coach.render.video import render_annotated_video

    run = RunDir.open(run_dir)
    run_json = run.load_run_json()
    lang = lang or run_json.get("lang", CONFIG.default_lang)

    findings = None
    findings_path = run.root / "judge" / "findings.json"
    if findings_path.exists():
        findings = json.loads(findings_path.read_text())

    video_out = run.root / "annotated.mp4"
    render_annotated_video(run, video_out, findings=findings, cfg=CONFIG, lang=lang)
    write_report(run, cfg=CONFIG, lang=lang)
    click.echo(f"report: {run.root / 'index.html'}")


@main.command()
@click.option("--port", default=8765)
def view(port: int) -> None:
    """Serve data/ locally (runs index + each run's index.html)."""
    from badminton_coach.viewer import serve

    serve(cfg=CONFIG, port=port)


@main.group("models")
def models_group() -> None:
    """Download/warm up model weights."""


@models_group.command("download")
def models_download() -> None:
    """Fetch every model weight the pipeline needs into data/models/. Safe to re-run (skips anything
    already present)."""
    import subprocess

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

    models_dir = CONFIG.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    click.echo("[1/4] rtmlib body-pose model (auto-downloads on first use)...")
    import os

    os.environ.setdefault("XDG_CACHE_HOME", str(models_dir))
    from rtmlib import BodyWithFeet

    BodyWithFeet(mode="performance", backend="onnxruntime", device="cpu")
    click.echo("      ok")

    racket_src = models_dir / "racketvision" / "src"
    if not (racket_src / "source" / "RacketPose").exists():
        click.echo("[2/4] cloning RacketVision (config files for the racket detector/pose models)...")
        racket_src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/OrcustD/RacketVision.git", str(racket_src)],
            check=True,
        )
    else:
        click.echo("[2/4] RacketVision source already present, skipping clone.")

    click.echo("[3/4] racket detector/pose + shuttle-tracker checkpoints from Hugging Face...")
    ckpt_dir = models_dir / "racketvision" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("epoch_300.pth", "best_PCK_epoch_90.pth", "balltrack_best.pth"):
        dest = ckpt_dir / filename
        if dest.exists():
            continue
        hf_hub_download(
            repo_id="linfeng302/RacketVision-Models",
            filename=filename,
            local_dir=ckpt_dir,
            cache_dir=str(models_dir / "racketvision" / ".cache" / "huggingface"),
        )
    click.echo("      ok")

    sam3db_src = models_dir / "sam3db_src"
    if not (sam3db_src / "sam_3d_body").exists():
        click.echo("[4/4] cloning SAM 3D Body (Python package source) + downloading the ViT-H checkpoint...")
        sam3db_src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/facebookresearch/sam-3d-body.git", str(sam3db_src)],
            check=True,
        )
    else:
        click.echo("[4/4] SAM 3D Body source already present, skipping clone.")

    sam3db_dir = models_dir / "sam3db_vith"
    sam3db_dir.mkdir(parents=True, exist_ok=True)
    ckpt_files = [
        ("model.ckpt", sam3db_dir / "model.ckpt"),
        ("model_config.yaml", sam3db_dir / "model_config.yaml"),
        ("assets/mhr_model.pt", sam3db_dir / "assets" / "mhr_model.pt"),
    ]
    try:
        for filename, dest in ckpt_files:
            if dest.exists():
                continue
            hf_hub_download(
                repo_id="facebook/sam-3d-body-vith",
                filename=filename,
                local_dir=sam3db_dir,
                cache_dir=str(sam3db_dir / ".cache" / "huggingface"),
            )
        click.echo("      ok")
    except (GatedRepoError, LocalEntryNotFoundError) as exc:
        raise click.ClickException(
            "facebook/sam-3d-body-vith is a gated Hugging Face repo. Accept its terms at "
            "https://huggingface.co/facebook/sam-3d-body-vith, then run `hf auth login` "
            f"(or set HF_TOKEN) and re-run `badminton-coach models download`. ({exc})"
        ) from exc

    click.echo("All model weights present.")
