"""Reference clips: pro slow-motion clears run through the exact same measurement pipeline as a
player's own footage, so their metrics are directly comparable (spec §3.6/§6.1). Personal use only —
`data/references/` is gitignored and never redistributed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from badminton_coach.config import CONFIG, Config
from badminton_coach.run import RunDir

_SLUG_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"


def _slugify(text: str) -> str:
    return "".join(c if c in _SLUG_SAFE else "-" for c in text).strip("-") or "ref"


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _fmt_time(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def add_reference(
    url_or_file: str,
    name: str,
    cfg: Config = CONFIG,
    start_s: float | None = None,
    end_s: float | None = None,
    device: str = "cuda",
    force: bool = False,
) -> RunDir:
    """Fetch (yt-dlp) or copy a source clip, optionally trim to [start_s, end_s], and run the full
    measurement pipeline against it, writing under `data/references/<name>/`. Idempotent per stage
    (re-running with the same source/trim reuses cached stage output, same as a player run)."""
    from badminton_coach.pipeline import run_measurement_pipeline

    ref_dir = cfg.references_dir / _slugify(name)
    ref_dir.mkdir(parents=True, exist_ok=True)
    run = RunDir(ref_dir)

    source_path = ref_dir / "source_raw.mp4"
    if _is_url(url_or_file):
        if not source_path.exists() or force:
            subprocess.run(
                [
                    "yt-dlp",
                    "-f",
                    "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]/best[height<=720]",
                    "-o",
                    str(source_path),
                    url_or_file,
                ],
                check=True,
            )
    else:
        src = Path(url_or_file)
        if not src.exists():
            raise FileNotFoundError(f"reference source not found: {src}")
        source_path = src

    clip_path = ref_dir / "clip.mp4"
    if start_s is not None or end_s is not None:
        if not clip_path.exists() or force:
            cmd = [cfg.ffmpeg_bin, "-y"]
            if start_s is not None:
                cmd += ["-ss", _fmt_time(start_s)]
            cmd += ["-i", str(source_path)]
            if end_s is not None:
                duration = end_s - (start_s or 0.0)
                cmd += ["-t", str(duration)]
            cmd += ["-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-an", str(clip_path)]
            subprocess.run(cmd, check=True, capture_output=True)
        clip_input = clip_path
    else:
        clip_input = source_path

    summary = run_measurement_pipeline([clip_input], run, cfg=cfg, device=device, force=force)

    from badminton_coach.metrics import compute_metrics_stage

    metrics_summary = compute_metrics_stage(run, cfg=cfg, references=[], force=force)
    summary["metrics"] = metrics_summary

    run.update_run_json(reference_name=name, source=url_or_file, start_s=start_s, end_s=end_s)
    return run


def load_references(cfg: Config = CONFIG) -> list[RunDir]:
    """All reference clips that have a `metrics/metrics.json` (i.e. finished the full pipeline)."""
    if not cfg.references_dir.exists():
        return []
    refs = []
    for p in sorted(cfg.references_dir.iterdir()):
        if p.is_dir() and (p / "metrics" / "metrics.json").exists():
            refs.append(RunDir(p))
    return refs


def load_reference_metric_values(references: list[RunDir]) -> dict[str, list[float]]:
    """Gathers, per metric name, every numeric value across every swing of every given reference run.
    Used both to fill each swing's `ref_mean/ref_min/ref_max/n_ref` and for the cross-attempt
    consistency check — computed once per `compute_metrics_stage` call, not per swing."""
    values: dict[str, list[float]] = {}
    for ref in references:
        metrics_path = ref.root / "metrics" / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text())
        for swing in data.get("swings", []):
            for name, mv in swing.get("metrics", {}).items():
                v = mv.get("value")
                if v is not None:
                    values.setdefault(name, []).append(v)
    return values
