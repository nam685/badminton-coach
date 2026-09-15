"""Shared measurement-pipeline runner (ingest -> measure -> track -> swings), used by both
`reference.py` (building reference clips) and the `analyze` CLI command (Task 10) so the two never
duplicate/drift on stage ordering.

Judge/render (the remaining pipeline stages) are deliberately not included here — they need a workspace
and player/lang context that only makes sense for an actual player run, never for a reference clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from badminton_coach.config import CONFIG, Config
from badminton_coach.ingest import ingest
from badminton_coach.measure.body import measure_body
from badminton_coach.measure.racket import measure_racket
from badminton_coach.measure.shuttle import track_shuttle
from badminton_coach.run import RunDir
from badminton_coach.swings import segment_swings
from badminton_coach.track import track


def run_measurement_pipeline(
    inputs: list[str | Path],
    run: RunDir,
    cfg: Config = CONFIG,
    device: str = "cuda",
    net_side_override: str | None = None,
    debug: bool = False,
    force: bool = False,
    skip_shuttle: bool = False,
) -> dict[str, Any]:
    """Runs ingest -> measure (body, racket, shuttle) -> track -> swings on `run`. Returns a dict of
    each stage's summary, plus `"errors"`: {stage_name: message} for any stage that failed non-fatally
    (currently only shuttle tracking is allowed to fail without aborting the run — a missing/poor
    shuttle signal is an expected, designed-for degradation, spec §10)."""
    summary: dict[str, Any] = {"errors": {}}

    ingest_result = ingest(inputs, run, cfg=cfg, force=force)
    summary["ingest"] = {"n_frames": ingest_result.n_frames, "fps": ingest_result.fps}
    fps = ingest_result.fps
    normalized_video = ingest_result.normalized_path

    summary["measure_body"] = measure_body(run, normalized_video, cfg=cfg, device=device, force=force, debug=debug)
    summary["measure_racket"] = measure_racket(run, normalized_video, cfg=cfg, force=force, debug=debug)

    if not skip_shuttle:
        try:
            summary["measure_shuttle"] = track_shuttle(run, normalized_video, cfg=cfg, device=device, force=force)
        except Exception as exc:  # noqa: BLE001 - shuttle is optional; every other failure should surface
            summary["errors"]["measure_shuttle"] = str(exc)[:1000]

    summary["track"] = track(run, cfg=cfg, fps=fps, net_side_override=net_side_override, force=force)
    summary["swings"] = segment_swings(run, cfg=cfg, fps=fps, force=force)

    return summary
