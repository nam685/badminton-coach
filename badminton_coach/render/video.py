"""Stage 7 (part 1) — the annotated video (spec §3.7): skeleton/racket/shuttle-trail/HUD every frame,
finding-driven red highlighting + banners, slow-motion around each swing's contact, encoded via an
ffmpeg pipe. Non-swing footage (before the first swing, between swings, after the last) is skipped —
the output is a focused highlight reel of the swings themselves, not a re-encode of the whole input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from badminton_coach.config import CONFIG, Config
from badminton_coach.i18n import t as i18n_t
from badminton_coach.measure.keypoints import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from badminton_coach.render.draw import draw_banner, draw_body, draw_hud, draw_racket, draw_shuttle_trail
from badminton_coach.run import RunDir
from badminton_coach.schema import Swing
from badminton_coach.video import iter_frames, write_video

SHUTTLE_TRAIL_LEN = 10

# Best-effort metric-name -> joint-index mapping, for highlighting the specific joints a finding is
# about (both sides, since we don't always know handedness at draw time from the metric name alone).
_METRIC_JOINT_HINTS: dict[str, list[int]] = {
    "elbow": [LEFT_ELBOW, RIGHT_ELBOW],
    "shoulder": [LEFT_SHOULDER, RIGHT_SHOULDER],
    "wrist": [LEFT_WRIST, RIGHT_WRIST],
    "hip": [LEFT_HIP, RIGHT_HIP],
    "ankle": [LEFT_ANKLE, RIGHT_ANKLE],
    "stance": [LEFT_ANKLE, RIGHT_ANKLE],
    "foot": [LEFT_ANKLE, RIGHT_ANKLE],
}


def _highlighted_joints(active_findings: list[dict]) -> set[int]:
    joints: set[int] = set()
    for f in active_findings:
        for m in f.get("evidence", {}).get("metrics", []):
            name = m.get("name", "")
            for hint, idxs in _METRIC_JOINT_HINTS.items():
                if hint in name:
                    joints.update(idxs)
    return joints


def _findings_by_swing_frame(findings: dict | None) -> dict[tuple[int, int], list[dict]]:
    """{(swing_index, frame): [finding, ...]} for quick per-frame lookup."""
    out: dict[tuple[int, int], list[dict]] = {}
    if not findings:
        return out
    for f in findings.get("findings", []):
        for ef in f.get("evidence", {}).get("frames", []):
            key = (ef["swing"], ef["frame"])
            out.setdefault(key, []).append(f)
    return out


def _active_finding_for_swing(findings: dict | None, swing_index: int) -> dict | None:
    """The highest-priority finding whose evidence.swings includes this swing (for the banner shown
    across the whole swing, not just its exact evidence frames)."""
    if not findings:
        return None
    by_id = {f["id"]: f for f in findings.get("findings", [])}
    for fid in findings.get("priority", []):
        f = by_id.get(fid)
        if f and swing_index in f.get("evidence", {}).get("swings", []):
            return f
    for f in findings.get("findings", []):
        if swing_index in f.get("evidence", {}).get("swings", []):
            return f
    return None


def render_annotated_video(
    run: RunDir,
    out_path: Path,
    findings: dict[str, Any] | None = None,
    cfg: Config = CONFIG,
    lang: str = "en",
) -> dict[str, Any]:
    track_dir = run.root / "track"
    swings_data = json.loads((run.root / "swings" / "swings.json").read_text())
    fps = swings_data["fps"]
    swings = [Swing(**s) for s in swings_data["swings"]]

    ingest = run.load_run_json().get("ingest", {})
    video = ingest["normalized_path"]

    player_smooth = np.load(track_dir / "player.npz")["smooth"]
    racket_smooth = np.load(track_dir / "racket.npz")["smooth"]
    shuttle_path = run.root / "measure" / "shuttle.csv"
    shuttle_df = pd.read_csv(shuttle_path) if shuttle_path.exists() else None

    active_by_frame = _findings_by_swing_frame(findings)

    def frame_generator():
        for swing in swings:
            active_finding = _active_finding_for_swing(findings, swing.index)
            contact_lo = swing.contact_frame - round(cfg.slow_mo_window_s * fps)
            contact_hi = swing.contact_frame + round(cfg.slow_mo_window_s * fps)

            for t in range(swing.prep_start_frame, swing.follow_end_frame + 1):
                if t < 0 or t >= player_smooth.shape[0]:
                    continue
                repeat = cfg.slow_mo_factor if contact_lo <= t <= contact_hi else 1

                frame_findings = active_by_frame.get((swing.index, t), [])
                highlighted = _highlighted_joints(frame_findings)

                base = next(iter_frames(video, start_frame=t, end_frame=t + 1))[1]
                img = draw_body(base, player_smooth[t], highlight_joints=highlighted)
                img = draw_racket(img, racket_smooth[t])

                if shuttle_df is not None:
                    window = shuttle_df[
                        (shuttle_df["Frame"] > t - SHUTTLE_TRAIL_LEN)
                        & (shuttle_df["Frame"] <= t)
                        & (shuttle_df["Visibility"] == 1)
                    ]
                    points = list(zip(window["X"].tolist(), window["Y"].tolist(), strict=True))
                    img = draw_shuttle_trail(img, points)

                phase_key = "contact" if t == swing.contact_frame else ("prep" if t < swing.contact_frame else "follow")
                phase = i18n_t(phase_key, lang)
                swing_label = i18n_t("swing", lang)
                img = draw_hud(img, [f"{swing_label} {swing.index}  {phase}  t={t / fps:.2f}s"])

                if active_finding:
                    img = draw_banner(img, active_finding["title"])

                for _ in range(repeat):
                    yield img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(frame_generator(), out_path, fps=fps, ffmpeg_bin=cfg.ffmpeg_bin)
    return {"n_swings": len(swings), "path": str(out_path)}
