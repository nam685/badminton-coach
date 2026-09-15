"""Renders the judge workspace's per-swing contact sheets and crops (spec §6.1): a grid of frames from
prep through follow-through with skeleton/racket drawn and timestamps/phase labels, plus close-up crops
around contact. Shared between the player's own run and each reference clip's run (both are just
`RunDir`s with the same `track/` + `swings/` artifacts).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from badminton_coach.render.draw import draw_body, draw_racket, draw_text
from badminton_coach.run import RunDir
from badminton_coach.schema import Swing
from badminton_coach.video import read_frame

GRID_COLS, GRID_ROWS = 4, 3
GRID_N = GRID_COLS * GRID_ROWS
THUMB_W = 220


def _phase_label(frame: int, swing: Swing) -> str:
    if frame == swing.contact_frame:
        return "CONTACT"
    if swing.prep_end_frame is not None and frame == swing.prep_end_frame:
        return "prep_end"
    if frame < swing.contact_frame:
        return "prep"
    return "follow"


def _annotated_frame(
    video: str, frame_idx: int, player_kpts: np.ndarray, racket_kpts: np.ndarray, fps: float, label: str
) -> np.ndarray:
    frame = read_frame(video, frame_idx)
    frame = draw_body(frame, player_kpts)
    frame = draw_racket(frame, racket_kpts)
    t_s = frame_idx / fps
    frame = draw_text(frame, f"{label}  f{frame_idx}  t={t_s:.2f}s", (8, 8), size=16, bg=(0, 0, 0))
    return frame


def render_contact_sheet(run: RunDir, swing: Swing, fps: float, out_path: Path) -> Path:
    ingest = run.load_run_json().get("ingest", {})
    video = ingest["normalized_path"]
    player_smooth = np.load(run.root / "track" / "player.npz")["smooth"]
    racket_smooth = np.load(run.root / "track" / "racket.npz")["smooth"]

    lo, hi = swing.prep_start_frame, swing.follow_end_frame
    frame_indices = sorted(set(np.linspace(lo, hi, GRID_N).round().astype(int).tolist()) | {swing.contact_frame})
    frame_indices = frame_indices[:GRID_N] if len(frame_indices) > GRID_N else frame_indices

    thumbs = []
    for t in frame_indices:
        t = min(max(t, 0), player_smooth.shape[0] - 1)
        annotated = _annotated_frame(video, t, player_smooth[t], racket_smooth[t], fps, _phase_label(t, swing))
        h, w = annotated.shape[:2]
        scale = THUMB_W / w
        thumb = cv2.resize(annotated, (THUMB_W, int(h * scale)))
        thumbs.append(thumb)

    while len(thumbs) < GRID_N:
        thumbs.append(np.zeros_like(thumbs[0]))

    rows = []
    for r in range(GRID_ROWS):
        row_thumbs = thumbs[r * GRID_COLS : (r + 1) * GRID_COLS]
        rows.append(np.concatenate(row_thumbs, axis=1))
    grid = np.concatenate(rows, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path


def render_crops(run: RunDir, swing: Swing, fps: float, out_dir: Path, radius: int = 3) -> list[Path]:
    ingest = run.load_run_json().get("ingest", {})
    video = ingest["normalized_path"]
    player_smooth = np.load(run.root / "track" / "player.npz")["smooth"]
    racket_smooth = np.load(run.root / "track" / "racket.npz")["smooth"]
    n_frames = player_smooth.shape[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in range(max(0, swing.contact_frame - radius), min(n_frames, swing.contact_frame + radius + 1)):
        annotated = _annotated_frame(video, t, player_smooth[t], racket_smooth[t], fps, _phase_label(t, swing))
        # crop to the player's bbox (with margin) when available, else keep the full frame
        xy = player_smooth[t]
        valid = ~np.isnan(xy[:, 0])
        if valid.sum() >= 3:
            pts = xy[valid]
            x1, y1 = max(0, int(pts[:, 0].min() - 40)), max(0, int(pts[:, 1].min() - 40))
            x2, y2 = (
                min(annotated.shape[1], int(pts[:, 0].max() + 40)),
                min(annotated.shape[0], int(pts[:, 1].max() + 40)),
            )
            if x2 > x1 and y2 > y1:
                annotated = annotated[y1:y2, x1:x2]
        p = out_dir / f"c_{t:06d}.jpg"
        cv2.imwrite(str(p), annotated)
        paths.append(p)
    return paths
