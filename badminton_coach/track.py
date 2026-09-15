"""Stage 2 — track/smooth: pick one player track out of `measure/body.npz`'s multiple detected people,
associate a racket with that player (which also gives handedness), fill small gaps, smooth, and
determine which way the net is (spec §3.3).

Player selection score per frame (spec: "bbox area × proximity to the racket handle × proximity to
previous frame's choice"), each factor normalized to roughly [0, 1] so the product is well-behaved:
  - normalized bbox area (derived from the keypoint bounding box — rtmlib's high-level API doesn't
    expose the detector's own box) relative to the largest person that frame
  - closeness to the nearest racket handle that frame (neutral 0.5 if no racket detected), boosted by
    that racket's own frame-to-frame motion — added after a real judge run caught the selector locking
    onto a bystander who was calmly *holding* a racket near a similarly-sized swinger (see
    docs/judge-eval-template.md's 2026-09-15 entry): proximity alone can't distinguish "holding" from
    "swinging", but motion can.
  - IoU with the previous frame's chosen bbox (neutral 0.5 for the first frame / right after a gap),
    with a relaxed floor when the racket signal above is decisive, so a strong sustained motion signal
    can override a bad early lock-in instead of "stickiness" keeping it forever
  - a final majority-vote pass (`_stabilize_selection`) cleans up isolated single-frame flips

**Known limitation** (found on real footage, not fully solved): during the fastest part of a swing, a
short run of frames (not just one) can still pick the wrong person before the signal above recovers —
greedy frame-by-frame scoring with a local stabilization pass isn't a full multi-object tracker. Good
enough to turn "wrong person for the whole swing" into "wrong person for a handful of frames near the
peak" in the one real case tested; likely worth revisiting with more real footage in Task 11 if it
recurs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from badminton_coach.config import CONFIG, Config
from badminton_coach.measure.keypoints import (
    BODY_NUM_KPTS,
    LEFT_WRIST,
    RACKET_HANDLE,
    RACKET_NUM_KPTS,
    RACKET_TOP,
    RIGHT_WRIST,
)
from badminton_coach.run import RunDir

STAGE_VERSION = "1"


# --- geometry helpers ---


def _keypoint_bbox(kpts_xy: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray | None:
    """Bounding box (x1,y1,x2,y2) of keypoints with score > thr; None if fewer than 2 are visible."""
    valid = scores > thr
    if valid.sum() < 2:
        return None
    pts = kpts_xy[valid]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def _bbox_area(bbox: np.ndarray | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a, area_b = _bbox_area(a), _bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _torso_length(kpts_xy: np.ndarray, scores: np.ndarray, thr: float) -> float | None:
    from badminton_coach.measure.keypoints import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

    idxs = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if any(scores[i] <= thr for i in idxs):
        return None
    mid_shoulder = (kpts_xy[LEFT_SHOULDER] + kpts_xy[RIGHT_SHOULDER]) / 2
    mid_hip = (kpts_xy[LEFT_HIP] + kpts_xy[RIGHT_HIP]) / 2
    return float(np.linalg.norm(mid_shoulder - mid_hip))


# --- player selection ---


@dataclass
class SelectionResult:
    chosen: np.ndarray  # [T] int, -1 if no person detected that frame
    scores: list[dict]  # per-frame debug info


def _racket_top_speeds(racket_kpts: np.ndarray, t: int, kpt_conf_thr: float) -> dict[int, float]:
    """{racket_index: px moved by its top point since frame t-1}, 0.0 for t==0 or no match found in the
    previous frame (matched by nearest racket-top position — a light heuristic, not full identity
    tracking, but good enough to tell "this racket just moved a lot" from "this racket has been still").
    """
    n_racket = racket_kpts.shape[1]
    speeds: dict[int, float] = {}
    if t == 0:
        return speeds
    prev_tops = [
        (r2, racket_kpts[t - 1, r2, RACKET_TOP, 0:2])
        for r2 in range(n_racket)
        if racket_kpts[t - 1, r2, RACKET_TOP, 2] > kpt_conf_thr
    ]
    if not prev_tops:
        return speeds
    for r in range(n_racket):
        if racket_kpts[t, r, RACKET_TOP, 2] <= kpt_conf_thr:
            continue
        top = racket_kpts[t, r, RACKET_TOP, 0:2]
        best_dist = min(np.linalg.norm(top - prev_top) for _, prev_top in prev_tops)
        speeds[r] = float(best_dist)
    return speeds


def select_player(
    body_kpts: np.ndarray,  # [T, P, 26, 3]
    racket_kpts: np.ndarray,  # [T, R, 5, 3]
    kpt_conf_thr: float,
) -> SelectionResult:
    """Player selection score per frame: normalized bbox area x racket-holding gate x racket-motion
    boost x continuity with the previous frame's choice (spec §3.3's factors, plus racket motion — see
    module docstring's note on why area/proximity alone aren't enough when a bystander is also holding a
    racket, e.g. a partner standing still nearby: someone calmly *holding* a racket scores the same as
    someone *swinging* one under proximity alone, so a fast-moving racket additionally boosts its
    holder's score, and a static one doesn't)."""
    n_frames, max_persons = body_kpts.shape[0], body_kpts.shape[1]
    chosen = np.full(n_frames, -1, dtype=np.int64)
    debug: list[dict] = []
    prev_bbox: np.ndarray | None = None

    for t in range(n_frames):
        n_racket = racket_kpts.shape[1]
        racket_handles = []
        for r in range(n_racket):
            if racket_kpts[t, r, RACKET_HANDLE, 2] > kpt_conf_thr:
                racket_handles.append((r, racket_kpts[t, r, RACKET_HANDLE, 0:2]))
        racket_speeds = _racket_top_speeds(racket_kpts, t, kpt_conf_thr)

        candidates = []
        for p in range(max_persons):
            xy = body_kpts[t, p, :, 0:2]
            sc = body_kpts[t, p, :, 2]
            if np.all(np.isnan(sc)):
                continue
            bbox = _keypoint_bbox(xy, np.nan_to_num(sc, nan=0.0), kpt_conf_thr)
            if bbox is None:
                continue
            candidates.append((p, bbox, xy, sc))

        if not candidates:
            debug.append({"frame": t, "chosen": -1, "n_candidates": 0})
            continue

        areas = np.array([_bbox_area(c[1]) for c in candidates])
        max_area = areas.max() if areas.max() > 0 else 1.0

        best_score, best_p, best_bbox = -1.0, -1, None
        frame_debug = []
        for (p, bbox, xy, sc), area in zip(candidates, areas, strict=True):
            norm_area = area / max_area

            if racket_handles:
                torso = _torso_length(xy, sc, kpt_conf_thr) or 100.0
                wrists = [xy[LEFT_WRIST], xy[RIGHT_WRIST]]
                nearest_r, min_dist = min(
                    ((r, np.linalg.norm(w - h)) for w in wrists for r, h in racket_handles), key=lambda x: x[1]
                )
                # A person either is or isn't holding the visible racket — use a steep gate rather than
                # a smooth 1/(1+d) falloff, which (normalized by each candidate's *own* torso length)
                # otherwise lets a large/close-to-camera background person "win" on area alone even when
                # a racket is clearly held by someone else: a bigger person's bigger torso shrinks their
                # relative distance-in-torso-lengths to *any* racket, including one that isn't theirs.
                ratio = min_dist / max(torso, 1e-6)
                racket_score = 1.0 if ratio < 1.5 else (0.3 if ratio < 3.0 else 0.02)
                # Boost by how fast *that* racket is moving: a calmly-held, static racket scores the
                # same on proximity alone as one mid-swing, which is exactly the failure mode that
                # let a stationary bystander with their own racket outscore the actual swinger.
                speed_norm = racket_speeds.get(nearest_r, 0.0) / max(torso, 1e-6)
                racket_score *= 0.4 + min(speed_norm, 1.2)
            else:
                racket_score = 0.5

            continuity_score = _iou(bbox, prev_bbox) if prev_bbox is not None else 0.5
            # Floor continuity rather than let it hit exactly 0 on a switch — but the floor itself scales
            # with how decisive the racket signal is: a *clearly, actively swinging* candidate (high
            # racket_score, which now includes the motion boost above) should be able to override a
            # previous pick that was only ever a coin-flip at frame 0 (before any motion data existed),
            # instead of "stickiness" permanently locking in whoever won that first ambiguous frame.
            continuity_floor = 0.7 if racket_score > 0.8 else 0.1
            if prev_bbox is not None:
                continuity_score = max(continuity_score, continuity_floor)

            score = norm_area * racket_score * max(continuity_score, 0.1)
            frame_debug.append({"person": p, "score": round(float(score), 4)})
            if score > best_score:
                best_score, best_p, best_bbox = score, p, bbox

        chosen[t] = best_p
        prev_bbox = best_bbox
        debug.append({"frame": t, "chosen": int(best_p), "n_candidates": len(candidates), "scores": frame_debug})

    return SelectionResult(chosen=_stabilize_selection(chosen), scores=debug)


def _stabilize_selection(chosen: np.ndarray, window: int = 5) -> np.ndarray:
    """Cleans up isolated single-frame flips (e.g. `..., 1, 1, 0, 1, 1, ...`) that survive the per-frame
    scoring even after the racket-motion/continuity improvements above — a brief tie or noisy frame can
    still flip the winner for one frame in the middle of an otherwise-consistent run. Replaces each
    frame's choice with the mode of a `window`-frame neighborhood (ties keep the original value); frames
    with no detection (-1) are left alone and excluded from neighbors' mode counts."""
    n = len(chosen)
    out = chosen.copy()
    half = window // 2
    for t in range(n):
        if chosen[t] == -1:
            continue
        lo, hi = max(0, t - half), min(n, t + half + 1)
        neighborhood = [v for v in chosen[lo:hi] if v != -1]
        if not neighborhood:
            continue
        values, counts = np.unique(neighborhood, return_counts=True)
        best = values[np.argmax(counts)]
        max_count = counts.max()
        # only override on a clear majority (not a tie) so we don't fight a real, sustained handoff
        if best != chosen[t] and np.sum(counts == max_count) == 1:
            out[t] = best
    return out


# --- racket association / handedness ---


def associate_racket(
    body_kpts: np.ndarray,  # [T, 26, 3] already player-selected
    racket_kpts: np.ndarray,  # [T, R, 5, 3]
    kpt_conf_thr: float,
    max_dist_torso: float = 3.0,
) -> tuple[np.ndarray, list[str | None]]:
    """Returns (racket_track[T,5,3] — the associated racket's keypoints per frame, NaN if none;
    wrist_side per frame: "left"/"right"/None)."""
    n_frames, n_racket = racket_kpts.shape[0], racket_kpts.shape[1]
    out = np.full((n_frames, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
    sides: list[str | None] = [None] * n_frames

    for t in range(n_frames):
        sc = body_kpts[t, :, 2]
        xy = body_kpts[t, :, 0:2]
        torso = _torso_length(xy, np.nan_to_num(sc, nan=0.0), kpt_conf_thr) or 100.0

        best_dist, best_r, best_side = np.inf, -1, None
        for r in range(n_racket):
            if racket_kpts[t, r, RACKET_HANDLE, 2] <= kpt_conf_thr:
                continue
            handle = racket_kpts[t, r, RACKET_HANDLE, 0:2]
            for side, idx in (("left", LEFT_WRIST), ("right", RIGHT_WRIST)):
                if sc[idx] <= kpt_conf_thr:
                    continue
                dist = float(np.linalg.norm(xy[idx] - handle))
                if dist < best_dist:
                    best_dist, best_r, best_side = dist, r, side

        if best_r >= 0 and best_dist <= max_dist_torso * torso:
            out[t] = racket_kpts[t, best_r]
            sides[t] = best_side

    return out, sides


def _majority_handedness(sides: list[str | None]) -> str | None:
    votes = [s for s in sides if s is not None]
    if not votes:
        return None
    left = votes.count("left")
    right = votes.count("right")
    return "left" if left > right else "right"


# --- gap fill + smoothing ---


def _fill_small_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear-interpolate NaN runs of length <= max_gap along axis 0; longer runs stay NaN."""
    out = x.copy()
    n = len(out)
    valid = ~np.isnan(out)
    if valid.all() or not valid.any():
        return out
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and j < n:
            out[i:j] = np.interp(np.arange(i, j), [i - 1, j], [out[i - 1], out[j]])
        i = j
    return out


def _smooth_with_gaps(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing applied independently to each contiguous non-NaN run."""
    out = x.copy()
    n = len(out)
    valid = ~np.isnan(out)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        run_len = j - i
        w = min(window, run_len if run_len % 2 == 1 else run_len - 1)
        if w >= polyorder + 1 and w >= 3:
            if w % 2 == 0:
                w -= 1
            out[i:j] = savgol_filter(out[i:j], w, min(polyorder, w - 1))
        i = j
    return out


def gap_fill_and_smooth(
    arr: np.ndarray, max_gap: int, window: int, polyorder: int, conf_thr: float
) -> dict[str, np.ndarray]:
    """arr: [T, K, 3] (x, y, score). Masks low-confidence points to NaN, gap-fills, smooths.
    Returns {"raw": [T,K,2] masked (pre-smooth), "smooth": [T,K,2], "conf": [T,K]}."""
    t, k, _ = arr.shape
    xy = arr[:, :, 0:2].copy()
    conf = arr[:, :, 2].copy()
    mask = conf <= conf_thr
    xy[mask] = np.nan

    raw = xy.copy()
    smooth = np.full_like(xy, np.nan)
    for ki in range(k):
        for axis in range(2):
            filled = _fill_small_gaps(xy[:, ki, axis], max_gap)
            smooth[:, ki, axis] = _smooth_with_gaps(filled, window, polyorder)

    return {"raw": raw, "smooth": smooth, "conf": conf}


# --- net direction ---


def determine_net_direction(
    racket_track: np.ndarray,  # [T, 5, 3]
    shuttle_df: pd.DataFrame | None,
    fps: float,
    net_side_override: str | None = None,
) -> tuple[str, str]:
    """Returns (net_side: "left"/"right", source: "shuttle"/"racket_travel"/"override"/"unknown")."""
    if net_side_override in ("left", "right"):
        return net_side_override, "override"

    top_xy = racket_track[:, RACKET_TOP, 0:2]
    valid = ~np.isnan(top_xy[:, 0])
    if valid.sum() < 3:
        return "right", "unknown"

    speed = np.full(len(top_xy), np.nan)
    for t in range(1, len(top_xy)):
        if valid[t] and valid[t - 1]:
            speed[t] = np.linalg.norm(top_xy[t] - top_xy[t - 1])
    if np.all(np.isnan(speed)):
        return "right", "unknown"
    peak_t = int(np.nanargmax(speed))

    # Look ~0.3s after the peak (frame-count window, not a fixed frame count, so this behaves the same
    # at 30fps and 60fps footage).
    window_frames = max(2, round(0.3 * fps))

    if shuttle_df is not None and len(shuttle_df) > 0:
        window = shuttle_df[(shuttle_df["Frame"] >= peak_t) & (shuttle_df["Frame"] <= peak_t + window_frames)]
        window = window[window["Visibility"] == 1]
        if len(window) >= 2:
            dx = window["X"].iloc[-1] - window["X"].iloc[0]
            if abs(dx) > 1e-6:
                return ("right" if dx > 0 else "left"), "shuttle"

    end_t = min(peak_t + max(2, window_frames // 2), len(top_xy) - 1)
    if valid[peak_t] and valid[end_t] and end_t > peak_t:
        dx = top_xy[end_t, 0] - top_xy[peak_t, 0]
        if abs(dx) > 1e-6:
            return ("right" if dx > 0 else "left"), "racket_travel"

    return "right", "unknown"


# --- stage entrypoint ---


def track(
    run: RunDir,
    cfg: Config = CONFIG,
    fps: float = 30.0,
    net_side_override: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    measure_dir = run.root / "measure"
    body_data = np.load(measure_dir / "body.npz")
    racket_data = np.load(measure_dir / "racket.npz")
    shuttle_path = measure_dir / "shuttle.csv"
    shuttle_df = pd.read_csv(shuttle_path) if shuttle_path.exists() else None

    body_kpts_all = body_data["kpts"]  # [T, P, 26, 3]
    racket_kpts_all = racket_data["kpts"]  # [T, R, 5, 3]

    # Python's built-in hash() on bytes is randomized per-process (PYTHONHASHSEED) unless disabled, so
    # it would never agree with a previous run's cached value — use a real deterministic digest instead.
    h = hashlib.sha256()
    h.update(body_kpts_all.tobytes())
    h.update(racket_kpts_all.tobytes())
    h.update(str(net_side_override).encode())
    input_hash = h.hexdigest()

    out_dir = run.root / "track"
    with run.stage("track", version=STAGE_VERSION, input_hash=input_hash, force=force) as st:
        if st.skip:
            data = np.load(out_dir / "player.npz")
            return {"n_frames": int(data["smooth"].shape[0])}

        out_dir.mkdir(parents=True, exist_ok=True)

        window = cfg.savgol_window_60fps if fps >= 45 else cfg.savgol_window_30fps

        selection = select_player(body_kpts_all, racket_kpts_all, cfg.body_kpt_conf_thr)
        n_frames = body_kpts_all.shape[0]

        player_raw = np.full((n_frames, BODY_NUM_KPTS, 3), np.nan, dtype=np.float32)
        for t in range(n_frames):
            if selection.chosen[t] >= 0:
                player_raw[t] = body_kpts_all[t, selection.chosen[t]]

        racket_assoc, wrist_sides = associate_racket(player_raw, racket_kpts_all, cfg.racket_kpt_conf_thr)

        player_gs = gap_fill_and_smooth(
            player_raw, cfg.gap_fill_max_frames, window, cfg.savgol_polyorder, cfg.body_kpt_conf_thr
        )
        racket_gs = gap_fill_and_smooth(
            racket_assoc, cfg.gap_fill_max_frames, window, cfg.savgol_polyorder, cfg.racket_kpt_conf_thr
        )

        handedness = _majority_handedness(wrist_sides) or "right"
        net_side, net_source = determine_net_direction(racket_assoc, shuttle_df, fps, net_side_override)

        torso_lengths = [
            _torso_length(player_gs["smooth"][t], np.full(BODY_NUM_KPTS, 1.0), 0.0) for t in range(n_frames)
        ]
        torso_lengths = [tl for tl in torso_lengths if tl is not None and not np.isnan(tl)]
        torso_median = float(np.median(torso_lengths)) if torso_lengths else None

        np.savez_compressed(
            out_dir / "player.npz",
            raw=player_raw,
            smooth=player_gs["smooth"],
            conf=player_gs["conf"],
        )
        np.savez_compressed(
            out_dir / "racket.npz",
            raw=racket_assoc[:, :, 0:2],
            smooth=racket_gs["smooth"],
            conf=racket_assoc[:, :, 2],
        )
        (out_dir / "selection.json").write_text(json.dumps(selection.scores, default=str))

        track_json = {
            "handedness": handedness,
            "net_side": net_side,
            "net_side_source": net_source,
            "torso_len_median_px": torso_median,
            "frac_frames_with_player": float((selection.chosen >= 0).mean()),
            "frac_frames_with_racket": float((~np.isnan(racket_assoc[:, RACKET_HANDLE, 0])).mean()),
        }
        (out_dir / "track.json").write_text(json.dumps(track_json, indent=2))

        run.mark_extra(st, **track_json)
        return {"n_frames": n_frames, **track_json}
