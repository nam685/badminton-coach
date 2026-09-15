"""Stage 3 — swings: segment the racket-speed signal into individual swing attempts, find each one's
contact frame (shuttle-proximity when available, else the speed peak), classify live vs. shadow, and
locate the prep/follow key instants (spec §3.4).

"prep_end" ("back-scratch"): the racket top is at its lowest point *in the frame* — i.e. the largest
y-pixel value (image y increases downward) — shortly before contact, matching the loaded position just
before the racket whips up and forward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from badminton_coach.config import CONFIG, Config
from badminton_coach.measure.keypoints import RACKET_TOP
from badminton_coach.metrics import racket_speed_series
from badminton_coach.run import RunDir
from badminton_coach.schema import Swing, SwingsFile

STAGE_VERSION = "1"


def _racket_arm_wrist_speed_series(
    player_smooth: np.ndarray, wrist_idx: int, fps: float, torso_len: float
) -> np.ndarray:
    wrist = player_smooth[:, wrist_idx, :]
    speed = np.full(len(wrist), np.nan)
    if torso_len <= 0:
        return speed
    for t in range(1, len(wrist)):
        if not np.any(np.isnan(wrist[t])) and not np.any(np.isnan(wrist[t - 1])):
            speed[t] = float(np.linalg.norm(wrist[t] - wrist[t - 1])) * fps / torso_len
    return speed


def find_swings(
    racket_smooth: np.ndarray,  # [T, 5, 2]
    player_smooth: np.ndarray,  # [T, 26, 2]
    shuttle_df: pd.DataFrame | None,
    fps: float,
    torso_len: float,
    handedness: str,
    cfg: Config = CONFIG,
) -> tuple[list[Swing], np.ndarray, str]:
    """Returns (swings, speed_series_used, speed_source: "racket_top"|"racket_arm_wrist")."""
    speed = racket_speed_series(racket_smooth, fps, torso_len)
    speed_source = "racket_top"
    if np.sum(~np.isnan(speed)) < max(3, len(speed) // 4):
        # Racket data too sparse — fall back to the racket-arm wrist speed (spec §3.4).
        from badminton_coach.measure.keypoints import LEFT_WRIST, RIGHT_WRIST

        wrist_idx = RIGHT_WRIST if handedness == "right" else LEFT_WRIST
        speed = _racket_arm_wrist_speed_series(player_smooth, wrist_idx, fps, torso_len)
        speed_source = "racket_arm_wrist"

    speed_filled = np.nan_to_num(speed, nan=0.0)
    min_sep = max(1, round(cfg.swing_min_separation_s * fps))
    peaks, _ = find_peaks(speed_filled, distance=min_sep, prominence=cfg.swing_speed_peak_prominence)

    n_frames = len(speed)
    radius = cfg.contact_search_radius_frames
    prep_frames = max(1, round(cfg.prep_window_s * fps))
    follow_frames = max(1, round(cfg.follow_window_s * fps))

    top_xy = racket_smooth[:, RACKET_TOP, :]

    swings: list[Swing] = []
    for i, peak in enumerate(sorted(peaks)):
        window_lo, window_hi = max(0, peak - radius), min(n_frames - 1, peak + radius)

        contact_frame, contact_source, mode = int(peak), "speed", "shadow"
        if shuttle_df is not None and len(shuttle_df) > 0 and not np.any(np.isnan(top_xy[window_lo : window_hi + 1])):
            window = shuttle_df[
                (shuttle_df["Frame"] >= window_lo)
                & (shuttle_df["Frame"] <= window_hi)
                & (shuttle_df["Visibility"] == 1)
            ]
            if len(window) > 0:
                best_dist, best_frame = np.inf, None
                for _, row in window.iterrows():
                    f = int(row["Frame"])
                    if f >= len(top_xy) or np.any(np.isnan(top_xy[f])):
                        continue
                    dist = float(np.linalg.norm(top_xy[f] - np.array([row["X"], row["Y"]]))) / max(torso_len, 1e-6)
                    if dist < best_dist:
                        best_dist, best_frame = dist, f
                if best_frame is not None and best_dist <= cfg.contact_shuttle_max_dist_torso:
                    contact_frame, contact_source, mode = best_frame, "shuttle", "live"

        prep_start = max(0, contact_frame - prep_frames)
        prep_search = top_xy[prep_start:contact_frame, 1]
        prep_end_frame = None
        if len(prep_search) > 0 and not np.all(np.isnan(prep_search)):
            prep_end_frame = prep_start + int(np.nanargmax(prep_search))

        follow_end_frame = min(n_frames - 1, contact_frame + follow_frames)

        swings.append(
            Swing(
                index=i,
                mode=mode,
                contact_frame=contact_frame,
                contact_time_s=contact_frame / fps,
                contact_source=contact_source,
                prep_start_frame=prep_start,
                prep_end_frame=prep_end_frame,
                follow_end_frame=follow_end_frame,
            )
        )

    return swings, speed, speed_source


def segment_swings(run: RunDir, cfg: Config = CONFIG, fps: float = 30.0, force: bool = False) -> dict[str, Any]:
    track_dir = run.root / "track"
    measure_dir = run.root / "measure"

    player_data = np.load(track_dir / "player.npz")
    racket_data = np.load(track_dir / "racket.npz")
    track_json = json.loads((track_dir / "track.json").read_text())
    shuttle_path = measure_dir / "shuttle.csv"
    shuttle_df = pd.read_csv(shuttle_path) if shuttle_path.exists() else None

    player_smooth = player_data["smooth"]
    racket_smooth = racket_data["smooth"]
    handedness = track_json["handedness"]
    net_side = track_json["net_side"]
    torso_len = track_json.get("torso_len_median_px") or 1.0

    import hashlib

    h = hashlib.sha256()
    h.update(player_smooth.tobytes())
    h.update(racket_smooth.tobytes())
    h.update(f"{fps}:{handedness}:{net_side}".encode())
    input_hash = h.hexdigest()

    out_dir = run.root / "swings"
    with run.stage("swings", version=STAGE_VERSION, input_hash=input_hash, force=force) as st:
        if st.skip:
            data = SwingsFile.model_validate_json((out_dir / "swings.json").read_text())
            n_live = sum(1 for s in data.swings if s.mode == "live")
            return {
                "n_swings": len(data.swings),
                "n_live": n_live,
                "speed_source": data.speed_source,
            }

        out_dir.mkdir(parents=True, exist_ok=True)
        swings, speed, speed_source = find_swings(
            racket_smooth, player_smooth, shuttle_df, fps, torso_len, handedness, cfg
        )

        swings_file = SwingsFile(
            fps=fps,
            handedness=handedness,
            net_side=net_side,
            net_side_source=track_json.get("net_side_source", "unknown"),
            torso_len_median_px=track_json.get("torso_len_median_px"),
            speed_source=speed_source,
            swings=swings,
        )
        (out_dir / "swings.json").write_text(swings_file.model_dump_json(indent=2))
        np.save(out_dir / "speed_series.npy", speed)

        n_live = sum(1 for s in swings if s.mode == "live")
        run.mark_extra(
            st, n_swings=len(swings), n_live=n_live, n_shadow=len(swings) - n_live, speed_source=speed_source
        )

        plots_dir = run.root / "plots"
        from badminton_coach.metrics import elbow_angle_series, hip_width_ratio, resolve_sides, shoulder_width_ratio

        sides = resolve_sides(handedness)
        elbow_series = elbow_angle_series(player_smooth, sides)
        sw_series = np.array(
            [shoulder_width_ratio(player_smooth[t], torso_len) or np.nan for t in range(len(player_smooth))]
        )
        hw_series = np.array(
            [hip_width_ratio(player_smooth[t], torso_len) or np.nan for t in range(len(player_smooth))]
        )
        for swing in swings:
            plot_swing(plots_dir, swing, speed, elbow_series, sw_series, hw_series, fps)

        return {"n_swings": len(swings), "n_live": n_live, "speed_source": speed_source}


def plot_swing(
    plots_dir: Path,
    swing: Swing,
    speed: np.ndarray,
    elbow_series: np.ndarray,
    shoulder_width_series: np.ndarray,
    hip_width_series: np.ndarray,
    fps: float,
) -> None:
    """Writes `plots/swing_<i>.png`: racket speed, elbow angle, shoulder/hip width ratio vs. time, with
    the contact frame marked. Debug/dev-convenience artifact, not consumed by any other stage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    lo = max(0, swing.prep_start_frame - 5)
    hi = min(len(speed) - 1, swing.follow_end_frame + 5)
    frames = np.arange(lo, hi + 1)
    t = frames / fps

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, speed[lo : hi + 1])
    axes[0].set_ylabel("racket speed\n(torso/s)")
    axes[1].plot(t, elbow_series[lo : hi + 1])
    axes[1].set_ylabel("elbow angle\n(deg)")
    axes[2].plot(t, shoulder_width_series[lo : hi + 1], label="shoulder")
    axes[2].plot(t, hip_width_series[lo : hi + 1], label="hip")
    axes[2].set_ylabel("width ratio")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right", fontsize="small")

    contact_t = swing.contact_frame / fps
    for ax in axes:
        ax.axvline(contact_t, color="red", linestyle="--", linewidth=1, label="contact")
        if swing.prep_end_frame is not None:
            ax.axvline(swing.prep_end_frame / fps, color="gray", linestyle=":", linewidth=1)

    fig.suptitle(f"swing {swing.index} ({swing.mode}, contact via {swing.contact_source})")
    fig.tight_layout()
    fig.savefig(plots_dir / f"swing_{swing.index}.png", dpi=100)
    plt.close(fig)
