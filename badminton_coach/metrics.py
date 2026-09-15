"""Stage 4 — metrics: pure functions computing per-swing technique metrics from smoothed keypoint arrays
(spec §3.5). All angle/geometry functions operate on a *single frame's* keypoints
(`player[26,3]`/`racket[5,3]`, x/y/score) and return `None` when required keypoints are missing (NaN).

**Coordinate convention**: image coordinates, x increases rightward, y increases *downward* (standard
image-plane convention). All distances are normalized by `torso_len` (the median
`|mid_shoulder - mid_hip|` over the session) unless noted, so metrics are comparable across camera
distances/resolutions but NOT across camera angles or sessions filmed from a different distance-to-torso
ratio (that's why "same camera position every session" matters — see the recording guide).

**Handedness / net-side sign conventions**: `handedness` ("left"/"right") selects which side's arm is the
*racket arm* for elbow/shoulder-abduction/contact metrics, and which side is the *non-racket* arm.
`net_dir_sign` (+1 if `net_side=="right"` else -1) makes "forward" (`contact_forward`) and "front foot"
consistently mean "toward the net" regardless of which way the player faces on screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from badminton_coach.measure.keypoints import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RACKET_BOTTOM,
    RACKET_HANDLE,
    RACKET_LEFT,
    RACKET_RIGHT,
    RACKET_TOP,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


@dataclass(frozen=True)
class Sides:
    """Body-landmark indices for the racket arm / non-racket arm, resolved from `handedness`."""

    shoulder: int
    elbow: int
    wrist: int
    hip: int
    knee: int
    ankle: int
    other_shoulder: int
    other_wrist: int
    other_hip: int
    other_ankle: int


def resolve_sides(handedness: str) -> Sides:
    if handedness == "left":
        return Sides(
            shoulder=LEFT_SHOULDER,
            elbow=LEFT_ELBOW,
            wrist=LEFT_WRIST,
            hip=LEFT_HIP,
            knee=LEFT_KNEE,
            ankle=LEFT_ANKLE,
            other_shoulder=RIGHT_SHOULDER,
            other_wrist=RIGHT_WRIST,
            other_hip=RIGHT_HIP,
            other_ankle=RIGHT_ANKLE,
        )
    return Sides(
        shoulder=RIGHT_SHOULDER,
        elbow=RIGHT_ELBOW,
        wrist=RIGHT_WRIST,
        hip=RIGHT_HIP,
        knee=RIGHT_KNEE,
        ankle=RIGHT_ANKLE,
        other_shoulder=LEFT_SHOULDER,
        other_wrist=LEFT_WRIST,
        other_hip=LEFT_HIP,
        other_ankle=LEFT_ANKLE,
    )


def net_dir_sign(net_side: str) -> float:
    return 1.0 if net_side == "right" else -1.0


# --- low-level geometry ---


def _valid(p: np.ndarray) -> bool:
    return p.shape[0] >= 2 and not np.any(np.isnan(p[:2]))


def _get(kpts: np.ndarray, idx: int) -> np.ndarray | None:
    p = kpts[idx]
    return p[:2] if _valid(p) else None


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Unsigned angle ABC (degrees), vertex at b, range [0, 180]."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_a)))


def angle_from_vertical_signed(p_low: np.ndarray, p_high: np.ndarray, sign: float) -> float:
    """Signed angle (degrees) of the vector p_low->p_high from straight-up vertical. 0 = vertical;
    positive = leaning toward the net (per `sign` = net_dir_sign(net_side)), negative = away from it.
    Caller contract: p_high is expected above p_low (smaller y) for the usual standing/upright case."""
    v = p_high - p_low  # y decreases upward in image coords, so "up" is (0, -1)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return 0.0
    up = np.array([0.0, -1.0])
    cos_a = np.clip(np.dot(v, up) / n, -1.0, 1.0)
    magnitude = math.degrees(math.acos(cos_a))
    horiz = v[0] * sign
    return magnitude if horiz >= 0 else -magnitude


# --- per-frame metrics ---


def elbow_angle(player: np.ndarray, sides: Sides) -> float | None:
    s, e, w = _get(player, sides.shoulder), _get(player, sides.elbow), _get(player, sides.wrist)
    if s is None or e is None or w is None:
        return None
    return angle_at(s, e, w)


def shoulder_abduction(player: np.ndarray, sides: Sides) -> float | None:
    h, s, e = _get(player, sides.hip), _get(player, sides.shoulder), _get(player, sides.elbow)
    if h is None or s is None or e is None:
        return None
    return angle_at(h, s, e)


def contact_height(player: np.ndarray, racket: np.ndarray, torso_len: float) -> dict[str, float] | None:
    """Positive = racket top above the reference point. Two variants: vs nose, vs racket-arm wrist."""
    nose = _get(player, NOSE)
    top = _get(racket, RACKET_TOP)
    if nose is None or top is None or torso_len <= 0:
        return None
    return {"vs_nose": (nose[1] - top[1]) / torso_len}


def contact_forward(
    player: np.ndarray, racket: np.ndarray, torso_len: float, sides: Sides, sign: float
) -> float | None:
    """Positive = racket top is toward the net relative to the racket-side shoulder."""
    shoulder = _get(player, sides.shoulder)
    top = _get(racket, RACKET_TOP)
    if shoulder is None or top is None or torso_len <= 0:
        return None
    return sign * (top[0] - shoulder[0]) / torso_len


def racket_shaft_angle(racket: np.ndarray, sign: float) -> float | None:
    handle, top = _get(racket, RACKET_HANDLE), _get(racket, RACKET_TOP)
    if handle is None or top is None:
        return None
    return angle_from_vertical_signed(handle, top, sign)


def racket_face_proxy(racket: np.ndarray) -> float | None:
    """|left-right| / |top-bottom|: larger => racket face more edge-on to the camera (foreshortened
    width), smaller => face more square to the camera. A rough visual proxy, not a measured angle."""
    left, right = _get(racket, RACKET_LEFT), _get(racket, RACKET_RIGHT)
    top, bottom = _get(racket, RACKET_TOP), _get(racket, RACKET_BOTTOM)
    if left is None or right is None or top is None or bottom is None:
        return None
    shaft_len = np.linalg.norm(top - bottom)
    if shaft_len < 1e-6:
        return None
    return float(np.linalg.norm(left - right) / shaft_len)


def trunk_lean(player: np.ndarray, sign: float) -> float | None:
    ls, rs = _get(player, LEFT_SHOULDER), _get(player, RIGHT_SHOULDER)
    lh, rh = _get(player, LEFT_HIP), _get(player, RIGHT_HIP)
    if ls is None or rs is None or lh is None or rh is None:
        return None
    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2
    return angle_from_vertical_signed(mid_hip, mid_shoulder, sign)


def shoulder_width_ratio(player: np.ndarray, torso_len: float) -> float | None:
    ls, rs = _get(player, LEFT_SHOULDER), _get(player, RIGHT_SHOULDER)
    if ls is None or rs is None or torso_len <= 0:
        return None
    return float(abs(ls[0] - rs[0]) / torso_len)


def hip_width_ratio(player: np.ndarray, torso_len: float) -> float | None:
    lh, rh = _get(player, LEFT_HIP), _get(player, RIGHT_HIP)
    if lh is None or rh is None or torso_len <= 0:
        return None
    return float(abs(lh[0] - rh[0]) / torso_len)


def non_racket_wrist_height(player: np.ndarray, sides: Sides, torso_len: float) -> float | None:
    shoulder, wrist = _get(player, sides.other_shoulder), _get(player, sides.other_wrist)
    if shoulder is None or wrist is None or torso_len <= 0:
        return None
    return (shoulder[1] - wrist[1]) / torso_len  # positive = wrist above shoulder


def stance_width(player: np.ndarray, torso_len: float) -> float | None:
    la, ra = _get(player, LEFT_ANKLE), _get(player, RIGHT_ANKLE)
    if la is None or ra is None or torso_len <= 0:
        return None
    return float(abs(la[0] - ra[0]) / torso_len)


def front_foot(player: np.ndarray, sign: float) -> str | None:
    """Which ankle is closer to the net ("left"/"right")."""
    la, ra = _get(player, LEFT_ANKLE), _get(player, RIGHT_ANKLE)
    if la is None or ra is None:
        return None
    # "closer to the net" = further along the net direction
    left_pos, right_pos = sign * la[0], sign * ra[0]
    return "left" if left_pos > right_pos else "right"


def jump_height(player: np.ndarray, standing_ankle_y: float, torso_len: float) -> float | None:
    la, ra = _get(player, LEFT_ANKLE), _get(player, RIGHT_ANKLE)
    if la is None or ra is None or torso_len <= 0 or standing_ankle_y is None:
        return None
    ankle_y = min(la[1], ra[1])  # smaller y = higher in the frame
    return max(0.0, (standing_ankle_y - ankle_y) / torso_len)


# --- time-series metrics (per frame across a window) ---


def racket_speed_series(racket_smooth: np.ndarray, fps: float, torso_len: float) -> np.ndarray:
    """Racket-top speed per frame (torso-lengths/second). `racket_smooth`: [T, 5, 2]."""
    top = racket_smooth[:, RACKET_TOP, :]
    speed = np.full(len(top), np.nan)
    if torso_len <= 0:
        return speed
    for t in range(1, len(top)):
        if not np.any(np.isnan(top[t])) and not np.any(np.isnan(top[t - 1])):
            speed[t] = float(np.linalg.norm(top[t] - top[t - 1])) * fps / torso_len
    return speed


def elbow_angle_series(player_smooth: np.ndarray, sides: Sides) -> np.ndarray:
    """`player_smooth`: [T, 26, 2] (no score column — `elbow_angle` only ever reads [:2])."""
    out = np.full(len(player_smooth), np.nan)
    for t in range(len(player_smooth)):
        val = elbow_angle(player_smooth[t], sides)
        out[t] = val if val is not None else float("nan")
    return out


# --- per-swing orchestration ---


def compute_swing_metrics(
    swing: "Swing",  # noqa: F821 - avoid a hard import cycle; badminton_coach.schema.Swing at runtime
    player_smooth: np.ndarray,  # [T, 26, 2]
    racket_smooth: np.ndarray,  # [T, 5, 2]
    speed_series: np.ndarray,  # [T], torso-lengths/s
    torso_len: float,
    handedness: str,
    net_side: str,
    fps: float,
) -> dict[str, float | str | None]:
    """Assembles the full flat metrics dict for one swing (spec §3.5), evaluated at the swing's key
    instants. Missing keypoints/instants yield `None` for that metric rather than raising."""
    sides = resolve_sides(handedness)
    sign = net_dir_sign(net_side)

    def player_at(frame: int | None) -> np.ndarray | None:
        if frame is None or frame < 0 or frame >= len(player_smooth):
            return None
        return player_smooth[frame]

    def racket_at(frame: int | None) -> np.ndarray | None:
        if frame is None or frame < 0 or frame >= len(racket_smooth):
            return None
        return racket_smooth[frame]

    prep_end, contact, follow_end = swing.prep_end_frame, swing.contact_frame, swing.follow_end_frame
    p_prep, p_contact, p_follow = player_at(prep_end), player_at(contact), player_at(follow_end)
    r_contact = racket_at(contact)

    out: dict[str, float | str | None] = {}

    out["elbow_angle_prep_end"] = elbow_angle(p_prep, sides) if p_prep is not None else None
    out["elbow_angle_contact"] = elbow_angle(p_contact, sides) if p_contact is not None else None
    out["elbow_angle_follow_end"] = elbow_angle(p_follow, sides) if p_follow is not None else None

    out["shoulder_abduction_prep_end"] = shoulder_abduction(p_prep, sides) if p_prep is not None else None
    out["shoulder_abduction_contact"] = shoulder_abduction(p_contact, sides) if p_contact is not None else None

    ch = contact_height(p_contact, r_contact, torso_len) if p_contact is not None and r_contact is not None else None
    out["contact_height_vs_nose"] = ch["vs_nose"] if ch else None

    out["contact_forward"] = (
        contact_forward(p_contact, r_contact, torso_len, sides, sign)
        if p_contact is not None and r_contact is not None
        else None
    )
    out["racket_shaft_angle_contact"] = racket_shaft_angle(r_contact, sign) if r_contact is not None else None
    out["racket_face_proxy_contact"] = racket_face_proxy(r_contact) if r_contact is not None else None

    out["trunk_lean_prep_end"] = trunk_lean(p_prep, sign) if p_prep is not None else None
    out["trunk_lean_contact"] = trunk_lean(p_contact, sign) if p_contact is not None else None

    out["shoulder_width_ratio_prep_end"] = shoulder_width_ratio(p_prep, torso_len) if p_prep is not None else None
    out["shoulder_width_ratio_contact"] = shoulder_width_ratio(p_contact, torso_len) if p_contact is not None else None
    out["hip_width_ratio_prep_end"] = hip_width_ratio(p_prep, torso_len) if p_prep is not None else None
    out["hip_width_ratio_contact"] = hip_width_ratio(p_contact, torso_len) if p_contact is not None else None

    out["non_racket_wrist_height_prep_end"] = (
        non_racket_wrist_height(p_prep, sides, torso_len) if p_prep is not None else None
    )

    out["stance_width_prep_end"] = stance_width(p_prep, torso_len) if p_prep is not None else None
    out["stance_width_contact"] = stance_width(p_contact, torso_len) if p_contact is not None else None

    front_prep = front_foot(p_prep, sign) if p_prep is not None else None
    front_contact = front_foot(p_contact, sign) if p_contact is not None else None
    out["front_foot_prep_end"] = front_prep
    out["front_foot_contact"] = front_contact
    out["weight_transfer"] = (
        (front_prep != front_contact) if front_prep is not None and front_contact is not None else None
    )

    standing_ankle_y = None
    if prep_end is not None and prep_end > swing.prep_start_frame:
        window = player_smooth[swing.prep_start_frame : prep_end + 1]
        ys = []
        for idx in (LEFT_ANKLE, RIGHT_ANKLE):
            col = window[:, idx, 1]
            ys.extend(col[~np.isnan(col)].tolist())
        if ys:
            standing_ankle_y = float(np.median(ys))
    out["jump_height_contact"] = (
        jump_height(p_contact, standing_ankle_y, torso_len)
        if p_contact is not None and standing_ankle_y is not None
        else None
    )

    window_lo, window_hi = swing.prep_start_frame, follow_end
    window_speed = speed_series[window_lo : window_hi + 1]
    out["peak_racket_speed"] = float(np.nanmax(window_speed)) if np.any(~np.isnan(window_speed)) else None

    out["time_prep_to_contact_s"] = (contact - prep_end) / fps if prep_end is not None else None
    out["follow_through_duration_s"] = (follow_end - contact) / fps

    return out


# --- stage entrypoint ---


def compute_metrics_stage(
    run,
    cfg=None,
    references: list | None = None,
    force: bool = False,
) -> dict:
    """Stage 4: load `track/*` + `swings/*`, compute every swing's metrics, compare against
    `references` (a list of other runs' `RunDir`s — typically pro reference clips; pass `[]`/`None` to
    skip comparison, e.g. when building a reference clip itself), and write `metrics/metrics.json`.
    """
    import hashlib
    import json as _json

    import numpy as _np

    from badminton_coach.config import CONFIG as _CONFIG
    from badminton_coach.reference import load_reference_metric_values
    from badminton_coach.schema import CrossAttemptStat, MetricsFile, MetricValue, Swing, SwingMetrics

    cfg = cfg or _CONFIG
    references = references or []
    ref_values = load_reference_metric_values(references) if references else {}

    def ref_stats(name: str) -> dict:
        values = ref_values.get(name)
        if not values:
            return {}
        return {
            "ref_mean": float(_np.mean(values)),
            "ref_min": float(min(values)),
            "ref_max": float(max(values)),
            "n_ref": len(values),
        }

    track_dir = run.root / "track"
    swings_dir = run.root / "swings"

    player_smooth = _np.load(track_dir / "player.npz")["smooth"]
    racket_smooth = _np.load(track_dir / "racket.npz")["smooth"]
    speed_series = _np.load(swings_dir / "speed_series.npy")
    track_json = _json.loads((track_dir / "track.json").read_text())
    swings_data = _json.loads((swings_dir / "swings.json").read_text())
    swings = [Swing(**s) for s in swings_data["swings"]]
    fps = swings_data["fps"]
    handedness = swings_data["handedness"]
    net_side = swings_data["net_side"]
    torso_len = track_json.get("torso_len_median_px") or 1.0

    h = hashlib.sha256()
    h.update(player_smooth.tobytes())
    h.update(racket_smooth.tobytes())
    h.update(str(sorted(r.root.name for r in references)).encode())
    input_hash = h.hexdigest()

    out_dir = run.root / "metrics"
    with run.stage("metrics", version="1", input_hash=input_hash, force=force) as st:
        if st.skip:
            data = MetricsFile.model_validate_json((out_dir / "metrics.json").read_text())
            return {"n_swings": len(data.swings)}

        out_dir.mkdir(parents=True, exist_ok=True)

        swing_metrics: list[SwingMetrics] = []
        raw_by_metric: dict[str, list[float]] = {}
        for swing in swings:
            raw = compute_swing_metrics(
                swing, player_smooth, racket_smooth, speed_series, torso_len, handedness, net_side, fps
            )
            metric_values: dict[str, MetricValue] = {}
            for name, value in raw.items():
                if isinstance(value, str) or isinstance(value, bool):
                    continue  # front_foot_*, weight_transfer are not numeric reference-comparable metrics
                metric_values[name] = MetricValue(value=value, **ref_stats(name))
                if value is not None:
                    raw_by_metric.setdefault(name, []).append(value)
            swing_metrics.append(
                SwingMetrics(
                    index=swing.index,
                    mode=swing.mode,
                    contact_frame=swing.contact_frame,
                    contact_time_s=swing.contact_time_s,
                    metrics=metric_values,
                )
            )

        cross_summary: dict[str, CrossAttemptStat] = {}
        for name, values in raw_by_metric.items():
            if len(values) == 0:
                continue
            mean = float(_np.mean(values))
            std = float(_np.std(values)) if len(values) > 1 else 0.0
            # Flag "inconsistent" only when we actually have >=2 attempts to measure spread across, and
            # a reference to judge "too much spread" against.
            consistency_flag = bool(len(values) > 1 and name in ref_values and std > 0.5 * abs(mean or 1.0))
            cross_summary[name] = CrossAttemptStat(mean=mean, std=std, n=len(values), consistency_flag=consistency_flag)

        metrics_file = MetricsFile(
            fps=fps,
            handedness=handedness,
            net_side=net_side,
            torso_len_median_px=track_json.get("torso_len_median_px"),
            swings=swing_metrics,
            cross_attempt_summary=cross_summary,
        )
        (out_dir / "metrics.json").write_text(metrics_file.model_dump_json(indent=2))

        run.mark_extra(st, n_swings=len(swings))
        return {"n_swings": len(swings)}
