"""3D rotation metrics from `body3d/joints.npz` (spec §3.4b/§3.5, Task 7b).

**Coordinate convention (confirmed against a real SAM 3D Body inference run on the test fixture — see
docs/body3d.md)**: camera-space, **Y increases downward** (matches the 2D image convention, NOT the
Y-up convention SMPL-family models typically use — verified empirically: a standing player's shoulders
have a *smaller* Y than their hips despite being physically higher). X/Z handedness is not independently
confirmed, but our math only needs the *horizontal* plane (X, Z) — the plane perpendicular to gravity —
so an unconfirmed X/Z handedness would only flip rotation-angle *magnitudes* between two frames if the
player rotated through more than 180°, which doesn't happen in a swing. The one thing that *does* depend
on it is the **sign** of `x_factor_deg` (which way the trunk is coiled) — hedged accordingly in the
rubric, per spec.

**Net direction in 3D**: assumes the camera's X-axis is aligned with image X (true for an upright, non-
rolled camera — the recording guide's "landscape, side-on" setup) and reuses `net_dir_sign(net_side)`
from `metrics.py` directly as the sign along camera-space X.

**MHR70 joint indices** (verified against the cloned `sam-3d-body` source,
`sam_3d_body/metadata/mhr70.py`): 5=left_shoulder, 6=right_shoulder, 9=left_hip, 10=right_hip.

**Backend-dependent indices**: `body3d`'s two possible backends do NOT share one joint layout past the
shoulders. `backend="sam3d_body"` is MHR70 order (hips at 9/10). `backend="rtmw3d"` (the fallback,
`rtmlib.Wholebody3d`/RTMW3D-x) is standard COCO-WholeBody order — shoulders happen to also be at 5/6
there, but **hips are at 11/12**, not 9/10 (9/10 are wrists in COCO order). Every function below that
indexes hips/shoulders takes a `backend` argument and resolves indices via `_indices_for_backend` rather
than assuming MHR70 — get this wrong and rtmw3d-fallback swings silently report wrist positions as hips.
"""

from __future__ import annotations

import math

import numpy as np

MHR70_LEFT_SHOULDER = 5
MHR70_RIGHT_SHOULDER = 6
MHR70_LEFT_HIP = 9
MHR70_RIGHT_HIP = 10

# COCO-WholeBody order (rtmw3d fallback): body block is standard COCO-17.
_COCO_LEFT_SHOULDER = 5
_COCO_RIGHT_SHOULDER = 6
_COCO_LEFT_HIP = 11
_COCO_RIGHT_HIP = 12


def _indices_for_backend(backend: str) -> tuple[int, int, int, int]:
    """Returns (left_shoulder, right_shoulder, left_hip, right_hip) for the given `body3d` backend."""
    if backend == "sam3d_body":
        return MHR70_LEFT_SHOULDER, MHR70_RIGHT_SHOULDER, MHR70_LEFT_HIP, MHR70_RIGHT_HIP
    if backend == "rtmw3d":
        return _COCO_LEFT_SHOULDER, _COCO_RIGHT_SHOULDER, _COCO_LEFT_HIP, _COCO_RIGHT_HIP
    raise ValueError(f"unknown body3d backend {backend!r} — add its joint layout to _indices_for_backend")


def _valid3(p: np.ndarray) -> bool:
    return p.shape[-1] >= 3 and not np.any(np.isnan(p[:3]))


def horizontal_rotation_deg(left: np.ndarray, right: np.ndarray, net_dir_sign: float) -> float | None:
    """Angle (degrees, [0, 90]) of the left->right line's horizontal-plane (X, Z) projection from the
    net direction (camera-space X axis). 0 = square to the net, 90 = fully sideways.

    (Unsigned by design: "how turned" is what the rubric needs; X-factor below carries the one place a
    sign matters, and even that is hedged.)
    """
    if not _valid3(left) or not _valid3(right):
        return None
    v = right[:3] - left[:3]
    x, _, z = v[0], v[1], v[2]
    if abs(x) < 1e-9 and abs(z) < 1e-9:
        return None
    # angle between (x, z) and the net-direction axis (net_dir_sign, 0)
    net_vec = np.array([net_dir_sign, 0.0])
    line_vec = np.array([x, z])
    n1, n2 = np.linalg.norm(net_vec), np.linalg.norm(line_vec)
    cos_a = np.clip(np.dot(net_vec, line_vec) / (n1 * n2), -1.0, 1.0)
    ang = math.degrees(math.acos(cos_a))
    # fold to [0, 90]: the shoulder/hip *line* has no intrinsic direction (left->right vs right->left
    # both describe the same body orientation), so >90 and its 180-complement mean the same rotation.
    return min(ang, 180.0 - ang)


def shoulder_rotation_deg(joints: np.ndarray, net_dir_sign: float, backend: str = "sam3d_body") -> float | None:
    """`joints`: [J, 3] (single frame, layout matching `backend`)."""
    ls, rs, _, _ = _indices_for_backend(backend)
    return horizontal_rotation_deg(joints[ls], joints[rs], net_dir_sign)


def hip_rotation_deg(joints: np.ndarray, net_dir_sign: float, backend: str = "sam3d_body") -> float | None:
    _, _, lh, rh = _indices_for_backend(backend)
    return horizontal_rotation_deg(joints[lh], joints[rh], net_dir_sign)


def x_factor_deg(shoulder_rot_deg: float | None, hip_rot_deg: float | None) -> float | None:
    """Trunk coil: shoulder rotation minus hip rotation. Positive = shoulders more sideways than hips
    (typical "coiled" loading position). Sign convention not independently verified — see module
    docstring; treat as directional-but-approximate."""
    if shoulder_rot_deg is None or hip_rot_deg is None:
        return None
    return shoulder_rot_deg - hip_rot_deg


def rotation_series(
    joints_series: np.ndarray,  # [T, J, 3]
    net_dir_sign: float,
    which: str = "shoulder",
    backend: str = "sam3d_body",
) -> np.ndarray:
    fn = shoulder_rotation_deg if which == "shoulder" else hip_rotation_deg
    out = np.full(len(joints_series), np.nan)
    for t in range(len(joints_series)):
        val = fn(joints_series[t], net_dir_sign, backend)
        out[t] = val if val is not None else np.nan
    return out


def angular_velocity_series(rotation_deg_series: np.ndarray, fps: float) -> np.ndarray:
    """deg/s, forward difference; NaN where either endpoint is missing."""
    out = np.full(len(rotation_deg_series), np.nan)
    for t in range(1, len(rotation_deg_series)):
        a, b = rotation_deg_series[t - 1], rotation_deg_series[t]
        if not np.isnan(a) and not np.isnan(b):
            out[t] = (b - a) * fps
    return out


def peak_abs_velocity_frame(velocity_series: np.ndarray, lo: int, hi: int) -> int | None:
    """Frame of maximum |angular velocity| within [lo, hi] (inclusive). None if all-NaN in range."""
    window = velocity_series[lo : hi + 1]
    if not np.any(~np.isnan(window)):
        return None
    return lo + int(np.nanargmax(np.abs(window)))


def sequence_ms(
    hip_rotation: np.ndarray,
    shoulder_rotation: np.ndarray,
    racket_speed: np.ndarray,
    fps: float,
    window_lo: int,
    window_hi: int,
) -> dict[str, float | None]:
    """Timestamps (ms, relative to `window_lo`) of peak hip angular velocity, peak shoulder angular
    velocity, and peak racket speed within [window_lo, window_hi] — the kinetic-chain check. A good
    sequence has hip -> shoulder -> racket strictly increasing, each 20-80ms apart (spec rubric); this
    function only reports the timestamps, the judge/rubric interprets them."""
    hip_vel = angular_velocity_series(hip_rotation, fps)
    shoulder_vel = angular_velocity_series(shoulder_rotation, fps)

    hip_t = peak_abs_velocity_frame(hip_vel, window_lo, window_hi)
    shoulder_t = peak_abs_velocity_frame(shoulder_vel, window_lo, window_hi)

    racket_window = racket_speed[window_lo : window_hi + 1]
    racket_t = None
    if np.any(~np.isnan(racket_window)):
        racket_t = window_lo + int(np.nanargmax(np.nan_to_num(racket_window, nan=-np.inf)))

    def to_ms(t: int | None) -> float | None:
        return None if t is None else (t - window_lo) * 1000.0 / fps

    return {"hip_ms": to_ms(hip_t), "shoulder_ms": to_ms(shoulder_t), "racket_ms": to_ms(racket_t)}


def trunk_lean_3d_deg(joints: np.ndarray, backend: str = "sam3d_body") -> float | None:
    """Angle (degrees) of the hip-mid -> shoulder-mid line from vertical (Y axis). 0 = perfectly
    upright. Unsigned (direction of lean isn't disambiguated in 3D without a second reference axis)."""
    ls_i, rs_i, lh_i, rh_i = _indices_for_backend(backend)
    ls, rs = joints[ls_i], joints[rs_i]
    lh, rh = joints[lh_i], joints[rh_i]
    if not all(_valid3(p) for p in (ls, rs, lh, rh)):
        return None
    mid_shoulder = (ls[:3] + rs[:3]) / 2
    mid_hip = (lh[:3] + rh[:3]) / 2
    v = mid_shoulder - mid_hip
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    up = np.array([0.0, -1.0, 0.0])  # Y increases downward — see module docstring
    cos_a = np.clip(np.dot(v, up) / n, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def compute_swing_metrics_3d(
    swing: "Swing",  # noqa: F821 - avoid a hard import cycle; badminton_coach.schema.Swing at runtime
    dense_joints: np.ndarray,  # [T, J, 3], NaN where body3d didn't cover that frame
    backend: str,
    net_dir_sign: float,
    speed_series: np.ndarray,  # [T], torso-lengths/s (same series metrics.py uses for peak_racket_speed)
    fps: float,
) -> dict[str, float | None]:
    """Flat dict of this swing's 3D rotation metrics (rubric §"kinetic chain"), named to match
    `metrics.py`'s `_prep_end`/`_contact` convention so they merge into the same `metrics` dict as every
    2D metric and get the same reference-comparison treatment. All `None` if `dense_joints` has no valid
    data in this swing's window (e.g. `backend="none"`)."""
    prep_end, contact = swing.prep_end_frame, swing.contact_frame

    def at(frame: int | None) -> np.ndarray | None:
        if frame is None or frame < 0 or frame >= len(dense_joints):
            return None
        return dense_joints[frame]

    j_prep, j_contact = at(prep_end), at(contact)

    shoulder_prep = shoulder_rotation_deg(j_prep, net_dir_sign, backend) if j_prep is not None else None
    shoulder_contact = shoulder_rotation_deg(j_contact, net_dir_sign, backend) if j_contact is not None else None
    hip_prep = hip_rotation_deg(j_prep, net_dir_sign, backend) if j_prep is not None else None
    hip_contact = hip_rotation_deg(j_contact, net_dir_sign, backend) if j_contact is not None else None

    out: dict[str, float | None] = {
        "shoulder_rotation_deg_prep_end": shoulder_prep,
        "shoulder_rotation_deg_contact": shoulder_contact,
        "hip_rotation_deg_prep_end": hip_prep,
        "hip_rotation_deg_contact": hip_contact,
        "x_factor_deg_prep_end": x_factor_deg(shoulder_prep, hip_prep),
        "x_factor_deg_contact": x_factor_deg(shoulder_contact, hip_contact),
        "trunk_lean_3d_deg_contact": trunk_lean_3d_deg(j_contact, backend) if j_contact is not None else None,
    }

    window_lo, window_hi = swing.prep_start_frame, contact
    if window_hi > window_lo:
        hip_series = rotation_series(dense_joints[window_lo : window_hi + 1], net_dir_sign, "hip", backend)
        shoulder_series = rotation_series(dense_joints[window_lo : window_hi + 1], net_dir_sign, "shoulder", backend)
        seq = sequence_ms(
            hip_series, shoulder_series, speed_series[window_lo : window_hi + 1], fps, 0, window_hi - window_lo
        )
        out["sequence_hip_ms"] = seq["hip_ms"]
        out["sequence_shoulder_ms"] = seq["shoulder_ms"]
        out["sequence_racket_ms"] = seq["racket_ms"]
    else:
        out["sequence_hip_ms"] = None
        out["sequence_shoulder_ms"] = None
        out["sequence_racket_ms"] = None

    return out
