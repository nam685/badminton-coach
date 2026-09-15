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
"""

from __future__ import annotations

import math

import numpy as np

MHR70_LEFT_SHOULDER = 5
MHR70_RIGHT_SHOULDER = 6
MHR70_LEFT_HIP = 9
MHR70_RIGHT_HIP = 10


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


def shoulder_rotation_deg(joints: np.ndarray, net_dir_sign: float) -> float | None:
    """`joints`: [J, 3] (single frame, MHR70 order)."""
    return horizontal_rotation_deg(joints[MHR70_LEFT_SHOULDER], joints[MHR70_RIGHT_SHOULDER], net_dir_sign)


def hip_rotation_deg(joints: np.ndarray, net_dir_sign: float) -> float | None:
    return horizontal_rotation_deg(joints[MHR70_LEFT_HIP], joints[MHR70_RIGHT_HIP], net_dir_sign)


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
) -> np.ndarray:
    fn = shoulder_rotation_deg if which == "shoulder" else hip_rotation_deg
    out = np.full(len(joints_series), np.nan)
    for t in range(len(joints_series)):
        val = fn(joints_series[t], net_dir_sign)
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


def trunk_lean_3d_deg(joints: np.ndarray) -> float | None:
    """Angle (degrees) of the hip-mid -> shoulder-mid line from vertical (Y axis). 0 = perfectly
    upright. Unsigned (direction of lean isn't disambiguated in 3D without a second reference axis)."""
    ls, rs = joints[MHR70_LEFT_SHOULDER], joints[MHR70_RIGHT_SHOULDER]
    lh, rh = joints[MHR70_LEFT_HIP], joints[MHR70_RIGHT_HIP]
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
