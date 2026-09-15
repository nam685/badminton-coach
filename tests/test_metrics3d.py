from __future__ import annotations

import numpy as np
import pytest

from badminton_coach import metrics3d as m3d


def _joints(shoulder_angle_deg: float, hip_angle_deg: float = 0.0) -> np.ndarray:
    """A minimal MHR70-shaped joint array (11 joints is enough to cover indices 0-10) with the
    shoulder/hip lines rotated by the given angles (degrees) in the horizontal X-Z plane, net direction
    assumed along +X. angle=0 -> line along X (square to net); angle=90 -> line along Z (sideways)."""
    j = np.full((11, 3), np.nan)
    half = 20.0

    def line(center_x: float, center_z: float, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
        rad = np.radians(angle_deg)
        dx, dz = half * np.cos(rad), half * np.sin(rad)
        left = np.array([center_x - dx, 100.0, center_z - dz])
        right = np.array([center_x + dx, 100.0, center_z + dz])
        return left, right

    j[m3d.MHR70_LEFT_SHOULDER], j[m3d.MHR70_RIGHT_SHOULDER] = line(0, 0, shoulder_angle_deg)
    j[m3d.MHR70_LEFT_HIP], j[m3d.MHR70_RIGHT_HIP] = line(0, -50, hip_angle_deg)
    return j


def test_shoulder_rotation_square_to_net_is_zero() -> None:
    j = _joints(shoulder_angle_deg=0.0)
    assert m3d.shoulder_rotation_deg(j, net_dir_sign=1.0) == pytest.approx(0.0, abs=1e-6)


def test_shoulder_rotation_sideways_is_90() -> None:
    j = _joints(shoulder_angle_deg=90.0)
    assert m3d.shoulder_rotation_deg(j, net_dir_sign=1.0) == pytest.approx(90.0, abs=1e-6)


def test_shoulder_rotation_45() -> None:
    j = _joints(shoulder_angle_deg=45.0)
    assert m3d.shoulder_rotation_deg(j, net_dir_sign=1.0) == pytest.approx(45.0, abs=1e-6)


def test_rotation_invariant_to_net_dir_sign() -> None:
    """Rotation *magnitude* (how turned) doesn't depend on which side the net is on — only x_factor's
    sign convention (not tested here) would."""
    j = _joints(shoulder_angle_deg=60.0)
    assert m3d.shoulder_rotation_deg(j, net_dir_sign=1.0) == pytest.approx(
        m3d.shoulder_rotation_deg(j, net_dir_sign=-1.0), abs=1e-6
    )


def test_shoulder_rotation_missing_joint_is_none() -> None:
    j = _joints(shoulder_angle_deg=30.0)
    j[m3d.MHR70_LEFT_SHOULDER] = np.nan
    assert m3d.shoulder_rotation_deg(j, net_dir_sign=1.0) is None


def test_hip_rotation_independent_of_shoulder() -> None:
    j = _joints(shoulder_angle_deg=80.0, hip_angle_deg=20.0)
    assert m3d.shoulder_rotation_deg(j, net_dir_sign=1.0) == pytest.approx(80.0, abs=1e-6)
    assert m3d.hip_rotation_deg(j, net_dir_sign=1.0) == pytest.approx(20.0, abs=1e-6)


def test_x_factor_positive_when_shoulders_more_turned() -> None:
    assert m3d.x_factor_deg(80.0, 20.0) == pytest.approx(60.0)
    assert m3d.x_factor_deg(20.0, 20.0) == pytest.approx(0.0)


def test_x_factor_none_propagates() -> None:
    assert m3d.x_factor_deg(None, 20.0) is None
    assert m3d.x_factor_deg(80.0, None) is None


def test_rotation_series_matches_per_frame() -> None:
    joints_series = np.stack([_joints(shoulder_angle_deg=a) for a in (0, 30, 60, 90)])
    series = m3d.rotation_series(joints_series, net_dir_sign=1.0, which="shoulder")
    np.testing.assert_allclose(series, [0, 30, 60, 90], atol=1e-6)


def test_angular_velocity_series_basic() -> None:
    rot = np.array([0.0, 10.0, 20.0, 20.0])
    vel = m3d.angular_velocity_series(rot, fps=10.0)
    assert np.isnan(vel[0])
    assert vel[1] == pytest.approx(100.0)  # 10 deg * 10 fps
    assert vel[3] == pytest.approx(0.0)


def test_angular_velocity_series_nan_on_gap() -> None:
    rot = np.array([0.0, np.nan, 20.0])
    vel = m3d.angular_velocity_series(rot, fps=10.0)
    assert np.isnan(vel[1])
    assert np.isnan(vel[2])


def test_peak_abs_velocity_frame_finds_peak() -> None:
    vel = np.array([np.nan, 5.0, -50.0, 10.0, 3.0])
    assert m3d.peak_abs_velocity_frame(vel, 0, 4) == 2


def test_peak_abs_velocity_frame_all_nan_returns_none() -> None:
    vel = np.full(5, np.nan)
    assert m3d.peak_abs_velocity_frame(vel, 0, 4) is None


def test_sequence_ms_orders_hip_before_shoulder_before_racket() -> None:
    n = 30
    fps = 30.0
    # hip rotates fast around frame 5, shoulder around frame 10, racket peaks around frame 15
    hip = np.concatenate([np.linspace(0, 60, 10), np.full(n - 10, 60.0)])
    shoulder = np.concatenate([np.full(8, 0.0), np.linspace(0, 80, 10), np.full(n - 18, 80.0)])
    racket_speed = np.zeros(n)
    racket_speed[15] = 50.0

    result = m3d.sequence_ms(hip, shoulder, racket_speed, fps, window_lo=0, window_hi=n - 1)
    assert result["hip_ms"] is not None
    assert result["shoulder_ms"] is not None
    assert result["racket_ms"] is not None
    assert result["hip_ms"] < result["shoulder_ms"] < result["racket_ms"]


def test_sequence_ms_missing_data_returns_none_entries() -> None:
    n = 10
    hip = np.full(n, np.nan)
    shoulder = np.full(n, np.nan)
    racket_speed = np.full(n, np.nan)
    result = m3d.sequence_ms(hip, shoulder, racket_speed, 30.0, 0, n - 1)
    assert result == {"hip_ms": None, "shoulder_ms": None, "racket_ms": None}


def test_trunk_lean_3d_upright_is_zero() -> None:
    # Y increases downward (confirmed against real SAM 3D Body output — see docs/body3d.md): shoulders
    # (physically higher) get the *smaller* Y.
    j = np.full((11, 3), np.nan)
    j[m3d.MHR70_LEFT_SHOULDER] = [-10, 100, 0]
    j[m3d.MHR70_RIGHT_SHOULDER] = [10, 100, 0]
    j[m3d.MHR70_LEFT_HIP] = [-10, 150, 0]
    j[m3d.MHR70_RIGHT_HIP] = [10, 150, 0]
    assert m3d.trunk_lean_3d_deg(j) == pytest.approx(0.0, abs=1e-6)


def test_trunk_lean_3d_tilted() -> None:
    j = np.full((11, 3), np.nan)
    j[m3d.MHR70_LEFT_SHOULDER] = [-10 + 20, 100, 0]
    j[m3d.MHR70_RIGHT_SHOULDER] = [10 + 20, 100, 0]
    j[m3d.MHR70_LEFT_HIP] = [-10, 150, 0]
    j[m3d.MHR70_RIGHT_HIP] = [10, 150, 0]
    val = m3d.trunk_lean_3d_deg(j)
    assert val is not None and val > 0
