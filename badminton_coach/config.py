"""Central configuration: paths, external binaries, and pipeline default thresholds.

Everything here is overridable via environment variables so the same code runs identically in dev,
in tests (pointed at a tmp dir), and later on the VPS (ticket 2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class Config:
    # --- paths ---
    data_dir: Path = field(default_factory=lambda: _env_path("BADMINTON_COACH_DATA", "./data"))

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def players_dir(self) -> Path:
        return self.data_dir / "players"

    @property
    def references_dir(self) -> Path:
        return self.data_dir / "references"

    @property
    def fixtures_dir(self) -> Path:
        return self.data_dir / "fixtures"

    # --- external binaries ---
    claude_bin: str = field(default_factory=lambda: os.environ.get("CLAUDE_BIN", "claude"))
    ffmpeg_bin: str = field(default_factory=lambda: os.environ.get("FFMPEG_BIN", "ffmpeg"))
    ffprobe_bin: str = field(default_factory=lambda: os.environ.get("FFPROBE_BIN", "ffprobe"))

    # --- video ingest ---
    inference_long_side: int = 1280
    preferred_fps: int = 60

    # --- measurement thresholds ---
    racket_det_score_thr: float = 0.3
    racket_kpt_conf_thr: float = 0.3
    body_kpt_conf_thr: float = 0.3
    low_conf_mask_thr: float = 0.3

    # --- tracking / smoothing ---
    gap_fill_max_frames: int = 5
    savgol_window_60fps: int = 7
    savgol_window_30fps: int = 5
    savgol_polyorder: int = 2

    # --- swing segmentation ---
    swing_min_separation_s: float = 0.8
    swing_speed_peak_prominence: float = 0.5  # torso-lengths/s, tuned during calibration (Task 11)
    contact_search_radius_frames: int = 8
    contact_shuttle_max_dist_torso: float = 1.0  # live vs shadow swing threshold
    prep_window_s: float = 0.6
    follow_window_s: float = 0.4

    # --- 3D body ---
    body3d_window_s: float = 0.8
    body3d_consistency_thr_torso: float = 0.15

    # --- judge ---
    judge_model: str = "opus"
    judge_effort: str = "high"
    judge_max_turns: int = 40
    judge_timeout_s: int = 1200
    judge_permission_mode: str = "dontAsk"

    # --- render ---
    slow_mo_factor: int = 4
    slow_mo_window_s: float = 0.25

    # --- language ---
    default_lang: str = "en"
    supported_langs: tuple[str, ...] = ("vi", "de", "en")


CONFIG = Config()
