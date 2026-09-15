"""Download and trim a short pro badminton-clear clip for use as a test fixture.

Personal-use only, per the design spec: the fixture lives under `data/fixtures/` (gitignored, never
redistributed) and is used solely to exercise the pipeline against real footage during development.

Run directly: `uv run python tests/fixtures/make_fixture.py [--start SS] [--duration SS]`
Tests that need it (`@pytest.mark.models`) call `ensure_fixture()` and skip if it's absent.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from badminton_coach.config import CONFIG

FIXTURE_URL = "https://www.youtube.com/watch?v=2bK27mv1Fq4"
FIXTURE_NAME = "clear_3s.mp4"
# Manually reviewed 2026-09-15: the source is "Clear Backcourt with Start" ultra-slow-motion footage;
# a title card covers 0-7s, the demonstrated swing (split-step -> contact -> follow-through) runs ~7-14s.
DEFAULT_START_S = 6.5
DEFAULT_DURATION_S = 8.0
SCALE_LONG_SIDE = 640


def fixture_path(cfg=CONFIG) -> Path:
    return cfg.fixtures_dir / FIXTURE_NAME


def ensure_fixture(cfg=CONFIG) -> Path | None:
    """Return the fixture path if it already exists on disk, else None (never downloads implicitly —
    tests should skip rather than trigger a network fetch)."""
    path = fixture_path(cfg)
    return path if path.exists() else None


def make_fixture(start_s: float = DEFAULT_START_S, duration_s: float = DEFAULT_DURATION_S, cfg=CONFIG) -> Path:
    cfg.fixtures_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cfg.fixtures_dir / "_raw_source.mp4"
    out_path = fixture_path(cfg)

    if not raw_path.exists():
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                # Prefer avc1 (H.264) explicitly, not just any mp4 -- an av01 (AV1) stream inside an mp4
                # container decoded cleanly for single-frame seeks but silently produced 0 frames when
                # decoded sequentially by the real pipeline on this machine. See badminton_coach/reference.py.
                "bestvideo[height<=720][vcodec^=avc1][ext=mp4]"
                "/best[height<=720][vcodec^=avc1][ext=mp4]"
                "/bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]/best[height<=720]",
                "-o",
                str(raw_path),
                FIXTURE_URL,
            ],
            check=True,
        )

    subprocess.run(
        [
            cfg.ffmpeg_bin,
            "-y",
            "-ss",
            str(start_s),
            "-i",
            str(raw_path),
            "-t",
            str(duration_s),
            "-vf",
            f"scale={SCALE_LONG_SIDE}:-2",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        check=True,
    )
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=DEFAULT_START_S)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    args = parser.parse_args()
    path = make_fixture(start_s=args.start, duration_s=args.duration)
    print(f"fixture written to {path}")
