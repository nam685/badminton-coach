from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from badminton_coach import video


def _make_synthetic_clip(
    path: Path,
    width: int = 320,
    height: int = 240,
    duration_s: float = 1.0,
    fps: int = 10,
    rotate_tag: int | None = None,
) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}:duration={duration_s}",
    ]
    if rotate_tag is not None:
        cmd += ["-metadata:s:v:0", f"rotate={rotate_tag}"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def test_probe_basic(tmp_path: Path) -> None:
    clip = _make_synthetic_clip(tmp_path / "clip.mp4", width=320, height=240, duration_s=1.0, fps=10)
    info = video.probe(clip)
    assert info.width == 320
    assert info.height == 240
    assert abs(info.fps - 10) < 0.01
    assert info.rotation_deg == 0
    assert 8 <= info.n_frames <= 12
    assert len(info.sha256) == 64


def test_display_width_height_swap_for_90_270() -> None:
    from badminton_coach.video import VideoInfo

    info = VideoInfo(
        path=Path("x.mp4"),
        sha256="0" * 64,
        fps=30.0,
        width=1920,
        height=1080,
        n_frames=10,
        duration_s=1.0,
        rotation_deg=90,
        codec_name="h264",
    )
    assert info.display_width == 1080
    assert info.display_height == 1920
    info180 = info.__class__(**{**info.__dict__, "rotation_deg": 180})
    assert info180.display_width == 1920
    assert info180.display_height == 1080


@pytest.mark.parametrize(
    "stream, expected",
    [
        ({"tags": {"rotate": "90"}}, 90),
        ({"tags": {"rotate": "180"}}, 180),
        ({"tags": {"rotate": "270"}}, 270),
        ({"tags": {"rotate": "0"}}, 0),
        ({}, 0),
        # ffprobe's Display Matrix side_data is counterclockwise-positive; our convention is
        # clockwise-positive, so rotation=-90 (i.e. 90 CW) maps to our rotation_deg=90.
        ({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90.0}]}, 90),
        ({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 90.0}]}, 270),
        ({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 180.0}]}, 180),
        ({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 0.0}]}, 0),
    ],
)
def test_resolve_rotation_parsing(stream: dict, expected: int) -> None:
    assert video._resolve_rotation(stream) == expected


def test_normalize_rotation_bakes_pixels_and_clears_tag(tmp_path: Path) -> None:
    # This ffmpeg build doesn't honor `-metadata:s:v:0 rotate=N` for mp4 muxing, so we can't produce
    # a real rotation-tagged input file to round-trip through probe(); instead we exercise
    # normalize_rotation() directly with an explicit rotation_deg (exactly how ingest.py calls it —
    # rotation_deg always comes from probe()'s already-parsed VideoInfo, never re-derived from the
    # output). See test_resolve_rotation_parsing above for the tag-parsing logic itself.
    clip = _make_synthetic_clip(tmp_path / "clip.mp4", width=320, height=240, duration_s=0.5)
    out = tmp_path / "normalized.mp4"
    video.normalize_rotation(clip, out, rotation_deg=90, width=240, height=320)
    out_info = video.probe(out)
    assert out_info.rotation_deg == 0
    assert out_info.width == 240
    assert out_info.height == 320


def test_normalize_rotation_180(tmp_path: Path) -> None:
    clip = _make_synthetic_clip(tmp_path / "clip.mp4", width=320, height=240, duration_s=0.5)
    out = tmp_path / "normalized.mp4"
    video.normalize_rotation(clip, out, rotation_deg=180, width=320, height=240)
    out_info = video.probe(out)
    assert out_info.rotation_deg == 0
    assert out_info.width == 320
    assert out_info.height == 240


def test_iter_frames_count_and_scale(tmp_path: Path) -> None:
    clip = _make_synthetic_clip(tmp_path / "clip.mp4", width=640, height=480, duration_s=1.0, fps=10)
    frames = list(video.iter_frames(clip, scale_long_side=320))
    assert 8 <= len(frames) <= 12
    idx0, frame0 = frames[0]
    assert idx0 == 0
    assert max(frame0.shape[:2]) == 320
    assert frame0.shape[2] == 3


def test_iter_frames_start_end_range(tmp_path: Path) -> None:
    clip = _make_synthetic_clip(tmp_path / "clip.mp4", width=320, height=240, duration_s=2.0, fps=10)
    frames = list(video.iter_frames(clip, start_frame=5, end_frame=10))
    indices = [i for i, _ in frames]
    assert indices == list(range(5, 10))


def test_read_frame_single(tmp_path: Path) -> None:
    clip = _make_synthetic_clip(tmp_path / "clip.mp4", width=320, height=240, duration_s=1.0, fps=10)
    frame = video.read_frame(clip, 3)
    assert frame.shape == (240, 320, 3)


def test_write_video_roundtrip(tmp_path: Path) -> None:
    frames = (np.full((100, 100, 3), i * 20 % 256, dtype=np.uint8) for i in range(15))
    out = tmp_path / "out.mp4"
    video.write_video(frames, out, fps=10)
    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    assert 13 <= count <= 15


def test_concat_videos(tmp_path: Path) -> None:
    a = _make_synthetic_clip(tmp_path / "a.mp4", width=160, height=120, duration_s=0.5, fps=10)
    b = _make_synthetic_clip(tmp_path / "b.mp4", width=160, height=120, duration_s=0.5, fps=10)
    out = tmp_path / "concat.mp4"
    video.concat_videos([a, b], out)
    info = video.probe(out)
    info_a = video.probe(a)
    info_b = video.probe(b)
    assert info.n_frames == pytest.approx(info_a.n_frames + info_b.n_frames, abs=1)
