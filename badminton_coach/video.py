"""Video I/O: ffprobe metadata, rotation-safe frame reading, ffmpeg-pipe encoding.

Rotation handling: rather than relying on OpenCV's build-dependent auto-rotation (inconsistent across
platforms/backends), we always resolve rotation ourselves from ffprobe metadata and, when non-zero,
physically re-encode a rotation-corrected copy via ffmpeg's `transpose`/`flip` filters (no rotation
metadata is copied into the output, so any downstream reader — OpenCV or otherwise — sees already-correct
pixels). See `resolve_rotation` / `normalize_rotation`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from badminton_coach.config import CONFIG


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    sha256: str
    fps: float
    width: int
    height: int
    n_frames: int
    duration_s: float
    rotation_deg: int  # 0, 90, 180, or 270 — degrees to rotate CLOCKWISE to display correctly
    codec_name: str

    @property
    def display_width(self) -> int:
        return self.height if self.rotation_deg in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation_deg in (90, 270) else self.height


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_frame_rate(rate_str: str) -> float:
    try:
        return float(Fraction(rate_str))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _resolve_rotation(stream: dict) -> int:
    """Normalize rotation to one of {0, 90, 180, 270}, meaning degrees clockwise to display correctly."""
    tags = stream.get("tags", {}) or {}
    if "rotate" in tags:
        try:
            deg = int(tags["rotate"]) % 360
            return deg if deg in (0, 90, 180, 270) else 0
        except ValueError:
            pass
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                # ffprobe's Display Matrix rotation is counterclockwise-positive; negate to get our
                # clockwise-positive convention, matching `tags.rotate`.
                deg = int(round(-float(sd["rotation"]))) % 360
                return deg if deg in (0, 90, 180, 270) else 0
            except ValueError:
                pass
    return 0


def probe(path: str | Path, ffprobe_bin: str | None = None) -> VideoInfo:
    """Read video metadata via ffprobe. Raises RuntimeError if ffprobe fails or no video stream is found."""
    path = Path(path)
    ffprobe_bin = ffprobe_bin or CONFIG.ffprobe_bin
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-select_streams",
            "v:0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()[:500]}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream found in {path}")
    stream = streams[0]

    fps = _parse_frame_rate(stream.get("r_frame_rate", "0/1"))
    width = int(stream["width"])
    height = int(stream["height"])
    duration_s = float(data.get("format", {}).get("duration") or stream.get("duration") or 0.0)

    n_frames_raw = stream.get("nb_frames")
    if n_frames_raw is not None:
        n_frames = int(n_frames_raw)
    elif fps > 0 and duration_s > 0:
        n_frames = round(duration_s * fps)
    else:
        n_frames = 0

    rotation_deg = _resolve_rotation(stream)
    codec_name = stream.get("codec_name", "unknown")

    return VideoInfo(
        path=path,
        sha256=sha256_file(path),
        fps=fps,
        width=width,
        height=height,
        n_frames=n_frames,
        duration_s=duration_s,
        rotation_deg=rotation_deg,
        codec_name=codec_name,
    )


_TRANSPOSE_FILTERS = {
    90: "transpose=1",  # 90 degrees clockwise
    180: "hflip,vflip",
    270: "transpose=2",  # 90 degrees counterclockwise
}


def normalize_rotation(
    in_path: str | Path,
    out_path: str | Path,
    rotation_deg: int,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    ffmpeg_bin: str | None = None,
) -> None:
    """Re-encode `in_path` into `out_path` with rotation baked into the pixels (no rotation metadata
    on the output) and, optionally, resampled to `fps` and/or scaled to `width`x`height`.

    No-ops are not handled here — callers should skip calling this when no transform is needed.
    """
    if rotation_deg not in (0, 90, 180, 270):
        raise ValueError(f"rotation_deg must be one of 0/90/180/270, got {rotation_deg}")
    ffmpeg_bin = ffmpeg_bin or CONFIG.ffmpeg_bin

    vf_parts: list[str] = []
    if rotation_deg != 0:
        vf_parts.append(_TRANSPOSE_FILTERS[rotation_deg])
    if width is not None and height is not None:
        vf_parts.append(f"scale={width}:{height}")
    if fps is not None:
        vf_parts.append(f"fps={fps}")

    cmd = [ffmpeg_bin, "-y", "-i", str(in_path)]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-an", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", str(out_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg normalize failed on {in_path}: {result.stderr.strip()[-1000:]}")


def concat_videos(segment_paths: list[str | Path], out_path: str | Path, ffmpeg_bin: str | None = None) -> None:
    """Concatenate already-matching (same codec/fps/resolution) video segments via the ffmpeg concat
    demuxer (stream copy, frame-accurate, no re-encode)."""
    ffmpeg_bin = ffmpeg_bin or CONFIG.ffmpeg_bin
    out_path = Path(out_path)
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in segment_paths))
    try:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip()[-1000:]}")
    finally:
        list_path.unlink(missing_ok=True)


def iter_frames(
    path: str | Path, scale_long_side: int | None = None, start_frame: int = 0, end_frame: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, bgr_frame) for a video already rotation-corrected (no metadata rotation).

    `scale_long_side`: if set, downscale so the longer side equals this value (aspect preserved),
    used for inference resolution. `None` keeps original resolution (used for rendering).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path}")
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        while True:
            if end_frame is not None and idx >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if scale_long_side is not None:
                frame = _scale_long_side(frame, scale_long_side)
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def _scale_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= long_side:
        return frame
    scale = long_side / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def read_frame(path: str | Path, index: int, scale_long_side: int | None = None) -> np.ndarray:
    """Random-access read of a single frame by index."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {index} from {path}")
        if scale_long_side is not None:
            frame = _scale_long_side(frame, scale_long_side)
        return frame
    finally:
        cap.release()


def write_video(
    frames: Iterator[np.ndarray], out_path: str | Path, fps: float, ffmpeg_bin: str | None = None
) -> None:
    """Encode a stream of BGR frames to an h264/yuv420p mp4 via an ffmpeg pipe."""
    ffmpeg_bin = ffmpeg_bin or CONFIG.ffmpeg_bin
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = next(frames, None)
    if first is None:
        raise ValueError("no frames to write")
    height, width = first.shape[:2]

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        proc.stdin.write(first.tobytes())
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {stderr.decode(errors='replace')[-1000:]}")
