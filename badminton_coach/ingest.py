"""Stage 0 — ingest: turn one or more raw phone clips into a single rotation-corrected `normalized.mp4`
plus `ingest.json` recording per-attempt facts (spec §3.1).

Multiple input files are treated as multiple attempts of the same session: each is individually
rotation-corrected and resampled to a common fps/resolution (the first input's, after rotation), then
concatenated frame-accurately. Per-attempt `frame_offset` lets downstream stages know which frames of
the combined video belong to which original file.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from badminton_coach.config import CONFIG, Config
from badminton_coach.run import RunDir
from badminton_coach.video import VideoInfo, concat_videos, normalize_rotation, probe

STAGE_VERSION = "1"


@dataclass
class IngestAttempt:
    file: str
    sha256: str
    fps: float
    width: int
    height: int
    rotation_deg: int
    duration_s: float
    frame_offset: int
    frame_count: int


@dataclass
class IngestResult:
    attempts: list[IngestAttempt]
    normalized_path: str
    fps: float
    width: int
    height: int
    n_frames: int

    def to_json(self) -> dict:
        return {
            "attempts": [asdict(a) for a in self.attempts],
            "normalized_path": self.normalized_path,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "n_frames": self.n_frames,
        }

    @classmethod
    def from_json(cls, data: dict) -> IngestResult:
        return cls(
            attempts=[IngestAttempt(**a) for a in data["attempts"]],
            normalized_path=data["normalized_path"],
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            n_frames=data["n_frames"],
        )


def _combined_input_hash(infos: list[VideoInfo]) -> str:
    h = hashlib.sha256()
    for info in infos:
        h.update(info.sha256.encode())
    return h.hexdigest()


def ingest(inputs: list[str | Path], run: RunDir, cfg: Config = CONFIG, force: bool = False) -> IngestResult:
    """Run the ingest stage, writing `<run>/ingest.json` and (when needed) `<run>/normalized.mp4` /
    `<run>/ingest/attempt_<i>.mp4`. Idempotent/cached via `RunDir.stage`."""
    if not inputs:
        raise ValueError("ingest requires at least one input video")

    infos = [probe(p, cfg.ffprobe_bin) for p in inputs]
    input_hash = _combined_input_hash(infos)

    with run.stage("ingest", version=STAGE_VERSION, input_hash=input_hash, force=force) as st:
        if st.skip:
            data = run.load_run_json()
            return IngestResult.from_json(data["ingest"])

        if len(infos) == 1 and infos[0].rotation_deg == 0:
            info = infos[0]
            result = IngestResult(
                attempts=[
                    IngestAttempt(
                        file=str(info.path),
                        sha256=info.sha256,
                        fps=info.fps,
                        width=info.width,
                        height=info.height,
                        rotation_deg=0,
                        duration_s=info.duration_s,
                        frame_offset=0,
                        frame_count=info.n_frames,
                    )
                ],
                normalized_path=str(info.path),
                fps=info.fps,
                width=info.width,
                height=info.height,
                n_frames=info.n_frames,
            )
        else:
            first = infos[0]
            target_fps = first.fps
            target_w, target_h = first.display_width, first.display_height

            attempts_dir = run.root / "ingest"
            attempts_dir.mkdir(parents=True, exist_ok=True)
            segment_paths: list[Path] = []
            attempts: list[IngestAttempt] = []
            frame_offset = 0
            for i, info in enumerate(infos):
                seg_path = attempts_dir / f"attempt_{i}.mp4"
                normalize_rotation(
                    info.path,
                    seg_path,
                    rotation_deg=info.rotation_deg,
                    fps=target_fps,
                    width=target_w,
                    height=target_h,
                    ffmpeg_bin=cfg.ffmpeg_bin,
                )
                seg_info = probe(seg_path, cfg.ffprobe_bin)
                attempts.append(
                    IngestAttempt(
                        file=str(info.path),
                        sha256=info.sha256,
                        fps=info.fps,
                        width=info.width,
                        height=info.height,
                        rotation_deg=info.rotation_deg,
                        duration_s=info.duration_s,
                        frame_offset=frame_offset,
                        frame_count=seg_info.n_frames,
                    )
                )
                segment_paths.append(seg_path)
                frame_offset += seg_info.n_frames

            normalized_path = run.root / "normalized.mp4"
            if len(segment_paths) == 1:
                segment_paths[0].replace(normalized_path)
            else:
                concat_videos(segment_paths, normalized_path, ffmpeg_bin=cfg.ffmpeg_bin)

            result = IngestResult(
                attempts=attempts,
                normalized_path=str(normalized_path),
                fps=target_fps,
                width=target_w,
                height=target_h,
                n_frames=frame_offset,
            )

        run.update_run_json(ingest=result.to_json())
        return result
