from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import badminton_coach.ingest as ingest_mod
from badminton_coach.config import Config
from badminton_coach.ingest import ingest
from badminton_coach.run import RunDir
from badminton_coach.video import probe as real_probe


def _make_clip(path: Path, width=320, height=240, duration_s=1.0, fps=10, rotate_tag=None) -> Path:
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration_s}"]
    if rotate_tag is not None:
        cmd += ["-metadata:s:v:0", f"rotate={rotate_tag}"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def test_ingest_single_no_rotation_is_passthrough(tmp_path: Path, cfg: Config) -> None:
    clip = _make_clip(tmp_path / "clip.mp4", duration_s=1.0, fps=10)
    run = RunDir.create("nam", cfg=cfg)
    result = ingest([clip], run, cfg=cfg)
    assert result.normalized_path == str(clip)
    assert len(result.attempts) == 1
    assert result.attempts[0].frame_offset == 0
    assert result.n_frames == result.attempts[0].frame_count
    assert (run.root / "run.json").exists()
    assert run.load_run_json()["ingest"]["normalized_path"] == str(clip)


def test_ingest_stores_absolute_path_even_when_given_relative(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    """Real bug found via a live judge run: a relative input path got persisted as-is, so
    `normalized_path` only resolved from the cwd `ingest()` happened to run in — broke `badminton-coach
    frames` when invoked (as the judge does) from a completely different cwd."""
    clip = _make_clip(tmp_path / "clip.mp4", duration_s=0.5, fps=10)
    monkeypatch.chdir(tmp_path)
    run = RunDir.create("nam", cfg=cfg)

    result = ingest(["clip.mp4"], run, cfg=cfg)  # relative to the cwd we just chdir'd into

    assert Path(result.normalized_path).is_absolute()
    assert Path(result.normalized_path) == clip.resolve()
    stored = run.load_run_json()["ingest"]["normalized_path"]
    assert Path(stored).is_absolute()


def test_ingest_single_with_rotation_normalizes(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    # This ffmpeg build doesn't honor `-metadata:s:v:0 rotate=N` for mp4 muxing (verified separately;
    # see tests/test_video.py), so we can't produce a real rotation-tagged input file. Instead, patch
    # probe() to report rotation_deg=90 for an otherwise-unrotated clip, which exercises ingest's
    # "rotation != 0 -> normalize" branch against a real ffmpeg call on real pixel data.
    clip = _make_clip(tmp_path / "clip.mp4", width=320, height=240, duration_s=0.5)

    def fake_probe(path, ffprobe_bin=None):
        info = real_probe(path, ffprobe_bin)
        return info.__class__(**{**info.__dict__, "rotation_deg": 90})

    monkeypatch.setattr(ingest_mod, "probe", fake_probe)

    run = RunDir.create("nam", cfg=cfg)
    result = ingest([clip], run, cfg=cfg)
    assert result.normalized_path != str(clip)
    assert Path(result.normalized_path).exists()
    assert result.width == 240  # swapped due to 90deg rotation
    assert result.height == 320


def test_ingest_multiple_files_concatenates_with_offsets(tmp_path: Path, cfg: Config) -> None:
    a = _make_clip(tmp_path / "a.mp4", width=320, height=240, duration_s=0.5, fps=10)
    b = _make_clip(tmp_path / "b.mp4", width=320, height=240, duration_s=0.5, fps=10)
    run = RunDir.create("nam", cfg=cfg)
    result = ingest([a, b], run, cfg=cfg)
    assert len(result.attempts) == 2
    assert result.attempts[0].frame_offset == 0
    assert result.attempts[1].frame_offset == result.attempts[0].frame_count
    assert result.n_frames == result.attempts[0].frame_count + result.attempts[1].frame_count
    assert Path(result.normalized_path).exists()


def test_ingest_is_cached_on_second_call(tmp_path: Path, cfg: Config) -> None:
    clip = _make_clip(tmp_path / "clip.mp4", duration_s=0.5)
    run = RunDir.create("nam", cfg=cfg)
    result1 = ingest([clip], run, cfg=cfg)
    status_after_first = run.stage_status("ingest")
    result2 = ingest([clip], run, cfg=cfg)
    assert result1.to_json() == result2.to_json()
    assert run.stage_status("ingest") == status_after_first  # not re-run (no new timing recorded)
