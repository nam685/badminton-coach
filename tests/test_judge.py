from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from badminton_coach.config import Config
from badminton_coach.judge.run import build_argv, run_judge
from badminton_coach.judge.workspace import build_history_md, build_workspace
from badminton_coach.measure.keypoints import BODY_NUM_KPTS, RACKET_NUM_KPTS
from badminton_coach.run import RunDir

READONLY_TOOLS_EXPECTED = {"Read", "Grep", "Glob"}


def test_build_argv_contains_required_flags() -> None:
    argv = build_argv("task text", "opus", "claude", "high", 40, '{"type":"object"}')
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "task text" in argv
    assert "--json-schema" in argv
    assert '{"type":"object"}' in argv
    assert "--allowedTools" in argv
    tools_idx = argv.index("--allowedTools")
    tools = argv[tools_idx + 1 :]
    # everything after --allowedTools up to the next flag
    tools = tools[: tools.index("--permission-mode")]
    assert READONLY_TOOLS_EXPECTED.issubset(set(tools))
    assert any("badminton-coach frames" in t for t in tools)
    assert "--strict-mcp-config" in argv
    assert "--max-turns" in argv
    assert "40" in argv


def _fake_stream_json_result(findings_dict: dict) -> str:
    result_msg = {"type": "result", "result": json.dumps(findings_dict), "model": "opus", "num_turns": 3}
    return json.dumps({"type": "system"}) + "\n" + json.dumps(result_msg) + "\n"


def _valid_findings() -> dict:
    return {
        "player": "nam",
        "shot": "clear",
        "handedness": "right",
        "measurement_quality": {"ok": True, "notes": "fine"},
        "swings": [{"index": 0, "mode": "live", "contact_frame": 10, "contact_time_s": 0.33, "one_line_verdict": "ok"}],
        "strengths": ["good footwork"],
        "findings": [
            {
                "id": "f1",
                "title": "Elbow bent at contact",
                "category": "elbow",
                "severity": 2,
                "pattern": "consistent",
                "evidence": {
                    "swings": [0],
                    "frames": [{"swing": 0, "frame": 10}],
                    "metrics": [{"name": "elbow_angle_contact", "value": 130.0, "reference": 160.0}],
                    "visual": "elbow clearly bent in the crop",
                },
                "explanation": "the arm is not extending enough at contact",
                "cue": "reach up and through the shuttle",
                "drill": "shadow swings focusing on full extension",
            }
        ],
        "priority": ["f1"],
        "progress_vs_history": None,
        "confidence": "medium",
    }


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def test_run_judge_success(cfg: Config, monkeypatch) -> None:
    run = RunDir.create("nam", cfg=cfg)
    workspace = run.root / "judge" / "workspace"
    workspace.mkdir(parents=True)

    fake_stdout = _fake_stream_json_result(_valid_findings())

    def fake_run(argv, cwd, capture_output, text, timeout):  # noqa: ARG001
        return subprocess.CompletedProcess(argv, 0, stdout=fake_stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_judge(run, workspace, lang="en", cfg=cfg)
    assert result["status"] == "done"
    assert result["n_findings"] == 1
    findings_path = run.root / "judge" / "findings.json"
    assert findings_path.exists()
    data = json.loads(findings_path.read_text())
    assert data["findings"][0]["title"] == "Elbow bent at contact"


def test_run_judge_nonzero_exit_fails_gracefully(cfg: Config, monkeypatch) -> None:
    run = RunDir.create("nam", cfg=cfg)
    workspace = run.root / "judge" / "workspace"
    workspace.mkdir(parents=True)

    def fake_run(argv, cwd, capture_output, text, timeout):  # noqa: ARG001
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_judge(run, workspace, cfg=cfg)
    assert result["status"] == "failed"
    assert (run.root / "judge" / "error.txt").exists()
    assert not (run.root / "judge" / "findings.json").exists()


def test_run_judge_invalid_schema_fails_gracefully(cfg: Config, monkeypatch) -> None:
    run = RunDir.create("nam", cfg=cfg)
    workspace = run.root / "judge" / "workspace"
    workspace.mkdir(parents=True)

    bad = _valid_findings()
    del bad["handedness"]  # required field missing
    fake_stdout = _fake_stream_json_result(bad)

    def fake_run(argv, cwd, capture_output, text, timeout):  # noqa: ARG001
        return subprocess.CompletedProcess(argv, 0, stdout=fake_stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_judge(run, workspace, cfg=cfg)
    assert result["status"] == "failed"
    assert result["reason"] == "invalid_schema"
    assert (run.root / "judge" / "raw_result.txt").exists()


def test_run_judge_timeout_fails_gracefully(cfg: Config, monkeypatch) -> None:
    run = RunDir.create("nam", cfg=cfg)
    workspace = run.root / "judge" / "workspace"
    workspace.mkdir(parents=True)

    def fake_run(argv, cwd, capture_output, text, timeout):  # noqa: ARG001
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_judge(run, workspace, cfg=cfg)
    assert result["status"] == "failed"
    assert result["reason"] == "timeout"


def test_run_judge_is_cached(cfg: Config, monkeypatch) -> None:
    run = RunDir.create("nam", cfg=cfg)
    workspace = run.root / "judge" / "workspace"
    workspace.mkdir(parents=True)
    calls = []

    def fake_run(argv, cwd, capture_output, text, timeout):  # noqa: ARG001
        calls.append(1)
        return subprocess.CompletedProcess(argv, 0, stdout=_fake_stream_json_result(_valid_findings()), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_judge(run, workspace, cfg=cfg)
    run_judge(run, workspace, cfg=cfg)
    assert len(calls) == 1


def test_build_history_md_empty_for_new_player(cfg: Config) -> None:
    assert build_history_md("brand-new-player", cfg) == ""


def test_build_history_md_summarizes_prior_findings(cfg: Config) -> None:
    old_run = RunDir.create("nam", cfg=cfg, slug="old")
    (old_run.root / "judge").mkdir(parents=True)
    (old_run.root / "judge" / "findings.json").write_text(json.dumps(_valid_findings()))

    history = build_history_md("nam", cfg)
    assert "Elbow bent at contact" in history
    assert "good footwork" in history


def test_build_history_md_excludes_current_run(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg, slug="current")
    (run.root / "judge").mkdir(parents=True)
    (run.root / "judge" / "findings.json").write_text(json.dumps(_valid_findings()))

    history = build_history_md("nam", cfg, exclude_run=run)
    assert history == ""


def _make_video(path: Path, n_frames=30, fps=30, w=320, h=240) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={w}x{h}:rate={fps}:duration={n_frames / fps}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _populate_run_for_workspace(run: RunDir, n: int = 30) -> None:
    video_path = run.root / "clip.mp4"
    _make_video(video_path, n_frames=n)
    run.update_run_json(ingest={"normalized_path": str(video_path), "width": 320, "height": 240})

    player = np.full((n, BODY_NUM_KPTS, 2), 100.0)
    racket = np.full((n, RACKET_NUM_KPTS, 2), 100.0)
    (run.root / "track").mkdir(parents=True)
    np.savez_compressed(run.root / "track" / "player.npz", raw=player, smooth=player, conf=np.ones((n, BODY_NUM_KPTS)))
    np.savez_compressed(
        run.root / "track" / "racket.npz", raw=racket, smooth=racket, conf=np.ones((n, RACKET_NUM_KPTS))
    )
    (run.root / "track" / "track.json").write_text(json.dumps({"handedness": "right", "net_side": "right"}))

    swing = {
        "index": 0,
        "mode": "live",
        "contact_frame": 15,
        "contact_time_s": 0.5,
        "contact_source": "shuttle",
        "prep_start_frame": 5,
        "prep_end_frame": 12,
        "follow_end_frame": 20,
    }
    (run.root / "swings").mkdir(parents=True)
    (run.root / "swings" / "swings.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "handedness": "right",
                "net_side": "right",
                "net_side_source": "unknown",
                "torso_len_median_px": 80.0,
                "speed_source": "racket_top",
                "swings": [swing],
            }
        )
    )

    (run.root / "metrics").mkdir(parents=True)
    (run.root / "metrics" / "metrics.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "handedness": "right",
                "net_side": "right",
                "torso_len_median_px": 80.0,
                "swings": [],
                "cross_attempt_summary": {},
            }
        )
    )


def test_build_workspace_creates_expected_layout(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    _populate_run_for_workspace(run)

    workspace = build_workspace(run, cfg=cfg, references=[], lang="vi", player="nam")

    assert (workspace / "metrics.json").exists()
    assert (workspace / "swings" / "0" / "contact_sheet.png").exists()
    assert (workspace / "swings" / "0" / "crops").is_dir()
    assert len(list((workspace / "swings" / "0" / "crops").iterdir())) > 0
    assert (workspace / "rubric.md").exists()
    assert (workspace / "history.md").exists()
    assert (workspace / "TASK.md").exists()
    assert "Vietnamese" in (workspace / "TASK.md").read_text()
    assert not (workspace / "CLAUDE.md").exists()
