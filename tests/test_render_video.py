from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from badminton_coach.config import Config
from badminton_coach.measure.keypoints import BODY_NUM_KPTS, RACKET_NUM_KPTS
from badminton_coach.render.video import _active_finding_for_swing, _highlighted_joints, render_annotated_video
from badminton_coach.run import RunDir


def _make_video(path: Path, n_frames=60, fps=30, w=320, h=240) -> None:
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


def _findings() -> dict:
    return {
        "findings": [
            {
                "id": "f1",
                "title": "Elbow bent at contact",
                "evidence": {
                    "swings": [0],
                    "frames": [{"swing": 0, "frame": 30}],
                    "metrics": [{"name": "elbow_angle_contact", "value": 130.0}],
                },
            }
        ],
        "priority": ["f1"],
    }


def test_highlighted_joints_maps_elbow_metric() -> None:
    from badminton_coach.measure.keypoints import LEFT_ELBOW, RIGHT_ELBOW

    joints = _highlighted_joints([{"evidence": {"metrics": [{"name": "elbow_angle_contact"}]}}])
    assert LEFT_ELBOW in joints
    assert RIGHT_ELBOW in joints


def test_highlighted_joints_empty_for_no_findings() -> None:
    assert _highlighted_joints([]) == set()


def test_active_finding_for_swing_picks_priority_order() -> None:
    data = _findings()
    result = _active_finding_for_swing(data, 0)
    assert result["id"] == "f1"


def test_active_finding_for_swing_none_when_no_match() -> None:
    data = _findings()
    assert _active_finding_for_swing(data, 5) is None


def test_active_finding_for_swing_none_findings() -> None:
    assert _active_finding_for_swing(None, 0) is None


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def _populate_run(run: RunDir, n: int = 60) -> None:
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

    swing = {
        "index": 0,
        "mode": "live",
        "contact_frame": 30,
        "contact_time_s": 1.0,
        "contact_source": "shuttle",
        "prep_start_frame": 20,
        "prep_end_frame": 27,
        "follow_end_frame": 40,
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


def test_render_annotated_video_produces_playable_file(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    _populate_run(run)

    out_path = run.root / "annotated.mp4"
    result = render_annotated_video(run, out_path, findings=_findings(), cfg=cfg, lang="vi")

    assert out_path.exists()
    assert result["n_swings"] == 1

    cap = cv2.VideoCapture(str(out_path))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    # 21 frames at 1x + slow_mo_factor(4)x around contact (0.25s window -> ~8 frames -> extra 3x each)
    assert count > 21


def test_render_annotated_video_without_findings(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    _populate_run(run)
    out_path = run.root / "annotated.mp4"
    result = render_annotated_video(run, out_path, findings=None, cfg=cfg)
    assert out_path.exists()
    assert result["n_swings"] == 1
