from __future__ import annotations

import json
from pathlib import Path

import pytest

from badminton_coach.config import Config
from badminton_coach.render.report import render_report_md, write_report
from badminton_coach.run import RunDir


def _findings() -> dict:
    return {
        "player": "nam",
        "shot": "clear",
        "handedness": "right",
        "measurement_quality": {"ok": True, "notes": ""},
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
        "progress_vs_history": "elbow angle improved from 120 to 130 degrees",
        "confidence": "medium",
    }


def _metrics() -> dict:
    return {
        "fps": 30.0,
        "handedness": "right",
        "net_side": "right",
        "torso_len_median_px": 80.0,
        "swings": [
            {
                "index": 0,
                "mode": "live",
                "contact_frame": 10,
                "contact_time_s": 0.33,
                "metrics": {"elbow_angle_contact": {"value": 130.0, "ref_mean": 160.0}},
            }
        ],
        "cross_attempt_summary": {},
    }


def test_render_report_md_with_findings() -> None:
    md = render_report_md(_findings(), _metrics(), lang="en")
    assert "Elbow bent at contact" in md
    assert "reach up and through the shuttle" in md
    assert "good footwork" in md
    assert "elbow angle improved" in md
    assert "elbow_angle_contact" in md


def test_render_report_md_without_findings() -> None:
    md = render_report_md(None, _metrics(), lang="en")
    assert "did not produce findings" in md
    assert "elbow_angle_contact" in md  # metrics table still present


def test_render_report_md_localized() -> None:
    md_vi = render_report_md(_findings(), None, lang="vi")
    md_de = render_report_md(_findings(), None, lang="de")
    assert md_vi != md_de


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


def test_write_report_creates_both_files(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    (run.root / "judge").mkdir(parents=True)
    (run.root / "judge" / "findings.json").write_text(json.dumps(_findings()))
    (run.root / "judge" / "stream.jsonl").write_text('{"type": "system"}\n')
    (run.root / "metrics").mkdir(parents=True)
    (run.root / "metrics" / "metrics.json").write_text(json.dumps(_metrics()))
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
                "swings": [],
            }
        )
    )

    result = write_report(run, cfg=cfg, lang="vi")
    assert Path(result["report_md"]).exists()
    assert Path(result["index_html"]).exists()
    html = Path(result["index_html"]).read_text()
    assert "Elbow bent at contact" in html
    assert '<html lang="vi">' in html


def test_write_report_without_judge_findings(cfg: Config) -> None:
    run = RunDir.create("nam", cfg=cfg)
    (run.root / "metrics").mkdir(parents=True)
    (run.root / "metrics" / "metrics.json").write_text(json.dumps(_metrics()))
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
                "swings": [],
            }
        )
    )
    result = write_report(run, cfg=cfg)
    assert Path(result["index_html"]).exists()
    assert "did not produce findings" in Path(result["index_html"]).read_text()
