from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from badminton_coach.config import Config
from badminton_coach.measure.body import measure_body
from badminton_coach.measure.keypoints import BODY_NUM_KPTS, LEFT_WRIST, RIGHT_WRIST
from badminton_coach.run import RunDir
from tests.fixtures.make_fixture import ensure_fixture

FIXTURE = ensure_fixture()
DEVICE = "cuda"


@pytest.mark.models
@pytest.mark.skipif(FIXTURE is None, reason="fixture not generated; run tests/fixtures/make_fixture.py")
def test_measure_body_on_fixture(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    run = RunDir.create("nam", cfg=cfg)
    summary = measure_body(run, FIXTURE, cfg=cfg, device=DEVICE)

    assert summary["n_frames"] > 50
    assert summary["max_persons"] >= 1

    data = np.load(run.root / "measure" / "body.npz")
    kpts = data["kpts"]  # [T, P, 26, 3]
    n_persons = data["n_persons"]  # [T]
    assert kpts.shape[2] == BODY_NUM_KPTS
    assert kpts.shape[3] == 3

    frames_with_person = int((n_persons >= 1).sum())
    assert frames_with_person / summary["n_frames"] >= 0.90

    # Best-detected person per frame (highest mean score), wrist confidence check (either hand — we
    # don't know handedness yet at this stage).
    wrist_scores = []
    for t in range(kpts.shape[0]):
        if n_persons[t] == 0:
            continue
        person_scores = np.nanmean(kpts[t, : n_persons[t], :, 2], axis=1)
        best = int(np.nanargmax(person_scores))
        lw, rw = kpts[t, best, LEFT_WRIST, 2], kpts[t, best, RIGHT_WRIST, 2]
        wrist_scores.append(max(lw, rw))
    wrist_scores = np.array(wrist_scores)
    frac_confident_wrist = float((wrist_scores > 0.3).mean())
    assert frac_confident_wrist > 0.5, f"only {frac_confident_wrist:.0%} of frames had a confident wrist"

    # run.json bookkeeping
    status = run.stage_status("measure_body")
    assert status["status"] == "done"
    assert status["fps"] > 0


@pytest.mark.models
@pytest.mark.skipif(FIXTURE is None, reason="fixture not generated; run tests/fixtures/make_fixture.py")
def test_measure_body_is_cached(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    run = RunDir.create("nam", cfg=cfg)
    measure_body(run, FIXTURE, cfg=cfg, device=DEVICE)
    status1 = run.stage_status("measure_body")
    measure_body(run, FIXTURE, cfg=cfg, device=DEVICE)
    status2 = run.stage_status("measure_body")
    assert status1 == status2  # not recomputed
