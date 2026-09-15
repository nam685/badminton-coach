"""Real end-to-end body3d test: runs the full measurement pipeline + estimate_3d against the fixture.
Slow (spawns the SAM 3D Body pinned-env subprocess) — kept separate from test_body3d.py's fast unit
tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from badminton_coach.config import Config
from badminton_coach.measure.body3d import estimate_3d
from badminton_coach.pipeline import run_measurement_pipeline
from badminton_coach.run import RunDir
from tests.fixtures.make_fixture import ensure_fixture

FIXTURE = ensure_fixture()


@pytest.mark.models
@pytest.mark.skipif(FIXTURE is None, reason="fixture not generated; run tests/fixtures/make_fixture.py")
def test_estimate_3d_end_to_end_on_fixture(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    run = RunDir.create("nam", cfg=cfg)
    run_measurement_pipeline([FIXTURE], run, cfg=cfg, device="cuda")

    result = estimate_3d(run, cfg=cfg, device="cuda")
    assert result["backend"] in ("sam3d_body", "rtmw3d", "none")

    data = np.load(run.root / "body3d" / "joints.npz", allow_pickle=True)
    print(
        f"\nbody3d backend={data['backend']}, n_frames={data['frame_indices'].shape[0]}, "
        f"consistency_frac={data.get('consistency_frac')}"
    )

    if str(data["backend"]) == "sam3d_body" and data["frame_indices"].shape[0] > 5:
        import json

        from badminton_coach import metrics3d as m3d
        from badminton_coach.metrics import net_dir_sign

        swings_data = json.loads((run.root / "swings" / "swings.json").read_text())
        net_sign = net_dir_sign(swings_data["net_side"])
        frame_indices = data["frame_indices"].tolist()
        joints = data["keypoints_3d"]

        for swing in swings_data["swings"]:
            contact, prep_end = swing["contact_frame"], swing["prep_end_frame"]
            if prep_end is None or contact not in frame_indices or prep_end not in frame_indices:
                continue
            j_contact = joints[frame_indices.index(contact)]
            j_prep = joints[frame_indices.index(prep_end)]
            rot_contact = m3d.shoulder_rotation_deg(j_contact, net_sign)
            rot_prep = m3d.shoulder_rotation_deg(j_prep, net_sign)
            print(f"swing {swing['index']}: shoulder rotation prep_end={rot_prep}, contact={rot_contact}")
