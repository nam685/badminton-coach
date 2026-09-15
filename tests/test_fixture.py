"""Sanity checks against the real downloaded pro-clear fixture (network-fetched, gitignored). Skipped
when the fixture hasn't been generated (`uv run python tests/fixtures/make_fixture.py`)."""

from __future__ import annotations

import pytest

from badminton_coach import video
from tests.fixtures.make_fixture import ensure_fixture

FIXTURE = ensure_fixture()


@pytest.mark.skipif(FIXTURE is None, reason="fixture not generated; run tests/fixtures/make_fixture.py")
def test_fixture_probes_cleanly() -> None:
    info = video.probe(FIXTURE)
    assert info.width > 0 and info.height > 0
    assert info.fps > 0
    assert info.duration_s > 3.0
    assert info.n_frames > 50


@pytest.mark.skipif(FIXTURE is None, reason="fixture not generated; run tests/fixtures/make_fixture.py")
def test_fixture_frame_count_matches_iteration() -> None:
    info = video.probe(FIXTURE)
    counted = sum(1 for _ in video.iter_frames(FIXTURE))
    # ffprobe's nb_frames can be off by a couple for some containers; allow small slack.
    assert abs(counted - info.n_frames) <= 2
