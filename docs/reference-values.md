# Reference values & calibration notes

Running log of manual observations from real runs, per spec §9 / plan Task 11. Updated as each task's
manual-check step is done.

## Task 3 — body pose (2026-09-15)

- Fixture (`data/fixtures/clear_3s.mp4`, 640px wide, 8s @ 25fps, 199 frames): `BodyWithFeet(mode="performance")`
  detected ≥1 person in ≥99% of frames, confident (>0.3) wrist keypoints in >90% of frames with a person.
- The footage has **two visually distinct figures** side by side throughout (same demonstration synced
  twice, or a front+side composite — not yet determined) — good incidental stress test for Task 6's
  player-selection logic (must consistently pick one track).
- Measured throughput: **~6.2 fps** on the RTX 3050 Ti with `mode="performance"` (YOLOX-x detector +
  RTMPose-x, 384x288) at this resolution — below the spec's original "≥15fps" aspiration (written before
  Nam clarified latency is a non-goal), but a full 5-attempt session (say 20s at 60fps = 1200 frames)
  still finishes in ~3-4 minutes, which is fine for offline analysis. Kept `performance` mode (most
  accurate) rather than trading down to `balanced`/`lightweight` for speed.
- Visual check (`measure/debug/body_*.jpg`): skeleton overlay tracks both figures correctly through the
  preparation phase; will re-check specifically at the contact frame once Task 7 (swing segmentation)
  identifies it precisely.
