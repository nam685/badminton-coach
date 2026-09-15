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

## Task 4 — racket detection + pose (2026-09-15)

- **Went straight to the mmdet/mmpose direct-inference fallback**, skipping the mmdeploy ONNX export
  attempt: validated the direct path works cleanly on the first real try (checkpoint load + real-frame
  inference), so there was no reason to burn the plan's "≤2h" mmdeploy budget chasing an alternative to
  something already proven to work. See `tools/export_racket_onnx/README.md` for the full reasoning and
  every environment-setup gotcha found along the way (mmcv flat-index wheel, numpy<2, setuptools<81,
  chumpy's `import pip` build quirk, and a real mmdet/mmpose `DefaultScope` collision bug when both
  libraries are used in one process).
- On the full fixture clip (199 frames): **88.9% of frames had a detected badminton racket** (score ≥
  0.3) — well above a "is this basically working" bar. The first ~15% of frames (the split-step/prep
  phase, before the racket is raised into clear view) accounts for most of the misses — expected, not a
  bug (spot-checked: an isolated 0-30 frame slice was only 27% detected, vs. 88.9% overall).
- Visual check (`measure/debug/racket_*.jpg`, generated from the cached npz): racket polygon and 5
  keypoints track correctly on the raised/swinging racket; a smaller, lower-confidence racket on the
  second, farther-from-camera player in frame is also correctly picked up (no keypoints drawn — below the
  0.3 keypoint-confidence threshold — but the bbox is there), a good sign the detector generalizes rather
  than overfitting to one scale/pose.
- Throughput: ~1.2 fps on CPU (single-threaded-ish mmdet/mmpose forward passes) — a 199-frame clip takes
  ~2.75 minutes. Acceptable for offline analysis (latency is not a concern per Nam).
- The fixture footage turned out to show **two different real people** (not a duplicated/mirrored
  composite as first guessed from Task 3's images alone) — one player prepping in the foreground, another
  executing the clear in the background. Confirms Task 6 (player selection) is not optional even on
  "clean" single-camera-position footage.
