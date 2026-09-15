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

## Task 5 — shuttle tracking / TrackNetV3 (2026-09-15)

- **Real bug found and fixed**: at the model's default `batch_size=20`, GPU memory sat at ~3.9/4.0GB and
  the run didn't error but appeared to hang (100% GPU util, no progress for minutes) — this is CUDA
  unified-memory paging thrashing near the VRAM ceiling rather than a clean allocation failure. Fixed by
  lowering the default `batch_size` to 8 (measured ~2.5GB peak) and adding an explicit
  `torch.cuda.OutOfMemoryError` handler that halves the batch size and, failing that, falls back to CPU.
  Confirmed fast (well under a minute) after the fix.
- **Low recall on the fixture**: only **1.5% of frames (3/199)** crossed the visibility threshold
  (confidence > 0.5), even though a shuttle is visibly present to the eye at several points (confirmed by
  manually inspecting a cropped frame). Diagnostic sweep of the raw (pre-threshold) heatmap max
  confidence across the clip showed it consistently low (0.08–0.43) with no clear spike — a real
  recall/domain-gap limitation on this footage, not a code bug (the same code path, math, and threshold
  as upstream `BallInferencer`; verified the model architecture and checkpoint loading independently).
  Plausible contributors: our per-clip median background (vs. RacketVision's dedicated empty-court
  captures), this footage's blur/compression, or a genuine domain gap from the training distribution.
  **This is exactly the risk the spec already flagged and already designed around** (§10: "shuttle
  tracking on amateur footage may be poor → contact falls back to racket-speed peak" and the shadow/live
  swing distinction in §3.4) — no pipeline change needed now; Task 11 calibration (with more/varied real
  footage) should revisit the confidence threshold and consider whether a court-specific median capture
  (rather than per-clip) helps, but that's a tuning question, not a correctness one.
- Unaffected: keypoint math, CSV schema, and caching all verified independently via unit tests + the
  real-checkpoint integration test.
- **CPU fallback is genuinely slow in this environment**: ~200 frames took several minutes on CPU
  (vs. seconds on GPU) — plausibly WSL2 CPU-virtualization overhead, not investigated further since
  latency isn't a concern for the rare real fallback case. Kept the fixture-based integration test on
  `device="cuda"` for a sane dev-loop runtime; the CPU code path itself is covered separately by a cheap
  synthetic 2-frame-clip test.

## Task 6 — tracking / smoothing / handedness / net direction (2026-09-15)

- **Real bug found and fixed**: the first draft of the player-selection score normalized "distance to
  the nearest racket handle" by *each candidate's own* torso length. That inverted the intended effect
  in the exact scenario the spec calls out (a bystander who appears larger/closer than the real player):
  a bigger candidate's bigger torso shrinks their relative distance-in-torso-lengths to *any* racket,
  including one held by someone else, letting them out-score the actual racket-holder on area alone.
  Fixed with a steep three-tier gate (holding / ambiguous / clearly-not-holding) instead of a smooth
  1/(1+d) falloff — "is this candidate holding the visible racket" is close to a binary fact and should
  score like one. Caught by a synthetic test built directly from the spec's own motivating scenario.
- **Real bug found and fixed**: `track()`'s cache-input-hash used Python's built-in `hash()` on the
  measurement arrays' bytes, which is randomized per-process (`PYTHONHASHSEED`) unless explicitly
  disabled — meaning the stage would never recognize its own previous output as cached across separate
  runs (always silently recomputing). Replaced with `hashlib.sha256`, matching every other stage in the
  codebase; grepped the rest of the package to confirm no other stage had the same mistake.
- Both bugs were caught before ever running against real data, by the unit tests written per the plan's
  own listed test scenarios — evidence those scenarios were worth specifying.

## Task 7 — swing segmentation, metrics, reference comparison (2026-09-15)

- Interpreted spec §3.4's "prep_end: racket top **lowest** y before contact" as *visually* lowest (i.e.
  the **largest** y-pixel value — image y increases downward) rather than the numerically smallest y,
  since the very next words are "i.e. the back-scratch": the back-scratch preparation position has the
  racket head dropped down low behind the body, which is a large y, not a small one. Documented the
  convention explicitly in `swings.py`'s module docstring so it isn't re-litigated later.
- Found (via `test_segment_swings_stage_writes_and_caches`) and fixed a real cache-consistency bug: the
  cached-skip branch of `segment_swings` returned a smaller summary dict (`{"n_swings"}` only) than the
  freshly-computed branch (`{"n_swings", "n_live", "speed_source"}`) — a caller re-running against an
  already-processed video would silently get a different-shaped result depending on whether the cache
  hit. Added `speed_source` to `SwingsFile` itself (previously only recorded in `run.json`'s internal
  stage bookkeeping) so the cached path reconstructs an identical summary from the on-disk artifact.
- **Real end-to-end run (`add_reference` on the fixture) passed and mostly looks right**: 2 swings
  detected, both `mode="shadow"` (consistent with Task 5's low shuttle recall on this footage). Swing 1
  (contact frame 95, peak racket speed 85.9 torso-lengths/s) was visually confirmed against a
  re-extracted video frame as a real contact/follow-through-looking moment, and its metrics are
  sensible: `contact_height_vs_nose` ≈ +0.03 (essentially level with the nose — plausible), `contact_forward`
  ≈ +2.0 torso-lengths (correctly signed toward the net), `racket_shaft_angle_contact` ≈ -6° (near
  vertical, as expected for a clear), `elbow_angle_contact` ≈ 130° (more bent than the "150–170°"
  coaching-literature ideal — plausibly genuine technique feedback rather than measurement error, but not
  independently confirmed). Swing 0's metrics are **not representative** — see the false-positive note
  below — don't read numbers off it.
- **Calibration finding for Task 11**: the *other* detected swing (frame 25) is a false positive — a
  visual check shows both players still in a split-step ready stance, rackets low, no swing happening.
  The racket's coupled motion during that foot-plant crossed the speed-peak prominence threshold
  (`swing_speed_peak_prominence: float = 0.5`, an uncalibrated placeholder per its own docstring). Not
  fixed now — it's explicitly a Task 11 concern once real footage is available — but a promising
  direction if it recurs: require a minimum *absolute* peak speed in addition to prominence, and/or
  cross-check against a large elbow-angle change over the same window (a real swing extends the arm; a
  foot-plant jitter doesn't).
