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

## Task 7b — 3D body (SAM 3D Body) + rotation metrics (2026-09-15)

- Full story in `docs/body3d.md`. Summary: switched from the spec's assumed DINOv3-H+ checkpoint (840M
  params, OOM-killed on this machine's 7.6GB RAM) to the ViT-H checkpoint (631M, same published
  accuracy) loaded via `torch.load(..., mmap=True)` — fixed the OOM outright (0.3s load vs. killed).
  fp16 doesn't work at all (`addmm_sparse_cuda` has no Half kernel) — runs fp32, which fits 4GB VRAM
  fine in practice. No CPU path exists in the library (`process_one_image` hardcodes `.to("cuda")`), so
  the real fallback chain is SAM 3D Body (GPU) -> RTMW3D, not "-> CPU ->" as originally planned.
- Coordinate convention confirmed empirically against real output: Y increases **downward** (matching
  2D image coordinates), not the Y-up convention originally assumed for SMPL-family models — fixed in
  `metrics3d.trunk_lean_3d_deg` before it was ever wired into the pipeline.
- The full pipeline's first end-to-end `estimate_3d()` run fell back to `backend="rtmw3d"` — traced to a
  transient resource-contention failure in the SAM 3D Body subprocess (confirmed: the identical
  subprocess call succeeds reliably in isolation) rather than a code bug. Found and fixed a real gap
  along the way: the subprocess wrapper discarded stderr on failure, making this kind of fallback
  silent; now writes a `*.subprocess_error.txt` with the cause.
- 17 tests (metrics3d) + 12 tests (body3d) passing, ruff clean, plus a real end-to-end pipeline run.
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

## Task 8 — judge (2026-09-15)

- **First real end-to-end run**: full pipeline through metrics (no body3d/references, for speed) against
  the test fixture, then a genuine `claude -p` call billed against the real subscription OAuth token —
  the first time the whole chain, including the actual judge call, ran together. Full narrative in
  `docs/judge-eval-template.md`; summary here per the usual per-task log.
- **Outcome was correct, not a pass**: the judge reported `measurement_quality.ok = false`, 0 findings,
  `confidence: "low"`, having noticed the skeleton overlay was drawn on a different person than the one
  actually swinging, and specifically named a mid-swing identity switch as the cause of an impossible
  metric (`contact_height_vs_nose = -0.98`). This validated the system prompt's "say so if measurement
  looks broken" rule working as designed, and incidentally worked as an integration-test oracle: it
  surfaced three real bugs that no synthetic-data unit test had caught.
- **Real bug found and fixed in `track.py`**: player selection locked onto a static bystander (also
  holding a racket) instead of the swinger, because the racket-proximity score couldn't distinguish
  "calmly holding" from "swinging," and continuity's stickiness then preserved a bad early pick. Fixed
  with a racket-motion signal that boosts the score of a candidate whose nearest racket head is actually
  moving, a continuity floor that relaxes when that signal is decisive, and a post-hoc majority-vote pass
  (`_stabilize_selection`) that cleans up isolated single-frame flips. Verified against the real
  persisted data (chosen-player array compared before/after) and with a new unit test
  (`test_select_player_prefers_swinging_racket_over_static_one`). Improved, not fully solved: a small
  residual block of frames right around the fastest motion can still pick wrong — documented as a known
  limitation in `track.py`'s module docstring, not silently claimed as fixed.
- **Real bug found and fixed in `ingest.py`**: `normalized_path` was persisted exactly as given, so it
  only resolved from whatever cwd `ingest()` happened to run in. Broke `badminton-coach frames` when the
  judge (correctly) invoked it from its own workspace directory. Fixed by resolving all input paths to
  absolute before anything is persisted; regression test
  `test_ingest_stores_absolute_path_even_when_given_relative`.
- **Real bug found and fixed in packaging**: `badminton-coach` wasn't on `PATH` outside the project's own
  `uv run` at all, so the judge's `Bash(badminton-coach frames:*)` tool call had nothing to run
  regardless of the path bug above. Fixed with `uv tool install --editable .`, now a required setup step
  (`README.md`, `CLAUDE.md`).
- **Not yet exercised**: genuine technique findings from correctly-tracked data — this run's
  `measurement_quality.ok=false` short-circuited before reaching that path. Next real-footage run
  (ideally with body3d and a reference clip) should cover it.

## Task 10 — CLI wiring, end-to-end, docs (2026-09-15)

- `cli.py` gained `analyze` (full pipeline orchestration: measurement stages -> body3d -> metrics ->
  judge -> render, with `--from-stage`/`--no-judge`/`--refs`/`--net-side`/`--skip-body3d`/`--skip-shuttle`
  and `--model`/`--effort` overrides), `judge`/`render` (re-run only that stage against an existing run
  dir), `view` (serves `data/`), and `models download` (rtmlib warm-up + `git clone` RacketVision/SAM 3D
  Body sources + Hugging Face checkpoint downloads, all idempotent — skips anything already present, and
  raises an actionable error naming the gated-repo URL and `hf auth login` if SAM 3D Body's terms haven't
  been accepted yet).
- `--from-stage`: since `run_measurement_pipeline()` only exposes one `force` flag for its whole span
  (ingest..swings), `analyze` forces that entire span whenever `--from-stage` names anything inside it,
  and forces only the requested stage onward otherwise (`track.py`/`metrics.py`/etc.'s own input-hash
  caching still naturally cascades a real change downstream regardless of this flag).
- `docs/recording-guide.md` written in vi/de/en per spec §10's last bullet (one side-on camera position,
  5+ attempts in one continuous clip).
- **Real end-to-end validation**: ran `badminton-coach analyze <fixture> --player ... --no-judge` for
  real (no stubbing) against `data/fixtures/clear_3s.mp4` — full pipeline through metrics via the actual
  installed CLI entrypoint, not a test harness. Found 3 swings, correctly resolved
  `handedness="left"`/`net_side="left"` (`net_side_source="racket_travel"`), and `body3d` fell back to
  `rtmw3d` (SAM 3D Body's subprocess didn't succeed under real system memory pressure from an unrelated
  process on this dev machine) — the fallback chain documented in Task 7b working exactly as designed
  under genuine resource contention, not a bug. `--no-judge` correctly produced no `report.md`/
  `index.html`/`annotated.mp4`.
- `tests/test_cli.py` (11 tests): every heavy stage function stubbed, exercises the CLI's own
  wiring/flag-translation logic only (force propagation per `--from-stage`, `--refs all|none|<names>`
  resolution, run.json `lang`/`shot` persistence, `judge`'s always-force behavior, `render`'s
  findings-or-None handling, `models download`'s skip-if-present and gated-repo error message).
- 173 tests passing, ruff clean.

## Fix: wire 3D hip/shoulder rotation metrics into metrics.json (2026-09-15)

Auditing before closing out Task 10 surfaced a real, live gap: Task 7b built and unit-tested
`metrics3d.py`'s rotation functions (`shoulder_rotation_deg`, `hip_rotation_deg`, `x_factor_deg`,
`sequence_ms`, `trunk_lean_3d_deg`) and `judge/rubric.md` already told the judge to read
`shoulder_rotation_deg`/`x_factor_deg`/`sequence_ms` out of `metrics.json` — but `compute_metrics_stage()`
never actually called any of `metrics3d.py`'s functions, so those fields never existed in a real
`metrics.json`. Hip/shoulder rotation is the one measurement the user explicitly asked to add ("hip/
shoulder rotation is important, add it") and it was silently dead: computed by `body3d`, never delivered
to the judge. Fixed:

- `compute_metrics_stage()` (`metrics.py`) now loads `body3d/joints.npz` when present, densifies the
  sparse per-window `keypoints_3d` array back to one row per video frame (NaN where uncovered), and — for
  any backend other than `"none"` — merges each swing's `compute_swing_metrics_3d()` output into the same
  flat metrics dict the 2D metrics already populate, so it gets identical reference-comparison and
  cross-attempt-consistency treatment for free. Absent/`--skip-body3d`/`backend="none"` runs are
  unaffected (the 3D keys simply don't appear, rather than erroring).
- New `compute_swing_metrics_3d()` in `metrics3d.py`: per swing, evaluates shoulder/hip rotation and
  X-factor at `prep_end` and `contact`, trunk lean at `contact`, and the hip→shoulder→racket kinetic
  sequence timestamps over `[prep_start_frame, contact_frame]` — named `shoulder_rotation_deg_prep_end`/
  `_contact`, `hip_rotation_deg_prep_end`/`_contact`, `x_factor_deg_prep_end`/`_contact`,
  `trunk_lean_3d_deg_contact`, `sequence_hip_ms`/`sequence_shoulder_ms`/`sequence_racket_ms` — matching
  `metrics.py`'s existing `_prep_end`/`_contact` naming convention instead of the rubric's looser bare
  names, which were ambiguous about which instant they meant.
- **Real bug found and fixed in the same pass, before it ever shipped data**: `body3d`'s two backends do
  not share one joint layout. `backend="sam3d_body"` is MHR70 order (hips at indices 9/10).
  `backend="rtmw3d"` (the documented GPU-failure fallback, `rtmlib.Wholebody3d`) is standard
  COCO-WholeBody order, where indices 9/10 are *wrists*, not hips — real hips are at 11/12. Shoulders
  happen to coincide at 5/6 in both layouts, which is exactly the kind of partial agreement that would
  have let this slip through casual testing. Every `metrics3d.py` function that indexes hips/shoulders
  now takes a `backend` argument and resolves indices via `_indices_for_backend()`; a swing measured
  under the rtmw3d fallback would otherwise have silently reported wrist position as hip rotation. Caught
  by writing the COCO-layout test fixture first and checking it actually disagreed with the MHR70 one at
  the hip (`test_hip_rotation_mhr70_indices_wrong_for_rtmw3d_data`) before trusting the fix.
- `judge/rubric.md` updated to name the exact `metrics.json` fields (was previously referencing a
  bare `shoulder_rotation_deg`/`x_factor_deg`/`sequence_ms` that were never going to appear under those
  names even after wiring, since real metrics need a `_prep_end`/`_contact` instant qualifier and
  `sequence_ms` is three separate numbers, not one object).
- 21 new tests (12 in `test_metrics3d.py`, 5 in `test_metrics_stage.py` end-to-end, plus coverage of the
  unknown-backend error path); 183 tests passing total, ruff clean. Not yet re-validated against a real
  judge run with real body3d data (the Task 8 real run skipped body3d for speed) — worth doing once real
  footage is available (Task 11).

## Picking a real side-on reference clip (2026-09-15)

Nam asked whether the two reference URLs named as *candidates* in the spec (§3, `2bK27mv1Fq4` and
`5tG-krOy1ro`) were actually side-on, having watched them and thought they looked front-on. Checked by
pulling real frames (not thumbnails) from both: **confirmed both are end-on/back-of-court views**, camera
facing straight down the court toward the net — not side-on, despite the spec listing them as
"candidates to verify (must be side-on)". Neither is usable as a reference for this system's 2D metrics,
which are only geometrically meaningful with the camera roughly perpendicular to the swing plane. This
also means `2bK27mv1Fq4` (used all session as the local test fixture) was never side-on either — the
pipeline *code paths* it exercised are unaffected (tracking/segmentation don't care about camera angle),
but the specific metric *values* quoted from it earlier in this doc (e.g. Task 7's elbow-angle numbers)
aren't representative of genuinely side-on footage and shouldn't be read as calibration data.

Searched for and verified two real side-on candidates by extracting actual frames (not thumbnails, which
can be staged/misleading) across each video's duration:

- **`S2brZPqx288`** (Howcast, "How to Hit a Forehand Overhead Clear"): cuts between several camera
  angles: this side-on segment only from an old `--start 65.5 --end 71.0` trim (now `--start 68.0 --end
  71.0`, four detected swings in the earlier wider window, two visually confirmed false positives —
  same uncalibrated-`swing_speed_peak_prominence` issue as Task 7's finding; tightened the trim to isolate
  just the one visually-confirmed clean contact). Added as reference `howcast-forehand-clear`.
- **`zC9WvCrfxHc`** ("Badminton: Clear Vorhand (seitlich)", jugendundsport.ch — a Swiss youth-sport body):
  genuinely side-on for its entire 20.5s duration, single fixed camera, plain dark background, no cuts —
  a materially cleaner source than the Howcast clip. Added as reference `jugendsport-clear-seitlich`,
  trimmed from `--start 1.0 --end 2.9` around one visually-confirmed clean contact (elbow_angle_contact
  ≈ 177°, contact_height_vs_nose ≈ +0.27, peak_racket_speed ≈ 61 torso-lengths/s, net_side resolved via
  the more reliable `"shuttle"` source rather than the `"racket_travel"` fallback — a side effect of the
  tighter, less ambiguous window). The untrimmed 20.5s clip found **10** "swings" via the same
  false-positive-prone segmentation (2 corroborating, ~5+ clear noise) — not fixed now, same documented
  Task 11 concern, but concretely demonstrates why every reference needs this same visual-confirmation +
  tight-trim treatment rather than trusting swing detection blindly on longer clips.

**Two real bugs found and fixed while doing this, both in code that had never been exercised for real
before now:**

- **`reference.py`'s yt-dlp format selector could silently pick a broken AV1 stream.** The `zC9WvCrfxHc`
  fetch first tried `bestvideo[height<=720][ext=mp4]/...`, which resolved to an av01 (AV1) stream that
  decoded cleanly for isolated single-frame seeks (`ffmpeg -ss X -frames:v 1`, used for eyeballing
  candidates) but produced **zero frames** through the pipeline's real sequential decode
  (`ffmpeg: "Missing Sequence Header"`), because `[ext=mp4]` doesn't exclude AV1-in-mp4. This wasn't a
  general "AV1 is broken here" problem — the *other* reference's source (`S2brZPqx288`) is also av01,
  same resolution, and decodes perfectly — something about this specific stream/segment was corrupted or
  malformed, not investigated further since forcing a known-reliable codec sidesteps the whole risk
  class. Fixed by preferring `vcodec^=avc1` (H.264) explicitly in both `reference.py` and
  `tests/fixtures/make_fixture.py`'s yt-dlp format strings, falling back to the old selector only if no
  avc1 stream exists.
- **The silent 0-frame decode cascaded into a confusing crash three stages later.** `measure_body()`
  wrote an empty `body.npz` instead of raising, so `measure_racket`/`measure_shuttle` also silently ran
  on nothing, and the actual failure only surfaced as `pandas.errors.EmptyDataError: No columns to parse
  from file` inside `track()` reading an empty `shuttle.csv` — three stages and zero useful context away
  from the real cause. Fixed: `measure_body()` now raises immediately with a clear message
  ("decoded 0 frames... a likely cause: an AV1-encoded source...") the moment it detects a 0-frame decode,
  matching this project's established pattern of failing loud and close to the cause rather than letting
  garbage propagate.
- **`add_reference()` silently reused a stale trim.** `clip.mp4` was only regenerated when it didn't
  exist yet or `force=True` was passed — calling `reference add` again with a *different* `--start`/
  `--end` (exactly what happened twice in this session, narrowing both reference clips' windows to cut
  out false positives) silently kept the first trim's `clip.mp4` and reran the whole pipeline on stale
  video. Also found in the same pass: the CLI's `reference add` had no `--force` flag at all, despite
  `add_reference()` accepting one. Fixed: `add_reference()` now compares the new `start_s`/`end_s` against
  what's stored in the run's own `run.json` from the previous call and retrims/reruns automatically on a
  real change, independent of `force`; added the missing `--force` CLI flag for the genuine
  re-fetch-from-scratch case.
- 2 new tests (`test_measure_body_raises_clearly_on_zero_decoded_frames`,
  `test_add_reference_retrims_when_window_changes_even_without_force`), 202 tests passing (full suite,
  including real-model/GPU tests), ruff clean.
