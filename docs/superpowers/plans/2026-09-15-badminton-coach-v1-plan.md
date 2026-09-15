# badminton-coach v1 — Implementation Plan

> **For agentic workers:** implement task-by-task (subagent-driven development if the superpowers plugin is
> available, otherwise one task per worktree/PR). Steps use checkbox (`- [ ]`) syntax. Read the design spec
> first: `docs/superpowers/specs/2026-09-15-badminton-coach-v1-design.md`. Do not redesign — if a step is
> blocked, finish the rest of the task, write the blocker into `docs/BLOCKERS.md`, and stop.
>
> **Rule of thumb (Nam): at every step, if the difficulty or the approach is uncertain, first ask "has someone
> already solved this?"** — search GitHub/PyPI/papers before writing it yourself, and prefer an existing,
> maintained solution. Known prior art per task is listed under **Prior art** below; extend it as you find more.

**Goal:** `badminton-coach analyze clip.mp4 --player nam` produces, locally on the RTX 3050 Ti, an
annotated video + report explaining what is wrong with the player's forehand overhead clear, using
RTMPose (body) + RacketVision (racket, shuttle) for measurement and `claude -p` (Opus, subscription OAuth
token) as the judge, with every intermediate artifact inspectable via `badminton-coach view`.

**Tech stack:** Python 3.12, `uv`, hatchling, `rtmlib`, `onnxruntime-gpu`, `torch` (cu12), OpenCV,
scipy, pandas, mmengine, huggingface-hub, yt-dlp, click, jinja2, matplotlib, pydantic; dev: pytest, ruff
(line-length 120, `select = ["E4","E7","E9","F","I","N","ARG"]`). Separate pinned Python 3.10 uv project
under `tools/export_racket_onnx/` for the one-time ONNX export.

## Global constraints
- Repo: `/home/namle685/projects/badminton-coach` (local git). Create GitHub repo `nam685/badminton-coach`
  (private) in Task 1. Work on feature branches; PR per task.
- All Python via `uv run`. All functions have return type annotations. No Django/web code (ticket 2).
- `data/` is gitignored. Never commit videos, weights, or reference clips.
- The judge is a `claude -p` subprocess (spec §6.2). No Agent SDK, no `anthropic` package.
- Every stage writes files under the run dir and is re-runnable alone (spec §8).
- Tests must pass on CPU without models except the smoke test, which is marked `@pytest.mark.models` and
  skipped when `data/models/` is absent.

## File structure (target)
See spec §5. Create files incrementally per task below.

---

### Task 1: Scaffold repo, env, GPU sanity
**Files:** `pyproject.toml`, `badminton_coach/__init__.py`, `badminton_coach/config.py`, `.gitignore`,
`README.md` (exists — extend), `CLAUDE.md` (exists), `Makefile`, `tests/test_config.py`.

- [ ] `cd /home/namle685/projects/badminton-coach && git init -q && git branch -M main` (skip if already a repo)
- [ ] `pyproject.toml` per spec §5 (name `badminton-coach`, package `badminton_coach`, script
      `badminton-coach = "badminton_coach.cli:main"`, ruff/pytest config as in aoe2coach).
      Pin `torch` to a cu12 build via a `[tool.uv.sources]`/index entry for `https://download.pytorch.org/whl/cu124`
      (or whatever cu12x index resolves; document the choice in README).
- [ ] `uv sync`; verify `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`.
- [ ] Verify onnxruntime CUDA EP: `uv run python -c "import torch, onnxruntime as ort; print(ort.get_available_providers())"`
      must include `CUDAExecutionProvider`. If it fails with cuDNN errors, add `onnxruntime.preload_dlls()`
      in `badminton_coach/__init__.py` (guarded) and document.
- [ ] `config.py`: `DATA_DIR` (env `BADMINTON_COACH_DATA`, default `./data`), `MODELS_DIR`, `CLAUDE_BIN`,
      defaults for thresholds (racket det score 0.3, kpt conf 0.3, swing min separation 0.8 s, etc.).
- [ ] `.gitignore`: `data/`, `.venv/`, `__pycache__/`, `*.mp4 *.mov *.onnx *.pth`, `runs/`.
- [ ] `Makefile`: `sync`, `test`, `lint`, `view`, `models`.
- [ ] `gh repo create nam685/badminton-coach --private --source . --push` after first commit.

### Task 2: Ingest + video I/O
**Files:** `badminton_coach/video.py`, `badminton_coach/ingest.py`, `badminton_coach/run.py` (run dir +
stage cache skeleton), `tests/test_ingest.py`, `tests/fixtures/make_fixture.py`.

- [ ] `video.py`: `probe(path) -> VideoInfo` via `ffprobe -print_format json` (fps, w, h, n_frames, rotation
      from `side_data_list` display matrix / `tags.rotate`); `iter_frames(path, scale_long_side=1280)`;
      `read_frame(path, index)`; `write_video(frames_iter, out, fps)` via ffmpeg pipe (h264, yuv420p, faststart).
- [ ] Rotation: test with a portrait phone clip; if OpenCV does not auto-rotate, normalize with
      `ffmpeg -vf transpose` into `<run>/normalized.mp4` and use that downstream. Record in `ingest.json`.
- [ ] `run.py`: `RunDir` class — `data/players/<player>/<YYYYMMDD-HHMMSS>[-slug]/`, `run.json` with stage
      statuses/timings/versions; `stage(name, version)` decorator that skips when outputs exist and inputs'
      sha256 + version match, unless `--force`/`--from-stage`.
- [ ] Fixture: `tests/fixtures/make_fixture.py` downloads reference candidate
      `https://www.youtube.com/watch?v=2bK27mv1Fq4` with `yt-dlp`, trims 3 s, downscales to 640px →
      `data/fixtures/clear_3s.mp4` (gitignored, cached). Tests skip if missing.
- [ ] Tests: probe on the fixture; frame count matches; rotation logic on a synthetic rotated clip made with ffmpeg.

### Task 3: Body pose (rtmlib)
**Files:** `badminton_coach/measure/__init__.py`, `badminton_coach/measure/body.py`, `tests/test_body.py`.

- [ ] `BodyEstimator(device='cuda')` wrapping `rtmlib.BodyWithFeet(mode='performance', backend='onnxruntime')`;
      `__call__(frame) -> (kpts[n,26,2], scores[n,26])`. Keypoint index constants for Halpe-26 in
      `measure/keypoints.py` (nose, shoulders, elbows, wrists, hips, knees, ankles, big toes, heels, neck, head-top…).
- [ ] `measure_body(run, video) -> body.npz` (object arrays per frame or padded `[T, max_persons, 26, 3]`
      with NaN padding — choose padded, store `n_persons[T]`). Progress bar (`tqdm`).
- [ ] Debug output with `--debug`: `measure/debug/body_<n>.jpg` every 10th frame with `rtmlib.draw_skeleton`.
- [ ] Benchmark on the fixture: log fps in `run.json`. Target ≥ 15 fps at 1280 on the 3050 Ti.
- [ ] Test (`@pytest.mark.models`): fixture → ≥ 90% frames with ≥1 person, wrist confidence > 0.3 in most frames.

**Prior art:** `rtmlib` already ships an RTMDet wrapper with `det_mode='multiclass'` + `det_categories` — try it
before writing the own ONNX wrapper; mmdeploy issues/discussions for RTMPose export errors; rtmlib author's
HF space `Tau-J` hosts exported ONNX models as format examples.

### Task 4: Racket model export (pinned env) + racket inference
**Files:** `tools/export_racket_onnx/{pyproject.toml,README.md,export.py,infer_frames.py}`,
`badminton_coach/measure/racket.py`, `badminton_coach/models.py` (downloads), `tests/test_racket.py`.

- [ ] `models.py`: `download_racket_checkpoints()` from HF `linfeng302/RacketVision-Models` (`checkpoints/epoch_300.pth`,
      `checkpoints/best_PCK_epoch_90.pth`, `checkpoints/balltrack_best.pth`) into `data/models/racketvision/`;
      also clone/fetch RacketVision configs (`source/RacketPose/configs/**`, `source/BallTrack/configs/**`) into
      `data/models/racketvision/src/` (git clone --depth 1 of `https://github.com/OrcustD/RacketVision`).
- [ ] Pinned export env: `tools/export_racket_onnx/pyproject.toml` with `requires-python = "==3.10.*"`,
      `torch==2.1.2`, `torchvision==0.16.2` (cu121 index or CPU — CPU is fine for export), `mmengine>=0.10.7`,
      `mmcv==2.1.0` from `https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html` (cp310 wheel exists)
      or the cpu/torch2.1 index if exporting on CPU, `mmdet>=3.0,<3.3`, `mmpose>=1.1`, `mmdeploy==1.3.1`,
      `onnx`, `onnxruntime`. `uv sync` in that dir. Document exact resolved versions in its README.
- [ ] `export.py`: run mmdeploy `tools/deploy.py` (from a shallow clone of mmdeploy at the matching tag) with
      `configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py` + RacketVision `rtmpose_m_racket.py` +
      `best_PCK_epoch_90.pth` → `data/models/racket_pose.onnx`; and `configs/mmdet/detection/detection_onnxruntime_dynamic.py`
      + `rtmdet_m_racket.py` + `epoch_300.pth` → `data/models/racket_det.onnx`. Verify each ONNX with
      onnxruntime on a fixture frame (outputs shaped `dets[1,N,5]`, `labels[1,N]`; pose `simcc_x/simcc_y`).
- [ ] **Fallback (only if export fails after ≤ 2 h):** `infer_frames.py` in the pinned env: mmdet
      `init_detector/inference_detector` + mmpose `init_model/inference_topdown`, filter label 0, write
      `racket.json` per frame; `racket.py` then calls it as a subprocess over a frames dir. Record the decision in
      `docs/BLOCKERS.md`.
- [ ] `racket.py`: `RacketDetector` (own ONNX wrapper: letterbox to 640, BGR→RGB per RacketVision config
      normalization — copy `mean/std` from `rtmdet_racket.py`; parse `dets`+`labels`, keep label 0, score ≥ 0.3,
      un-letterbox) + `rtmlib.RTMPose(onnx_model=..., model_input_size=(256,256))` for 5 kpts. Output
      `racket.npz`: `[T, max_rackets, 5, 3]` + bboxes + scores.
- [ ] Debug PNGs every 10th frame with the racket polygon; test on fixture: racket detected in ≥ 60% of frames
      (record the actual number in `run.json`; it is a measurement, not a hard gate).

### Task 5: Shuttle tracking (TrackNetV3)
**Files:** `badminton_coach/measure/tracknet/` (vendored from RacketVision `source/BallTrack`, MIT notice
kept), `badminton_coach/measure/shuttle.py`, `tests/test_shuttle.py`.

- [ ] Vendor only what the model needs (model definitions, config `tracknetv3_base.py`, pre/post-processing);
      register via `mmengine.registry.MODELS` as upstream does. Confirm it imports in the main env (torch 2.x).
- [ ] `shuttle.py`: `track_shuttle(run, video) -> shuttle.csv` (Frame, X, Y, Visibility, Confidence) in
      inference resolution; sequence stacking as upstream `inference.py`; `torch.autocast` on cuda.
- [ ] Debug: trail overlay every 10th frame. Test on fixture: visibility>0 in some frames; no crash on a clip
      shorter than `seq_len`.

**Prior art:** multi-person tracking = ByteTrack via the `supervision` package (`sv.ByteTrack`) rather than
hand-rolled IoU matching; One-Euro filter / Savitzky–Golay from `scipy.signal`.

### Task 6: Tracking, smoothing, handedness, net direction
**Files:** `badminton_coach/track.py`, `tests/test_track.py`.

- [ ] Player selection per spec §3.3 → `track/selection.json`, `track/player.npz` `[T,26,3]`.
- [ ] Racket association → `track/racket.npz` `[T,5,3]`; handedness majority vote → `track/track.json`
      (`handedness`, `net_side`, `torso_len_median_px`).
- [ ] Gap fill (≤5 frames), confidence mask, Savitzky–Golay; keep `raw` and `smooth` arrays.
- [ ] Net direction: shuttle x-velocity after the largest racket-speed peak → else racket-`top` travel
      direction after the peak (shadow swings) → else CLI `--net-side`. Record which source was used.
- [ ] Tests on synthetic sequences: two persons, background person larger but far from racket → correct choice;
      handedness vote; gap interpolation; smoothing preserves peaks within tolerance.

**Prior art:** `scipy.signal.find_peaks` for swing peaks; DTW (`tslearn`/`dtaidistance`) if contact-only
alignment proves too coarse; golf/tennis swing-analysis repos for X-factor / kinematic-sequence code.

### Task 7: Swing segmentation + metrics + reference comparison
**Files:** `badminton_coach/swings.py`, `badminton_coach/metrics.py`, `badminton_coach/reference.py`,
`badminton_coach/schema.py` (pydantic models for `swings.json`, `metrics.json`), `tests/test_swings.py`,
`tests/test_metrics.py`.

- [ ] `swings.py` per spec §3.4 (speed signal, peaks, contact frame with `contact_source`, per-swing
      `mode` live/shadow, key instants, windows). Shadow swings must work with `shuttle.csv` entirely empty. Plot `plots/swing_<i>.png` (elbow angle, racket speed, shoulder/hip width ratio; contact marked).
- [ ] `metrics.py` per spec §3.5 — pure functions on `[26,3]`/`[5,3]` arrays; every metric documented in a
      docstring with its sign convention. `metrics.json` via pydantic.
- [ ] `reference.py`: `add_reference(url_or_file, name, start, end)` (yt-dlp download → trim → run stages
      ingest→metrics under `data/references/<name>/`); `load_references()`; `compare(metrics, refs)` →
      per-metric `{value, ref_mean, ref_min, ref_max, n_ref}`; cross-attempt summary.
- [ ] Tests: synthetic keypoint trajectories with known angles (e.g. straight arm → 180°), synthetic speed trace
      with 3 peaks → 3 swings, contact chooses shuttle-proximity frame when shuttle present and the speed peak
      when absent (mode=shadow), net direction from racket travel when no shuttle, torso normalization
      invariance under scaling, handedness mirror (left-hander flips `contact_forward` sign correctly).
- [ ] Manual check on the pro fixture: metrics plausible (contact above head, elbow ~150–170° at contact).
      Write the observed values into `docs/reference-values.md`.

**Prior art:** `gaomingqi/sam-body4d` — training-free *temporally consistent* video HMR built on SAM 3D Body
(+ SAM 3 tracking); evaluate it before writing own smoothing/consistency logic. `yangtiming/Fast-SAM-3D-Body`
(MIT) for speed. OpenCap / biomechanics toolkits for joint-angle conventions.

### Task 7b: 3D body (SAM 3D Body) + rotation metrics
**Files:** `badminton_coach/measure/body3d.py`, `badminton_coach/metrics3d.py`, `tests/test_body3d.py`,
`tests/test_metrics3d.py`, `docs/body3d.md`.

- [ ] Prereq (Nam): accept the gated terms on `https://huggingface.co/facebook/sam-3d-body-dinov3` and run
      `hf auth login`. `models download` fetches the checkpoint + `assets/mhr_model.pt` into `data/models/sam3db/`.
- [ ] Install `sam-3d-body` from `https://github.com/facebookresearch/sam-3d-body` (follow its INSTALL.md; pin
      the commit in `pyproject.toml`). If its torch pin conflicts with the main env, isolate it like the racket
      export: `tools/body3d_env/` pinned uv project + subprocess over a frames dir → `joints.npz`. Record the
      decision in `docs/body3d.md`.
- [ ] `body3d.py`: `estimate_3d(run, swings, frames, player_bboxes) -> body3d/joints.npz` per spec §3.4b
      (windows only, bbox prompt, fp16, OOM → CPU → RTMW3D fallback chain; log backend + s/frame in `run.json`).
      2D-consistency check vs rtmlib, masking, interpolation, smoothing.
- [ ] `metrics3d.py`: shoulder/hip rotation vs net direction in the gravity-horizontal plane, X-factor, ranges,
      angular velocities and peak timestamps (`sequence_ms`), 3D trunk lean and shoulder abduction (spec §3.5).
      Pure functions on `[T, J, 3]`; joint-name mapping for the MHR skeleton (and RTMW3D) in `measure/keypoints.py`.
- [ ] Merge into `metrics.json` (fields optional; `null` with `body3d.backend = "none"` if the stage failed).
- [ ] Reference clips also run `body3d`; `reference.py` includes the 3D metrics in deltas.
- [ ] Tests: synthetic 3D skeleton rotated by known angles → rotation metrics exact; sequence ordering from
      synthetic angular-velocity peaks; consistency mask on injected outliers. `@pytest.mark.models`: on the
      fixture, shoulder rotation at contact is smaller (more square) than at prep_end.
- [ ] Manual: check the top-down inset (Task 9) against the optional from-behind clip when Nam records one;
      note agreement in `docs/reference-values.md`.

### Task 8: Judge — workspace, prompt, invocation, schema
**Files:** `badminton_coach/judge/{__init__,workspace,prompt,run}.py`, `badminton_coach/judge/rubric.md`,
`badminton_coach/judge/schema.json`, `badminton_coach/render/draw.py` (needed for contact sheets — see Task 9;
implement the drawing primitives here first), `tests/test_judge.py`.

- [ ] `draw.py` primitives: `draw_body(frame, kpts, scores, color_overrides)`, `draw_racket`, `draw_shuttle_trail`,
      `draw_angle_arc(frame, a, b, c, label)`, `draw_hud`, `draw_text(frame, text, xy, lang)` via PIL with a
      bundled/system TTF that has Vietnamese diacritics and umlauts (DejaVu Sans / Noto Sans) — never
      `cv2.putText` for user-facing text. Colors: green default, red when highlighted.
- [ ] `workspace.py`: build the workspace exactly as spec §6.1 (contact sheets 4×3 grid with timestamps + phase
      labels, crops, plots copied, references, `rubric.md`, `history.md` from earlier `findings.json` of the
      same player, `TASK.md`). Keep it under `<run>/judge/workspace/`. No `CLAUDE.md` inside.
- [ ] `rubric.md`: copy spec §6.3 verbatim, then refine wording only.
- [ ] `schema.json`: spec §6.5 as a strict JSON Schema; pydantic mirror in `schema.py`.
- [ ] `prompt.py`: `JUDGE_SYSTEM` per spec §6.4 including the language rule; `build_task(..., lang)` for
      `TASK.md` (states the language explicitly, e.g. "Write all user-facing text in Vietnamese").
- [ ] `badminton_coach/i18n/{vi,de,en}.json` + `i18n.t(key, lang)`; test that all three files have identical key sets.
- [ ] Manual: run the judge once with `--lang vi` and once with `--lang de` on the fixture; confirm the JSON
      validates and the text is natural (ask Nam to spot-check the Vietnamese).
- [ ] `run.py`: build argv exactly as spec §6.2 (copy `_build_argv` style from
      `~/projects/aoe2coach/aoe2coach/coach.py`); `subprocess.run(..., cwd=workspace, timeout=1200)`; write
      `judge/stream.jsonl`, `judge/argv.json`, parse final result → validate → `judge/findings.json`; on any
      failure write `judge/error.txt`, set `run.json.judge = "failed"`, return `None` (pipeline continues).
      Include `claude` version and model id from the stream in `run.json`.
- [ ] `frames` CLI subcommand (spec §4) so the judge can request more frames; it must only write inside the
      workspace and print the paths it wrote.
- [ ] Tests: argv construction (allowedTools, json-schema, cwd), findings validation (valid/invalid samples),
      workspace layout on a synthetic run (no models needed — use saved fixture arrays in `tests/data/`).
- [ ] Manual: run the judge on the pro fixture and on Nam's clip (when available); save outputs; write
      `judge/eval.md` template (finding → true/false/partly, notes). See spec §9.

### Task 9: Render — annotated video + report + viewer
**Files:** `badminton_coach/render/{video,report}.py`, `badminton_coach/render/templates/{run.html,index.html}`,
`badminton_coach/viewer.py`, `tests/test_render.py`.

- [ ] `render/video.py`: per spec §3.7 — skeleton/racket/shuttle/arcs/HUD every frame, top-down rotation inset
      from `body3d` (shoulder + hip bars vs net direction, degrees), finding-driven red
      windows + banners, 4× slow-mo around each contact (frame duplication), reference contact still side-by-side
      at contact, ffmpeg pipe encode → `annotated.mp4`. Works with `findings=None` (no red, no banners).
- [ ] `render/report.py`: `report.md` + `index.html` (jinja2): video, per-swing contact sheet + plot, findings
      table with evidence links to frames, strengths, metrics table with reference deltas, collapsible judge
      transcript (`stream.jsonl` rendered as text), run metadata.
- [ ] `viewer.py`: `badminton-coach view` → regenerate `data/index.html` (players → runs, newest first, with
      top finding) and serve `data/` with `http.server` on `--port`.
- [ ] Tests: render 20 synthetic frames with fake findings → mp4 exists and decodes with the right frame
      count; report renders with and without findings.

### Task 10: CLI wiring, end-to-end, docs
**Files:** `badminton_coach/cli.py`, `README.md`, `docs/recording-guide.md`, `tests/test_cli.py`,
`tests/test_e2e.py` (`@pytest.mark.models`).

- [ ] `cli.py` (click) with every command in spec §4; `analyze` runs stages in order with caching/`--from-stage`/
      `--no-judge`; multiple inputs = one run with attempts concatenated (keep per-attempt frame offsets in `ingest.json`).
- [ ] `models download`: rtmlib models warm-up + racket ONNX presence check + TrackNet weights + SAM 3D Body
      (gated HF: clear error message telling the user to accept terms and `hf auth login`).
- [ ] README: install (uv, torch cu12 index, onnxruntime-gpu/cuDNN note), `claude setup-token` →
      `CLAUDE_CODE_OAUTH_TOKEN`, quick start, data layout, dev loop (`--from-stage`, `judge`, `render`, `view`),
      recording guide link, licenses (RacketVision MIT, rtmlib Apache-2.0).
- [ ] `docs/recording-guide.md` in **vi, de and en** (spec §10 last bullet: one side-on camera position, 5+
      attempts in one continuous clip).
- [ ] `--lang` on `analyze`/`judge`/`render`, persisted in `run.json`; `report.md`/`index.html` chrome via i18n.
- [ ] E2E on the pro fixture with `--no-judge` in CI-less local run; full run with judge manually; attach
      the resulting `report.md` excerpt to the PR.
- [ ] Update `CLAUDE.md` with the commands and structure as built.

### Task 11 (optional, after Nam's first clip): calibration pass
- [ ] Run on Nam's clip(s); fill `judge/eval.md`; tune thresholds in `config.py`, rubric wording, prompt.
      Record before/after in `docs/reference-values.md`. Re-run stability check (spec §9.3).
