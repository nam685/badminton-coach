# badminton-coach v1 — Measure with specialists, judge with Claude (Design Spec)

**Date:** 2026-09-15
**Status:** Approved for implementation (research + design done by Fable 5.1; implementation by cheaper models)
**Scope:** Ticket 1 — standalone local CLI. Ticket 2 (nam685.de/badminton web app, users, history) is a
separate GitHub issue on `nam685/nam-website` and is **out of scope here** except for the forward-compatible
data layout in §7.
**Precedent:** `~/projects/aoe2coach` (pure library + `claude -p` agent in a per-run workspace, auth via
`CLAUDE_CODE_OAUTH_TOKEN`), and `nam-website/docs/superpowers/specs/2026-06-22-aoe2-coach-v2-design.md`.

## 1. Why

Nam does an overhead **clear**; it feels right; the coach says it's wrong, and *what* is wrong changes
week to week. Friends have the same problem. Existing products either give numbers without explanation
(SportsBox-style 3D golf apps) or vague "form scores". The gap: **precise measurement + a frontier model
that explains like a coach**, on video from a phone.

The thesis (same as the AoE2 coach program): **the judge's quality is bounded by the measurement bundle**.
A frontier VLM alone cannot localize joints to the pixel, cannot measure an angle within ~20°, cannot count
frames, and will confidently hallucinate "elbow bent at contact". Specialist models can't *interpret*.
So: specialists measure, Claude judges, the renderer draws from the specialists' coordinates and Claude's
findings.

No ablation study (Nam's call). Ship SOTA measurement + Claude judge.

## 2. Research summary (what was verified, 2026-09-15)

| Need | Choice | Why / verified facts |
|---|---|---|
| Body pose (2D) | **RTMPose via `rtmlib`** (`BodyWithFeet`, 26 kpts Halpe, mode `performance`) | pip-installable, ONNX, `device='cuda'` via onnxruntime-gpu, feet keypoints for footwork, same model family as the racket model. MediaPipe rejected (weaker on fast sport motion, no racket). |
| Racket | **RacketVision** (AAAI 2026, MIT, weights on HF `linfeng302/RacketVision-Models`): RTMDet-M racket detector (3 classes, badminton = cat **0**, input 640×640) + RTMPose-M racket pose (**5 kpts: top, bottom, handle, left, right**, input 256×256, SimCC) | Only public badminton racket keypoint model. Configs are MMPose configs → exportable to ONNX with mmdeploy, then run with rtmlib/onnxruntime in the main env. |
| Shuttle | **TrackNetV3** from RacketVision `source/BallTrack` (`balltrack_best.pth`) | Same repo; needs only torch + mmengine (registry) + cv2. Output per frame: X, Y, Visibility, Confidence. |
| 3D body (hip/shoulder rotation) | **SAM 3D Body** (Meta, `facebookresearch/sam-3d-body`, HF `facebook/sam-3d-body-dinov3`, SAM License, **gated** — accept terms on HF + `hf auth login`), run **only on swing windows**, per-image, smoothed by us | Nam: rotation is important → in v1. Bundles its own MHR body model (no SMPL-X registration). Outputs `pred_keypoints_3d` (camera space), `pred_keypoints_2d`, `pred_cam_t`, `focal_length`. 840M-param DINOv3-H+ (or 631M ViT-H) → run in fp16 to fit 4 GB; seconds/frame is acceptable (latency is a non-goal). Fallback/cheap path: `rtmlib.Wholebody3d` (RTMW3D-x, 133 kpts with depth, MPJPE ~57 mm H3WB) in the same ONNX stack. Optional speedup: Fast-SAM-3D-Body (ECCV 2026, MIT, ~10× ). GVHMR/WHAM rejected: SMPL registration + DPVO/detectron2 deps. Racket stays 2D. |
| Judge | **`claude -p` subprocess**, cwd = per-run workspace, `--allowedTools Read Grep Glob` (+ one scoped Bash tool), `--json-schema` for structured findings, `--output-format stream-json` saved for debugging | Precedent `aoe2coach/coach.py::_build_argv`. Auth: `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` (Pro/Max, one-year token — docs: "can only make model requests", fine here). Installed CLI 2.1.272 has `--json-schema`, `--effort`, `--permission-mode`, `--max-turns`, `--append-system-prompt`. No Agent SDK / API key needed. |
| GPU | RTX 3050 Ti 4 GB, WSL2, CUDA 12.0 driver, `nvidia-smi` works | ONNX RTMPose-l fits easily. onnxruntime-gpu ≥1.19 needs CUDA 12 + **cuDNN 9** libs: get them from the `torch` cu12 wheels (`import torch` first, or `onnxruntime.preload_dlls()`), never from the system. |
| Reference "good examples" | Pro slow-motion clips via `yt-dlp` run through the *same* measurement pipeline | Candidates to verify (must be side-on): `https://www.youtube.com/watch?v=2bK27mv1Fq4`, `https://www.youtube.com/watch?v=5tG-krOy1ro`. Personal use only — `data/` is gitignored, never redistributed. |
| Coaching rubric | Written into `rubric.md` (§6.3) from coaching sources (Badminton Insight forehand clear guide + general biomechanics) with explicit hedges | The judge cites the rubric, never invents thresholds. |

Sources: rtmlib README; RacketVision README/configs/`download_checkpoints.py`/`tools/inference.py`;
mmpose "how to deploy"; Claude Code docs (authentication, Agent SDK python); onnxruntime CUDA EP docs;
badminton-insight.com forehand clear guide. See the chat log of 2026-09-15 for links.

## 3. Pipeline

```
video(s) ─► 0 ingest        normalize (rotation fix, fps, hash) ─► frames on demand
        ─► 1 measure        body 2D (rtmlib) + racket (ONNX) + shuttle (TrackNetV3) per frame
        ─► 2 track/smooth   player selection, gap fill, smoothing, handedness
        ─► 3 swings         segment attempts, contact frame per swing, phase windows
        ─► 3b body3d        SAM 3D Body on the player's bbox, swing windows only → 3D joints, smoothed
        ─► 4 metrics        per-swing metrics at key instants (2D + 3D rotation), normalized; reference deltas
        ─► 5 workspace      metrics.json + annotated key frames + contact sheets + plots + rubric + history
        ─► 6 judge          claude -p (agent, Read/Grep/Glob + frames tool) → findings.json (schema)
        ─► 7 render         annotated.mp4 (skeleton, racket, arcs, red/green, slow-mo) + report.md + index.html
```

Every stage writes its artifacts into the run directory (§7) and can be re-run alone
(`--from-stage`, cached by input hash). This is the "see all intermediate results" requirement.

### 3.1 Ingest
- Inputs: one or more files (phone MP4/MOV) or a directory. Multiple files = multiple attempts of the same
  player in one session.
- `ffprobe` for fps/rotation; phone videos carry a rotation tag — verify OpenCV applies it
  (`CAP_PROP_ORIENTATION_AUTO`); if not, `ffmpeg -vf transpose` to a normalized copy in the run dir.
- Keep native fps (60 preferred). Downscale for inference to long side 1280; keep original for rendering.
- Record `ingest.json`: file, sha256, fps, width, height, n_frames, duration, rotation applied.

### 3.2 Measure
- **Body**: `rtmlib.BodyWithFeet(mode='performance', backend='onnxruntime', device='cuda')` → per frame
  `[n_persons, 26, 2]` + scores. Persist all persons; selection happens in 3.3.
- **Racket**: own thin ONNX wrapper for RTMDet (letterbox 640, outputs `dets[N,5]` + `labels[N]`, keep
  `label==0` badminton, score ≥ 0.3) → `rtmlib.RTMPose(onnx, model_input_size=(256,256))` on the bbox → 5 kpts.
  Persist all rackets + scores.
- **Shuttle**: TrackNetV3 (vendored `BallTrack` code, MIT) → per frame `(x, y, vis, conf)`.
- Output: `measure/body.npz`, `measure/racket.npz`, `measure/shuttle.csv` in *inference resolution*
  coordinates plus the scale factor to original.
- **Model export (one-time tool, separate env)**: `tools/export_racket_onnx/` with its own uv project pinned
  to Python 3.10, `torch==2.1.2`, `mmcv==2.1.0` (openmmlab cu121/torch2.1 wheel index), `mmdet<3.3`,
  `mmpose>=1.3`, `mmdeploy==1.3.1`; run `tools/deploy.py` with `pose-detection_simcc_onnxruntime_dynamic.py`
  for the pose ckpt (`best_PCK_epoch_90.pth`) and `detection_onnxruntime_dynamic.py` for the det ckpt
  (`epoch_300.pth`). Output `data/models/racket_det.onnx`, `data/models/racket_pose.onnx`.
  **Fallback** if export fights back: keep the pinned env and run a small `infer_frames.py` (mmdet
  `init_detector/inference_detector` + mmpose `init_model/inference_topdown`, mirroring RacketVision's
  `tools/inference.py`) as a subprocess over a frames dir → JSON. Same output contract.

### 3.3 Track / smooth
- **Player selection**: friends will film with people in the background. Per frame score each person by
  bbox area × proximity to the racket handle (if a racket is detected) × proximity to previous frame's
  choice (IoU). Keep one track; record `track/selection.json` (index chosen per frame) for debugging.
- **Racket association**: the racket whose `handle` is nearest the chosen player's wrist; that wrist's side,
  majority-voted over the clip, is **handedness**.
- **Gap fill + smoothing**: linear interpolation for gaps ≤ 5 frames; mask low-confidence points (< 0.3);
  Savitzky–Golay (window ~7 at 60 fps, 5 at 30) on positions. Keep raw alongside smoothed.
- **Net direction**: sign of shuttle x-velocity after contact (primary), fallback CLI flag `--net-side left|right`.

### 3.4 Swings
- Signal: racket `top` speed (px/s, normalized by torso length/s); fallback racket-wrist speed.
- Peaks above threshold with ≥ 0.8 s separation = one swing each.
- **Contact frame**: within ±8 frames of the peak, the frame minimizing shuttle–racket-head distance when
  the shuttle is visible; else the frame of peak racket speed. Record `contact_source` (`shuttle`|`speed`).
- Key instants per swing: `prep_end` (racket `top` lowest y before contact, i.e. "back-scratch"),
  `contact`, `follow_end` (contact + 0.4 s). Windows: `prep` = contact − 0.6 s → prep_end.
- **Shadow vs live**: a swing is `mode="live"` if the shuttle is visible within 1 torso-length of the racket
  head in the ±8 frames around the peak, else `mode="shadow"`; a clip with no live swing is a shadow session.
  Shadow swings are a first-class input (consistent, no feed variation): contact = peak racket speed, and
  the metrics are read as "the intended contact point". Not available for shadow swings: shuttle-relative
  timing (early/late) and anything about where the shuttle went. The judge is told the mode per swing.
- Output `swings/swings.json`.

### 3.4b Body 3D (`body3d` stage)
- Runs **after swings** so it only processes prep→follow windows (±0.8 s around each contact, ≈100 frames per
  swing at 60 fps), on the original-resolution frames, using the 2D player bbox (§3.3) as the box prompt so
  the 3D estimate is of the *same* person. Model: SAM 3D Body (`process_one_image` per frame), fp16 on CUDA;
  if it OOMs on 4 GB, the run degrades to (a) CPU (slow, allowed) or (b) `rtmlib.Wholebody3d` — the stage
  records `body3d.backend` and the judge is told the quality tier.
- Output `body3d/joints.npz`: `[T_window, J, 3]` camera-space joints per swing + `pred_cam_t`, `focal_length`,
  plus the model's own 2D keypoints for a consistency check against rtmlib (median px error logged).
- Post: mask frames where 3D↔2D consistency is bad (> 0.15 torso in px), interpolate, Savitzky–Golay.
- Camera up-vector: assume the phone is level (static tripod/leaning) so image-up ≈ gravity-up; expose
  `--camera-tilt-deg` for correction; hedge in the rubric.

### 3.5 Metrics (per swing, at key instants, all normalized by torso length = |mid_shoulder − mid_hip|)
Racket arm = handedness side. Angles in degrees, image plane.
- `elbow_angle` (shoulder–elbow–wrist) at prep_end, contact, follow_end
- `shoulder_abduction` (hip–shoulder–elbow) at prep_end and contact
- `contact_height` = (nose_y − racket_top_y) / torso (positive = above head); also wrist version
- `contact_forward` = (racket_top_x − shoulder_x) / torso, signed toward the net (+ = in front)
- `racket_shaft_angle` = angle of handle→top vs vertical at contact; `racket_face_proxy` = |left−right| / |top−bottom| (foreshortening → face open/closed toward camera; hedge)
- `trunk_lean` = hip-mid→shoulder-mid vs vertical, at prep_end and contact
- **3D rotation (from `body3d`)** — angles in the horizontal plane (perpendicular to gravity-up) relative to
  the net direction; 0° = shoulders/hips square to the net, 90° = fully sideways:
  `shoulder_rotation_deg`, `hip_rotation_deg` at prep_end, contact, follow_end;
  `x_factor_deg` = shoulder − hip separation at prep_end (trunk coil);
  `hip_rotation_range_deg`, `shoulder_rotation_range_deg` = prep_end → contact change;
  `sequence_ms` = timestamps of peak hip angular velocity, peak shoulder angular velocity, peak racket speed
  (proximal→distal ordering with 20–80 ms gaps = good kinetic chain; simultaneous or reversed = "arm swing");
  `trunk_lean_3d_deg`, `shoulder_abduction_3d_deg` at contact (replaces the 2D versions when available).
- `shoulder_width_ratio` and `hip_width_ratio` (2D projected width / torso) are kept as a **cross-check** of
  the 3D rotation (they must move in the same direction); disagreement lowers `measurement_quality`.
- `non_racket_wrist_height` = (shoulder_y − wrist_y)/torso at prep_end (arm up for balance)
- `stance_width` (ankles) at prep_end and contact; `front_foot` (which ankle is nearer the net) at
  prep_end vs contact → `weight_transfer` boolean; `jump_height` (min ankle y vs standing)
- `peak_racket_speed` (torso/s), `time_prep_to_contact_s`, `follow_through_duration_s`
- Time series (per frame) of elbow angle, racket speed, shoulder/hip width ratio → `plots/`.
- **Reference deltas**: for each metric, `{value, ref_mean, ref_min, ref_max}` over the reference swings.
- **Cross-attempt summary**: mean/std per metric over all swings in the session; consistency flag if
  std is large relative to references.
Output `metrics/metrics.json` (schema in `badminton_coach/schema.py`).

### 3.6 Workspace + judge → see §6.
### 3.7 Render
- Per frame: skeleton (26 kpts, rtmlib `draw_skeleton` style), racket 5-pt polygon, shuttle trail (last
  10 frames), elbow-angle arc with value, contact-height line at contact, HUD with swing index / phase /
  timestamp.
- Rotation overlay: a small top-down inset (shoulder line + hip line as two rotating bars vs the net
  direction, with the degree values) during prep→contact, driven by `body3d`.
- Findings drive color: joints/frames referenced by a finding are red in that window; green otherwise.
  Finding title as a banner during its `frame_range`.
- Per swing: normal speed prep → **4× slow-mo** for contact ± 0.25 s → normal follow-through.
- Reference panel at contact: side-by-side still of the best-matching reference swing's contact frame
  (ghost overlay is v2).
- Encode with ffmpeg pipe: h264, `yuv420p`, `-movflags +faststart` (plays in browsers).
- `report.md` (findings in priority order, evidence, drills, metrics table) and `index.html` (video,
  contact sheets, plots, findings, collapsible judge transcript, raw metrics).

## 4. CLI (`badminton-coach`, package `badminton_coach`)

```
badminton-coach analyze <video...> --player nam --lang vi|de|en [--shot clear] [--net-side auto] [--refs all]
                                   [--model opus] [--effort high] [--from-stage measure] [--no-judge]
badminton-coach reference add <url|file> --name viktor-2018 [--start 0:12 --end 0:20]
badminton-coach reference list
badminton-coach judge <run-dir>                 # re-run judge only (prompt iteration)
badminton-coach render <run-dir>                # re-render only
badminton-coach frames <run-dir> --swing 2 --from 0.40 --to 0.55 --step 1 [--crop upper]   # judge's tool
badminton-coach view [--port 8765]              # serves data/ (runs index + per-run index.html)
badminton-coach models download                 # rtmlib auto-downloads; racket ONNX + tracknet from HF
```
Config via `pyproject`-free `badminton_coach/config.py` defaults + env `BADMINTON_COACH_DATA` (default
`./data`), `CLAUDE_BIN` (default `claude`), `CLAUDE_CODE_OAUTH_TOKEN` (required for judge).

## 5. Package layout (flat, hatchling, like aoe2coach)

```
badminton-coach/
  pyproject.toml            # deps: rtmlib, onnxruntime-gpu, torch (cu12), opencv-python-headless, numpy,
                            #       scipy, pandas, mmengine, huggingface-hub, yt-dlp, click, jinja2,
                            #       matplotlib, pydantic; dev: pytest, ruff (line-length 120)
  badminton_coach/
    cli.py  config.py  schema.py  run.py (stage orchestration, caching)
    ingest.py  video.py
    measure/  body.py  racket.py  shuttle.py  (+ vendored tracknet/ from RacketVision BallTrack, MIT)
    track.py  swings.py  metrics.py  reference.py
    judge/  workspace.py  prompt.py  run.py  rubric.md  schema.json
    render/  draw.py  video.py  report.py  templates/
  tools/export_racket_onnx/  (own pinned uv project, see §3.2)
  tests/  (unit tests on synthetic sequences + smoke test on a fixture clip)
  docs/superpowers/{specs,plans}/
  data/  (gitignored: models/, references/, players/, fixtures cache)
```

## 6. The judge

### 6.1 Workspace (per run; kept on disk under the run dir as `judge/workspace/`)
```
metrics.json            all swings, key-instant metrics, reference deltas, cross-attempt summary
swings/<i>/contact_sheet.png     12-frame grid prep→follow, skeleton+racket drawn, timestamps, phase labels
swings/<i>/frames/f_<n>.png      annotated frames at 4× density around contact
swings/<i>/crops/c_<n>.png       upper-body crops at contact ± 3 frames
plots/swing_<i>.png              elbow angle, racket speed, hip/shoulder rotation (3D) vs time, peak markers
                                  for hip / shoulder / racket (kinetic-chain sequence); contact marked
references/<name>/contact_sheet.png + metrics.json     the "good example(s)"
rubric.md               technique checklist for the shot type, with hedges (§6.3)
history.md              previous sessions of this player: date, top findings, key metric means (empty in v1
                        unless prior runs exist under data/players/<player>/) — forward-compat with ticket 2
TASK.md                 thin trigger; the system prompt carries the contract
```
The judge may also call `badminton-coach frames …` (allowlisted `Bash(badminton-coach frames:*)`) to
pull more frames into `swings/<i>/frames/` and Read them — this is the "let it scrub the footage for
minutes" capability without the Agent SDK. Cwd = workspace; no `CLAUDE.md` inside; `--strict-mcp-config`
to avoid user MCP servers.

### 6.2 Invocation
```
claude -p <TASK.md contents> --model opus --effort high --output-format stream-json --verbose
       --json-schema <findings schema> --append-system-prompt <JUDGE_SYSTEM>
       --allowedTools Read Grep Glob "Bash(badminton-coach frames:*)"
       --permission-mode dontAsk --max-turns 40 --strict-mcp-config
```
cwd = workspace, timeout 20 min. Save `judge/stream.jsonl` (full transcript = debug view), parse the final
`result` message → `judge/findings.json` (validated with pydantic against `judge/schema.json`; on failure
save raw and mark the run `judge_failed`, never crash the pipeline — render still runs without findings).
`dontAsk` is the pinned deny-not-hang mode (from aoe2coach).

### 6.3 `rubric.md` (forehand overhead clear — v1 content, judge cites it; hedged)
- Camera is side-on. Elbow angle, contact height/forwardness, racket shaft angle, timing are 2D image-plane
  measurements and reliable. Shoulder/hip **rotation, X-factor and the hip→shoulder→racket sequence come from
  monocular 3D** (`body3d`) — good enough for "did the hips turn ~60° or ~10°" and for ordering, not for
  ±5° claims; quote the numbers as approximate and lean on the reference clips' values. If `body3d` is
  absent or low-quality for a swing, fall back to the 2D width proxy and say "appears".
- Rotation targets (soft, prefer references): sideways at prep_end (shoulders ≳ 70–90° to the net, hips
  somewhat less → positive X-factor); by contact shoulders near square (≲ 20°); hips lead shoulders, shoulders
  lead the racket (sequence_ms strictly increasing).
- Preparation: sideways stance (small shoulder-width ratio), racket arm elbow ~90° and up, racket head
  behind ("back-scratch"), **non-racket arm raised** toward the shuttle for balance and timing,
  weight on the back (racket-side) foot.
- Sequence: legs → hips → trunk → shoulder → elbow → forearm pronation → contact. Elbow stays back until the
  trunk has turned (elbow leads, racket lags).
- Contact: **high, above the head, slightly in front** (~half a metre in front per coaching sources; in
  torso units roughly +0.2…+0.5 forward, height > +0.6 above nose for the racket head — treat as
  soft ranges, prefer the reference clips' numbers); arm **nearly** extended (not locked: ~150–170°, "too
  straight = shoulder only", "too bent = no rotational power"); racket face square, shaft near-vertical
  at contact for a clear.
- Follow-through: racket comes down and across the body, weight transfers to the front foot (scissor kick
  for a jump clear: feet switch in the air, land on the non-racket-side foot first), recovery to base.
- Common faults to look for: contact behind the head / too low; arm too bent or locked; no hip/trunk
  turn (facing the net throughout); non-racket arm dropped or never raised; panhandle grip (visible as
  racket face parallel to the body on the way up); no weight transfer; late/slow preparation; stopping the
  swing at contact.
- Grip cannot be verified from this footage — only mention if the racket-face proxy strongly suggests it.

### 6.4 System prompt contract (`judge/prompt.py::JUDGE_SYSTEM`, draft)
- You are a badminton coach operating as an agent with file tools; inputs are in your cwd.
- **Process**: read `metrics.json` and `rubric.md` → read every `contact_sheet.png` (yours + references) →
  for each swing look at `crops/` at contact → pull more frames with the frames tool when uncertain →
  compare to references and to `history.md` → decide.
- **Rules**: every finding must cite ≥1 swing + frame and ≥1 metric or a specific visual observation;
  numbers come only from `metrics.json` (never estimate angles yourself); a fault seen in 1 of N attempts is
  "inconsistent", in most attempts is a "pattern"; state what is *good* too (1–2 items); prioritize by
  impact on the clear (contact point > rotation/sequence > arm > non-racket arm > footwork); ≤ 4 findings;
  for `mode="shadow"` swings, judge the position at peak racket speed as the intended contact point and never
  comment on timing vs the shuttle or shuttle flight;
  one drill per finding; hedge anything based on the rotation proxy; if measurement looks broken (racket
  never detected, contact_source=speed everywhere, missing swings) say so first.
- **Output**: only via the JSON schema. **Language**: all user-facing strings (`title`, `explanation`,
  `cue`, `drill`, `one_line_verdict`, `strengths`, `progress_vs_history`, `measurement_quality.notes`) are
  written in the player's language (`lang` in `TASK.md`: `vi`, `de` or `en`) — natural coaching language, not
  a translation of English; badminton terms as a coach in that language would say them (e.g. vi: "cú đánh
  cao sâu / cầu cao sâu", de: "Clear / Überkopf-Clear"). Enum fields (`category`, `pattern`, `confidence`)
  stay English. `rubric.md` and the system prompt stay English (the model reads them fine).

### 6.5 Findings schema (`judge/schema.json`)
```
{ player, shot, handedness, measurement_quality: {ok: bool, notes},
  swings: [{index, mode: "live"|"shadow", contact_frame, contact_time_s, one_line_verdict}],
  strengths: [string],
  findings: [{ id, title, category: enum[contact_point, rotation_sequence, elbow, racket, non_racket_arm,
               footwork_weight_transfer, timing, follow_through, other],
               severity: 1|2|3, pattern: "consistent"|"inconsistent"|"single",
               evidence: { swings: [int], frames: [{swing, frame}], metrics: [{name, value, reference}],
                           visual: string },
               explanation, cue (one sentence to say to yourself), drill }],
  priority: [finding ids], progress_vs_history: string|null, confidence: "low"|"medium"|"high" }
```

### 6.6 Language
Users may speak only Vietnamese or only German. `--lang` (per player, stored in `run.json`; ticket 2 stores it
on the player) drives: the judge's user-facing strings (§6.4), the banners/HUD labels burned into
`annotated.mp4`, and `report.md`/`index.html` chrome. Static UI strings live in `badminton_coach/i18n/{vi,de,en}.json`
(small: phase names, "contact", "swing", "finding", "drill", "strengths", table headers). Fonts: use a TTF
with Vietnamese diacritics (e.g. DejaVu Sans or Noto Sans, bundled or system) via PIL for text on frames —
OpenCV's Hershey fonts cannot render Vietnamese or umlauts.

## 7. Data layout (forward-compatible with ticket 2)
```
data/models/                          rtmlib cache, racket_det.onnx, racket_pose.onnx, balltrack_best.pth
data/references/<name>/               source.mp4, ingest.json, measure/, swings/, metrics/, contact sheets
data/players/<player>/<YYYYMMDD-HHMMSS>[-<slug>]/    one run = one session (one or more attempts)
    ingest.json  measure/  track/  swings/  metrics/  plots/  judge/{workspace/,stream.jsonl,findings.json}
    annotated.mp4  report.md  index.html  run.json (stage status, timings, versions)
data/index.html                       generated by `view`
```
`history.md` for the judge is built from earlier `findings.json` files of the same player. Ticket 2 will
put a DB + web UI on top of exactly this layout (player = identified user, run = timestamped upload).

## 8. Dev conveniences (explicit requirement)
- Every stage → files; `--from-stage` resume; stage cache keyed by video sha256 + stage version string.
- `badminton-coach view` = tiny static server over `data/` (runs index, per-run page with the video, contact
  sheets, plots, findings, judge transcript). No framework.
- `judge` and `render` re-runnable alone for prompt/drawing iteration without re-inference.
- `--debug` keeps intermediate PNGs for *every* frame (default: only key frames).
- Logging with timings per stage into `run.json`.

## 9. Evaluation (no ablation; sanity gates)
1. **Pro clip through the pipeline** → judge should report high measurement quality and ≤1 minor finding.
   If it "finds" faults in a world-class clear, the pipeline or prompt is wrong.
2. **Nam's clip** → findings should overlap what the coach says; Nam scores each finding true/false in
   `judge/eval.md` (manual). Iterate on prompt/rubric with `badminton-coach judge` only.
3. **Re-run stability**: run the judge 3× on the same workspace; top finding should agree.
4. Unit tests: angle math, swing segmentation on synthetic speed traces, contact selection with/without
   shuttle, schema validation, reference deltas, handedness vote.
5. Smoke test: 3-second fixture clip end-to-end with `--no-judge` (CI-safe, CPU).

## 10. Risks / open decisions
- **mmdeploy export** may fail on version drift → fallback subprocess in the pinned env (§3.2). Budget one task.
- **Racket detection recall** on amateur phone footage (motion blur) — unknown; mitigations: 60 fps
  recording, gap interpolation, speed-based contact fallback, and the judge is told when racket data is thin.
- **SAM 3D Body on 4 GB VRAM**: 840M params → fp16 mandatory; if it still OOMs, CPU (hours are acceptable)
  or the RTMW3D fallback. Gated HF repo: Nam accepts the terms once and runs `hf auth login`. Per-image
  model → our own temporal smoothing; monocular depth is prior-driven → rotation numbers are approximate,
  the rubric says so.
- **Shuttle tracking** on amateur footage may be poor → contact falls back to racket-speed peak.
- **onnxruntime-gpu + cuDNN 9** on WSL2: install torch cu12 wheels and `import torch` before onnxruntime
  (or `onnxruntime.preload_dlls()`); document in README.
- **Subscription token**: `claude setup-token` → `CLAUDE_CODE_OAUTH_TOKEN`; one year; personal use in a
  first-party CLI, same as aoe2coach.
- Recording guidance for users (README, in vi/de/en): **one camera position, side-on** (perpendicular to the
  swing, racket-arm side toward the camera), landscape, whole body + racket in frame the whole time, 60 fps,
  **5+ attempts in one continuous clip** from that same position, plain background if possible. Same angle
  every time so metrics are comparable within a session and across sessions/references. A second clip from
  behind the player (rotation/footwork) is optional and stored, but v1 only analyzes the side-on view.

## 11. Out of scope (v1)
Web app / users / history UI (ticket 2), 3D racket (racket stays 2D), ghost overlay of the reference on the user's video, shots
other than the forehand overhead clear (the pipeline is shot-agnostic; only `rubric.md` and metric
thresholds are clear-specific — add `rubric_<shot>.md` later), real-time, mobile.
