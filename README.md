# badminton-coach

Record your badminton stroke on a phone → get an annotated video and a coach-style report of what is wrong.

**How it works:** specialist models measure (RTMPose body pose, RacketVision racket keypoints + TrackNetV3
shuttle, SAM 3D Body for hip/shoulder rotation), the pipeline turns that into per-swing metrics and
compares them to pro reference clips, and Claude (Opus, via `claude -p` with a subscription OAuth token)
judges the evidence like a coach. The renderer draws the findings on the video. Every intermediate result
is written to disk and viewable with `badminton-coach view`.

Status: **all 10 tasks of the v1 implementation plan done** — measurement pipeline, tracking,
swing/metric extraction, 3D rotation, the judge, rendering, and full CLI wiring all work end-to-end,
validated against real footage (see `docs/reference-values.md` and `docs/judge-eval-template.md`). Task
11 (calibration against Nam's own footage) is optional and still open. See
`docs/superpowers/specs/2026-09-15-badminton-coach-v1-design.md` and
`docs/superpowers/plans/2026-09-15-badminton-coach-v1-plan.md`.

v1 targets the forehand overhead **clear**, local GPU (RTX 3050 Ti, WSL2). A web version for friends
(nam685.de/badminton) is a separate ticket on `nam685/nam-website`.

## Setup

```bash
uv sync
uv tool install --editable .          # REQUIRED: puts `badminton-coach` on PATH globally — the judge
                                       # agent invokes it by bare name from its own workspace directory,
                                       # not the repo root (found via a real judge run — see
                                       # docs/judge-eval-template.md)
claude setup-token                    # once, if you haven't: writes a long-lived OAuth token
export CLAUDE_CODE_OAUTH_TOKEN=...    # put it in .env (gitignored) — the judge needs this to run
badminton-coach models download       # fetches rtmlib/RacketVision/TrackNet/SAM 3D Body weights into
                                       # data/models/ (SAM 3D Body is a gated HF repo: accept its terms
                                       # at huggingface.co/facebook/sam-3d-body-vith and `hf auth login`
                                       # first, or the command tells you to when it hits the gate)
```

GPU notes (torch cu12 wheels, `import torch` before `onnxruntime`/`onnxruntime.preload_dlls()`), the
onnxruntime/rtmlib packaging workaround, and every environment-specific decision are documented inline in
`pyproject.toml` and each module's own docstring — see also `docs/body3d.md` and
`tools/export_racket_onnx/README.md` for the two pinned sub-environments.

## Quick start

Record following `docs/recording-guide.md` (vi/de/en), then:

```bash
badminton-coach analyze clip.mp4 --player nam --lang en
```

This runs the full pipeline (ingest, measurement, tracking, swings, metrics, 3D rotation, the judge, and
rendering) and prints the run directory — open `<run>/index.html` for the report, or `badminton-coach
view` to browse every run.

Multiple attempts filmed in one continuous clip need no special handling (`swing segmentation` finds each
one); multiple separate video files are also accepted (`analyze a.mp4 b.mp4 ... --player nam`) and are
concatenated into one run with per-attempt frame offsets kept in `run.json`.

## Dev loop

```bash
badminton-coach analyze clip.mp4 --player nam --no-judge   # measurement + metrics only, no billed judge call
badminton-coach analyze clip.mp4 --player nam --from-stage track   # force recompute from `track` onward
badminton-coach judge data/players/nam/<run>      # re-run only the judge (prompt/rubric iteration)
badminton-coach render data/players/nam/<run>     # re-render only (video + report), no new judge call
badminton-coach reference add <url|file> --name viktor-2018 --start 0:12 --end 0:20
badminton-coach reference list
badminton-coach view --port 8765                  # serve data/ locally
```

Every stage caches its own output in the run dir keyed on a content hash of its inputs — re-running
`analyze` on an unchanged run is fast (mostly cache hits); `--from-stage <stage>` forces recompute from
that stage onward regardless.

## Data layout

`data/` is gitignored (models, reference clips, player runs). A run lives at
`data/players/<player>/<timestamp>/`, with one subdirectory per stage (`measure/`, `track/`, `swings/`,
`metrics/`, `body3d/`, `judge/`) plus `run.json` (stage status/caching bookkeeping) and, once rendered,
`annotated.mp4`/`report.md`/`index.html`.

## Licenses

- RacketVision (racket detector/pose + shuttle tracker, vendored/called as a subprocess): MIT.
- rtmlib (body pose): Apache-2.0.
- SAM 3D Body: see `data/models/sam3db_src/LICENSE` after `models download` (Meta's own license terms;
  the checkpoint is gated and requires accepting its terms on Hugging Face separately).
