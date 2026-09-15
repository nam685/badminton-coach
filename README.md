# badminton-coach

Record your badminton stroke on a phone → get an annotated video and a coach-style report of what is wrong.

**How it works:** specialist models measure (RTMPose body pose, RacketVision racket keypoints + TrackNetV3
shuttle, SAM 3D Body for hip/shoulder rotation), the pipeline turns that into per-swing metrics and
compares them to pro reference clips, and Claude (Opus, via `claude -p` with a subscription OAuth token)
judges the evidence like a coach. The renderer draws the findings on the video. Every intermediate result
is written to disk and viewable with `badminton-coach view`.

Status: **Tasks 1-9 of the implementation plan done** (measurement pipeline, tracking, swing/metric
extraction, 3D rotation, the judge, and rendering all work end-to-end, validated against real footage —
see `docs/reference-values.md` and `docs/judge-eval-template.md`). Task 10 (final CLI wiring for
`analyze`, full docs) is next. See `docs/superpowers/specs/2026-09-15-badminton-coach-v1-design.md` and
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
```

Models are currently fetched ad hoc by each stage (rtmlib auto-downloads on first use; RacketVision and
SAM 3D Body checkpoints via `hf download`, see `tools/export_racket_onnx/README.md` and
`docs/body3d.md`) — a single `badminton-coach models download` command that does all of this is Task 10.

GPU notes, the onnxruntime/rtmlib packaging workaround, and everything else are documented inline in
`pyproject.toml` and each module's own docstring.
