# badminton-coach — Project Instructions

Standalone CLI: measure a badminton stroke from phone video (RTMPose + RacketVision + TrackNetV3), let
Claude judge it (`claude -p` agent in a per-run workspace), render an annotated video + report.

## Read first
- Spec: `docs/superpowers/specs/2026-09-15-badminton-coach-v1-design.md` (the contract — do not redesign)
- Plan: `docs/superpowers/plans/2026-09-15-badminton-coach-v1-plan.md` (task-by-task, checkboxes)
- Precedent for the judge subprocess: `~/projects/aoe2coach/aoe2coach/coach.py` (`_build_argv`, `run_claude_coach`)

## Code style
- Python 3.12, `uv run` for everything, ruff line-length 120, return type annotations on all functions
- Flat package `badminton_coach/`; pure functions on numpy arrays for metrics; pydantic for JSON contracts
- No web framework, no Django, no Agent SDK, no `anthropic` package (judge = `claude -p` subprocess)
- Every pipeline stage writes files into the run dir and can be re-run alone (`--from-stage`)

## Data
- `data/` is gitignored: models, reference clips, player runs. Never commit videos or weights.
- Run dir layout is fixed by spec §7 (`data/players/<player>/<timestamp>/…`) — ticket 2 builds on it.

## After editing badminton_coach/
- Re-run `uv tool install --editable .` is NOT needed for code changes (editable install already
  reflects them), but IS needed once if you haven't installed it yet — the judge agent invokes
  `badminton-coach frames` by bare command name from its own workspace directory (not the repo root),
  so it must be on `PATH` globally, not just runnable via `uv run` from here. See README.md.
- After touching `track.py`, `ingest.py`, or anything the judge reads, sanity-check against real data if
  you have any — several real bugs (wrong-player tracking, relative-path persistence) were only caught
  by an actual judge run, not by unit tests on synthetic data. See `docs/judge-eval-template.md`.

## Commands (as built — Task 10)
```
badminton-coach analyze <video...> --player nam --lang vi|de|en [--shot clear] [--net-side auto|left|right]
                                   [--refs all|none|name1,name2] [--model opus] [--effort high]
                                   [--from-stage <stage>] [--no-judge] [--skip-body3d] [--skip-shuttle]
badminton-coach judge <run-dir> [--lang ...] [--refs ...] [--model ...] [--effort ...]   # always forces a fresh judge call
badminton-coach render <run-dir> [--lang ...]              # re-render only, no new judge call
badminton-coach reference add <url|file> --name viktor-2018 [--start 0:12 --end 0:20]
badminton-coach reference list
badminton-coach frames <workspace> --swing 2 --from 0.40 --to 0.55 --step 1 [--crop upper]   # judge's own tool
badminton-coach view [--port 8765]
badminton-coach models download    # rtmlib + RacketVision (git clone + HF checkpoints) + SAM 3D Body (gated HF)
```
`--from-stage` (see `STAGE_ORDER` in `cli.py`) forces recompute from that stage onward; earlier stages
still use their on-disk cache. `run_measurement_pipeline()` (`pipeline.py`) only takes one `force` flag
for its whole span (ingest..swings), so `analyze` forces that whole span whenever `--from-stage` names
anything inside it.

## Rule of thumb
- At every step, if difficulty or approach is uncertain, first ask "has someone already solved this?" — search
  before building; prefer maintained existing solutions. See **Prior art** notes in the plan.

## Blockers
- If a task step is blocked, finish the rest, write the blocker to `docs/BLOCKERS.md`, stop.
