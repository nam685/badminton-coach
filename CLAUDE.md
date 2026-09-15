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

## Rule of thumb
- At every step, if difficulty or approach is uncertain, first ask "has someone already solved this?" — search
  before building; prefer maintained existing solutions. See **Prior art** notes in the plan.

## Blockers
- If a task step is blocked, finish the rest, write the blocker to `docs/BLOCKERS.md`, stop.
