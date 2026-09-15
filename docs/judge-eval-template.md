# Judge evaluation log

Manual scoring template (spec §9): for each real judge run, list every finding and score it
true/false/partly, plus overall impressions. Copy this template into a run's own
`judge/eval.md` when doing a real evaluation pass (Task 11, or ad hoc).

## Run: `data/players/<player>/<timestamp>/`

- Source clip:
- Findings.json: `judge/findings.json`

| # | Finding title | True / False / Partly | Notes |
|---|---|---|---|
| 1 | | | |

**Strengths listed — accurate?**

**Priority order — did it pick the right thing to fix first?**

**Anything the judge missed that a human coach would have caught?**

**Anything hallucinated (not grounded in metrics.json/rubric.md)?**

---

## Session log (2026-09-15) — first real end-to-end judge run

Ran the full pipeline (ingest → measure_body/racket/shuttle → track → swings → metrics, **no** body3d
and **no** references, for speed) against the test fixture, then the real `claude -p` judge with a
genuine `CLAUDE_CODE_OAUTH_TOKEN`. This was the first time the whole chain — including the actual
subscription-billed judge call — ran together.

**Outcome: `measurement_quality.ok = false`, 0 findings, `confidence: "low"`.** The judge correctly:

1. Noticed the skeleton overlay was drawn on a **different person** than the one swinging in the raw
   frames (a two-person clip — see Task 4's earlier note) and refused to report technique findings
   about a bystander.
2. Called out a **mid-swing identity switch** in swing 0 as the specific cause of two impossible
   numbers (`contact_height_vs_nose = -0.98`, i.e. the racket a full torso-length *below* the nose).
3. Cross-checked the metrics against the plots and correctly said the elbow-angle trace (124-133°
   throughout) doesn't look like a real swing.
4. Noted `body3d`/`sequence_ms`/`weight_transfer` were entirely absent (expected — skipped for speed in
   this run) and that no reference clips were loaded (`n_ref = 0` everywhere) — both correct, both
   flagged rather than silently glossed over.
5. Attempted to use the `frames` tool to pull more evidence and correctly reported it failing
   ("couldn't open the source video from the workspace") rather than silently proceeding without it.

**This is the intended behavior for broken measurement input** — the system prompt's rule ("if
measurement looks broken... say so first... lower your confidence") worked exactly as designed, and the
judge's diagnosis of *why* it was broken was accurate and specific enough to debug from directly.

**Two real bugs found from this one run, both fixed same-day:**

- **Player selection picked the wrong (static) person for most of the clip.** Two people are visible
  throughout (one swinging, one a mostly-still bystander who also happens to be holding a racket); the
  selection score (bbox area × racket-proximity × frame-to-frame continuity) couldn't tell "calmly
  holding a racket" from "actively swinging one," and continuity's stickiness then locked in whichever
  person won the very first (low-information) frame. Fixed with two changes to `track.py`: (1) a
  racket-*motion* signal (how far the nearest racket's head moved since the previous frame) now boosts
  a candidate's score, so a static held racket no longer scores like a swinging one; (2) the continuity
  floor is relaxed when a candidate's racket signal is decisive, so a strong, sustained motion signal
  can override a bad early lock-in instead of being permanently stuck. A post-hoc majority-vote pass
  (`_stabilize_selection`) additionally cleans up isolated single-frame flips. Verified on the real data:
  the correct player is now chosen for the large majority of the swing (was: the wrong player for
  nearly the whole thing); a small residual block of frames around the fastest motion is still
  occasionally wrong — a known, documented limitation (see `track.py`'s module docstring and
  `docs/reference-values.md`'s Task 8 entry), not silently claimed as fully solved.
- **`ingest()` persisted a relative path.** `normalized_path` was stored exactly as given, so it only
  resolved correctly from whatever cwd `ingest()` happened to run in — broke `badminton-coach frames`
  when the judge (correctly) invoked it from its own workspace directory, a different cwd. Fixed by
  resolving all input paths to absolute before anything is persisted.
- Also found and fixed: `badminton-coach` wasn't on `PATH` at all outside the project's own `uv run` —
  the judge's `Bash(badminton-coach frames:*)` tool call had no working command to run regardless of the
  path bug above. Fixed with `uv tool install --editable .` (now documented as a required setup step —
  see `README.md`/`CLAUDE.md`).

**Not yet re-verified**: whether the judge produces genuine technique findings once given correctly-
tracked data (this run's `measurement_quality.ok=false` short-circuited before reaching that). Next
real-data run (ideally with body3d and a reference clip too) should exercise that path.
