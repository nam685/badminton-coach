# badminton-coach

Record your badminton stroke on a phone → get an annotated video and a coach-style report of what is wrong.

**How it works:** specialist models measure (RTMPose body pose, RacketVision racket keypoints + TrackNetV3
shuttle), the pipeline turns that into per-swing metrics and compares them to pro reference clips, and
Claude (Opus, via `claude -p` with a subscription OAuth token) judges the evidence like a coach. The
renderer draws the findings on the video. Every intermediate result is written to disk and viewable with
`badminton-coach view`.

Status: **design done, implementation pending.** See
`docs/superpowers/specs/2026-09-15-badminton-coach-v1-design.md` and
`docs/superpowers/plans/2026-09-15-badminton-coach-v1-plan.md`.

v1 targets the forehand overhead **clear**, local GPU (RTX 3050 Ti, WSL2). A web version for friends
(nam685.de/badminton) is a separate ticket on `nam685/nam-website`.
