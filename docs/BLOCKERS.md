# Blockers

## SAM 3D Body checkpoint download — gated repo, terms not yet accepted (Task 7b)

`hf download facebook/sam-3d-body-dinov3 --local-dir data/models/sam3db` fails with:
```
Error: Access denied. This repository requires approval.
```
`hf auth whoami` confirms we're logged in as `nam685`, and downloading just `README.md` from the same
repo succeeds (model cards on gated repos are usually public) — the checkpoint files themselves are
still gated behind an approval Nam hasn't clicked through yet.

**Action needed (Nam):** open https://huggingface.co/facebook/sam-3d-body-dinov3 while logged in, click
"Agree and access repository", fill the short form. Then re-run `uv run badminton-coach models download`
(or `hf download facebook/sam-3d-body-dinov3 --local-dir data/models/sam3db` directly).

Everything else in Task 7b (the `body3d.py`/`metrics3d.py` code, `sam-3d-body` package install) can and
did proceed without the checkpoint; only the actual model download/inference is blocked until approval.
