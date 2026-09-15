# Racket inference environment (pinned Python 3.10)

**Decision (2026-09-15): direct mmdet/mmpose inference, not mmdeploy ONNX export.**

The spec's plan allowed mmdeploy ONNX export to fail and fall back to a subprocess over mmdet/mmpose's
own Python inference APIs, budgeted "≤ 2h" before switching. Instead of attempting mmdeploy first, the
mmdet/mmpose path was validated directly and worked cleanly on the first real checkpoint+frame test (see
below) — so this environment runs that path permanently rather than spending the mmdeploy budget on an
approach that already has a working, "someone already solved this" alternative (RacketVision's own
`tools/inference.py` uses exactly this API). `racket.py` in the main package calls `infer_frames.py`
here as a subprocess over the run's frames; there is no ONNX racket model.

## What's pinned and why (all discovered empirically while setting this up)

- **Python 3.10, CPU-only torch 2.1.2/torchvision 0.16.2** (`download.pytorch.org/whl/cpu`) — matches
  RacketVision's documented pin. CPU-only is deliberate: it sidesteps RacketVision's own documented
  "`mmcv` may fail to build/match on CUDA≥12.8" warning entirely (this machine's driver reports CUDA 13),
  and speed doesn't matter for what is, per run, a few hundred CPU forward passes through small models.
- **`mmcv==2.1.0`** from the OpenMMLab CPU/torch2.1 flat wheel index (`[[tool.uv.index]] format="flat"`,
  `url=".../mmcv/dist/cpu/torch2.1/index.html"`) — the plain PyPI `mmcv` package doesn't ship prebuilt
  ops; this is OpenMMLab's own prebuilt-wheel index, matched to (torch version, CPU/CUDA).
- **`numpy<2`** — torch 2.1.2 was compiled against NumPy 1.x; NumPy 2.x installed alongside it raises
  `_ARRAY_API not found` on import. RacketVision's README explicitly pins this too.
- **`setuptools<81`** — `mmpose.apis` imports the legacy `pkg_resources` module at import time; recent
  setuptools (this environment resolved 84.0.0 by default) removed it in favor of `importlib.metadata`.
- **`chumpy` build fix** (`[tool.uv.extra-build-dependencies]`) — a transitive mmpose dependency whose
  `setup.py` does `import pip` at build time, which uv's isolated build env doesn't provide by default.

## Setup

```bash
cd tools/export_racket_onnx
uv sync
```

## Usage

```bash
uv run python infer_frames.py --video <path> --out <racket.json> [--start N] [--end N] \
    [--det-score-thr 0.3] [--device cpu]
```

Runs RTMDet-M (badminton-racket class only) + RTMPose-M (5 keypoints: top, bottom, handle, left, right)
over the given frame range of `<video>`, writing one JSON record per detected racket per frame. Called as
a subprocess by `badminton_coach.measure.racket` in the main environment — see that module for the exact
argv and output-parsing contract.

## Verified working (2026-09-15)

Loaded both checkpoints (`epoch_300.pth` detector, `best_PCK_epoch_90.pth` pose) against the
inference-only configs (`configs/detection/rtmdet_m_racket_infer.py`,
`configs/pose/rtmpose_m_racket_infer.py`) from the cloned RacketVision source
(`data/models/racketvision/src/`), and ran both on a real frame from the test fixture: detector found the
racket at score 0.86 (label 0, badminton_racket), pose model returned 5 plausible keypoints (scores
0.66-0.93) geometrically consistent with a racket (top/bottom on the shaft axis, left/right forming the
frame width between them).
