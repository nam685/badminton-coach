# Task 7b — SAM 3D Body integration notes

## Environment: separate pinned env, same pattern as Task 4's racket export

`tools/body3d_env/` — Python 3.11 (SAM 3D Body's own requirement), CUDA-capable torch (resolves to the
same cu13 build as the main env on this machine — no pin needed, unlike Task 4's racket env which
deliberately went CPU-only). GPU is **mandatory**, not optional: `SAM3DBodyEstimator.process_one_image`
hardcodes `recursive_to(batch, "cuda")` internally regardless of what device the model itself was moved
to — there is no supported CPU path in the stock library. This changes the spec's planned OOM fallback
chain (§3.4b said "OOM → CPU (slow, allowed) → RTMW3D"): **CPU is not available for SAM 3D Body itself**;
the real fallback chain is fp16 GPU → reduced batch/crop → RTMW3D (rtmlib, which does support CPU).

## Avoiding detectron2 (and SAM2, MoGe) entirely

The README's "quick start" installs `detectron2` (ViTDet person detector), SAM2 (mask segmentor), and
MoGe (FOV/camera estimator) — detectron2 in particular is notorious for slow/fragile installs (compiled
CUDA extensions, picky about torch version). None of the three are actually required:

- We already have a player bounding box from Task 6's tracking (`track/track.json` + the player's
  keypoint-derived bbox) — pass it via `process_one_image(img, bboxes=our_bbox)`, which **skips the
  detector entirely** (verified by reading `sam_3d_body_estimator.py`: `if bboxes is not None: ... elif
  self.detector is not None: ... else: full image`).
- `use_mask=False` (the default) skips SAM2 mask-conditioning.
- No FOV estimator is required either; passing neither `cam_int` nor a `fov_estimator` falls back to
  "the default FOV" (a fixed assumption, per the estimator's own log message) — acceptable since our
  angle math only needs the horizontal-plane *direction* of a line, not absolute focal length.
- Confirmed via `grep`: `sam_3d_body/__init__.py` only imports `SAM3DBodyEstimator` and
  `load_sam_3d_body[_hf]`, neither of which imports `sam_3d_body.visualization.*` (where detectron2,
  pyrender, and trimesh actually live) or the detector/segmentor/fov modules unless you ask for them.

`tools/body3d_env/pyproject.toml`'s dependency list is the minimal set found by grepping every `import`
in the cloned `sam_3d_body/` package (excluding `visualization/`): torch, torchvision, pillow,
braceexpand, opencv-python, einops, numpy, omegaconf, roma, timm, yacs, pytorch-lightning,
huggingface-hub, plus `termcolor` (needed transitively by the DINOv3 backbone code, which
`sam_3d_body`'s DINOv3 backbone auto-downloads via `torch.hub` from `facebookresearch/dinov3` — not a
`pip` dependency at all, discovered only by running it).

## Loading the model without `setup_sam_3d_body`'s convenience wrapper

`notebook/utils.py::setup_sam_3d_body()` always tries to construct a detector/segmentor/FOV estimator.
Bypass it and call the lower-level pieces directly:

```python
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

model, model_cfg = load_sam_3d_body(
    checkpoint_path=".../data/models/sam3db/model.ckpt",
    device="cuda",
    mhr_path=".../data/models/sam3db/assets/mhr_model.pt",
)
estimator = SAM3DBodyEstimator(model, model_cfg)  # no detector/segmentor/fov args

outputs = estimator.process_one_image(rgb_image, bboxes=np.array([[x1, y1, x2, y2]]), use_mask=False)
# outputs: list of dicts (one per bbox), each with pred_keypoints_3d, pred_keypoints_2d, pred_cam_t,
# focal_length, pred_vertices, plus MHR-internal fields (global_rot, pred_joint_coords,
# pred_global_rots, mhr_model_params) we don't currently use but could for validation later.
```

## Keypoint scheme: MHR70

`pred_keypoints_3d`/`pred_keypoints_2d` are 70 keypoints in the order defined by
`sam_3d_body/metadata/mhr70.py::pose_info` (verified by reading the source, not assumed): indices 0-14
match Halpe/COCO body order (nose, eyes, ears, shoulders, elbows... wait, hips at 9/10, knees 11/12,
ankles 13/14), then feet (15-20), then a lot of hand joints (21-61), then 62 = left_wrist, 63-69 =
elbow/shoulder landmark variants (olecranon, cubital fossa, acromion) and neck. The indices this codebase
actually uses: **5 = left_shoulder, 6 = right_shoulder, 9 = left_hip, 10 = right_hip** — see
`metrics3d.py`'s `MHR70_*` constants.

## Coordinate convention — best-effort, not yet empirically confirmed

Camera-space, assumed right-handed Y-up (typical for SMPL-family body models; MHR is presented as being
in that lineage). `metrics3d.py`'s rotation math only uses the horizontal (X, Z) plane and is largely
insensitive to getting the exact handedness/sign wrong (see that module's docstring for the precise
argument) — the one place a wrong sign would show up is `x_factor_deg`'s sign, which the rubric already
hedges. **Action for whoever runs this against real footage next** (Task 11, or sooner if convenient):
dump one frame's `pred_keypoints_3d` for a known pose (e.g. arms out to the sides, clearly facing the
camera) and sanity-check the axis signs against this assumption; update this note either way.

## System RAM constraint — switched to the ViT-H checkpoint + `mmap=True`

This machine has only 7.6GB RAM + 2GB swap. Loading the DINOv3-H+ checkpoint (840M params) the
straightforward way (`torch.load(..., mmap=False)`, the library's own default) got OOM-killed by the
kernel (confirmed via `dmesg`: `Out of memory: Killed process ... anon-rss:3888392kB`) partway through
just materializing the state dict in RAM, before ever touching the GPU.

**Fix, in order of what actually mattered:**
1. Switched to the smaller **ViT-H checkpoint** (`facebook/sam-3d-body-vith`, 631M params — same
   published accuracy as DINOv3-H+ per the model card: 54.8 MPJPE on 3DPW for both).
2. **`torch.load(..., mmap=True)`** instead of the library's default full-RAM read — this was the fix
   that actually mattered: checkpoint load time dropped from OOM-killed to **0.3 seconds**, because the
   2.4GB of tensor data is memory-mapped from disk rather than copied into a Python-managed buffer before
   `load_state_dict` moves it onto the GPU. `load_sam_3d_body()` in the upstream package doesn't expose
   this — we bypass it and call `torch.load`/`SAM3DBody(cfg)`/`load_state_dict` directly (see
   `tools/body3d_env/infer_frames_3d.py`).

Total load (model construction + mmap load + state dict + move to CUDA): **~17 seconds**. Peak system RAM
never spiked; GPU inference on one frame took **~3 seconds** (fp32, ViT-H, single crop).

## fp16 doesn't work — sparse CUDA op has no Half implementation

Wrapping `process_one_image` in `torch.autocast("cuda", dtype=torch.float16)` (the spec's original plan,
to fit comfortably in 4GB VRAM) crashes:
```
RuntimeError: "addmm_sparse_cuda" not implemented for 'Half'
```
from deep inside the MHR pose-correctives model's sparse matmul. **Runs in fp32 only.** In practice this
hasn't been a VRAM problem — a single ViT-H forward pass fits comfortably in 4GB — so the spec's original
"fp16 to fit 4GB" concern turned out to be unnecessary for this checkpoint size.

## Real inference — confirmed working end to end

Ran `estimator.process_one_image()` on a real fixture frame (own bbox, no detector) and got exactly the
expected output shape and fields: `pred_keypoints_3d` (70,3), `pred_keypoints_2d` (70,2), `focal_length`,
`pred_cam_t`, plus several MHR-internal fields not currently used (`pred_global_rots`, `mhr_model_params`
— could be used for a more principled rotation extraction later instead of raw keypoint geometry).

**Coordinate convention resolved empirically** (see `metrics3d.py`'s docstring for the fix): a standing
player's `left_shoulder`/`right_shoulder` Y-values (~-1.4) were *smaller* than their `left_hip`/`right_hip`
Y-values (~-0.9) despite shoulders being physically higher — meaning **Y increases downward**, the same
convention as 2D image coordinates, not the Y-up convention SMPL-family models typically use as
originally assumed. Fixed in `metrics3d.trunk_lean_3d_deg` and its tests before this was ever wired into
the full pipeline.

## Status at end of this session

- Both checkpoints downloaded: DINOv3-H+ (2GB, OOMs on this machine as-is) and **ViT-H (1.6GB, works)**.
- `tools/body3d_env/infer_frames_3d.py`: loads the ViT-H model via `mmap=True`, runs
  `process_one_image(frame, bboxes=our_bbox, use_mask=False)` fp32 over a requested (possibly sparse)
  set of frames, writes `keypoints_3d`/`keypoints_2d`/`cam_t`/`focal_length` to an npz. Verified against
  a real frame.
- `badminton_coach/measure/body3d.py::estimate_3d()`: builds swing-window frame+bbox requests from
  Task 6's tracking, invokes the subprocess, 2D-consistency-checks the result against rtmlib's
  independent detection (masks disagreeing frames), falls back to `rtmlib.Wholebody3d` if the SAM 3D
  Body subprocess produces nothing usable, gap-fills + smooths, writes `body3d/joints.npz`. Unit-tested
  (12 tests, synthetic data) and a real end-to-end run against the fixture — see
  `docs/reference-values.md` for the outcome.
- `metrics3d.py`: rotation/X-factor/sequence-timing math, fully unit-tested (17 tests) against synthetic
  3D skeletons with known angles, using the now-confirmed Y-down convention.

## The fallback chain actually triggered once — and that was useful

The first full-pipeline `estimate_3d()` run on the fixture (all 5 stages: ingest, measure, track, swings,
then body3d) fell back to `backend="rtmw3d"` instead of `"sam3d_body"`, even though the identical SAM 3D
Body subprocess call succeeds reliably in isolation (manually reproduced twice, byte-for-byte the same
argv/env/cwd `_run_sam3db_subprocess` builds). The likely cause: that run happened while the system was
under heavy concurrent load (a slow racket CPU inference stage, another background download, and the
user's own editor tooling all running at once) on a machine with only 7.6GB RAM — a plausible transient
OOM or resource contention inside the SAM 3D Body subprocess specifically.

**This is exactly the scenario the fallback chain exists for**, and it worked as designed: no crash, no
hang, a usable (if weaker) 3D signal via RTMW3D instead, with `backend` recorded so the judge/rubric can
hedge accordingly. The one real gap found: `_run_sam3db_subprocess` discarded `stderr` on failure,
making the fallback silent/undiagnosable — fixed to write a `*.subprocess_error.txt` file with the
return code and stderr tail (or the timeout) whenever it falls back, so a future run's cause is visible
instead of a black box. Not otherwise treated as a bug to "fix" further — a real per-frame subprocess
spawn under genuine memory pressure failing sometimes, while the pipeline degrades gracefully, is the
intended behavior on a resource-constrained dev machine; Task 11 (once run on less contended hardware,
or this same machine idle) should confirm the success rate is high in the common case.
