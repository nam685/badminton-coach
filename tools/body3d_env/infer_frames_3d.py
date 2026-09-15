"""Run SAM 3D Body over a specific (sparse) set of frames + bboxes from a video. Runs in the pinned
Python-3.11 env (this directory) — invoked as a subprocess by `badminton_coach.measure.body3d` in the
main environment. See docs/body3d.md for why this env exists and what it deliberately skips
(detectron2/SAM2/MoGe — we pass our own bbox, no detector/segmentor/FOV needed).

**fp32 only**: sparse CUDA matmul (used inside the MHR pose-correctives model) has no fp16
implementation (`RuntimeError: "addmm_sparse_cuda" not implemented for 'Half'`, confirmed empirically) —
`torch.autocast(dtype=float16)` around `process_one_image` crashes. Contrary to the original spec
wording ("run in fp16 to fit 4GB"), this only runs in fp32; measured VRAM use was well within 4GB for
the ViT-H (631M param) checkpoint on a single 256x256-ish crop, so this hasn't been a problem in
practice. If it does OOM on a particular frame, the caller's fallback chain handles it (see body3d.py).

Input (--request, a JSON file): {"frames": [int, ...], "bboxes": [[x1,y1,x2,y2], ...]} (same length,
frames need not be contiguous — only the frames inside swing windows are ever requested).
Output (--out, an .npz file): frame_indices[N], keypoints_3d[N,70,3], keypoints_2d[N,70,2],
cam_t[N,3], focal_length[N]. Frames the model fails on are simply omitted (frame_indices is the source
of truth for which frames succeeded).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
DEFAULT_SAM3DB_SRC = _REPO_ROOT / "data" / "models" / "sam3db_src"
DEFAULT_CHECKPOINT = _REPO_ROOT / "data" / "models" / "sam3db_vith" / "model.ckpt"
DEFAULT_MHR = _REPO_ROOT / "data" / "models" / "sam3db_vith" / "assets" / "mhr_model.pt"
DEFAULT_CONFIG = _REPO_ROOT / "data" / "models" / "sam3db_vith" / "model_config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sam3db-src", type=Path, default=DEFAULT_SAM3DB_SRC)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--mhr", type=Path, default=DEFAULT_MHR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.sam3db_src))

    import torch
    from sam_3d_body import SAM3DBodyEstimator
    from sam_3d_body.models.meta_arch import SAM3DBody
    from sam_3d_body.utils.checkpoint import load_state_dict
    from sam_3d_body.utils.config import get_config

    request = json.loads(args.request.read_text())
    frames_wanted: list[int] = request["frames"]
    bboxes: list[list[float]] = request["bboxes"]

    model_cfg = get_config(str(args.config))
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(args.mhr)
    model_cfg.freeze()

    model = SAM3DBody(model_cfg)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False, mmap=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_state_dict(model, state_dict, strict=False)
    model = model.to("cuda")
    model.eval()

    estimator = SAM3DBodyEstimator(model, model_cfg)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {args.video}")

    out_frame_indices: list[int] = []
    out_kpts3d: list[np.ndarray] = []
    out_kpts2d: list[np.ndarray] = []
    out_cam_t: list[np.ndarray] = []
    out_focal: list[float] = []

    started = time.monotonic()
    for frame_idx, bbox in zip(frames_wanted, bboxes, strict=True):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bbox_arr = np.array([bbox], dtype=np.float32)
        try:
            outputs = estimator.process_one_image(frame_rgb, bboxes=bbox_arr, use_mask=False, inference_type="body")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        if not outputs:
            continue
        o = outputs[0]
        out_frame_indices.append(frame_idx)
        out_kpts3d.append(np.asarray(o["pred_keypoints_3d"], dtype=np.float32))
        out_kpts2d.append(np.asarray(o["pred_keypoints_2d"], dtype=np.float32))
        out_cam_t.append(np.asarray(o["pred_cam_t"], dtype=np.float32))
        out_focal.append(float(o["focal_length"]))

    cap.release()
    elapsed = time.monotonic() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if out_frame_indices:
        np.savez_compressed(
            args.out,
            frame_indices=np.array(out_frame_indices, dtype=np.int64),
            keypoints_3d=np.stack(out_kpts3d),
            keypoints_2d=np.stack(out_kpts2d),
            cam_t=np.stack(out_cam_t),
            focal_length=np.array(out_focal, dtype=np.float32),
        )
    else:
        np.savez_compressed(
            args.out,
            frame_indices=np.zeros(0, dtype=np.int64),
            keypoints_3d=np.zeros((0, 70, 3), dtype=np.float32),
            keypoints_2d=np.zeros((0, 70, 2), dtype=np.float32),
            cam_t=np.zeros((0, 3), dtype=np.float32),
            focal_length=np.zeros(0, dtype=np.float32),
        )

    summary = {
        "n_requested": len(frames_wanted),
        "n_succeeded": len(out_frame_indices),
        "elapsed_s": round(elapsed, 2),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
