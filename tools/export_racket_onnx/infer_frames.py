"""Run RacketVision's RTMDet-M (racket detection) + RTMPose-M (5-keypoint racket pose) over a frame
range of a video. Runs in the pinned Python-3.10 env (see README.md) — invoked as a subprocess by
`badminton_coach.measure.racket` in the main environment.

Output JSON (written to --out): {"<frame_index>": [{"bbox": [x1,y1,x2,y2], "bbox_score": float,
"keypoints": [[x,y], ...5], "keypoint_scores": [...5]}, ...]} — one entry per frame that had at least one
detection above --det-score-thr, ordered by descending bbox_score within a frame.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

BADMINTON_RACKET_LABEL = 0

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_RACKETVISION_SRC = _REPO_ROOT / "data" / "models" / "racketvision" / "src" / "source" / "RacketPose"
_CHECKPOINTS = _REPO_ROOT / "data" / "models" / "racketvision" / "checkpoints"

DEFAULT_DET_CONFIG = _RACKETVISION_SRC / "configs" / "detection" / "rtmdet_m_racket_infer.py"
DEFAULT_DET_CHECKPOINT = _CHECKPOINTS / "epoch_300.pth"
DEFAULT_POSE_CONFIG = _RACKETVISION_SRC / "configs" / "pose" / "rtmpose_m_racket_infer.py"
DEFAULT_POSE_CHECKPOINT = _CHECKPOINTS / "best_PCK_epoch_90.pth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="exclusive; default = end of video")
    p.add_argument("--det-config", type=Path, default=DEFAULT_DET_CONFIG)
    p.add_argument("--det-checkpoint", type=Path, default=DEFAULT_DET_CHECKPOINT)
    p.add_argument("--pose-config", type=Path, default=DEFAULT_POSE_CONFIG)
    p.add_argument("--pose-checkpoint", type=Path, default=DEFAULT_POSE_CHECKPOINT)
    p.add_argument("--det-score-thr", type=float, default=0.3)
    p.add_argument("--max-rackets-per-frame", type=int, default=3)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from mmdet.apis import inference_detector, init_detector
    from mmengine.registry import DefaultScope
    from mmpose.apis import inference_topdown, init_model

    # mmdet and mmpose each set a process-global `DefaultScope` (to "mmdet"/"mmpose" respectively) when
    # their model is constructed; whichever was built *last* silently wins for registry lookups (e.g.
    # `PackDetInputs`) inside the *other* library's inference calls unless the scope is re-asserted right
    # before each call. Known mm-ecosystem gotcha when mixing mmdet+mmpose in one process.
    det_model = init_detector(str(args.det_config), str(args.det_checkpoint), device=args.device)
    pose_model = init_model(str(args.pose_config), str(args.pose_checkpoint), device=args.device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {args.video}")

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    results: dict[str, list[dict]] = {}
    idx = args.start
    started = time.monotonic()
    n_processed = 0

    while True:
        if args.end is not None and idx >= args.end:
            break
        ok, frame = cap.read()
        if not ok:
            break

        with DefaultScope.overwrite_default_scope("mmdet"):
            det_result = inference_detector(det_model, frame)
        inst = det_result.pred_instances
        scores = inst.scores.cpu().numpy()
        labels = inst.labels.cpu().numpy()
        bboxes = inst.bboxes.cpu().numpy()

        keep = (labels == BADMINTON_RACKET_LABEL) & (scores >= args.det_score_thr)
        kept_scores = scores[keep]
        kept_bboxes = bboxes[keep]
        order = np.argsort(-kept_scores)[: args.max_rackets_per_frame]

        frame_dets = []
        for i in order:
            bbox = kept_bboxes[i]
            bbox_score = float(kept_scores[i])
            with DefaultScope.overwrite_default_scope("mmpose"):
                pose_results = inference_topdown(pose_model, frame, bboxes=bbox[None, :])
            r = pose_results[0]
            frame_dets.append(
                {
                    "bbox": bbox.tolist(),
                    "bbox_score": bbox_score,
                    "keypoints": r.pred_instances.keypoints[0].tolist(),
                    "keypoint_scores": r.pred_instances.keypoint_scores[0].tolist(),
                }
            )

        if frame_dets:
            results[str(idx)] = frame_dets

        idx += 1
        n_processed += 1

    cap.release()

    elapsed = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))

    n_with_detection = len(results)
    summary = {
        "n_frames_processed": n_processed,
        "n_frames_with_racket": n_with_detection,
        "elapsed_s": round(elapsed, 2),
        "fps": round(n_processed / elapsed, 2) if elapsed > 0 else 0.0,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
