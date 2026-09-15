"""Stage 1c — shuttle: TrackNetV3 (vendored from RacketVision, `measure/tracknet/`), run on our own
video's frames directly (no dataset directory layout needed) with a **median background computed from
the clip itself** (there is no separate "empty court" capture available for our footage, unlike
RacketVision's benchmark dataset — the median-across-frames approximation works because the court/net
background is static while the player and shuttle move).

Preprocessing/postprocessing (batch construction, heatmap contour peak-finding, confidence calculation)
mirrors RacketVision's `source/BallTrack/inference.py::BallInferencer` — see LICENSE_NOTICE.md.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from badminton_coach.config import CONFIG, Config
from badminton_coach.measure.tracknet import SEQ_LEN, TRACKNET_HEIGHT, TRACKNET_WIDTH, TrackNetV3
from badminton_coach.run import RunDir
from badminton_coach.video import sha256_file

STAGE_VERSION = "1"

CHECKPOINT_PATH = CONFIG.data_dir / "models" / "racketvision" / "checkpoints" / "balltrack_best.pth"


def _remove_ddp_prefix(state_dict: dict) -> dict:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def compute_median(video_path: str | Path, n_samples: int = 60) -> np.ndarray:
    """Median background frame (BGR, original resolution) sampled evenly across the clip."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames = max(n_frames, 1)
        step = max(n_frames // n_samples, 1)
        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        if not frames:
            raise RuntimeError(f"no frames read from {video_path}")
        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    finally:
        cap.release()


class ShuttleTracker:
    def __init__(self, checkpoint_path: str | Path = CHECKPOINT_PATH, device: str = "cuda", thre: float = 0.5) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = TrackNetV3()
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(_remove_ddp_prefix(state))
        self.model.to(self.device).eval()
        self.thre = thre

    def _predict_location(self, heatmap: np.ndarray) -> tuple[int, int, int, int, float]:
        mask = heatmap > self.thre
        if mask.max() == 0:
            return 0, 0, 0, 0, 0.0
        mask_img = mask.astype("uint8") * 255
        cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0, 0, 0, 0, 0.0
        rects = [cv2.boundingRect(c) for c in cnts]
        x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
        conf = float(np.mean(heatmap[y : y + h, x : x + w]))
        return x, y, w, h, conf

    def __call__(self, video_path: str | Path, median: np.ndarray | None = None, batch_size: int = 8) -> list[dict]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video {video_path}")
        raw_frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                raw_frames.append(frame)
        finally:
            cap.release()
        if not raw_frames:
            return []

        img_h, img_w = raw_frames[0].shape[:2]
        sx, sy = img_w / TRACKNET_WIDTH, img_h / TRACKNET_HEIGHT

        if median is None:
            median = compute_median(video_path)
        median_r = cv2.resize(median, (TRACKNET_WIDTH, TRACKNET_HEIGHT)) / 255.0
        median_r = np.moveaxis(np.expand_dims(median_r, 0), -1, 1)  # (1, 3, H, W)

        resized = (
            np.array([cv2.resize(f, (TRACKNET_WIDTH, TRACKNET_HEIGHT)) for f in raw_frames], dtype=np.float32) / 255.0
        )

        results: list[dict] = []
        n = len(resized)
        i = 0
        current_batch_size = batch_size
        while i < n:
            end = min(n, i + current_batch_size)
            batch = []
            for k in range(i, end):
                fids = [max(0, j) for j in range(k - SEQ_LEN, k)]
                seq = resized[fids]
                data = np.concatenate([deepcopy(median_r), np.moveaxis(seq, -1, 1)], 0).reshape(
                    -1, TRACKNET_HEIGHT, TRACKNET_WIDTH
                )
                batch.append(data)
            batch_t = torch.from_numpy(np.array(batch)).float().to(self.device)

            try:
                with torch.no_grad():
                    if self.device.type == "cuda":
                        with torch.amp.autocast("cuda"):
                            preds = self.model(frames=batch_t)
                    else:
                        preds = self.model(frames=batch_t)
                preds = preds.detach().cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                # 4GB VRAM is tight for this model at batch_size=20 (observed ~3.9GB used on the test
                # fixture) — back off the batch size and retry the same range; fall back to CPU (slow,
                # but latency is not a concern) if a single-sample batch still doesn't fit.
                torch.cuda.empty_cache()
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    continue
                self.device = torch.device("cpu")
                self.model.to(self.device)
                continue

            for j in range(preds.shape[0]):
                hm = preds[j][0]
                x, y, w, h, conf = self._predict_location(hm)
                cx = int((x + w / 2) * sx)
                cy = int((y + h / 2) * sy)
                vis = 0 if cx == 0 and cy == 0 else 1
                results.append({"Frame": i + j, "X": cx, "Y": cy, "Visibility": vis, "Confidence": round(conf, 4)})
            i = end
        return results


def track_shuttle(
    run: RunDir,
    normalized_video: str | Path,
    cfg: Config = CONFIG,
    device: str = "cuda",
    force: bool = False,
) -> dict[str, Any]:
    """Run shuttle tracking over `normalized_video`, writing `measure/shuttle.csv`
    (Frame, X, Y, Visibility, Confidence — original-resolution pixel coords)."""
    input_hash = f"{sha256_file(Path(normalized_video))}:{device}"

    out_dir = run.root / "measure"
    csv_path = out_dir / "shuttle.csv"

    with run.stage("measure_shuttle", version=STAGE_VERSION, input_hash=input_hash, force=force) as st:
        if st.skip:
            df = pd.read_csv(csv_path)
            return {"n_frames": len(df), "frac_visible": float((df["Visibility"] == 1).mean()) if len(df) else 0.0}

        out_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = cfg.data_dir / "models" / "racketvision" / "checkpoints" / "balltrack_best.pth"
        if not checkpoint_path.exists():
            raise RuntimeError(f"{checkpoint_path} not found — run `badminton-coach models download` first")

        tracker = ShuttleTracker(checkpoint_path=checkpoint_path, device=device)
        results = tracker(normalized_video)

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

        frac_visible = float((df["Visibility"] == 1).mean()) if len(df) else 0.0
        run.mark_extra(st, n_frames=len(df), frac_visible=round(frac_visible, 3), device=device)
        return {"n_frames": len(df), "frac_visible": round(frac_visible, 3)}
