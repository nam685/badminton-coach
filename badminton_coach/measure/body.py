"""Stage 1a — body pose: RTMPose (Halpe-26, feet included) via `rtmlib.BodyWithFeet`, run over every
frame of the ingested clip at inference resolution. Persists every detected person per frame (player
selection happens later, in `track.py`) — spec §3.2/§3.3.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from badminton_coach.config import CONFIG, Config
from badminton_coach.measure.keypoints import BODY_NUM_KPTS
from badminton_coach.run import RunDir
from badminton_coach.video import iter_frames, sha256_file

STAGE_VERSION = "1"


class BodyEstimator:
    """Thin wrapper around `rtmlib.BodyWithFeet`. Constructing it downloads/caches the ONNX models on
    first use (rtmlib's own cache, under `~/.cache/rtmlib` by default)."""

    def __init__(self, device: str = "cuda", mode: str = "performance", backend: str = "onnxruntime") -> None:
        # rtmlib caches downloaded ONNX models under $XDG_CACHE_HOME/rtmlib (default ~/.cache/rtmlib).
        # Point it at our data dir so all model weights live in one place (spec §7's `data/models/`)
        # and `--models models_dir absent` is a meaningful "not downloaded yet" gate for tests.
        os.environ.setdefault("XDG_CACHE_HOME", str(CONFIG.models_dir))
        from rtmlib import BodyWithFeet

        self._model = BodyWithFeet(mode=mode, backend=backend, device=device)

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (keypoints[n_persons, 26, 2], scores[n_persons, 26])."""
        return self._model(frame)


def measure_body(
    run: RunDir,
    normalized_video: str | Path,
    cfg: Config = CONFIG,
    scale_long_side: int | None = None,
    device: str = "cuda",
    force: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Run body pose estimation over every frame of `normalized_video`, writing `measure/body.npz`
    (`kpts[T, max_persons, 26, 3]` — x, y, score; NaN-padded — and `n_persons[T]`).

    Returns a small summary dict (also recorded into `run.json`'s stage entry).
    """
    scale_long_side = scale_long_side or cfg.inference_long_side
    input_hash = f"{sha256_file(Path(normalized_video))}:{scale_long_side}:{device}"

    out_dir = run.root / "measure"
    out_path = out_dir / "body.npz"

    with run.stage("measure_body", version=STAGE_VERSION, input_hash=input_hash, force=force) as st:
        if st.skip:
            data = np.load(out_path)
            return {"n_frames": int(data["n_persons"].shape[0]), "max_persons": int(data["kpts"].shape[1])}

        out_dir.mkdir(parents=True, exist_ok=True)
        estimator = BodyEstimator(device=device)

        per_frame_kpts: list[np.ndarray] = []
        per_frame_scores: list[np.ndarray] = []

        if debug:
            debug_dir = out_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

        n_frames_total = 0
        import time

        started = time.monotonic()
        for idx, frame in tqdm(iter_frames(normalized_video, scale_long_side=scale_long_side), desc="body pose"):
            kpts, scores = estimator(frame)
            per_frame_kpts.append(kpts)
            per_frame_scores.append(scores)
            n_frames_total += 1

            if debug and idx % 10 == 0:
                from rtmlib import draw_skeleton

                img = draw_skeleton(frame.copy(), kpts, scores, kpt_thr=cfg.body_kpt_conf_thr)
                import cv2

                cv2.imwrite(str(debug_dir / f"body_{idx:06d}.jpg"), img)

        elapsed = time.monotonic() - started
        fps = n_frames_total / elapsed if elapsed > 0 else 0.0

        max_persons = max((k.shape[0] for k in per_frame_kpts), default=0)
        max_persons = max(max_persons, 1)

        kpts_arr = np.full((n_frames_total, max_persons, BODY_NUM_KPTS, 3), np.nan, dtype=np.float32)
        n_persons_arr = np.zeros(n_frames_total, dtype=np.int32)
        for t, (kpts, scores) in enumerate(zip(per_frame_kpts, per_frame_scores, strict=True)):
            n = kpts.shape[0]
            n_persons_arr[t] = n
            if n == 0:
                continue
            kpts_arr[t, :n, :, 0:2] = kpts
            kpts_arr[t, :n, :, 2] = scores

        np.savez_compressed(out_path, kpts=kpts_arr, n_persons=n_persons_arr, scale_long_side=scale_long_side)

        run.mark_extra(st, fps=round(fps, 2), n_frames=n_frames_total, max_persons=max_persons, device=device)
        return {"n_frames": n_frames_total, "max_persons": max_persons, "fps": round(fps, 2)}
