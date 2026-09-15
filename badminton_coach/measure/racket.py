"""Stage 1b — racket: RacketVision's RTMDet-M + RTMPose-M (5 keypoints: top, bottom, handle, left,
right), run via a subprocess into the pinned Python-3.10 environment at `tools/export_racket_onnx/`
(see that directory's README for why this is a direct mmdet/mmpose call rather than an ONNX export).

Runs against the run's *original-resolution* normalized video (not the inference-scaled copy used for
body pose) — the racket detector's own config already resizes/pads to 640x640 internally, so handing it
full resolution costs nothing extra and loses no detail.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from badminton_coach.config import CONFIG, Config
from badminton_coach.measure.keypoints import RACKET_NUM_KPTS, RACKET_SKELETON_EDGES
from badminton_coach.run import RunDir
from badminton_coach.video import iter_frames, sha256_file

STAGE_VERSION = "1"

_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "export_racket_onnx"
_INFER_SCRIPT = _TOOLS_DIR / "infer_frames.py"

DEFAULT_TIMEOUT_S = 3600


def run_racket_subprocess(
    video: str | Path,
    out_json: str | Path,
    start: int = 0,
    end: int | None = None,
    det_score_thr: float | None = None,
    device: str = "cpu",
    timeout: int = DEFAULT_TIMEOUT_S,
    max_rackets_per_frame: int = 3,
) -> dict[str, Any]:
    """Invoke `infer_frames.py` in the pinned env. Returns its summary dict (frames processed, fps, …).

    Raises RuntimeError with the subprocess's stderr tail on failure.
    """
    det_score_thr = CONFIG.racket_det_score_thr if det_score_thr is None else det_score_thr
    # Paths must be absolute: the subprocess runs with cwd=_TOOLS_DIR, not the caller's cwd.
    argv = [
        "uv",
        "run",
        "python",
        str(_INFER_SCRIPT),
        "--video",
        str(Path(video).resolve()),
        "--out",
        str(Path(out_json).resolve()),
        "--start",
        str(start),
        "--det-score-thr",
        str(det_score_thr),
        "--device",
        device,
        "--max-rackets-per-frame",
        str(max_rackets_per_frame),
    ]
    if end is not None:
        argv += ["--end", str(end)]

    # Strip VIRTUAL_ENV (and any stray UV_* env pointing at the main env) so the nested `uv run`
    # unambiguously targets the pinned env's own .venv instead of warning and falling back.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV" and not k.startswith("UV_")}

    result = subprocess.run(argv, cwd=str(_TOOLS_DIR), capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"racket inference subprocess failed: {result.stderr.strip()[-2000:]}")

    # The script's stdout is a single JSON summary line, possibly preceded by library warnings on
    # stderr (already captured above) — take the last non-empty stdout line.
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"racket inference subprocess produced no summary output: {result.stderr[-1000:]}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"racket inference subprocess summary was not JSON: {lines[-1]!r}") from exc


def measure_racket(
    run: RunDir,
    normalized_video: str | Path,
    cfg: Config = CONFIG,
    device: str = "cpu",
    force: bool = False,
    max_rackets_per_frame: int = 3,
    debug: bool = False,
) -> dict[str, Any]:
    """Run racket detection+pose over every frame of `normalized_video`, writing `measure/racket.npz`
    (`kpts[T, max_rackets, 5, 3]`, `bboxes[T, max_rackets, 4]`, `bbox_scores[T, max_rackets]`, all
    NaN-padded) and `measure/racket_raw.json` (the subprocess's raw per-frame output, kept for debugging).
    """
    input_hash = f"{sha256_file(Path(normalized_video))}:{cfg.racket_det_score_thr}:{device}"

    out_dir = run.root / "measure"
    npz_path = out_dir / "racket.npz"
    raw_json_path = out_dir / "racket_raw.json"

    with run.stage("measure_racket", version=STAGE_VERSION, input_hash=input_hash, force=force) as st:
        if st.skip:
            data = np.load(npz_path)
            return {"n_frames": int(data["kpts"].shape[0]), "max_rackets": int(data["kpts"].shape[1])}

        out_dir.mkdir(parents=True, exist_ok=True)

        if not _INFER_SCRIPT.exists():
            raise RuntimeError(
                f"{_INFER_SCRIPT} not found — is the pinned env set up? See tools/export_racket_onnx/README.md"
            )

        summary = run_racket_subprocess(
            normalized_video,
            raw_json_path,
            det_score_thr=cfg.racket_det_score_thr,
            device=device,
            max_rackets_per_frame=max_rackets_per_frame,
        )

        raw: dict[str, list[dict]] = json.loads(raw_json_path.read_text())
        n_frames_total = summary["n_frames_processed"]

        max_rackets = max((len(v) for v in raw.values()), default=0)
        max_rackets = max(max_rackets, 1)

        kpts_arr = np.full((n_frames_total, max_rackets, RACKET_NUM_KPTS, 3), np.nan, dtype=np.float32)
        bboxes_arr = np.full((n_frames_total, max_rackets, 4), np.nan, dtype=np.float32)
        bbox_scores_arr = np.full((n_frames_total, max_rackets), np.nan, dtype=np.float32)

        for frame_str, dets in raw.items():
            t = int(frame_str)
            if t >= n_frames_total:
                continue
            for i, det in enumerate(dets[:max_rackets]):
                kpts = np.array(det["keypoints"], dtype=np.float32)
                scores = np.array(det["keypoint_scores"], dtype=np.float32)
                kpts_arr[t, i, :, 0:2] = kpts
                kpts_arr[t, i, :, 2] = scores
                bboxes_arr[t, i] = np.array(det["bbox"], dtype=np.float32)
                bbox_scores_arr[t, i] = det["bbox_score"]

        np.savez_compressed(npz_path, kpts=kpts_arr, bboxes=bboxes_arr, bbox_scores=bbox_scores_arr)

        if debug:
            _write_debug_overlays(normalized_video, kpts_arr, bboxes_arr, out_dir / "debug")

        frac_with_racket = summary["n_frames_with_racket"] / n_frames_total if n_frames_total else 0.0
        run.mark_extra(
            st,
            fps=summary.get("fps", 0.0),
            n_frames=n_frames_total,
            max_rackets=max_rackets,
            frac_frames_with_racket=round(frac_with_racket, 3),
            device=device,
        )
        return {
            "n_frames": n_frames_total,
            "max_rackets": max_rackets,
            "frac_frames_with_racket": round(frac_with_racket, 3),
        }


def _write_debug_overlays(video: str | Path, kpts_arr: np.ndarray, bboxes_arr: np.ndarray, debug_dir: Path) -> None:
    import cv2

    debug_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in iter_frames(video):
        if idx % 10 != 0 or idx >= kpts_arr.shape[0]:
            continue
        img = frame.copy()
        for r in range(kpts_arr.shape[1]):
            if np.isnan(bboxes_arr[idx, r]).all():
                continue
            x1, y1, x2, y2 = bboxes_arr[idx, r].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 1)
            pts = kpts_arr[idx, r, :, 0:2]
            scores = kpts_arr[idx, r, :, 2]
            for a, b in RACKET_SKELETON_EDGES:
                if scores[a] > CONFIG.racket_kpt_conf_thr and scores[b] > CONFIG.racket_kpt_conf_thr:
                    pa, pb = tuple(pts[a].astype(int)), tuple(pts[b].astype(int))
                    cv2.line(img, pa, pb, (255, 255, 255), 1)
            for k in range(pts.shape[0]):
                if scores[k] > CONFIG.racket_kpt_conf_thr:
                    cv2.circle(img, tuple(pts[k].astype(int)), 3, (0, 0, 255), -1)
        cv2.imwrite(str(debug_dir / f"racket_{idx:06d}.jpg"), img)
