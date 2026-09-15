"""Keypoint index/name constants shared across measurement, tracking, metrics, and rendering.

Body: Halpe-26 (rtmlib `BodyWithFeet`), verified 2026-09-15 against the installed rtmlib 0.0.16
source (`rtmlib/visualization/skeleton/halpe26.py`) — index order confirmed, not assumed.
Racket: RacketVision's 5-keypoint scheme (top, bottom, handle, left, right), verified against the
RacketVision repo's `rtmpose_m_racket.py` config (cloned into `data/models/racketvision/src/`).
"""

from __future__ import annotations

# --- Body: Halpe-26 ---

BODY_NAMES: list[str] = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
    "head",  # 17
    "neck",  # 18
    "hip",  # 19 (pelvis midpoint)
    "left_big_toe",  # 20
    "right_big_toe",  # 21
    "left_small_toe",  # 22
    "right_small_toe",  # 23
    "left_heel",  # 24
    "right_heel",  # 25
]
BODY_NUM_KPTS = len(BODY_NAMES)
BODY_IDX: dict[str, int] = {name: i for i, name in enumerate(BODY_NAMES)}

# Convenience aliases used throughout metrics.py / track.py
NOSE = BODY_IDX["nose"]
LEFT_SHOULDER = BODY_IDX["left_shoulder"]
RIGHT_SHOULDER = BODY_IDX["right_shoulder"]
LEFT_ELBOW = BODY_IDX["left_elbow"]
RIGHT_ELBOW = BODY_IDX["right_elbow"]
LEFT_WRIST = BODY_IDX["left_wrist"]
RIGHT_WRIST = BODY_IDX["right_wrist"]
LEFT_HIP = BODY_IDX["left_hip"]
RIGHT_HIP = BODY_IDX["right_hip"]
LEFT_KNEE = BODY_IDX["left_knee"]
RIGHT_KNEE = BODY_IDX["right_knee"]
LEFT_ANKLE = BODY_IDX["left_ankle"]
RIGHT_ANKLE = BODY_IDX["right_ankle"]
NECK = BODY_IDX["neck"]
PELVIS = BODY_IDX["hip"]

# Left/right swap map (for the handedness mirror used by metrics.py sign conventions)
BODY_SWAP: dict[int, int] = {}
for _a, _b in [
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_big_toe", "right_big_toe"),
    ("left_small_toe", "right_small_toe"),
    ("left_heel", "right_heel"),
]:
    BODY_SWAP[BODY_IDX[_a]] = BODY_IDX[_b]
    BODY_SWAP[BODY_IDX[_b]] = BODY_IDX[_a]

BODY_SKELETON_EDGES: list[tuple[int, int]] = [
    (BODY_IDX["left_ankle"], BODY_IDX["left_knee"]),
    (BODY_IDX["left_knee"], BODY_IDX["left_hip"]),
    (BODY_IDX["left_hip"], BODY_IDX["hip"]),
    (BODY_IDX["right_ankle"], BODY_IDX["right_knee"]),
    (BODY_IDX["right_knee"], BODY_IDX["right_hip"]),
    (BODY_IDX["right_hip"], BODY_IDX["hip"]),
    (BODY_IDX["head"], BODY_IDX["neck"]),
    (BODY_IDX["neck"], BODY_IDX["hip"]),
    (BODY_IDX["neck"], BODY_IDX["left_shoulder"]),
    (BODY_IDX["left_shoulder"], BODY_IDX["left_elbow"]),
    (BODY_IDX["left_elbow"], BODY_IDX["left_wrist"]),
    (BODY_IDX["neck"], BODY_IDX["right_shoulder"]),
    (BODY_IDX["right_shoulder"], BODY_IDX["right_elbow"]),
    (BODY_IDX["right_elbow"], BODY_IDX["right_wrist"]),
    (BODY_IDX["left_ankle"], BODY_IDX["left_big_toe"]),
    (BODY_IDX["left_ankle"], BODY_IDX["left_small_toe"]),
    (BODY_IDX["left_ankle"], BODY_IDX["left_heel"]),
    (BODY_IDX["right_ankle"], BODY_IDX["right_big_toe"]),
    (BODY_IDX["right_ankle"], BODY_IDX["right_small_toe"]),
    (BODY_IDX["right_ankle"], BODY_IDX["right_heel"]),
]

# --- Racket: RacketVision 5-keypoint scheme ---

RACKET_NAMES: list[str] = ["top", "bottom", "handle", "left", "right"]
RACKET_NUM_KPTS = len(RACKET_NAMES)
RACKET_IDX: dict[str, int] = {name: i for i, name in enumerate(RACKET_NAMES)}
RACKET_TOP = RACKET_IDX["top"]
RACKET_BOTTOM = RACKET_IDX["bottom"]
RACKET_HANDLE = RACKET_IDX["handle"]
RACKET_LEFT = RACKET_IDX["left"]
RACKET_RIGHT = RACKET_IDX["right"]

# Verified 2026-09-15 against RacketVision's actual inference config
# (data/models/racketvision/src/source/RacketPose/configs/pose/rtmpose_m_racket_infer.py `skeleton_info`).
RACKET_SKELETON_EDGES: list[tuple[int, int]] = [
    (RACKET_TOP, RACKET_LEFT),
    (RACKET_TOP, RACKET_RIGHT),
    (RACKET_BOTTOM, RACKET_LEFT),
    (RACKET_BOTTOM, RACKET_RIGHT),
    (RACKET_BOTTOM, RACKET_HANDLE),
    (RACKET_BOTTOM, RACKET_TOP),
    (RACKET_LEFT, RACKET_RIGHT),
]
