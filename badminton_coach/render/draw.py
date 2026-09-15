"""Drawing primitives shared by contact sheets (judge workspace, Task 8) and the annotated video
(Task 9). All text goes through `draw_text` (PIL + DejaVu Sans, bundled with matplotlib — already a
dependency) since OpenCV's built-in fonts (`cv2.putText`) cannot render Vietnamese diacritics or German
umlauts. All keypoint/box drawing takes plain numpy arrays (no coupling to the npz file layouts) so
these functions work equally on live pipeline output and on hand-built test fixtures.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from badminton_coach.measure.keypoints import BODY_SKELETON_EDGES, RACKET_SKELETON_EDGES

GREEN = (0, 200, 0)
RED = (0, 0, 220)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 220)

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        import matplotlib

        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        _FONT_CACHE[size] = ImageFont.truetype(path, size)
    return _FONT_CACHE[size]


def draw_text(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    size: int = 18,
    color: tuple[int, int, int] = WHITE,
    bg: tuple[int, int, int] | None = None,
    anchor: str = "la",
) -> np.ndarray:
    """Draws `text` (any language DejaVu Sans covers — Vietnamese, German, etc.) at `xy` (top-left by
    default; see PIL's `anchor`). `frame` is BGR (OpenCV convention); `color`/`bg` are BGR too, for
    drop-in compatibility with the rest of this module. Returns a new array (does not mutate in place)."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _font(size)
    rgb = (color[2], color[1], color[0])
    if bg is not None:
        bbox = draw.textbbox(xy, text, font=font, anchor=anchor)
        pad = 3
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(bg[2], bg[1], bg[0]))
    draw.text(xy, text, font=font, fill=rgb, anchor=anchor)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_body(
    frame: np.ndarray,
    kpts: np.ndarray,  # [26, 2] or [26, 3]
    scores: np.ndarray | None = None,  # [26], required if kpts has no score column
    kpt_thr: float = 0.3,
    color: tuple[int, int, int] = GREEN,
    highlight_joints: set[int] | None = None,
    highlight_color: tuple[int, int, int] = RED,
    radius: int = 3,
    thickness: int = 2,
) -> np.ndarray:
    """Draws the Halpe-26 skeleton. `highlight_joints`: joint indices to draw in `highlight_color`
    instead of `color` (used to flag the specific joints a judge finding is about)."""
    xy = kpts[:, 0:2]
    sc = kpts[:, 2] if kpts.shape[-1] >= 3 else scores
    if sc is None:
        sc = np.ones(len(xy))
    highlight_joints = highlight_joints or set()

    out = frame.copy()
    valid = ~np.isnan(xy[:, 0]) & (sc > kpt_thr)
    for a, b in BODY_SKELETON_EDGES:
        if valid[a] and valid[b]:
            c = highlight_color if (a in highlight_joints or b in highlight_joints) else color
            pa, pb = tuple(xy[a].astype(int)), tuple(xy[b].astype(int))
            cv2.line(out, pa, pb, c, thickness, cv2.LINE_AA)
    for i in range(len(xy)):
        if valid[i]:
            c = highlight_color if i in highlight_joints else color
            cv2.circle(out, tuple(xy[i].astype(int)), radius, c, -1, cv2.LINE_AA)
    return out


def draw_racket(
    frame: np.ndarray,
    kpts: np.ndarray,  # [5, 2] or [5, 3]
    scores: np.ndarray | None = None,
    kpt_thr: float = 0.3,
    color: tuple[int, int, int] = WHITE,
    bbox: np.ndarray | None = None,
    thickness: int = 2,
) -> np.ndarray:
    xy = kpts[:, 0:2]
    sc = kpts[:, 2] if kpts.shape[-1] >= 3 else scores
    if sc is None:
        sc = np.ones(len(xy))

    out = frame.copy()
    if bbox is not None and not np.any(np.isnan(bbox)):
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    valid = ~np.isnan(xy[:, 0]) & (sc > kpt_thr)
    for a, b in RACKET_SKELETON_EDGES:
        if valid[a] and valid[b]:
            pa, pb = tuple(xy[a].astype(int)), tuple(xy[b].astype(int))
            cv2.line(out, pa, pb, color, thickness, cv2.LINE_AA)
    for i in range(len(xy)):
        if valid[i]:
            cv2.circle(out, tuple(xy[i].astype(int)), 3, RED, -1, cv2.LINE_AA)
    return out


def draw_shuttle_trail(
    frame: np.ndarray,
    points: list[tuple[float, float]],  # oldest -> newest
    color: tuple[int, int, int] = YELLOW,
) -> np.ndarray:
    out = frame.copy()
    n = len(points)
    for i, (x, y) in enumerate(points):
        if math.isnan(x) or math.isnan(y):
            continue
        alpha = (i + 1) / max(n, 1)
        radius = max(1, int(2 + 3 * alpha))
        cv2.circle(out, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    return out


def draw_angle_arc(
    frame: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    label: str | None = None,
    color: tuple[int, int, int] = YELLOW,
    radius: int = 30,
) -> np.ndarray:
    """Draws an arc at vertex `b` spanning the angle ABC, with an optional text label (the numeric
    value — pass it pre-formatted, e.g. f"{angle:.0f}deg", so this stays language-agnostic)."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return frame
    out = frame.copy()
    ang1 = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
    ang2 = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]))
    center = tuple(b.astype(int))
    cv2.ellipse(out, center, (radius, radius), 0, ang1, ang2, color, 2, cv2.LINE_AA)
    if label:
        out = draw_text(out, label, (center[0] + radius, center[1]), size=14, color=color, bg=(0, 0, 0))
    return out


def draw_hud(
    frame: np.ndarray,
    lines: list[str],
    xy: tuple[int, int] = (10, 10),
    size: int = 16,
    color: tuple[int, int, int] = WHITE,
) -> np.ndarray:
    out = frame
    x, y = xy
    line_height = int(size * 1.4)
    for i, line in enumerate(lines):
        out = draw_text(out, line, (x, y + i * line_height), size=size, color=color, bg=(0, 0, 0))
    return out


def draw_banner(
    frame: np.ndarray,
    text: str,
    color: tuple[int, int, int] = RED,
    size: int = 20,
) -> np.ndarray:
    """A full-width banner at the bottom of the frame (used for finding titles during their frame
    range)."""
    h, w = frame.shape[:2]
    out = frame.copy()
    bar_h = int(size * 1.8)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    out = draw_text(out, text, (w // 2, h - bar_h // 2), size=size, color=color, anchor="mm")
    return out
