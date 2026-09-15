"""Pydantic models for `swings/swings.json` and `metrics/metrics.json` (spec §3.4/§3.5)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Swing(BaseModel):
    index: int
    mode: Literal["live", "shadow"]
    contact_frame: int
    contact_time_s: float
    contact_source: Literal["shuttle", "speed"]
    prep_start_frame: int
    prep_end_frame: int | None
    follow_end_frame: int


class SwingsFile(BaseModel):
    fps: float
    handedness: str
    net_side: str
    net_side_source: str
    torso_len_median_px: float | None
    speed_source: str
    swings: list[Swing]


class MetricValue(BaseModel):
    value: float | None
    ref_mean: float | None = None
    ref_min: float | None = None
    ref_max: float | None = None
    n_ref: int = 0


class SwingMetrics(BaseModel):
    index: int
    mode: Literal["live", "shadow"]
    contact_frame: int
    contact_time_s: float
    metrics: dict[str, MetricValue]


class CrossAttemptStat(BaseModel):
    mean: float | None
    std: float | None
    n: int
    consistency_flag: bool = False


class MetricsFile(BaseModel):
    fps: float
    handedness: str
    net_side: str
    torso_len_median_px: float | None
    shot: str = "clear"
    swings: list[SwingMetrics]
    cross_attempt_summary: dict[str, CrossAttemptStat] = {}
