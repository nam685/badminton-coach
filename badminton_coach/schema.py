"""Pydantic models for `swings/swings.json`, `metrics/metrics.json` (spec §3.4/§3.5), and the judge's
`findings.json` (spec §6.5 — mirrors `judge/schema.json`, the JSON Schema handed to `claude -p
--json-schema`; keep the two in sync by hand, there are few enough fields that this is easy)."""

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


# --- judge findings.json ---

FindingCategory = Literal[
    "contact_point",
    "rotation_sequence",
    "elbow",
    "racket",
    "non_racket_arm",
    "footwork_weight_transfer",
    "timing",
    "follow_through",
    "other",
]


class MeasurementQuality(BaseModel):
    ok: bool
    notes: str


class SwingVerdict(BaseModel):
    index: int
    mode: Literal["live", "shadow"]
    contact_frame: int
    contact_time_s: float
    one_line_verdict: str


class EvidenceFrame(BaseModel):
    swing: int
    frame: int


class EvidenceMetric(BaseModel):
    name: str
    value: float
    reference: float | None = None


class Evidence(BaseModel):
    swings: list[int]
    frames: list[EvidenceFrame]
    metrics: list[EvidenceMetric]
    visual: str


class Finding(BaseModel):
    id: str
    title: str
    category: FindingCategory
    severity: Literal[1, 2, 3]
    pattern: Literal["consistent", "inconsistent", "single"]
    evidence: Evidence
    explanation: str
    cue: str
    drill: str


class FindingsFile(BaseModel):
    player: str
    shot: str
    handedness: Literal["left", "right"]
    measurement_quality: MeasurementQuality
    swings: list[SwingVerdict]
    strengths: list[str]
    findings: list[Finding]
    priority: list[str]
    progress_vs_history: str | None = None
    confidence: Literal["low", "medium", "high"]
