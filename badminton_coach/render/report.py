"""Stage 7 (part 2) — `report.md` + `index.html` (spec §3.7/§8): findings in priority order with
evidence and drills, the metrics table with reference deltas, and (in the HTML) the video, contact
sheets, plots, and a collapsible judge transcript — everything needed to review a run without touching a
terminal, per the explicit "see all intermediate results" dev-convenience requirement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jinja2

from badminton_coach.config import CONFIG, Config
from badminton_coach.i18n import t as i18n_t
from badminton_coach.run import RunDir

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(_TEMPLATES_DIR), autoescape=True)


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def render_report_md(findings: dict[str, Any] | None, metrics: dict[str, Any] | None, lang: str = "en") -> str:
    lines: list[str] = []
    if findings is None:
        lines.append("# Report\n")
        lines.append(
            "_The judge did not produce findings for this run (see judge/error.txt). "
            "Metrics and the annotated video are still available._\n"
        )
    else:
        lines.append(f"# {findings['player']} — {findings['shot']} ({findings['handedness']}-handed)\n")
        mq = findings["measurement_quality"]
        if not mq["ok"]:
            lines.append(f"> **Measurement quality flagged:** {mq['notes']}\n")
        if findings.get("progress_vs_history"):
            lines.append(f"**{i18n_t('progress', lang)}:** {findings['progress_vs_history']}\n")

        if findings.get("strengths"):
            lines.append(f"## {i18n_t('strengths', lang)}")
            for s in findings["strengths"]:
                lines.append(f"- {s}")
            lines.append("")

        by_id = {f["id"]: f for f in findings.get("findings", [])}
        ordered = [by_id[fid] for fid in findings.get("priority", []) if fid in by_id]
        ordered += [f for f in findings.get("findings", []) if f["id"] not in findings.get("priority", [])]

        lines.append(f"## {i18n_t('findings', lang)}")
        for f in ordered:
            lines.append(f"### {f['title']} ({f['category']}, severity {f['severity']}, {f['pattern']})")
            lines.append(f["explanation"])
            lines.append(f"\n**{i18n_t('cue', lang)}:** {f['cue']}")
            lines.append(f"\n**{i18n_t('drill', lang)}:** {f['drill']}")
            ev = f["evidence"]
            metric_bits = ", ".join(
                f"{m['name']}={m['value']:.1f}"
                + (f" (ref {m['reference']:.1f})" if m.get("reference") is not None else "")
                for m in ev.get("metrics", [])
            )
            if metric_bits:
                lines.append(f"\n_{i18n_t('evidence', lang)}: {metric_bits} — {ev.get('visual', '')}_")
            lines.append("")

    if metrics:
        lines.append(f"## {i18n_t('metrics', lang)}")
        lines.append(f"| {i18n_t('swings', lang)} | metric | {i18n_t('value', lang)} | {i18n_t('reference', lang)} |")
        lines.append("|---|---|---|---|")
        for swing in metrics.get("swings", []):
            for name, mv in swing.get("metrics", {}).items():
                if mv.get("value") is None:
                    continue
                ref = f"{mv['ref_mean']:.2f}" if mv.get("ref_mean") is not None else "—"
                lines.append(f"| {swing['index']} | {name} | {mv['value']:.2f} | {ref} |")

    return "\n".join(lines) + "\n"


def render_index_html(
    run: RunDir,
    findings: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    swings_data: dict[str, Any] | None,
    stream_text: str | None,
    lang: str = "en",
) -> str:
    template = _ENV.get_template("run.html")
    return template.render(
        run_name=run.root.name,
        findings=findings,
        metrics=metrics,
        swings=swings_data.get("swings", []) if swings_data else [],
        stream_text=stream_text,
        lang=lang,
        t=lambda key: i18n_t(key, lang),
        has_video=(run.root / "annotated.mp4").exists(),
    )


def write_report(run: RunDir, cfg: Config = CONFIG, lang: str = "en") -> dict[str, Any]:  # noqa: ARG001
    # `cfg` isn't read directly here (paths all come from `run.root`) but is kept for consistency with
    # every other stage-entrypoint function's signature, and in case a future config knob is needed.
    findings = _load_json(run.root / "judge" / "findings.json")
    metrics = _load_json(run.root / "metrics" / "metrics.json")
    swings_data = _load_json(run.root / "swings" / "swings.json")
    stream_path = run.root / "judge" / "stream.jsonl"
    stream_text = stream_path.read_text() if stream_path.exists() else None

    report_md = render_report_md(findings, metrics, lang)
    (run.root / "report.md").write_text(report_md)

    index_html = render_index_html(run, findings, metrics, swings_data, stream_text, lang)
    (run.root / "index.html").write_text(index_html)

    return {"report_md": str(run.root / "report.md"), "index_html": str(run.root / "index.html")}
