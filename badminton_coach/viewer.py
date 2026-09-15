"""`badminton-coach view` (spec §4/§8): a tiny static server over `data/` for browsing every run's
artifacts (video, contact sheets, plots, findings, judge transcript) without touching a terminal —
regenerates `data/index.html` (players -> runs, newest first, each with its top finding) on every call.
"""

from __future__ import annotations

import functools
import http.server
import json

from badminton_coach.config import CONFIG, Config

_INDEX_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>badminton-coach runs</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 700px; margin: 2rem auto; padding: 0 1rem;
         background: #111; color: #eee; }}
  h1 {{ color: #fff; }}
  a {{ color: #6cf; }}
  .run {{ padding: 0.5rem 0; border-bottom: 1px solid #333; }}
  .meta {{ color: #999; font-size: 0.9em; }}
</style></head><body>
<h1>badminton-coach runs</h1>
{body}
</body></html>
"""


def _run_summary(run_dir) -> dict:
    run_json = run_dir / "run.json"
    findings_path = run_dir / "judge" / "findings.json"
    top_finding = None
    if findings_path.exists():
        try:
            data = json.loads(findings_path.read_text())
            findings = data.get("findings", [])
            priority = data.get("priority", [])
            by_id = {f["id"]: f for f in findings}
            if priority and priority[0] in by_id:
                top_finding = by_id[priority[0]]["title"]
            elif findings:
                top_finding = findings[0]["title"]
        except (json.JSONDecodeError, KeyError):
            pass
    lang = None
    if run_json.exists():
        try:
            lang = json.loads(run_json.read_text()).get("lang")
        except json.JSONDecodeError:
            pass
    return {
        "name": run_dir.name,
        "top_finding": top_finding,
        "lang": lang,
        "has_index": (run_dir / "index.html").exists(),
    }


def build_index(cfg: Config = CONFIG) -> str:
    sections = []
    if cfg.players_dir.exists():
        for player_dir in sorted(cfg.players_dir.iterdir()):
            if not player_dir.is_dir():
                continue
            runs = sorted((r for r in player_dir.iterdir() if r.is_dir()), reverse=True)
            if not runs:
                continue
            sections.append(f"<h2>{player_dir.name}</h2>")
            for run_dir in runs:
                s = _run_summary(run_dir)
                rel = f"players/{player_dir.name}/{run_dir.name}/index.html"
                label = s["top_finding"] or ("no findings yet" if not s["has_index"] else "no findings")
                link = f'<a href="{rel}">{run_dir.name}</a>' if s["has_index"] else run_dir.name
                sections.append(f'<div class="run">{link}<div class="meta">{label}</div></div>')
    if not sections:
        sections.append("<p>No runs yet.</p>")
    return _INDEX_TEMPLATE.format(body="\n".join(sections))


def write_index(cfg: Config = CONFIG) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "index.html").write_text(build_index(cfg))


def serve(cfg: Config = CONFIG, port: int = 8765) -> None:
    write_index(cfg)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(cfg.data_dir))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {cfg.data_dir} at http://127.0.0.1:{port}/index.html")
        httpd.serve_forever()
