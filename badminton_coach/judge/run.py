"""Invoke `claude -p` against a judge workspace (spec §6.2), matching the argv-building pattern of
`~/projects/aoe2coach/aoe2coach/coach.py::_build_agentic_argv` — a headless, read-only-tools agent run
with structured JSON-Schema output.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from badminton_coach.config import CONFIG, Config
from badminton_coach.judge.prompt import JUDGE_SYSTEM, build_task
from badminton_coach.run import RunDir
from badminton_coach.schema import FindingsFile

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"

READONLY_TOOLS = ["Read", "Grep", "Glob", "Bash(badminton-coach frames:*)"]


def build_argv(
    task_prompt: str,
    model: str,
    claude_bin: str,
    effort: str,
    max_turns: int,
    json_schema: str,
) -> list[str]:
    return [
        claude_bin,
        "-p",
        task_prompt,
        "--model",
        model,
        "--effort",
        effort,
        "--output-format",
        "stream-json",
        "--verbose",
        "--json-schema",
        json_schema,
        "--append-system-prompt",
        JUDGE_SYSTEM,
        "--allowedTools",
        *READONLY_TOOLS,
        "--permission-mode",
        CONFIG.judge_permission_mode,
        "--max-turns",
        str(max_turns),
        "--strict-mcp-config",
    ]


def _extract_result_text(stream_lines: list[dict]) -> tuple[str | None, dict[str, Any]]:
    """The final `result`-type message in a `stream-json` transcript holds the assistant's last
    message (which, with --json-schema, is schema-conformant JSON text) plus run metadata."""
    for msg in reversed(stream_lines):
        if msg.get("type") == "result":
            return msg.get("result"), msg
    return None, {}


def run_judge(
    run: RunDir,
    workspace: Path,
    lang: str = "en",
    cfg: Config = CONFIG,
    force: bool = False,
) -> dict[str, Any]:
    """Runs the judge against `workspace`, writing (inside `run.root/"judge"`): `argv.json`,
    `stream.jsonl` (full transcript, debug), and — on success — `findings.json` (validated). On any
    failure, writes `error.txt` and returns `{"status": "failed", ...}`; never raises (render still runs
    without findings)."""
    judge_dir = run.root / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    findings_path = judge_dir / "findings.json"
    stream_path = judge_dir / "stream.jsonl"

    import hashlib

    input_hash = hashlib.sha256(f"{workspace}:{lang}:{cfg.judge_model}:{cfg.judge_effort}".encode()).hexdigest()

    with run.stage("judge", version="1", input_hash=input_hash, force=force) as st:
        if st.skip:
            if findings_path.exists():
                return {"status": "done"}
            return {"status": "failed"}

        task_prompt = build_task(lang)
        schema_text = _SCHEMA_PATH.read_text()
        argv = build_argv(
            task_prompt, cfg.judge_model, cfg.claude_bin, cfg.judge_effort, cfg.judge_max_turns, schema_text
        )
        (judge_dir / "argv.json").write_text(json.dumps(argv, indent=2))

        try:
            result = subprocess.run(
                argv, cwd=str(workspace), capture_output=True, text=True, timeout=cfg.judge_timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            (judge_dir / "error.txt").write_text(f"judge timed out after {cfg.judge_timeout_s}s\n{exc}")
            run.mark_extra(st, status="failed", reason="timeout")
            return {"status": "failed", "reason": "timeout"}

        stream_path.write_text(result.stdout)

        if result.returncode != 0:
            (judge_dir / "error.txt").write_text(f"claude exited {result.returncode}\nstderr:\n{result.stderr[-4000:]}")
            run.mark_extra(st, status="failed", reason="nonzero_exit")
            return {"status": "failed", "reason": "nonzero_exit"}

        stream_lines = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                stream_lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        result_text, result_meta = _extract_result_text(stream_lines)
        if not result_text:
            (judge_dir / "error.txt").write_text("no 'result' message found in the judge's stream-json output")
            run.mark_extra(st, status="failed", reason="no_result")
            return {"status": "failed", "reason": "no_result"}

        try:
            parsed = json.loads(result_text)
            findings = FindingsFile.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001 - report and degrade, never crash the pipeline
            (judge_dir / "raw_result.txt").write_text(result_text)
            (judge_dir / "error.txt").write_text(f"findings failed schema validation: {exc}")
            run.mark_extra(st, status="failed", reason="invalid_schema")
            return {"status": "failed", "reason": "invalid_schema"}

        findings_path.write_text(findings.model_dump_json(indent=2))
        run.mark_extra(
            st,
            status="done",
            model=result_meta.get("model"),
            num_turns=result_meta.get("num_turns"),
            total_cost_usd=result_meta.get("total_cost_usd"),
            n_findings=len(findings.findings),
        )
        return {"status": "done", "n_findings": len(findings.findings)}
