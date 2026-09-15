from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

import badminton_coach.cli as cli_mod
from badminton_coach.config import Config
from badminton_coach.run import RunDir


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_input_video(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"not a real video, pipeline is stubbed")
    return p


def _stub_pipeline_stages(monkeypatch, calls: dict) -> None:
    """Replaces every heavy pipeline stage with a fast fake and records what analyze/judge/render passed
    it, so these tests exercise the CLI's own wiring/flag logic without any real inference or claude -p
    call."""
    import badminton_coach.judge.run as judge_run_mod
    import badminton_coach.judge.workspace as workspace_mod
    import badminton_coach.measure.body3d as body3d_mod
    import badminton_coach.metrics as metrics_mod
    import badminton_coach.pipeline as pipeline_mod
    import badminton_coach.render.report as report_mod
    import badminton_coach.render.video as video_mod

    def fake_pipeline(inputs, run, cfg, device, net_side_override, force, skip_shuttle):  # noqa: ARG001
        calls["pipeline_force"] = force
        calls["skip_shuttle"] = skip_shuttle
        calls["net_side_override"] = net_side_override
        return {"ingest": {"n_frames": 1, "fps": 30.0}, "errors": {}}

    def fake_estimate_3d(run, cfg, device, force):  # noqa: ARG001
        calls["body3d_force"] = force
        return {"backend": "none"}

    def fake_compute_metrics_stage(run, cfg, references, force):  # noqa: ARG001
        calls["metrics_force"] = force
        calls["references"] = sorted(r.root.name for r in references)
        return {"n_swings": 0}

    def fake_build_workspace(run, cfg, references, lang, player):  # noqa: ARG001
        calls["workspace_lang"] = lang
        calls["workspace_player"] = player
        ws = run.root / "judge" / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def fake_run_judge(run, workspace, lang, cfg, force):  # noqa: ARG001
        calls["judge_force"] = force
        calls["judge_lang"] = lang
        return {"status": "done"}

    def fake_render_annotated_video(run, out_path, findings, cfg, lang):  # noqa: ARG001
        calls["render_lang"] = lang
        calls["render_findings"] = findings
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake")
        return {"ok": True}

    def fake_write_report(run, cfg, lang):  # noqa: ARG001
        (run.root / "report.md").write_text("stub")
        (run.root / "index.html").write_text("<html></html>")
        return {"report_md": str(run.root / "report.md")}

    monkeypatch.setattr(pipeline_mod, "run_measurement_pipeline", fake_pipeline)
    monkeypatch.setattr(body3d_mod, "estimate_3d", fake_estimate_3d)
    monkeypatch.setattr(metrics_mod, "compute_metrics_stage", fake_compute_metrics_stage)
    monkeypatch.setattr(workspace_mod, "build_workspace", fake_build_workspace)
    monkeypatch.setattr(judge_run_mod, "run_judge", fake_run_judge)
    monkeypatch.setattr(video_mod, "render_annotated_video", fake_render_annotated_video)
    monkeypatch.setattr(report_mod, "write_report", fake_write_report)


def _stub_references(monkeypatch, names: list[str], cfg: Config) -> None:
    import badminton_coach.reference as reference_mod

    refs = []
    for name in names:
        root = cfg.references_dir / name
        root.mkdir(parents=True, exist_ok=True)
        refs.append(RunDir(root))
    monkeypatch.setattr(reference_mod, "load_references", lambda cfg=cfg: refs)  # noqa: ARG005


def test_analyze_default_wiring(runner, monkeypatch, tmp_path, cfg) -> None:
    calls: dict = {}
    _stub_pipeline_stages(monkeypatch, calls)
    _stub_references(monkeypatch, ["viktor-2018"], cfg)
    monkeypatch.setattr(cli_mod, "CONFIG", cfg)
    video = _make_input_video(tmp_path)

    result = runner.invoke(cli_mod.main, ["analyze", str(video), "--player", "nam", "--lang", "vi"])

    assert result.exit_code == 0, result.output
    assert calls["pipeline_force"] is False
    assert calls["references"] == ["viktor-2018"]
    assert calls["workspace_lang"] == "vi"
    assert calls["judge_force"] is False
    assert calls["render_lang"] == "vi"

    [run_dir] = list((cfg.players_dir / "nam").iterdir())
    run_json = json.loads((run_dir / "run.json").read_text())
    assert run_json["lang"] == "vi"
    assert run_json["shot"] == "clear"
    assert (run_dir / "index.html").exists()


def test_analyze_no_judge_skips_judge_and_render(runner, monkeypatch, tmp_path, cfg) -> None:
    calls: dict = {}
    _stub_pipeline_stages(monkeypatch, calls)
    _stub_references(monkeypatch, [], cfg)
    monkeypatch.setattr(cli_mod, "CONFIG", cfg)
    video = _make_input_video(tmp_path)

    result = runner.invoke(cli_mod.main, ["analyze", str(video), "--player", "nam", "--no-judge"])

    assert result.exit_code == 0, result.output
    assert "metrics_force" in calls  # metrics still ran
    assert "judge_force" not in calls
    assert "render_lang" not in calls


def test_analyze_refs_none_skips_reference_loading(runner, monkeypatch, tmp_path, cfg) -> None:
    calls: dict = {}
    _stub_pipeline_stages(monkeypatch, calls)
    video = _make_input_video(tmp_path)
    monkeypatch.setattr(cli_mod, "CONFIG", cfg)

    import badminton_coach.reference as reference_mod

    def fail_if_called(cfg=cfg):  # noqa: ARG001
        raise AssertionError("load_references should not be called when --refs none")

    monkeypatch.setattr(reference_mod, "load_references", fail_if_called)

    result = runner.invoke(cli_mod.main, ["analyze", str(video), "--player", "nam", "--refs", "none", "--no-judge"])

    assert result.exit_code == 0, result.output
    assert calls["references"] == []


def test_analyze_from_stage_within_measurement_pipeline_forces_it(runner, monkeypatch, tmp_path, cfg) -> None:
    calls: dict = {}
    _stub_pipeline_stages(monkeypatch, calls)
    _stub_references(monkeypatch, [], cfg)
    monkeypatch.setattr(cli_mod, "CONFIG", cfg)
    video = _make_input_video(tmp_path)

    result = runner.invoke(
        cli_mod.main, ["analyze", str(video), "--player", "nam", "--from-stage", "track", "--no-judge"]
    )

    assert result.exit_code == 0, result.output
    assert calls["pipeline_force"] is True
    assert calls["body3d_force"] is True
    assert calls["metrics_force"] is True


def test_analyze_from_stage_after_measurement_only_forces_downstream(runner, monkeypatch, tmp_path, cfg) -> None:
    calls: dict = {}
    _stub_pipeline_stages(monkeypatch, calls)
    _stub_references(monkeypatch, [], cfg)
    monkeypatch.setattr(cli_mod, "CONFIG", cfg)
    video = _make_input_video(tmp_path)

    result = runner.invoke(cli_mod.main, ["analyze", str(video), "--player", "nam", "--from-stage", "metrics"])

    assert result.exit_code == 0, result.output
    assert calls["pipeline_force"] is False
    assert calls["body3d_force"] is False
    assert calls["metrics_force"] is True
    assert calls["judge_force"] is True


def test_analyze_net_side_override_passed_through(runner, monkeypatch, tmp_path, cfg) -> None:
    calls: dict = {}
    _stub_pipeline_stages(monkeypatch, calls)
    _stub_references(monkeypatch, [], cfg)
    monkeypatch.setattr(cli_mod, "CONFIG", cfg)
    video = _make_input_video(tmp_path)

    result = runner.invoke(cli_mod.main, ["analyze", str(video), "--player", "nam", "--net-side", "left", "--no-judge"])

    assert result.exit_code == 0, result.output
    assert calls["net_side_override"] == "left"


def test_judge_command_always_forces_a_fresh_run(runner, monkeypatch, cfg) -> None:
    calls: dict = {}
    _stub_pipeline_stages(monkeypatch, calls)
    _stub_references(monkeypatch, [], cfg)
    run = RunDir.create("nam", cfg=cfg)
    run.update_run_json(lang="de", player="nam")

    result = runner.invoke(cli_mod.main, ["judge", str(run.root)])

    assert result.exit_code == 0, result.output
    assert calls["judge_force"] is True
    assert calls["judge_lang"] == "de"  # picked up from run.json, not re-specified on the command line


def test_render_command_uses_existing_findings(runner, monkeypatch, cfg) -> None:
    calls: dict = {}
    _stub_pipeline_stages(monkeypatch, calls)
    run = RunDir.create("nam", cfg=cfg)
    run.update_run_json(lang="en")
    (run.root / "judge").mkdir(parents=True)
    findings = {"findings": [{"id": "f1"}]}
    (run.root / "judge" / "findings.json").write_text(json.dumps(findings))

    result = runner.invoke(cli_mod.main, ["render", str(run.root)])

    assert result.exit_code == 0, result.output
    assert calls["render_findings"] == findings
    assert calls["render_lang"] == "en"


def test_render_command_without_findings_passes_none(runner, monkeypatch, cfg) -> None:
    calls: dict = {}
    _stub_pipeline_stages(monkeypatch, calls)
    run = RunDir.create("nam", cfg=cfg)

    result = runner.invoke(cli_mod.main, ["render", str(run.root)])

    assert result.exit_code == 0, result.output
    assert calls["render_findings"] is None


def test_models_download_skips_files_already_present(runner, monkeypatch, cfg) -> None:
    monkeypatch.setattr(cli_mod, "CONFIG", cfg)

    class FakeBodyWithFeet:
        def __init__(self, **kwargs) -> None:  # noqa: ARG002
            pass

    import rtmlib

    monkeypatch.setattr(rtmlib, "BodyWithFeet", FakeBodyWithFeet)

    clone_calls = []

    def fake_run(argv, check):  # noqa: ARG001
        # simulate a successful clone by creating the destination the caller checks for next time
        dest = Path(argv[-1])
        clone_calls.append(argv)
        if "RacketVision" in argv[-2] or "RacketVision" in str(dest):
            (dest / "source" / "RacketPose").mkdir(parents=True, exist_ok=True)
        else:
            (dest / "sam_3d_body").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    hf_calls = []
    # pre-create one racket checkpoint to prove it's skipped
    ckpt_dir = cfg.models_dir / "racketvision" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "epoch_300.pth").write_bytes(b"already here")

    def fake_hf_hub_download(*, repo_id, filename, local_dir, cache_dir):  # noqa: ARG001
        hf_calls.append((repo_id, filename))
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded")
        return str(dest)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    result = runner.invoke(cli_mod.main, ["models", "download"])

    assert result.exit_code == 0, result.output
    # the pre-existing checkpoint was not re-downloaded
    assert ("linfeng302/RacketVision-Models", "epoch_300.pth") not in hf_calls
    assert ("linfeng302/RacketVision-Models", "best_PCK_epoch_90.pth") in hf_calls
    assert ("linfeng302/RacketVision-Models", "balltrack_best.pth") in hf_calls
    assert ("facebook/sam-3d-body-vith", "model.ckpt") in hf_calls
    assert any("RacketVision" in str(c) for c in clone_calls)
    assert any("sam-3d-body" in str(c) for c in clone_calls)


def test_models_download_gated_repo_gives_actionable_error(runner, monkeypatch, cfg) -> None:
    from huggingface_hub.errors import GatedRepoError

    monkeypatch.setattr(cli_mod, "CONFIG", cfg)

    class FakeBodyWithFeet:
        def __init__(self, **kwargs) -> None:  # noqa: ARG002
            pass

    import rtmlib

    monkeypatch.setattr(rtmlib, "BodyWithFeet", FakeBodyWithFeet)

    def fake_run(argv, check):  # noqa: ARG001
        dest = Path(argv[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "source" / "RacketPose").mkdir(parents=True, exist_ok=True)
        (dest / "sam_3d_body").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_hf_hub_download(*, repo_id, filename, local_dir, cache_dir):  # noqa: ARG001
        if repo_id == "facebook/sam-3d-body-vith":
            resp = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co"))
            raise GatedRepoError("access denied", response=resp)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded")
        return str(dest)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    result = runner.invoke(cli_mod.main, ["models", "download"])

    assert result.exit_code != 0
    assert "gated" in result.output.lower()
    assert "hf auth login" in result.output
