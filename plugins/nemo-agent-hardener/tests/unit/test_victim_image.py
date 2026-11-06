# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for running the victim from a pre-built image."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _doubles import make_job_context, make_sdk
from nemo_agent_hardener_plugin.cli import _shared, war_game
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import benign_suite, execution
from nemo_agent_hardener_plugin.jobs import run as run_module
from nemo_agent_hardener_plugin.jobs import synth_benign as synth_module
from nemo_agent_hardener_plugin.jobs.victim_image import VictimImage
from typer.testing import CliRunner

IMAGE = "registry.example.com/agent:1"


def _provisioned_config(tmp_path: Path) -> AgentHardenerConfig:
    cfg = AgentHardenerConfig(
        venv_path=tmp_path / "venv",
        garak_venv_path=tmp_path / "garak-venv",
        operator_env_file=tmp_path / "operator.env",
    )
    cfg.agent_hardener_bin.parent.mkdir(parents=True, exist_ok=True)
    cfg.agent_hardener_bin.touch()
    cfg.garak_python.parent.mkdir(parents=True, exist_ok=True)
    cfg.garak_python.touch()
    return cfg


def _manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    return manifest


def _capture_commands(monkeypatch: pytest.MonkeyPatch, *, output: str = "") -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_execute(cmd: list[str], *_a: Any, **_k: Any) -> tuple[Any, str, Any]:
        commands.append(cmd)
        return SimpleNamespace(returncode=0), output, None

    monkeypatch.setattr(execution._common, "execute", fake_execute)
    return commands


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_victim_image_from_spec_reads_the_ref_and_drift_flag() -> None:
    assert VictimImage.from_spec({}) is None
    assert VictimImage.from_spec({"victim_image": None, "allow_context_drift": False}) is None
    assert VictimImage.from_spec({"victim_image": IMAGE}) == VictimImage(IMAGE)
    assert VictimImage.from_spec({"victim_image": IMAGE, "allow_context_drift": True}) == VictimImage(IMAGE, True)


def test_victim_image_args() -> None:
    assert VictimImage(IMAGE).args() == ["--victim-image", IMAGE]
    assert VictimImage(IMAGE, allow_context_drift=True).args() == ["--victim-image", IMAGE, "--allow-context-drift"]


def test_run_command_without_an_image_is_unchanged() -> None:
    cmd = execution._run_command(
        "/venv/bin/agent-hardener",
        "m.yaml",
        benign_suite="suite.csv",
        env_file=".env",
        rounds=2,
        replay_args=["--replay", "hits.jsonl"],
        reuse=True,
    )

    assert cmd == [
        "/venv/bin/agent-hardener",
        "run",
        "--config",
        "m.yaml",
        "--reuse",
        "--benign-suite",
        "suite.csv",
        "--env-file",
        ".env",
        "--rounds",
        "2",
        "--replay",
        "hits.jsonl",
    ]


def test_run_command_selects_the_image() -> None:
    cmd = execution._run_command("ah", "m.yaml", victim_image=VictimImage(IMAGE, allow_context_drift=True))

    assert cmd == ["ah", "run", "--config", "m.yaml", "--victim-image", IMAGE, "--allow-context-drift"]


def test_one_shot_war_game_passes_the_image_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(run_module.AgentHardenerConfig, "get", lambda: _provisioned_config(tmp_path))
    commands = _capture_commands(monkeypatch)
    sdk = make_sdk(entities=SimpleNamespace(create=lambda *a, **k: SimpleNamespace(name="run-x")))
    job = run_module.AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)

    result = job.run(
        {"config": str(manifest), "victim_image": IMAGE, "allow_context_drift": True},
        ctx=make_job_context(tmp_path),
        sdk=sdk,
    )

    assert result["status"] == "completed", result
    assert [c[1] for c in commands] == ["run"]
    assert _flag_value(commands[0], "--victim-image") == IMAGE
    assert "--allow-context-drift" in commands[0]


def test_one_shot_war_game_without_an_image_adds_no_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(run_module.AgentHardenerConfig, "get", lambda: _provisioned_config(tmp_path))
    commands = _capture_commands(monkeypatch)
    sdk = make_sdk(entities=SimpleNamespace(create=lambda *a, **k: SimpleNamespace(name="run-x")))
    job = run_module.AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)

    job.run({"config": str(manifest)}, ctx=make_job_context(tmp_path), sdk=sdk)

    assert "--victim-image" not in commands[0]
    assert "--allow-context-drift" not in commands[0]


def test_service_war_game_passes_the_same_image_to_up_and_the_reused_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(run_module.AgentHardenerConfig, "get", lambda: _provisioned_config(tmp_path))
    commands = _capture_commands(monkeypatch)

    @contextlib.contextmanager
    def fake_launch(*_a: Any, **_k: Any):
        yield object()

    monkeypatch.setattr(execution, "launch_synth_service", fake_launch)
    monkeypatch.setattr(execution, "drive_synth_hitl", lambda *_a, **_k: "/x/requests.csv")
    monkeypatch.setattr(benign_suite, "read_suite", lambda _p: [{"tool": "clock", "payload": "t", "label": "benign"}])
    monkeypatch.setattr(benign_suite, "write_suite", lambda path, suite: None)
    job = run_module.AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)

    result = job.run(
        {"config": str(manifest), "driver": "service", "victim_image": IMAGE},
        ctx=make_job_context(tmp_path),
        sdk=None,
    )

    assert result["status"] == "completed", result
    up, run, down = commands
    assert (up[1], run[1], down[1]) == ("up", "run", "down")
    assert _flag_value(up, "--victim-image") == _flag_value(run, "--victim-image") == IMAGE
    assert "--reuse" in run
    assert "--victim-image" not in down
    assert "--allow-context-drift" not in up + run


def test_service_war_game_with_a_cached_suite_passes_the_image_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = _capture_commands(monkeypatch)
    monkeypatch.setattr(benign_suite, "write_suite", lambda path, suite: None)
    sdk = SimpleNamespace(entities=SimpleNamespace(create=lambda *a, **k: SimpleNamespace(name="run-x")))

    execution._run_service_driven(
        str(_manifest(tmp_path)),
        None,
        _provisioned_config(tmp_path),
        make_job_context(tmp_path),
        sdk,
        "clockbot",
        1,
        manifest_id="m1",
        cached_suite=[{"tool": "clock", "payload": "t", "label": "benign", "rationale": "", "persona": ""}],
        victim_image=VictimImage(IMAGE),
    )

    assert [c[1] for c in commands] == ["run"]
    assert _flag_value(commands[0], "--victim-image") == IMAGE


def test_synth_benign_command_passes_the_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _capture_commands(monkeypatch)
    monkeypatch.setattr(execution, "_benign_requests_csv", lambda root: root / "requests.csv")

    execution.run_synth_benign(
        "ah",
        str(_manifest(tmp_path)),
        None,
        {},
        make_job_context(tmp_path),
        interview="auto",
        victim_image=VictimImage(IMAGE, allow_context_drift=True),
    )

    assert commands[0][:2] == ["ah", "synth-benign"]
    assert commands[0][-4:] == ["--yes", "--victim-image", IMAGE, "--allow-context-drift"]


def test_synth_benign_command_without_an_image_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _capture_commands(monkeypatch)
    monkeypatch.setattr(execution, "_benign_requests_csv", lambda root: root / "requests.csv")
    manifest = str(_manifest(tmp_path))

    execution.run_synth_benign("ah", manifest, ".env", {}, make_job_context(tmp_path), interview="skip")

    assert commands[0] == ["ah", "synth-benign", "--config", manifest, "--env-file", ".env", "--no-interactive"]


def test_synth_job_hands_the_spec_image_to_synth_benign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(synth_module.AgentHardenerConfig, "get", lambda: _provisioned_config(tmp_path))
    monkeypatch.setattr(synth_module, "_materialize_manifest", lambda *_a, **_k: str(_manifest(tmp_path)))
    monkeypatch.setattr(synth_module, "_agent_staleness_warning", lambda *_a, **_k: None)
    monkeypatch.setattr(synth_module, "_manifest_secret_values", lambda *_a, **_k: {})
    monkeypatch.setattr(synth_module, "_effective_models", lambda *_a, **_k: None)
    monkeypatch.setattr(synth_module._common, "job_runtime_env", lambda *_a, **_k: {})
    monkeypatch.setattr(synth_module, "read_and_persist_suite", lambda *_a, **_k: [])
    seen: dict[str, Any] = {}

    def fake_run_synth(*_a: Any, interview: str, victim_image: VictimImage | None = None) -> Path:
        seen["victim_image"] = victim_image
        return tmp_path / "requests.csv"

    monkeypatch.setattr(synth_module, "run_synth_benign", fake_run_synth)
    job = synth_module.AgentHardenerSynthBenignJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)

    out = job.run(
        {"manifest_id": "m1", "victim_image": IMAGE, "allow_context_drift": True},
        ctx=make_job_context(tmp_path),
        sdk=object(),
    )

    assert out["status"] == "completed", out
    assert seen["victim_image"] == VictimImage(IMAGE, allow_context_drift=True)


# ── SDK ──────────────────────────────────────────────────────────────────────


def _capture_war_game_spec(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}

    class _Job:
        name = "war-game"

        def run(self, spec: dict, **_kwargs: Any) -> dict:
            captured["spec"] = spec
            return {"status": "completed"}

    monkeypatch.setattr(sdk_module, "AgentHardenerRunJob", _Job)
    return captured


def test_sdk_run_puts_the_image_in_the_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured = _capture_war_game_spec(monkeypatch)

    sdk_module.AgentHardenerPluginResource(make_sdk()).run(
        manifest_id="m1", victim_image=IMAGE, allow_context_drift=True
    )

    assert captured["spec"]["victim_image"] == IMAGE
    assert captured["spec"]["allow_context_drift"] is True


def test_sdk_run_without_an_image_leaves_the_spec_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured = _capture_war_game_spec(monkeypatch)

    sdk_module.AgentHardenerPluginResource(make_sdk()).run(manifest_id="m1")

    assert "victim_image" not in captured["spec"]
    assert "allow_context_drift" not in captured["spec"]


def test_sdk_run_allows_drift_for_the_manifests_own_image(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured = _capture_war_game_spec(monkeypatch)

    sdk_module.AgentHardenerPluginResource(make_sdk()).run(manifest_id="m1", allow_context_drift=True)

    assert captured["spec"]["allow_context_drift"] is True
    assert "victim_image" not in captured["spec"]


@pytest.mark.parametrize("method", ["submit", "sanity_check"])
def test_sdk_submissions_put_the_image_in_the_spec(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}

    class _Scheduler:
        def submit_remote(self, _job: Any, spec: dict, **_kwargs: Any) -> dict:
            captured["spec"] = spec
            return {"name": "job-x"}

    monkeypatch.setattr(sdk_module, "NemoJobScheduler", _Scheduler)
    monkeypatch.setattr(sdk_module, "base_url", lambda: "http://localhost:8080")
    resource = sdk_module.AgentHardenerPluginResource(make_sdk())
    extra: dict[str, Any] = (
        {"mitigations": {"defenses": []}, "selected_defense_ids": [], "replay_hitlog_fileset": "default/hits"}
        if method == "sanity_check"
        else {}
    )

    getattr(resource, method)(manifest_id="m1", victim_image=IMAGE, **extra)

    assert captured["spec"]["victim_image"] == IMAGE


def test_sdk_synth_benign_puts_the_image_in_the_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}

    class _Job:
        name = "synth"

        def run(self, spec: dict, **_kwargs: Any) -> dict:
            captured["spec"] = spec
            return {"status": "completed"}

    monkeypatch.setattr(sdk_module, "AgentHardenerSynthBenignJob", _Job)

    sdk_module.AgentHardenerPluginResource(make_sdk()).synth_benign(
        manifest_id="m1", victim_image=IMAGE, allow_context_drift=True
    )

    assert captured["spec"]["victim_image"] == IMAGE
    assert captured["spec"]["allow_context_drift"] is True


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cli(monkeypatch: pytest.MonkeyPatch, resource: Any) -> Any:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    monkeypatch.setattr(_shared.checks, "require_preflight", lambda _c, **_k: None)
    monkeypatch.setattr(_shared, "make_sdk", lambda _u: SimpleNamespace(agent_hardener=resource))
    monkeypatch.setattr(_shared, "base_url", lambda: "http://localhost:8080")
    monkeypatch.setattr(war_game, "missing_secrets", lambda _p, env_files: [])
    monkeypatch.setattr(
        _shared.AgentHardenerConfig,
        "get",
        classmethod(lambda _cls: SimpleNamespace(default_workspace="default", operator_env_file=Path(".env"))),
    )
    return cli_main.AgentHardenerCLI().get_cli()


def _recording_resource(captured: dict[str, Any]) -> Any:
    def record(name: str, result: Any) -> Any:
        return lambda **kwargs: captured.update({name: kwargs}) or result

    return SimpleNamespace(
        run=record("run", {"status": "completed"}),
        synth_benign=record("synth", {"status": "completed", "suite_size": 1}),
        sanity_check=record("sanity", {"name": "job-x"}),
    )


def _mitigations_file(tmp_path: Path) -> str:
    path = tmp_path / "mitigations.json"
    path.write_text(json.dumps({"defenses": [{"id": "g1", "kind": "guardrail"}]}), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    ("command", "key"),
    [
        (["run", "--manifest-id", "m1"], "run"),
        (["synth-benign", "--manifest-id", "m1"], "synth"),
        (
            ["sanity-check", "--manifest-id", "m1", "--mitigations", "{mitigations}", "--replay-hitlog", "d/h"],
            "sanity",
        ),
    ],
)
def test_cli_war_game_commands_forward_the_image(
    command: list[str], key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    app = _cli(monkeypatch, _recording_resource(captured))
    args = [arg.replace("{mitigations}", _mitigations_file(tmp_path)) for arg in command]

    result = CliRunner().invoke(app, [*args, "--victim-image", f" {IMAGE} ", "--allow-context-drift"])

    assert result.exit_code == 0, result.output
    assert captured[key]["victim_image"] == IMAGE
    assert captured[key]["allow_context_drift"] is True


@pytest.mark.parametrize(
    "command",
    [["run", "--manifest-id", "m1"], ["synth-benign", "--manifest-id", "m1"]],
)
def test_cli_war_game_commands_default_to_a_local_build(command: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    app = _cli(monkeypatch, _recording_resource(captured))

    result = CliRunner().invoke(app, command)

    assert result.exit_code == 0, result.output
    recorded = next(iter(captured.values()))
    assert recorded["victim_image"] is None
    assert recorded["allow_context_drift"] is False


@pytest.mark.parametrize(("extra", "needs_docker"), [([], True), (["--victim-image", IMAGE], False)])
@pytest.mark.parametrize("command", [["run", "--manifest-id", "m1"], ["synth-benign", "--manifest-id", "m1"]])
def test_cli_preflight_needs_docker_only_without_an_image(
    command: list[str], extra: list[str], needs_docker: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight: list[bool] = []
    app = _cli(monkeypatch, _recording_resource({}))
    monkeypatch.setattr(_shared.checks, "require_preflight", lambda _c, *, needs_docker: preflight.append(needs_docker))

    result = CliRunner().invoke(app, [*command, *extra])

    assert result.exit_code == 0, result.output
    assert preflight == [needs_docker]


@pytest.mark.parametrize("value", ["", "  ", "agent image", "-x"])
def test_cli_rejects_a_bad_image_reference(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    app = _cli(monkeypatch, _recording_resource(captured))

    result = CliRunner().invoke(app, ["run", "--manifest-id", "m1", f"--victim-image={value}"])

    assert result.exit_code == 2
    assert "image reference" in result.output
    assert captured == {}


def test_cli_rejects_allow_context_drift_for_a_local_manifest_without_an_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    app = _cli(monkeypatch, _recording_resource(captured))

    result = CliRunner().invoke(app, ["run", "--config", str(_manifest(tmp_path)), "--allow-context-drift"])

    assert result.exit_code == 1
    assert "--allow-context-drift needs --victim-image" in result.output
    assert captured == {}


def test_cli_allows_drift_for_a_saved_manifests_image(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    app = _cli(monkeypatch, _recording_resource(captured))

    result = CliRunner().invoke(app, ["run", "--manifest-id", "m1", "--allow-context-drift"])

    assert result.exit_code == 0, result.output
    assert captured["run"]["victim_image"] is None
    assert captured["run"]["allow_context_drift"] is True
