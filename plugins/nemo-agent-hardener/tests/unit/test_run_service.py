# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the service-driven war-game orchestration: sandbox up -> synth HITL -> reuse-benign run."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from _doubles import make_job_context, make_sdk
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import artifacts, benign_suite, execution
from nemo_agent_hardener_plugin.jobs import manifest as manifest_mod
from nemo_agent_hardener_plugin.jobs import run as run_module
from nemo_agent_hardener_plugin.jobs.errors import AgentHardenerRunError
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext


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


def _ctx(tmp_path: Path) -> JobContext:
    return make_job_context(tmp_path)


def test_service_driven_flow_sequences_up_hitl_then_reuse_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")

    monkeypatch.setattr(run_module.AgentHardenerConfig, "get", lambda: _provisioned_config(tmp_path))

    commands: list[list[str]] = []

    def fake_execute(cmd, _env, _log_path, _ctx, *, artifact_name):  # noqa: ANN001, ANN202 - test stub
        commands.append(cmd)
        return SimpleNamespace(returncode=0), "log-tail", None

    monkeypatch.setattr(run_module._common, "execute", fake_execute)

    @contextlib.contextmanager
    def fake_launch(*_a: Any, **_k: Any):
        yield object()  # client is unused because drive_synth_hitl is stubbed

    monkeypatch.setattr(execution, "launch_synth_service", fake_launch)

    hitl_calls: list[str] = []
    monkeypatch.setattr(
        execution, "drive_synth_hitl", lambda _c, cfg, *_a, **_k: hitl_calls.append(cfg) or "/x/requests.csv"
    )
    # The reviewed suite from the serve step is read, written to a distinct CSV, and handed to `run`.
    monkeypatch.setattr(
        benign_suite,
        "read_suite",
        lambda _p: [{"tool": "clock", "payload": "t", "label": "benign", "rationale": "", "persona": ""}],
    )
    monkeypatch.setattr(benign_suite, "write_suite", lambda path, suite: None)

    job = run_module.AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)
    result = job.run({"config": str(manifest), "driver": "service"}, ctx=_ctx(tmp_path), sdk=None)

    assert result["status"] == "completed"
    assert hitl_calls == [str(manifest)]  # HITL ran once, for this manifest
    # up first, then the war-game reusing the warm sandbox with the reviewed suite handed in as a file.
    assert commands[0][1] == "up"
    assert commands[1][1] == "run"
    assert "--reuse" in commands[1] and "--reuse-benign" not in commands[1]
    assert commands[1][commands[1].index("--benign-suite") + 1].endswith("benign-suite.csv")


def test_service_driven_reuses_cached_suite_and_skips_interview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    cfg = _provisioned_config(tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        run_module._common,
        "execute",
        lambda cmd, *a, **k: (commands.append(cmd), (SimpleNamespace(returncode=0), "", None))[1],
    )
    monkeypatch.setattr(
        execution, "drive_synth_hitl", lambda *a, **k: pytest.fail("interview must be skipped when cached")
    )
    written: dict[str, Any] = {}
    monkeypatch.setattr(benign_suite, "write_suite", lambda path, suite: written.update(path=str(path), suite=suite))
    sdk = SimpleNamespace(entities=SimpleNamespace(create=lambda *a, **k: SimpleNamespace(name="run-x")))

    outcome = execution._run_service_driven(
        str(manifest),
        None,
        cfg,
        _ctx(tmp_path),
        sdk,
        "clockbot",
        1,
        manifest_id="m1",
        cached_suite=[{"tool": "clock", "payload": "time?", "label": "benign", "rationale": "", "persona": ""}],
        stop_after_synth=False,
    )

    assert outcome.status == "completed"
    assert written["suite"][0]["tool"] == "clock"  # cached suite written to the temp CSV handed to agent-hardener
    # A single self-contained war-game (no separate `up` whose forward would collide with the attack's).
    assert [c[1] for c in commands] == ["run"]
    # The cached suite is passed explicitly (agent-hardener seeds it), not via the on-disk `--reuse-benign` cache.
    assert commands[0][commands[0].index("--benign-suite") + 1].endswith("benign-suite.csv")
    assert "--reuse" not in commands[0]
    assert "--rounds" not in commands[0]  # default (1 round) omits the flag


def test_service_driven_passes_rounds_flag_when_multi_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    cfg = _provisioned_config(tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        run_module._common,
        "execute",
        lambda cmd, *a, **k: (commands.append(cmd), (SimpleNamespace(returncode=0), "", None))[1],
    )
    monkeypatch.setattr(benign_suite, "write_suite", lambda path, suite: None)
    sdk = SimpleNamespace(entities=SimpleNamespace(create=lambda *a, **k: SimpleNamespace(name="run-x")))

    execution._run_service_driven(
        str(manifest),
        None,
        cfg,
        _ctx(tmp_path),
        sdk,
        "clockbot",
        1,
        manifest_id="m1",
        cached_suite=[{"tool": "clock", "payload": "t", "label": "b", "rationale": "", "persona": ""}],
        stop_after_synth=False,
        rounds=3,
    )

    assert commands[0][commands[0].index("--rounds") + 1] == "3"  # manifest's rounds passed to `run`


def test_service_driven_generate_persists_suite_and_stops_before_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    cfg = _provisioned_config(tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        run_module._common,
        "execute",
        lambda cmd, *a, **k: (commands.append(cmd), (SimpleNamespace(returncode=0), "", None))[1],
    )

    @contextlib.contextmanager
    def fake_launch(*_a: Any, **_k: Any):
        yield object()

    monkeypatch.setattr(execution, "launch_synth_service", fake_launch)
    monkeypatch.setattr(execution, "drive_synth_hitl", lambda *a, **k: "/x/requests.csv")
    rows = [{"tool": "clock", "payload": "time?", "label": "benign", "rationale": "", "persona": ""}]
    monkeypatch.setattr(benign_suite, "read_suite", lambda _path: rows)

    persisted: dict[str, Any] = {}
    sdk = SimpleNamespace(
        entities=SimpleNamespace(
            create=lambda *a, **k: SimpleNamespace(name="run-x"),
            get_entity_by_name=lambda **k: SimpleNamespace(data={}),
            update_entity_by_name=lambda **k: persisted.update(k["data"]),
        )
    )

    outcome = execution._run_service_driven(
        str(manifest),
        None,
        cfg,
        _ctx(tmp_path),
        sdk,
        "clockbot",
        1,
        manifest_id="m1",
        cached_suite=[],
        stop_after_synth=True,
    )

    assert outcome.status == "completed"
    assert persisted["benign_suite"] == rows  # reviewed suite cached on the manifest
    # Generation stops before the attack "run" and tears its sandbox down (frees the victim-port forward).
    assert [c[1] for c in commands] == ["up", "down"]


async def test_compile_builds_subprocess_step_carrying_the_spec() -> None:
    spec = run_module.WarGameSpec(config="agent-hardener.yaml", driver="service")
    job_spec = await run_module.AgentHardenerRunJob.compile(
        workspace="default", spec=spec, entity_client=None, job_name=None, async_sdk=None
    )

    steps = list(job_spec["steps"])
    assert len(steps) == 1
    step = steps[0]
    assert step["name"] == "war-game"
    assert step["executor"]["provider"] == "subprocess"
    assert step["executor"]["command"] == ["python", "-m", "nemo_agent_hardener_plugin.tasks.war_game"]
    assert step["config"] == {
        "config": "agent-hardener.yaml",
        "manifest_id": None,
        "env_file": None,
        "driver": "service",
        "stop_after_synth": False,
        "replay_hitlog_fileset": None,
        "benign_suite_fileset": None,
        "port": None,
        "defenders": None,
        "attack_intensity": None,
        "rounds": None,
        "validate_only": False,
        "defense_guardrails": None,
        "defense_policy": None,
        "source_run": None,
        "models": None,
    }


async def test_compile_precreates_run_record_for_service_manifest() -> None:
    created: dict[str, Any] = {}

    class FakeEntityClient:
        async def get(self, _entity_type: Any, *, name: str, workspace: str) -> Any:
            assert name == "m1"
            return SimpleNamespace(agent="default/clockbot", port=8000)

        async def create(self, entity: Any) -> Any:
            created["entity"] = entity
            return SimpleNamespace(name="agent-hardener-run-abc")

    spec = run_module.WarGameSpec(manifest_id="m1", driver="service")
    job_spec = await run_module.AgentHardenerRunJob.compile(
        workspace="default", spec=spec, entity_client=FakeEntityClient(), job_name="job-1", async_sdk=None
    )

    # The pre-created run's name rides in the step config so the worker reuses it (no second record).
    assert list(job_spec["steps"])[0]["config"]["run_name"] == "agent-hardener-run-abc"
    assert created["entity"].job_id == "job-1"
    assert created["entity"].agent == "default/clockbot"
    assert created["entity"].status == "running"


async def test_compile_skips_precreation_when_generating_suite() -> None:
    async def _fail(*_a: Any, **_k: Any) -> Any:
        pytest.fail("generate-only (stop_after_synth) runs must not pre-create a run record")

    spec = run_module.WarGameSpec(manifest_id="m1", driver="service", stop_after_synth=True)
    job_spec = await run_module.AgentHardenerRunJob.compile(
        workspace="default",
        spec=spec,
        entity_client=SimpleNamespace(get=_fail, create=_fail),
        job_name="job-1",
        async_sdk=None,
    )
    assert "run_name" not in list(job_spec["steps"])[0]["config"]


def test_materialize_manifest_writes_yaml_from_agent_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = SimpleNamespace(data={"agent": "default/clockbot"})
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    monkeypatch.setattr(
        manifest_mod,
        "resolve_agent_to_manifest",
        lambda *_a, **_k: SimpleNamespace(manifest={"agent": {"name": "clockbot", "port": 8000}}, warnings=[]),
    )
    ctx = make_job_context(tmp_path)

    path = manifest_mod._materialize_manifest(sdk, "clockbot-hardening", ctx)

    assert path.endswith("agent-hardener.yaml")
    assert "clockbot" in (tmp_path / "agent-hardener.yaml").read_text(encoding="utf-8")


def test_materialize_manifest_needs_sdk(tmp_path: Path) -> None:
    ctx = make_job_context(tmp_path)
    with pytest.raises(RuntimeError, match="platform SDK"):
        manifest_mod._materialize_manifest(None, "m", ctx)


def test_materialize_project_manifest_repoints_project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stored_yaml = "agent:\n  name: research\n  project_dir: .\n  workflow: pkg/workflow.yaml\n  port: 8000\n"
    record = SimpleNamespace(
        data={"source_type": "project", "project_fileset": "default/bundle", "manifest_yaml": stored_yaml}
    )
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    restored = tmp_path / "restored-project"
    monkeypatch.setattr(manifest_mod, "download_and_extract_project", lambda *_a, **_k: restored)
    ctx = make_job_context(tmp_path)

    path = manifest_mod._materialize_manifest(sdk, "research-hardening", ctx)

    written = (tmp_path / "agent-hardener.yaml").read_text(encoding="utf-8")
    assert path.endswith("agent-hardener.yaml")
    # project_dir is repointed at the freshly restored bundle; the workflow stays project-relative.
    assert f"project_dir: {restored}" in written
    assert "workflow: pkg/workflow.yaml" in written


def test_materialize_project_manifest_requires_fileset_and_yaml(tmp_path: Path) -> None:
    record = SimpleNamespace(data={"source_type": "project", "manifest_yaml": "agent: {}"})
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    ctx = make_job_context(tmp_path)
    with pytest.raises(RuntimeError, match="project_fileset"):
        manifest_mod._materialize_manifest(sdk, "research-hardening", ctx)


def _capturing_sdk() -> tuple[NeMoPlatform, list[dict[str, Any]]]:
    """A fake SDK whose entity create/update calls capture the recorded run data."""
    captured: list[dict[str, Any]] = []
    entities = SimpleNamespace(
        create=lambda _t, *, workspace, data: (captured.append(data), SimpleNamespace(name="run-1"))[1],
        update_entity_by_name=lambda *, name, entity_type, workspace, data: captured.append(data),
    )
    return make_sdk(entities), captured


def test_run_boundary_records_classified_subprocess_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    monkeypatch.setattr(run_module.AgentHardenerConfig, "get", lambda: _provisioned_config(tmp_path))

    def failing_execute(_cmd, env, _log_path, _ctx, *, artifact_name):  # noqa: ANN001, ANN202 - test stub
        # agent-hardener writes a structured cause to the error file, then exits non-zero.
        Path(env["AGENT_HARDENER_ERROR_FILE"]).write_text(
            json.dumps({"category": "sandbox", "message": "docker down", "remediation": "start docker"}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1), "log", None

    monkeypatch.setattr(run_module._common, "execute", failing_execute)
    sdk, captured = _capturing_sdk()
    job = run_module.AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)

    result = job.run({"config": str(manifest)}, ctx=_ctx(tmp_path), sdk=sdk)

    assert result["status"] == "failed"
    assert result["error"]["category"] == "sandbox"
    assert "docker down" in result["error"]["message"]
    assert captured[-1]["status"] == "failed"
    assert captured[-1]["error_category"] == "sandbox"
    assert captured[-1]["error_remediation"] == "start docker"


def test_failure_preserves_the_precreated_agent_and_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An update replaces the whole record, so a failure must carry the pre-created facts forward.

    Otherwise the Hardening list shows the failed run targeting no agent at all.
    """
    cfg = AgentHardenerConfig(
        venv_path=tmp_path / "venv", garak_venv_path=tmp_path / "garak", operator_env_file=tmp_path / "op.env"
    )
    monkeypatch.setattr(run_module.AgentHardenerConfig, "get", lambda: cfg)  # unprovisioned → fails early

    captured: list[dict[str, Any]] = []
    precreated = SimpleNamespace(data={"agent": "default/clockbot", "port": 9000, "status": "running"})
    sdk = SimpleNamespace(
        entities=SimpleNamespace(
            get_entity_by_name=lambda **_k: precreated,
            create=lambda _t, *, workspace, data: (captured.append(data), SimpleNamespace(name="run-1"))[1],
            update_entity_by_name=lambda *, name, entity_type, workspace, data: captured.append(data),
        )
    )
    job = run_module.AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)

    result = job.run({"manifest_id": "m1", "run_name": "run-1"}, ctx=_ctx(tmp_path), sdk=sdk)

    assert result["status"] == "failed"
    assert captured[-1]["status"] == "failed"
    assert captured[-1]["agent"] == "default/clockbot"  # not blanked
    assert captured[-1]["port"] == 9000
    assert "clockbot" in captured[-1]["summary"]


def test_failure_without_a_precreated_record_has_no_agent_to_preserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AgentHardenerConfig(
        venv_path=tmp_path / "venv", garak_venv_path=tmp_path / "garak", operator_env_file=tmp_path / "op.env"
    )
    monkeypatch.setattr(run_module.AgentHardenerConfig, "get", lambda: cfg)
    sdk, captured = _capturing_sdk()
    job = run_module.AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)

    job.run({"config": str(tmp_path / "nope.yaml")}, ctx=_ctx(tmp_path), sdk=sdk)

    assert captured[-1]["agent"] == ""  # nothing was ever recorded to carry forward


def test_run_boundary_classifies_unprovisioned_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    # A config whose venvs were never provisioned (bins absent) trips require_provisioned before any subprocess.
    cfg = AgentHardenerConfig(
        venv_path=tmp_path / "venv", garak_venv_path=tmp_path / "garak", operator_env_file=tmp_path / "op.env"
    )
    monkeypatch.setattr(run_module.AgentHardenerConfig, "get", lambda: cfg)
    sdk, captured = _capturing_sdk()
    job = run_module.AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)

    result = job.run({"config": str(manifest)}, ctx=_ctx(tmp_path), sdk=sdk)

    assert result["status"] == "failed"
    assert result["error"]["category"] == "provisioning"
    assert captured[-1]["error_category"] == "provisioning"
    assert captured[-1]["status"] == "failed"


def test_service_driven_requires_a_platform_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    monkeypatch.setattr(run_module.AgentHardenerConfig, "get", lambda: _provisioned_config(tmp_path))
    monkeypatch.setattr(run_module._common, "execute", lambda *a, **k: (SimpleNamespace(returncode=0), "", None))

    ctx = _ctx(tmp_path)
    ctx.job_id = None  # local run_local: no submitted job to drive status_details HITL
    job = run_module.AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)
    # run() no longer raises: the boundary classifies the failure and surfaces it as a failed result.
    result = job.run({"config": str(manifest), "driver": "service"}, ctx=ctx, sdk=None)
    assert result["status"] == "failed"
    assert "submitted platform job" in result["error"]["message"]


def test_replay_args_empty_without_fileset(tmp_path: Path) -> None:
    assert artifacts._replay_args(None, sdk=object(), ctx=_ctx(tmp_path)) == []


def test_replay_args_downloads_fileset_and_points_replay_at_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(_sdk: Any, ref: str, dest: Path) -> Path:
        assert ref == "default/hits"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "attack.hitlog.jsonl").write_text("{}\n", encoding="utf-8")
        return dest

    monkeypatch.setattr(artifacts, "download_fileset", fake_download)

    args = artifacts._replay_args("default/hits", sdk=object(), ctx=_ctx(tmp_path))

    assert args[0] == "--replay"
    assert args[1].endswith("attack.hitlog.jsonl")


def test_replay_args_raises_when_fileset_has_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifacts, "download_fileset", lambda _s, _r, dest: (dest.mkdir(parents=True), dest)[1])
    with pytest.raises(AgentHardenerRunError, match="contained no file"):
        artifacts._replay_args("default/hits", sdk=object(), ctx=_ctx(tmp_path))


def test_service_driven_replay_appends_replay_to_run_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    cfg = _provisioned_config(tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        run_module._common,
        "execute",
        lambda cmd, *a, **k: (commands.append(cmd), (SimpleNamespace(returncode=0), "", None))[1],
    )
    monkeypatch.setattr(benign_suite, "write_suite", lambda path, suite: None)
    sdk = SimpleNamespace(entities=SimpleNamespace(create=lambda *a, **k: SimpleNamespace(name="run-x")))

    execution._run_service_driven(
        str(manifest),
        None,
        cfg,
        _ctx(tmp_path),
        sdk,
        "clockbot",
        1,
        manifest_id="m1",
        cached_suite=[{"tool": "clock", "payload": "t", "label": "b", "rationale": "", "persona": ""}],
        stop_after_synth=False,
        replay_args=["--replay", "/tmp/hits.hitlog.jsonl"],
    )

    # Replay composes with the cached-suite fast path: still passes the benign suite, plus --replay <path>.
    assert "--benign-suite" in commands[0]
    assert commands[0][commands[0].index("--replay") + 1] == "/tmp/hits.hitlog.jsonl"


def test_save_hitlog_fileset_uploads_newest_hitlog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hitlog = (
        tmp_path / ".agent-hardener" / "run-logs" / "run-1" / "round_1" / "garak" / "agent-breaker.uuid.hitlog.jsonl"
    )
    hitlog.parent.mkdir(parents=True)
    hitlog.write_text("{}\n", encoding="utf-8")

    uploaded: dict[str, Any] = {}

    def fake_upload(_sdk: Any, path: Path, *, workspace: str) -> str:
        uploaded.update(path=path, workspace=workspace)
        return f"{workspace}/hitlog-abc"

    monkeypatch.setattr(artifacts, "upload_file_to_fileset", fake_upload)

    ref = artifacts._save_hitlog_fileset(object(), _ctx(tmp_path), "default")

    assert ref == "default/hitlog-abc"
    assert uploaded["path"] == hitlog


def test_save_hitlog_fileset_empty_when_no_hitlog(tmp_path: Path) -> None:
    assert artifacts._save_hitlog_fileset(object(), _ctx(tmp_path), "default") == ""


def test_uploaded_benign_suite_downloads_fileset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(_sdk: Any, ref: str, dest: Path) -> Path:
        assert ref == "default/suite"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "requests.csv").write_text("tool,payload\nclock,now\n", encoding="utf-8")
        return dest

    monkeypatch.setattr(artifacts, "download_fileset", fake_download)
    path = artifacts._uploaded_benign_suite("default/suite", sdk=object(), ctx=_ctx(tmp_path))
    assert path is not None and path.endswith("requests.csv")


def test_uploaded_benign_suite_none_without_fileset(tmp_path: Path) -> None:
    assert artifacts._uploaded_benign_suite(None, sdk=object(), ctx=_ctx(tmp_path)) is None


def test_prepare_invocation_appends_benign_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n", encoding="utf-8")
    cfg = _provisioned_config(tmp_path)
    monkeypatch.setattr(run_module._common, "build_subprocess_env", lambda _c, _e=None: {})
    monkeypatch.setattr(run_module._common, "check_victim_secrets", lambda *a, **k: None)

    with_suite, _ = execution._prepare_invocation(str(manifest), None, cfg, None, "/tmp/suite.csv")
    assert with_suite[with_suite.index("--benign-suite") + 1] == "/tmp/suite.csv"

    without_suite, _ = execution._prepare_invocation(str(manifest), None, cfg, None, None)
    assert "--benign-suite" not in without_suite  # unchanged one-shot behavior when none supplied


def test_one_shot_forwards_benign_suite_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n", encoding="utf-8")
    cfg = _provisioned_config(tmp_path)
    monkeypatch.setattr(run_module._common, "build_subprocess_env", lambda _c, _e=None: {})
    monkeypatch.setattr(run_module._common, "check_victim_secrets", lambda *a, **k: None)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        run_module._common,
        "execute",
        lambda cmd, *a, **k: (commands.append(cmd), (SimpleNamespace(returncode=0), "", None))[1],
    )

    execution._run_one_shot(str(manifest), None, cfg, _ctx(tmp_path), None, benign_suite="/tmp/suite.csv")
    assert commands[0][commands[0].index("--benign-suite") + 1] == "/tmp/suite.csv"


def test_service_driven_uploaded_suite_overrides_and_skips_synth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    cfg = _provisioned_config(tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        run_module._common,
        "execute",
        lambda cmd, *a, **k: (commands.append(cmd), (SimpleNamespace(returncode=0), "", None))[1],
    )
    monkeypatch.setattr(
        execution, "drive_synth_hitl", lambda *a, **k: pytest.fail("uploaded suite must skip synthesis")
    )
    sdk = SimpleNamespace(entities=SimpleNamespace(create=lambda *a, **k: SimpleNamespace(name="run-x")))

    # No cached suite, but an uploaded suite override is supplied → still the explicit-suite path (no `up`/synth).
    execution._run_service_driven(
        str(manifest),
        None,
        cfg,
        _ctx(tmp_path),
        sdk,
        "clockbot",
        1,
        manifest_id="m1",
        cached_suite=[],
        benign_suite_override="/tmp/uploaded-suite.csv",
    )

    assert [c[1] for c in commands] == ["run"]
    assert commands[0][commands[0].index("--benign-suite") + 1] == "/tmp/uploaded-suite.csv"


def test_apply_manifest_overrides_maps_intensity_and_selects_defenders() -> None:
    manifest: dict[str, Any] = {"agent": {"name": "clockbot", "workflow": "workflow.yaml"}, "backends": []}
    manifest_mod._apply_manifest_overrides(manifest, {"attack_intensity": "thorough", "defenders": ["openshell"]})
    assert manifest["garak"] == {"generations": 5, "max_attempts_per_tool": 10}
    # Subset selection replaces the defender list; entries carry capabilities (validator requires it) but
    # omit the unused config block.
    entry = manifest["overrides"]["defenders"][0]
    assert entry["name"] == "openshell-policy-defender"
    assert entry["capabilities"]  # non-empty — agent-hardener's SessionConfig validator requires it
    assert "config" not in entry


def test_apply_manifest_overrides_standard_and_empty_are_noops() -> None:
    manifest = {"agent": {"name": "x", "workflow": "w"}, "backends": []}
    manifest_mod._apply_manifest_overrides(manifest, {"attack_intensity": "standard", "defenders": []})
    assert "garak" not in manifest  # standard = engine defaults
    assert "overrides" not in manifest  # empty selection = agent-hardener defaults


def test_apply_manifest_overrides_keeps_guardrails_without_a_workflow() -> None:
    """A workflow-less victim (every BYO one) must still get the guardrails defender.

    It writes Relay plugin config Agent Hardener owns, not a file the agent contains. Gating it on
    `agent.workflow` made a Studio run that *selected* defenders score 0 blocked, while selecting
    none — which falls through to agent-hardener's defaults — hardened normally.
    """
    manifest = {"agent": {"name": "x"}, "backends": []}
    manifest_mod._apply_manifest_overrides(manifest, {"defenders": ["guardrails"]})

    names = [entry["name"] for entry in manifest["overrides"]["defenders"]]
    assert names == ["defender-guardrails"]


def test_selected_defenders_match_agent_hardeners_own_defaults() -> None:
    """The plugin mirrors agent-hardener's defender list; drifting to a superseded implementation
    silently downgrades every Studio run that picks defenders."""
    implementations = {key: entry["implementation"] for key, entry in manifest_mod.DEFENDER_ENTRIES.items()}

    assert implementations["openshell"].startswith("agent_hardener.agents.defenders.openshell_defender_v2")
    assert implementations["guardrails"].startswith("agent_hardener.agents.defenders.guardrails_defender_v2")


def test_apply_manifest_overrides_applies_explicit_port_only() -> None:
    manifest = {"agent": {"name": "x", "port": 8000, "workflow": "w"}, "backends": []}
    manifest_mod._apply_manifest_overrides(manifest, {"port": 9001})
    assert manifest["agent"]["port"] == 9001
    # No port in data → leave the resolver-derived port untouched.
    manifest2 = {"agent": {"name": "x", "port": 8000, "workflow": "w"}, "backends": []}
    manifest_mod._apply_manifest_overrides(manifest2, {"defenders": []})
    assert manifest2["agent"]["port"] == 8000


def test_materialize_manifest_overlays_per_run_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stored manifest config: standard intensity, port 8000. Override to thorough + port 9100 for this run only.
    record = SimpleNamespace(data={"agent": "default/clockbot", "attack_intensity": "standard", "port": 8000})
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    monkeypatch.setattr(
        manifest_mod,
        "resolve_agent_to_manifest",
        lambda *_a, **_k: SimpleNamespace(manifest={"agent": {"name": "clockbot", "port": 8000}}, warnings=[]),
    )
    ctx = make_job_context(tmp_path)

    manifest_mod._materialize_manifest(sdk, "m1", ctx, {"attack_intensity": "thorough", "port": 9100})

    written = yaml.safe_load((tmp_path / "agent-hardener.yaml").read_text(encoding="utf-8"))
    assert written["garak"] == {"generations": 5, "max_attempts_per_tool": 10}  # thorough override applied
    assert written["agent"]["port"] == 9100  # per-run port override applied; manifest entity untouched


def test_service_driven_records_manifest_id_on_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n  port: 1\n", encoding="utf-8")
    cfg = _provisioned_config(tmp_path)
    monkeypatch.setattr(run_module._common, "execute", lambda *a, **k: (SimpleNamespace(returncode=0), "", None))
    monkeypatch.setattr(benign_suite, "write_suite", lambda path, suite: None)

    created: dict[str, Any] = {}
    sdk = SimpleNamespace(
        entities=SimpleNamespace(
            create=lambda _t, *, workspace, data: created.update(data) or SimpleNamespace(name="run-x")
        )
    )

    execution._run_service_driven(
        str(manifest),
        None,
        cfg,
        _ctx(tmp_path),
        sdk,
        "clockbot",
        1,
        manifest_id="clockbot-hardening",
        cached_suite=[{"tool": "c", "payload": "t", "label": "b", "rationale": "", "persona": ""}],
    )

    assert created["manifest_id"] == "clockbot-hardening"


GUARDRAILS_TOML = 'version = 1\n[[components]]\nkind = "agent_hardener.pre_tool_verifier"\n'


def test_seed_validation_manifest_zeros_defenders_and_seeds_baseline(tmp_path: Path) -> None:
    manifest_path = tmp_path / "agent-hardener.yaml"
    manifest_path.write_text(yaml.safe_dump({"agent": {"name": "clockbot", "port": 1}}), encoding="utf-8")

    manifest_mod._seed_validation_manifest(str(manifest_path), GUARDRAILS_TOML, "version: 2\n", _ctx(tmp_path))

    seeded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    overrides = seeded["overrides"]
    # Zero defenders → agent-hardener generates no new mitigations (frozen validation).
    assert overrides["defenders"] == []
    # Composed policy seeded as the victim's baseline policy.
    assert overrides["victim_control"]["config"]["policy_path"].endswith("composed-policy.yaml")
    assert overrides["storage"]["victim_policy_path"].endswith("composed-policy.yaml")
    assert (tmp_path / "composed-policy.yaml").read_text(encoding="utf-8") == "version: 2\n"
    # The composed guardrails become the baseline on all three paths agent-hardener derives from the
    # plugins file — what defenders base on, what seeds init/, and what is uploaded into the victim.
    composed = tmp_path / "composed-plugins.toml"
    assert composed.read_text(encoding="utf-8") == GUARDRAILS_TOML
    assert overrides["target"]["agent_relay_plugins"] == str(composed)
    assert overrides["storage"]["victim_relay_plugins_path"] == str(composed)
    assert overrides["victim_control"]["config"]["uploads"] == [f"{composed}:/etc/nemo-relay/plugins.toml"]


def test_seed_validation_manifest_without_policy_only_zeros_defenders(tmp_path: Path) -> None:
    manifest_path = tmp_path / "agent-hardener.yaml"
    manifest_path.write_text(yaml.safe_dump({"agent": {"name": "x"}}), encoding="utf-8")

    manifest_mod._seed_validation_manifest(str(manifest_path), None, None, _ctx(tmp_path))

    seeded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert seeded["overrides"] == {"defenders": []}
    assert not (tmp_path / "composed-policy.yaml").exists()


# ── stored manifest settings survive re-materialization ──────────────────


def _capture_resolve(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record the kwargs `_materialize_agent_manifest` hands to the resolver."""
    seen: dict[str, Any] = {}

    def fake_resolve(_ref, **kwargs):
        seen.update(kwargs)
        # project_dir is the scaffold resolution writes; the legacy-upgrade path freezes it.
        return SimpleNamespace(
            manifest={"agent": {"name": "clockbot", "port": 8000}},
            warnings=[],
            project_dir=kwargs["manifest_dir"] / ".agent-hardener-agents" / "clockbot",
        )

    monkeypatch.setattr(manifest_mod, "resolve_agent_to_manifest", fake_resolve)
    return seen


def test_stored_egress_and_secrets_are_reapplied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest is rebuilt from the agent ref every run; omitted settings are silently re-derived.

    Egress is the one that bites: without it the victim's outbound calls are dropped, so tool-using
    attacks no-op while the run still reports success.
    """
    record = SimpleNamespace(
        data={"agent": "default/clockbot", "egress": ["en.wikipedia.org"], "secrets": ["MY_TOKEN"]}
    )
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    seen = _capture_resolve(monkeypatch)

    manifest_mod._materialize_manifest(sdk, "clockbot-hardening", make_job_context(tmp_path))

    assert seen["egress"] == ["en.wikipedia.org"]
    assert seen["secrets"] == ["MY_TOKEN"]


def test_unset_egress_and_secrets_still_derive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """None means 'derive', so manifests that set neither behave exactly as before."""
    record = SimpleNamespace(data={"agent": "default/clockbot"})
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    seen = _capture_resolve(monkeypatch)

    manifest_mod._materialize_manifest(sdk, "clockbot-hardening", make_job_context(tmp_path))

    assert seen["egress"] is None
    assert seen["secrets"] is None


def test_project_manifest_applies_stored_egress_and_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A project manifest is rebuilt from the stored YAML, so PATCHed settings must be applied onto it.

    Without this the agent path honours an edit and the project path silently ignores it — the PATCH
    returns 200 either way.
    """
    record = SimpleNamespace(
        data={
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "manifest_yaml": "agent:\n  name: research\n  project_dir: .\n  egress:\n  - old.example\n",
            "egress": ["en.wikipedia.org", "api.example.com:80"],
            "secrets": ["MY_TOKEN"],
        }
    )
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    monkeypatch.setattr(manifest_mod, "download_and_extract_project", lambda *_a, **_k: tmp_path / "proj")

    path = manifest_mod._materialize_manifest(sdk, "research-lab", make_job_context(tmp_path))
    agent = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["agent"]

    assert agent["egress"] == ["en.wikipedia.org", "api.example.com:80"]  # replaces the baked-in value
    assert agent["secrets"] == ["MY_TOKEN"]


def test_project_manifest_keeps_its_yaml_when_nothing_is_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No stored override leaves whatever agent-hardener's own `init` baked into the manifest."""
    record = SimpleNamespace(
        data={
            "source_type": "project",
            "project_fileset": "default/proj-bundle",
            "manifest_yaml": "agent:\n  name: research\n  project_dir: .\n  egress:\n  - baked.example\n",
        }
    )
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    monkeypatch.setattr(manifest_mod, "download_and_extract_project", lambda *_a, **_k: tmp_path / "proj")

    path = manifest_mod._materialize_manifest(sdk, "research-lab", make_job_context(tmp_path))
    agent = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["agent"]

    assert agent["egress"] == ["baked.example"]


# ── frozen targets: one materialization path ─────────────────────────────────


def _frozen_agent_record(**extra: Any) -> SimpleNamespace:
    return SimpleNamespace(
        data={
            "agent": "default/clockbot",
            "agent_fileset": "default/agent-fs-1",
            "manifest_yaml": "agent:\n  name: clockbot\n  project_dir: /gone\n  port: 8000\n",
            **extra,
        }
    )


def test_frozen_agent_manifest_never_re_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored bundle is the target. Re-resolving would reintroduce silent drift and dropped settings."""
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: _frozen_agent_record()))
    monkeypatch.setattr(manifest_mod, "download_and_extract_project", lambda *_a, **_k: tmp_path / "restored")
    seen = _capture_resolve(monkeypatch)

    path = manifest_mod._materialize_manifest(sdk, "clockbot-hardening", make_job_context(tmp_path))
    agent = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["agent"]

    assert seen == {}, "a frozen manifest must not call the agent resolver"
    assert agent["project_dir"] == str(tmp_path / "restored")  # repointed at the restored bundle
    assert agent["secrets_file"].endswith(".env")


def test_frozen_manifest_takes_the_current_gateway_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gateway belongs to the platform we run on, not to the frozen target."""
    record = _frozen_agent_record()
    # backends is top-level on agent-hardener's AgentManifest; nesting it under `agent` is extra_forbidden.
    record.data["manifest_yaml"] = (
        "agent:\n  name: clockbot\n  project_dir: /gone\n"
        "backends:\n- name: nemo-gateway\n  host: stale.invalid\n  ports: [1]\n"
    )
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    monkeypatch.setattr(manifest_mod, "download_and_extract_project", lambda *_a, **_k: tmp_path / "restored")
    monkeypatch.setattr(manifest_mod, "gateway_backend", lambda _u: {"name": "nemo-gateway", "host": "fresh.local"})

    path = manifest_mod._materialize_manifest(sdk, "clockbot-hardening", make_job_context(tmp_path))
    rendered = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    assert "backends" not in rendered["agent"], "nested backends fails AgentManifest validation"
    assert [b["host"] for b in rendered["backends"]] == ["fresh.local"]  # replaced, not appended


def test_legacy_manifest_re_resolves_then_freezes_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifests made before freezing keep working, and upgrade on their first run — no user action."""
    record = SimpleNamespace(data={"agent": "default/clockbot"})  # no agent_fileset
    updated: dict[str, Any] = {}
    sdk = SimpleNamespace(
        entities=SimpleNamespace(
            get_entity_by_name=lambda **_k: record,
            update_entity_by_name=lambda **kw: updated.update(kw),
        )
    )
    seen = _capture_resolve(monkeypatch)
    monkeypatch.setattr(manifest_mod, "upload_project_dir", lambda _s, _d, *, workspace: "default/new-fs")

    manifest_mod._materialize_manifest(sdk, "clockbot-hardening", make_job_context(tmp_path))

    assert seen != {}, "the legacy path still resolves"
    assert updated["data"]["agent_fileset"] == "default/new-fs"
    assert updated["data"]["manifest_yaml"]


def test_legacy_upgrade_failure_does_not_fail_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The war-game matters more than the upgrade; a storage hiccup must not lose the run."""
    record = SimpleNamespace(data={"agent": "default/clockbot"})
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    _capture_resolve(monkeypatch)

    def _boom(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("storage down")

    monkeypatch.setattr(manifest_mod, "upload_project_dir", _boom)

    path = manifest_mod._materialize_manifest(sdk, "clockbot-hardening", make_job_context(tmp_path))

    assert Path(path).exists()


@pytest.mark.parametrize(
    ("source", "record_extra"),
    [
        pytest.param("agent", {"agent_fileset": "default/agent-fs-1"}, id="agent-source"),
        pytest.param("project", {"project_fileset": "default/proj-fs-1"}, id="project-source"),
    ],
)
def test_stored_env_reaches_both_sources(
    source: str, record_extra: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug class this unification exists to kill: a setting honoured on one source, ignored on the other."""
    record = SimpleNamespace(
        data={
            "source_type": source,
            "manifest_yaml": "agent:\n  name: a\n  project_dir: .\n  env:\n    KEEP: me\n",
            "env": {"BACKEND_URL": "http://host.docker.internal:8086", "KEEP": "overridden"},
            **record_extra,
        }
    )
    sdk = SimpleNamespace(entities=SimpleNamespace(get_entity_by_name=lambda **_k: record))
    monkeypatch.setattr(manifest_mod, "download_and_extract_project", lambda *_a, **_k: tmp_path / "restored")

    path = manifest_mod._materialize_manifest(sdk, "m1", make_job_context(tmp_path))
    env = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["agent"]["env"]

    assert env["BACKEND_URL"] == "http://host.docker.internal:8086"
    assert env["KEEP"] == "overridden", "stored env wins over what was baked in at init"


def test_prepare_invocation_forwards_rounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The native (CLI) path must be able to ask for multi-round hardening, not just the service path."""
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: clockbot\n", encoding="utf-8")
    cfg = _provisioned_config(tmp_path)
    monkeypatch.setattr(run_module._common, "build_subprocess_env", lambda _c, _e=None: {})
    monkeypatch.setattr(run_module._common, "check_victim_secrets", lambda *a, **k: None)

    multi, _ = execution._prepare_invocation(str(manifest), None, cfg, None, None, None, 3)
    assert multi[multi.index("--rounds") + 1] == "3"

    single, _ = execution._prepare_invocation(str(manifest), None, cfg, None, None, None, 1)
    assert "--rounds" not in single  # agent-hardener's own default; only emitted when asked for
