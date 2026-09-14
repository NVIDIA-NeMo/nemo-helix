# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from nemo_agents_plugin.cli import (
    MAX_ETHOS_STAGED_BYTES,
    MAX_ETHOS_STAGED_FILES,
    AgentsCLI,
    _agents_client,
    _collect_text_agent_artifacts,
    _spec_package_warning,
    _upload_ethos_fileset,
)
from nemo_agents_plugin.entities import AGENT_SPEC_FILENAME
from typer.testing import CliRunner


class _ValidatedAgentConfig:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return self._config


class _FakeFilesResponse:
    def __init__(self, paths: Sequence[str]) -> None:
        self._body = SimpleNamespace(data=[SimpleNamespace(path=path) for path in paths])

    def data(self) -> SimpleNamespace:
        return self._body


class _FakeEthosFiles:
    def __init__(self, existing_paths: Sequence[str] = (), *, delete_error: Exception | None = None) -> None:
        self.existing_paths = existing_paths
        self.delete_error = delete_error
        self.deleted: list[str] = []

    def list_files(self, *, name: str, workspace: str) -> _FakeFilesResponse:
        assert name == "fabric-agent-ethos"
        assert workspace == "default"
        return _FakeFilesResponse(self.existing_paths)

    def delete_file(self, *, path: str, name: str, workspace: str) -> None:
        assert name == "fabric-agent-ethos"
        assert workspace == "default"
        self.deleted.append(path)
        if self.delete_error is not None:
            raise self.delete_error


def _upload_ethos_snapshot(agent_root: Path, *, existing_paths: Sequence[str] = ()) -> tuple[set[str], list[str]]:
    files = _FakeEthosFiles(existing_paths)
    sdk = SimpleNamespace(files=files)
    uploaded: set[str] = set()

    def _capture_upload(local_dir: Path, *, fileset: str, workspace: str, sdk: Any) -> None:
        assert fileset == "fabric-agent-ethos"
        assert workspace == "default"
        uploaded.update(path.relative_to(local_dir).as_posix() for path in local_dir.rglob("*") if path.is_file())

    with (
        patch("nemo_agents_plugin.cli._platform_sdk", return_value=sdk),
        patch("nemo_agents_plugin.cli.client_from_platform", return_value=files),
        patch("nemo_agents_plugin.jobs.fileset_io.upload_to_fileset", _capture_upload),
    ):
        _upload_ethos_fileset(
            agent_name="fabric-agent",
            workspace="default",
            agent_root=agent_root,
            base_url="http://test",
        )

    return uploaded, files.deleted


def _install_mock_transport(
    handler, *, on_create: Callable[[dict[str, Any]], None] | None = None
) -> AbstractContextManager[Any]:
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    class _Client(real_client):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if on_create is not None:
                on_create(kwargs)
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    return patch("nemo_agents_plugin.cli.httpx.Client", _Client)


def test_no_args_prints_help_successfully() -> None:
    app = AgentsCLI().get_cli()
    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "Agent lifecycle management" in result.stdout


def test_agents_client_preserves_legacy_cli_timeout() -> None:
    captured_timeout: list[float | None] = []

    with (
        patch("nemo_agents_plugin.cli._resolve_context_headers", return_value={}),
        _install_mock_transport(
            lambda _request: httpx.Response(200, json={}),
            on_create=lambda kwargs: captured_timeout.append(kwargs.get("timeout")),
        ),
    ):
        client = _agents_client("http://test", "default")

    assert captured_timeout == [30]
    assert client._timeout == 30
    assert client._client.timeout == httpx.Timeout(30)


def test_run_starts_nat_server_for_nat_config(tmp_path: Path) -> None:
    config = tmp_path / "workflow.yaml"
    config.write_text("workflow:\n  _type: chat_completion\n")

    app = AgentsCLI().get_cli()
    with patch("subprocess.run") as run:
        result = CliRunner().invoke(
            app,
            ["run", "--agent-config", str(config), "--host", "127.0.0.1", "--port", "8081"],
        )

    assert result.exit_code == 0, result.stderr
    run.assert_called_once_with(
        [
            "nat",
            "start",
            "fastapi",
            "--config_file",
            config.name,
            "--host",
            "127.0.0.1",
            "--port",
            "8081",
        ],
        check=True,
        cwd=tmp_path,
    )


def test_run_starts_fabric_server_for_platform_config(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text("config_format: nemo-agents-spec-v1\nname: fabric-agent\n")

    app = AgentsCLI().get_cli()
    with patch("subprocess.run") as run:
        result = CliRunner().invoke(
            app,
            ["run", "--agent-config", str(config), "--host", "127.0.0.1", "--port", "8081"],
        )

    assert result.exit_code == 0, result.stderr
    run.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "nemo_agents_plugin.fabric.server",
            "--agent-config",
            config.name,
            "--host",
            "127.0.0.1",
            "--port",
            "8081",
        ],
        check=True,
        cwd=tmp_path,
    )


def test_run_rejects_empty_yaml_config(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text("")

    app = AgentsCLI().get_cli()
    with patch("subprocess.run") as run:
        result = CliRunner().invoke(app, ["run", "--agent-config", str(config)])

    assert result.exit_code == 1
    assert f"agent config {config} root must be a YAML mapping" in result.stderr
    run.assert_not_called()


def test_list_404_prints_request_context_and_hint() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["list", "--base-url", "http://test"])

    assert result.exit_code == 1
    assert "Error: GET agent API failed: HTTP 404 Not Found" in result.stderr
    assert "Request: GET http://test/apis/agents/v2/workspaces/default/agents" in result.stderr
    assert "Target: agents API route /apis/agents/v2/workspaces/default/agents" in result.stderr
    assert "route may not be deployed" in result.stderr


def test_optimize_targets_agents_route() -> None:
    captured: dict[str, Any] = {}

    from nemo_platform_plugin.commands import add_job_commands
    from nemo_platform_plugin.scheduler import submit_path_for

    OptimizeJob = import_module("nemo_optimization.jobs.optimize").OptimizeJob
    assert submit_path_for(OptimizeJob, workspace="default") == "/apis/agents/v2/workspaces/default/jobs/optimize"

    def _submit_remote(_self, job_cls, spec, **kwargs):
        captured["job_cls"] = job_cls
        captured["spec"] = spec
        captured["base_url"] = kwargs["base_url"]
        captured["workspace"] = kwargs["workspace"]
        return {"name": "optimize-123"}

    agents_cli = AgentsCLI()
    app = agents_cli.get_cli()
    add_job_commands(app, {"agents.optimize": OptimizeJob}, cli=agents_cli)
    with patch("nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote", _submit_remote):
        result = CliRunner().invoke(
            app,
            [
                "optimize",
                "--optimize-config",
                "/tmp/optimize.yml",
                "--agent",
                "react-agent",
                "--base-url",
                "http://test",
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert captured["job_cls"] is OptimizeJob
    assert captured["base_url"] == "http://test"
    assert captured["workspace"] == "default"
    assert captured["spec"]["agent"] == "react-agent"
    assert captured["spec"]["optimize_config"] == "/tmp/optimize.yml"

    legacy_result = CliRunner().invoke(app, ["optimize", "submit"])
    assert legacy_result.exit_code == 2
    assert "No such command 'submit'" in legacy_result.output


def test_optimize_prepare_fileset_stays_under_optimize_command() -> None:
    from nemo_platform_plugin.commands import add_job_commands

    OptimizeJob = import_module("nemo_optimization.jobs.optimize").OptimizeJob

    agents_cli = AgentsCLI()
    app = agents_cli.get_cli()
    add_job_commands(app, {"agents.optimize": OptimizeJob}, cli=agents_cli)

    result = CliRunner().invoke(app, ["optimize", "prepare-fileset", "--help"])

    assert result.exit_code == 0, result.output
    assert "--source" in result.output
    assert "--fileset" in result.output

    top_level_result = CliRunner().invoke(app, ["prepare-fileset", "--help"])
    assert top_level_result.exit_code != 0


def test_agent_jobs_do_not_register_legacy_run_submit_verbs() -> None:
    import click
    from nemo_agents_plugin.jobs.analyze_batch import AnalyzeBatchJob
    from nemo_agents_plugin.jobs.evaluate_agent import EvaluateAgentJob
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteJob
    from nemo_agents_plugin.jobs.execute import ExecuteAgentJob
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsJob
    from nemo_agents_plugin.jobs.package_agent import PackageAgentJob
    from nemo_platform_plugin.commands import add_job_commands
    from nemo_platform_plugin.job import NemoJob
    from typer.main import get_command

    OptimizeJob = import_module("nemo_optimization.jobs.optimize").OptimizeJob
    jobs: dict[str, type[NemoJob]] = {
        "agents.analyze": AnalyzeBatchJob,
        "agents.evaluate": EvaluateAgentJob,
        "agents.evaluate-suite": EvaluateSuiteJob,
        "agents.execute": ExecuteAgentJob,
        "agents.optimize": OptimizeJob,
        "agents.optimize-skills": OptimizeSkillsJob,
        "agents.package-agent": PackageAgentJob,
    }

    agents_cli = AgentsCLI()
    app = agents_cli.get_cli()
    add_job_commands(app, jobs, cli=agents_cli)
    command = get_command(app)

    assert isinstance(command, click.Group)
    flat_job_names = {job_cls.name for job_cls in jobs.values()} - {"optimize"}
    for job_name in flat_job_names:
        job_command = command.commands[job_name]
        assert not isinstance(job_command, click.Group)
        for legacy_verb in ("run", "submit"):
            legacy_result = CliRunner().invoke(app, [job_name, legacy_verb])
            assert legacy_result.exit_code == 2
            assert "Got unexpected extra argument" in legacy_result.output

    optimize_command = command.commands["optimize"]
    assert isinstance(optimize_command, click.Group)
    assert set(optimize_command.commands) == {"prepare-fileset"}

    optimize_help = CliRunner().invoke(app, ["optimize", "--help"])
    assert optimize_help.exit_code == 0, optimize_help.output
    assert "Precedence:" in optimize_help.output

    for job_cls in jobs.values():
        job_command = command.commands[job_cls.name]
        if isinstance(job_command, click.Group):
            assert "run" not in job_command.commands
            assert "submit" not in job_command.commands


@pytest.mark.parametrize("placeholder", ["${NEMO_DEFAULT_MODEL}", "$NEMO_DEFAULT_MODEL"])
def test_create_resolves_default_model_placeholder(tmp_path, placeholder: str) -> None:
    """`nemo agents create` resolves NEMO_DEFAULT_MODEL before POST.

    Regression for AIRCORE-613: the agents service has no user context at
    deploy time, so an unresolved literal would be persisted on the Agent.
    Covers both braced ``${VAR}`` and bare ``$VAR`` forms supported by
    ``expand_env_vars``.
    """
    import json as _json

    config = tmp_path / "agent.yml"
    config.write_text(f"llms:\n  llm:\n    _type: openai\n    model_name: {placeholder}\n")

    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = req.read()
        return httpx.Response(200, json={"name": "calc"})

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.utils.get_default_model", return_value="nvidia-nemotron-3-super-v3"),
    ):
        result = CliRunner().invoke(
            app, ["create", "--name", "calc", "--agent-config", str(config), "--base-url", "http://test"]
        )

    assert result.exit_code == 0, result.stderr
    sent = _json.loads(captured["body"])
    assert sent["config"]["llms"]["llm"]["model_name"] == "nvidia-nemotron-3-super-v3"
    assert sent["config_format"] == "nat-workflow-v1"


def test_create_validates_platform_agent_config_before_post(tmp_path) -> None:
    import json as _json

    config = tmp_path / "agent.yaml"
    config.write_text(
        "\n".join(
            [
                "config_format: nemo-agents-spec-v1",
                "name: fabric-agent",
                "default_harness: hermes",
                "harnesses:",
                "  hermes:",
                "    kind: hermes",
                "models:",
                "  default:",
                "    provider: nvidia",
                "    model: ${NEMO_DEFAULT_MODEL}",
                "",
            ]
        )
    )
    normalized_config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes", "settings": {}}},
        "models": {"default": {"provider": "nvidia", "model": "user-default-model"}},
        "environment": {"provider": "local"},
    }
    captured: dict[str, Any] = {}

    async def _validate_platform_agent_config(config_dict: dict[str, Any], *, base_dir: Path):
        captured["validated_config"] = config_dict
        captured["base_dir"] = base_dir
        return type("ValidationResult", (), {"agent_config": _ValidatedAgentConfig(normalized_config)})()

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = req.read()
        return httpx.Response(200, json={"name": "fabric-agent"})

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.utils.get_default_model", return_value="user-default-model"),
        patch("nemo_agents_plugin.fabric.validation.validate_platform_agent_config", _validate_platform_agent_config),
        patch("nemo_agents_plugin.cli._upload_ethos_fileset") as mock_upload,
    ):
        result = CliRunner().invoke(
            app,
            ["create", "--name", "fabric-agent", "--agent-config", str(config), "--base-url", "http://test"],
        )

    assert result.exit_code == 0, result.stderr
    sent = _json.loads(captured["body"])
    assert captured["base_dir"] == tmp_path
    assert captured["validated_config"]["config_format"] == "nemo-agents-spec-v1"
    assert captured["validated_config"]["models"]["default"]["model"] == "user-default-model"
    assert sent["config"] == normalized_config
    assert sent["config_format"] == "nemo-agents-spec-v1"
    mock_upload.assert_called_once_with(
        agent_name="fabric-agent",
        workspace="default",
        agent_root=tmp_path,
        base_url="http://test",
    )


def test_create_fabric_uploads_ethos_fileset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "agents" / "fabric-agent-spec"
    package.mkdir(parents=True)
    config = package / "agent.yaml"
    config.write_text(
        "\n".join(
            [
                "config_format: nemo-agents-spec-v1",
                "name: fabric-agent",
                "default_harness: hermes",
                "harnesses:",
                "  hermes:",
                "    kind: hermes",
                "",
            ]
        )
    )
    (package / AGENT_SPEC_FILENAME).write_text("# Contract\n")
    monkeypatch.chdir(tmp_path)
    normalized_config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes", "settings": {}}},
        "environment": {"provider": "local"},
    }

    async def _validate_platform_agent_config(config_dict: dict[str, Any], *, base_dir: Path):
        del config_dict, base_dir
        return type("ValidationResult", (), {"agent_config": _ValidatedAgentConfig(normalized_config)})()

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        return httpx.Response(200, json={"name": "fabric-agent"})

    uploaded: dict[str, Any] = {}

    def fake_upload(local_dir: Path, *, fileset: str, workspace: str, sdk: Any) -> None:
        uploaded["files"] = {path.relative_to(local_dir).as_posix() for path in local_dir.rglob("*") if path.is_file()}
        uploaded["fileset"] = fileset
        uploaded["workspace"] = workspace
        uploaded["sdk_base_url"] = sdk.base_url

    files = _FakeEthosFiles([AGENT_SPEC_FILENAME], delete_error=FileNotFoundError())

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.fabric.validation.validate_platform_agent_config", _validate_platform_agent_config),
        patch("nemo_agents_plugin.jobs.fileset_io.upload_to_fileset", fake_upload),
        patch("nemo_agents_plugin.cli.client_from_platform", return_value=files),
        patch("nemo_agents_plugin.cli._platform_sdk") as mock_sdk,
    ):
        mock_sdk.return_value = SimpleNamespace(base_url="http://test", files=files)
        result = CliRunner().invoke(
            app,
            ["create", "--name", "fabric-agent", "--agent-config", str(config), "--base-url", "http://test"],
        )

    assert result.exit_code == 0, result.stderr
    warning = _spec_package_warning("fabric-agent", config)
    assert result.stderr.splitlines()[-len(warning) :] == list(warning)
    assert uploaded["files"] == {"agent.yaml"}
    assert files.deleted == [AGENT_SPEC_FILENAME]
    assert uploaded["fileset"] == "fabric-agent-ethos"
    assert uploaded["workspace"] == "default"
    assert uploaded["sdk_base_url"] == "http://test"


def test_spec_package_warning_points_at_nemo_ethos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "agents" / "acme-bot-spec"
    package.mkdir(parents=True)
    config = package / "agent.yaml"
    config.write_text("name: acme-bot\n")
    (package / AGENT_SPEC_FILENAME).write_text("# spec\n")
    monkeypatch.chdir(tmp_path)

    assert _spec_package_warning("acme-bot", config) == (
        "Warning: This package uses AGENT-SPEC.md.",
        "Run the nemo-ethos skill to write ETHOS.md, then delete the acme-bot-spec package.",
    )
    assert _spec_package_warning("acme-bot", tmp_path / "agent.yaml") == ()
    escaped = tmp_path / "escaped-spec"
    escaped.mkdir()
    (escaped / "agent.yaml").write_text("name: escaped\n")
    (escaped / AGENT_SPEC_FILENAME).write_text("# spec\n")
    assert _spec_package_warning("../escaped", escaped / "agent.yaml") == ()


def test_agents_cli_has_no_ethos_migrate_command() -> None:
    result = CliRunner().invoke(AgentsCLI().get_cli(), ["ethos", "migrate", "--help"])

    assert result.exit_code != 0


def test_collect_text_agent_artifacts_allows_small_agent_root(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text("name: a\n")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "SKILL.md").write_text("# skill\n")

    _collect_text_agent_artifacts(tmp_path)


def test_upload_ethos_fileset_skips_non_utf8_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    agent_root = tmp_path / "agent"
    agent_root.mkdir()
    (agent_root / "agent.yaml").write_text("name: fabric-agent\n", encoding="utf-8")
    cache = agent_root / "__pycache__"
    cache.mkdir()
    (cache / "tool.pyc").write_bytes(b"\xff\xfe\x00binary")

    uploaded, deleted = _upload_ethos_snapshot(agent_root)

    assert uploaded == {"agent.yaml"}
    assert deleted == []
    assert "skipping non-UTF-8 agent artifact '__pycache__/tool.pyc'" in capsys.readouterr().err


def test_upload_ethos_fileset_replaces_stale_runtime_artifacts(tmp_path: Path) -> None:
    agent_root = tmp_path / "agent"
    skill = agent_root / "skills" / "review"
    skill.mkdir(parents=True)
    (agent_root / "agent.yaml").write_text("name: fabric-agent\n", encoding="utf-8")
    (skill / "SKILL.md").write_text("# Review\n", encoding="utf-8")

    uploaded, deleted = _upload_ethos_snapshot(
        agent_root,
        existing_paths=[
            "ETHOS.md",
            AGENT_SPEC_FILENAME,
            "agent.yaml",
            "skills/old/SKILL.md",
            "__pycache__/tool.pyc",
        ],
    )

    assert deleted == [AGENT_SPEC_FILENAME, "agent.yaml", "skills/old/SKILL.md", "__pycache__/tool.pyc"]
    assert uploaded == {"agent.yaml", "skills/review/SKILL.md"}


def test_upload_ethos_fileset_preserves_remote_ethos_over_local(tmp_path: Path) -> None:
    agent_root = tmp_path / "agent"
    agent_root.mkdir()
    (agent_root / "agent.yaml").write_text("name: fabric-agent\n", encoding="utf-8")
    (agent_root / "ETHOS.md").write_text("# Local Ethos\n", encoding="utf-8")
    remote = {"ETHOS.md": b"# Remote Ethos\n"}
    files = _FakeEthosFiles(list(remote))
    sdk = SimpleNamespace(files=files)

    def _capture_upload(local_dir: Path, **_: Any) -> None:
        remote.update(
            (path.relative_to(local_dir).as_posix(), path.read_bytes())
            for path in local_dir.rglob("*")
            if path.is_file()
        )

    with (
        patch("nemo_agents_plugin.cli._platform_sdk", return_value=sdk),
        patch("nemo_agents_plugin.cli.client_from_platform", return_value=files),
        patch("nemo_agents_plugin.jobs.fileset_io.upload_to_fileset", _capture_upload),
    ):
        _upload_ethos_fileset(
            agent_name="fabric-agent",
            workspace="default",
            agent_root=agent_root,
            base_url="http://test",
        )

    assert remote == {"ETHOS.md": b"# Remote Ethos\n", "agent.yaml": b"name: fabric-agent\n"}
    assert files.deleted == []


def test_collect_text_agent_artifacts_rejects_oversized_agent_root(tmp_path: Path) -> None:
    (tmp_path / "big.bin").write_bytes(b"x" * (MAX_ETHOS_STAGED_BYTES + 1))

    with pytest.raises(ValueError, match="byte limit for container config delivery"):
        _collect_text_agent_artifacts(tmp_path)


def test_collect_text_agent_artifacts_rejects_too_many_files(tmp_path: Path) -> None:
    for index in range(MAX_ETHOS_STAGED_FILES + 1):
        (tmp_path / f"f{index}.txt").write_text("x")

    with pytest.raises(ValueError, match="more than"):
        _collect_text_agent_artifacts(tmp_path)


def test_collect_text_agent_artifacts_rejects_file_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"x" * (MAX_ETHOS_STAGED_BYTES + 1))
    agent_root = tmp_path / "agent"
    agent_root.mkdir()
    (agent_root / "agent.yaml").write_text("name: a\n")
    (agent_root / "link.bin").symlink_to(outside)

    with pytest.raises(ValueError, match="contains symlink 'link.bin'"):
        _collect_text_agent_artifacts(agent_root)


def test_collect_text_agent_artifacts_rejects_directory_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("x")
    agent_root = tmp_path / "agent"
    agent_root.mkdir()
    (agent_root / "agent.yaml").write_text("name: a\n")
    (agent_root / "linkdir").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="contains symlink 'linkdir'"):
        _collect_text_agent_artifacts(agent_root)


def test_create_fabric_rolls_back_agent_when_fileset_upload_fails(tmp_path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text(
        "\n".join(
            [
                "config_format: nemo-agents-spec-v1",
                "name: fabric-agent",
                "default_harness: hermes",
                "harnesses:",
                "  hermes:",
                "    kind: hermes",
                "",
            ]
        )
    )
    normalized_config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes", "settings": {}}},
        "environment": {"provider": "local"},
    }
    methods: list[str] = []

    async def _validate_platform_agent_config(config_dict: dict[str, Any], *, base_dir: Path):
        del config_dict, base_dir
        return type("ValidationResult", (), {"agent_config": _ValidatedAgentConfig(normalized_config)})()

    def handler(req: httpx.Request) -> httpx.Response:
        methods.append(req.method)
        if req.method == "POST":
            return httpx.Response(200, json={"name": "fabric-agent"})
        if req.method == "DELETE":
            return httpx.Response(204)
        raise AssertionError(f"unexpected {req.method}")

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.fabric.validation.validate_platform_agent_config", _validate_platform_agent_config),
        patch(
            "nemo_agents_plugin.cli._upload_ethos_fileset",
            side_effect=RuntimeError("upload boom"),
        ),
        patch("nemo_agents_plugin.cli._platform_sdk") as mock_sdk,
    ):
        result = CliRunner().invoke(
            app,
            ["create", "--name", "fabric-agent", "--agent-config", str(config), "--base-url", "http://test"],
        )

    assert result.exit_code == 1
    assert "failed to upload Ethos fileset" in result.stderr
    # Rollback removes the agent entity only; the Ethos fileset is durable and may
    # already hold an ETHOS.md written before this agent existed.
    assert methods == ["POST", "DELETE"]
    mock_sdk.assert_not_called()


def test_create_fabric_reports_rollback_failure(tmp_path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text(
        "\n".join(
            [
                "config_format: nemo-agents-spec-v1",
                "name: fabric-agent",
                "default_harness: hermes",
                "harnesses:",
                "  hermes:",
                "    kind: hermes",
                "",
            ]
        )
    )
    normalized_config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes", "settings": {}}},
        "environment": {"provider": "local"},
    }

    async def _validate_platform_agent_config(config_dict: dict[str, Any], *, base_dir: Path):
        del config_dict, base_dir
        return type("ValidationResult", (), {"agent_config": _ValidatedAgentConfig(normalized_config)})()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            return httpx.Response(200, json={"name": "fabric-agent"})
        return httpx.Response(500, json={"detail": "delete exploded"})

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.fabric.validation.validate_platform_agent_config", _validate_platform_agent_config),
        patch(
            "nemo_agents_plugin.cli._upload_ethos_fileset",
            side_effect=RuntimeError("upload boom"),
        ),
    ):
        result = CliRunner().invoke(
            app,
            ["create", "--name", "fabric-agent", "--agent-config", str(config), "--base-url", "http://test"],
        )

    assert result.exit_code == 1
    assert "failed to roll back agent 'fabric-agent'" in result.stderr
    assert "nemo agents delete fabric-agent" in result.stderr


def test_create_nat_does_not_upload_ethos_fileset(tmp_path) -> None:
    config = tmp_path / "agent.yml"
    config.write_text("llms:\n  llm:\n    _type: openai\n    model_name: nvidia-nemotron-3-super-v3\n")

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        return httpx.Response(200, json={"name": "calc"})

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.cli._upload_ethos_fileset") as mock_upload,
        patch("nemo_agents_plugin.utils.get_default_model", return_value="nvidia-nemotron-3-super-v3"),
    ):
        result = CliRunner().invoke(
            app, ["create", "--name", "calc", "--agent-config", str(config), "--base-url", "http://test"]
        )

    assert result.exit_code == 0, result.stderr
    mock_upload.assert_not_called()


def test_create_rejects_unsupported_config_format(tmp_path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text("config_format: custom-v2\nname: custom-agent\n")

    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not POST unsupported config_format")

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["create", "--name", "custom-agent", "--agent-config", str(config), "--base-url", "http://test"],
        )

    assert result.exit_code == 1
    assert "unsupported config_format 'custom-v2'" in result.stderr


@pytest.mark.parametrize("placeholder", ["${NEMO_DEFAULT_MODEL}", "$NEMO_DEFAULT_MODEL"])
def test_create_aborts_when_default_model_missing(tmp_path, placeholder: str) -> None:
    """If no default model is selected, refuse to POST a config with an unresolved
    NEMO_DEFAULT_MODEL placeholder (braced or bare). Regression for AIRCORE-613."""
    config = tmp_path / "agent.yml"
    config.write_text(f"llms:\n  llm:\n    _type: openai\n    model_name: {placeholder}\n")

    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not POST when placeholder is unresolved")

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.utils.get_default_model", return_value=None),
    ):
        result = CliRunner().invoke(
            app, ["create", "--name", "calc", "--agent-config", str(config), "--base-url", "http://test"]
        )

    assert result.exit_code == 1
    assert "${NEMO_DEFAULT_MODEL}" in result.stderr
    assert "nemo setup" in result.stderr


def test_invoke_with_custom_timeout() -> None:
    """--timeout is threaded through to the httpx client."""
    captured_timeout: list[float | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "test", "choices": []})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler, on_create=lambda kw: captured_timeout.append(kw.get("timeout"))):
        result = CliRunner().invoke(
            app,
            ["invoke", "--agent", "calc", "--input", "hi", "--base-url", "http://test", "--timeout", "42"],
        )

    assert result.exit_code == 0, result.stderr
    assert captured_timeout[0] == 42.0


def test_invoke_timeout_error_message() -> None:
    """Timeout errors print actionable guidance mentioning --timeout."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=req)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app, ["invoke", "--agent", "calc", "--input", "hi", "--base-url", "http://test", "--timeout", "5"]
        )

    assert result.exit_code == 1
    assert "timed out" in result.stderr.lower()
    assert "--timeout" in result.stderr


def test_local_invoke_runs_fabric_config_once(tmp_path: Path) -> None:
    import json as _json

    from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult

    config = tmp_path / "agent.yaml"
    config.write_text(
        "\n".join(
            [
                "config_format: nemo-agents-spec-v1",
                "name: fabric-agent",
                "default_harness: hermes",
                "harnesses:",
                "  hermes:",
                "    kind: hermes",
                "models:",
                "  default:",
                "    provider: openai",
                "    model: openai/gpt-5.4",
                "",
            ]
        )
    )
    captured: dict[str, Any] = {}

    async def _invoke_agent_config_once(agent_config: Any, inputs: list[Any], *, base_dir: Path):
        captured["agent_config"] = agent_config
        captured["inputs"] = inputs
        captured["base_dir"] = base_dir
        return [
            FabricRuntimeResult(
                status="succeeded",
                output={"response": "hello"},
                response="hello",
                runtime_id="runtime-1",
                invocation_id="invocation-1",
                request_id="request-1",
            )
        ]

    app = AgentsCLI().get_cli()
    with patch("nemo_agents_plugin.fabric.invocation.invoke_agent_config_once", _invoke_agent_config_once):
        result = CliRunner().invoke(app, ["invoke", "--agent-config", str(config), "--input", "hello"])

    assert result.exit_code == 0, result.stderr
    assert captured["base_dir"] == tmp_path
    assert captured["inputs"] == ["hello"]
    assert captured["agent_config"].config_format == "nemo-agents-spec-v1"
    assert captured["agent_config"].name == "fabric-agent"
    parsed = _json.loads(result.stdout)
    assert parsed["status"] == "succeeded"
    assert parsed["response"] == "hello"
    assert parsed["runtime_id"] == "runtime-1"


def test_local_invoke_fabric_config_exits_nonzero_on_failed_result(tmp_path: Path) -> None:
    from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult

    config = tmp_path / "agent.yaml"
    config.write_text(
        "\n".join(
            [
                "config_format: nemo-agents-spec-v1",
                "name: fabric-agent",
                "default_harness: hermes",
                "harnesses:",
                "  hermes:",
                "    kind: hermes",
                "models:",
                "  default:",
                "    provider: openai",
                "    model: openai/gpt-5.4",
                "",
            ]
        )
    )

    async def _invoke_agent_config_once(agent_config: Any, inputs: list[Any], *, base_dir: Path):
        del agent_config, inputs, base_dir
        return [
            FabricRuntimeResult(
                status="failed",
                error={"stage": "invoke", "message": "adapter failed"},
                events=[{"kind": "invocation_end"}],
                request_id="request-1",
            ),
            FabricRuntimeResult(
                status="succeeded",
                response="later result",
                request_id="request-2",
            ),
        ]

    app = AgentsCLI().get_cli()
    with patch("nemo_agents_plugin.fabric.invocation.invoke_agent_config_once", _invoke_agent_config_once):
        result = CliRunner().invoke(app, ["invoke", "--agent-config", str(config), "--input", "hello"])

    assert result.exit_code == 1
    assert '"status": "failed"' in result.stdout
    assert "adapter failed" in result.stdout
    assert "later result" in result.stdout
    assert "request-2" in result.stdout


def test_platform_invoke_writes_clean_json_to_stdout() -> None:
    """`nemo agents invoke --agent` returns JSON on stdout with no spinner bleed.

    AIRCORE-574: the spinner must render only on stderr so consumers can pipe
    stdout to `jq`. CliRunner's non-TTY stderr auto-disables the spinner, so
    here we just verify the response JSON is intact on stdout.
    """
    import json as _json

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["invoke", "--agent", "calc", "--input", "ping", "--base-url", "http://test"])

    assert result.exit_code == 0, result.stderr
    parsed = _json.loads(result.stdout)
    assert parsed["choices"][0]["message"]["content"] == "hi"


def test_platform_invoke_accepts_no_progress_flag() -> None:
    """`--no-progress` is wired through and doesn't break invocation."""
    import json as _json

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["invoke", "--agent", "calc", "--input", "ping", "--no-progress", "--base-url", "http://test"],
        )

    assert result.exit_code == 0, result.stderr
    assert _json.loads(result.stdout)["choices"][0]["message"]["content"] == "hi"


def test_list_connection_error_prints_request_context_and_hint() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=req)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["list", "--base-url", "http://test"])

    assert result.exit_code == 1
    assert "Error: GET agent API failed: connection refused" in result.stderr
    assert "Request: GET http://test/apis/agents/v2/workspaces/default/agents" in result.stderr
    assert "Target: agents API route /apis/agents/v2/workspaces/default/agents" in result.stderr
    assert "nemo config view" in result.stderr


# ---------------------------------------------------------------------------
# environment / environment-spec / compute-spec commands
# ---------------------------------------------------------------------------


def test_deploy_forwards_environment_ref() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = req.read()
        return httpx.Response(201, json={"name": "d1", "status": "pending"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            [
                "deploy",
                "--agent",
                "a1",
                "--environment",
                "default/env1",
                "--no-wait",
                "--base-url",
                "http://test",
            ],
        )

    assert result.exit_code == 0, result.stderr
    import json as _j

    body = _j.loads(captured["body"])
    assert body["agent"] == "a1"
    assert body["environment"] == "default/env1"


def test_deploy_forwards_image_entrypoint_mode() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = req.read()
        return httpx.Response(201, json={"name": "d1", "status": "pending"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            [
                "deploy",
                "--agent",
                "a1",
                "--mode",
                "docker",
                "--image",
                "hand-built-agent:latest",
                "--use-image-entrypoint",
                "--no-wait",
                "--base-url",
                "http://test",
            ],
        )

    assert result.exit_code == 0, result.stderr
    import json as _j

    body = _j.loads(captured["body"])
    assert body["agent"] == "a1"
    assert body["deployment_mode"] == "docker"
    assert body["image"] == "hand-built-agent:latest"
    assert body["use_image_entrypoint"] is True


def test_deploy_rejects_image_entrypoint_for_subprocess() -> None:
    called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(201, json={"name": "d1", "status": "pending"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["deploy", "--agent", "a1", "--use-image-entrypoint", "--no-wait", "--base-url", "http://test"],
        )

    assert result.exit_code == 2
    assert "--use-image-entrypoint requires --mode docker or k8s." in result.stderr
    assert not called


def test_deploy_rejects_empty_environment() -> None:
    called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(201, json={"name": "d1", "status": "pending"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["deploy", "--agent", "a1", "--environment", "   ", "--no-wait", "--base-url", "http://test"],
        )

    assert result.exit_code == 2
    assert "--environment must not be empty." in result.stderr
    assert not called  # no API call made


def test_environment_spec_create_from_file(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"env": {"LOG_LEVEL": "debug"}, "secrets": {"TOK": "default/tok"}}')
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        captured["body"] = req.read()
        return httpx.Response(201, json={"name": "ben"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["environment-specs", "create", "ben", "--spec-file", str(spec), "--base-url", "http://test"],
        )

    assert result.exit_code == 0, result.stderr
    import json as _j

    assert captured["url"].endswith("/environment-specs")
    body = _j.loads(captured["body"])
    assert body["name"] == "ben"
    assert body["env"] == {"LOG_LEVEL": "debug"}
    assert body["secrets"] == {"TOK": "default/tok"}


def test_environment_spec_create_from_inline_json() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = req.read()
        return httpx.Response(201, json={"name": "prod"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["environment-specs", "create", "prod", "--spec", '{"provider": "local"}', "--base-url", "http://test"],
        )

    assert result.exit_code == 0, result.stderr
    import json as _j

    body = _j.loads(captured["body"])
    assert body == {"provider": "local", "name": "prod"}


def test_environment_spec_create_reports_validation_errors_without_traceback() -> None:
    called = False

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(201, json={"name": "bad"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["environment-specs", "create", "bad", "--spec", '{"env":{"PORT":1}}', "--base-url", "http://test"],
        )

    assert result.exit_code == 2
    assert "Error: POST agent API validation failed:" in result.stderr
    assert "env.PORT" in result.stderr
    assert "Traceback" not in result.stderr
    assert not called


def test_environment_spec_create_rejects_both_sources(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    app = AgentsCLI().get_cli()
    result = CliRunner().invoke(
        app,
        ["environment-specs", "create", "x", "--spec-file", str(spec), "--spec", "{}", "--base-url", "http://test"],
    )
    assert result.exit_code == 2
    assert "only one of --spec-file or --spec" in result.stderr


def test_environment_create_with_ref_flags() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        captured["body"] = req.read()
        return httpx.Response(201, json={"name": "env1"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            [
                "environments",
                "create",
                "env1",
                "--environment-spec",
                "default/ben",
                "--compute-spec",
                "default/big",
                "--base-url",
                "http://test",
            ],
        )

    assert result.exit_code == 0, result.stderr
    import json as _j

    assert captured["url"].endswith("/environments")
    body = _j.loads(captured["body"])
    assert body["name"] == "env1"
    assert body["environment_spec"] == "default/ben"
    assert body["compute_spec"] == "default/big"


def test_compute_spec_create_and_list() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.setdefault("methods", []).append(req.method)
        if req.method == "POST":
            captured["body"] = req.read()
            return httpx.Response(201, json={"name": "big"})
        return httpx.Response(
            200,
            json={
                "data": [{"name": "big", "workspace": "default"}],
                "pagination": {
                    "page": 1,
                    "page_size": 1,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        create = CliRunner().invoke(
            app,
            [
                "compute-specs",
                "create",
                "big",
                "--spec",
                '{"resources": {"limits": {"cpu": "2"}}}',
                "--base-url",
                "http://test",
            ],
        )
        listing = CliRunner().invoke(app, ["compute-specs", "list", "--format", "json", "--base-url", "http://test"])

    assert create.exit_code == 0, create.stderr
    assert listing.exit_code == 0, listing.stderr
    import json as _j

    assert _j.loads(captured["body"])["name"] == "big"


def test_environment_delete_confirmation() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "DELETE"
        return httpx.Response(204)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["environments", "delete", "env1", "--yes", "--base-url", "http://test"])

    assert result.exit_code == 0, result.stderr
    assert "Environment 'env1' deleted." in result.stdout


def test_environment_spec_create_routes_through_typed_client() -> None:
    """The CLI builds the request through the source-owned typed client."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["method"] = req.method
        captured["url"] = str(req.url)
        captured["body"] = req.read()
        return httpx.Response(201, request=req, json={"name": "ben"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            [
                "environment-specs",
                "create",
                "ben",
                "--spec",
                '{"env": {"LOG_LEVEL": "debug"}}',
                "--workspace",
                "team-a",
                "--base-url",
                "http://test",
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/apis/agents/v2/workspaces/team-a/environment-specs")
    assert json.loads(captured["body"]) == {"name": "ben", "env": {"LOG_LEVEL": "debug"}}


def test_compute_spec_create_via_typed_client_translates_http_error() -> None:
    """A server error raised inside the typed client uses the CLI's rich message."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            [
                "compute-specs",
                "create",
                "big",
                "--spec",
                '{"resources": {"limits": {"cpu": "2"}}}',
                "--base-url",
                "http://test",
            ],
        )

    assert result.exit_code == 1
    assert "POST agent API" in result.stderr


def test_environment_spec_create_via_typed_client_sends_auth_header() -> None:
    """The CLI threads its resolved auth header through the typed client."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["auth"] = req.headers.get("authorization")
        return httpx.Response(201, json={"name": "ben"})

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch(
            "nemo_agents_plugin.cli._resolve_context_headers",
            return_value={"Authorization": "Bearer tok-xyz"},
        ),
    ):
        result = CliRunner().invoke(
            app,
            [
                "environment-specs",
                "create",
                "ben",
                "--spec",
                '{"provider": "local"}',
                "--base-url",
                "http://test",
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert captured["auth"] == "Bearer tok-xyz"
