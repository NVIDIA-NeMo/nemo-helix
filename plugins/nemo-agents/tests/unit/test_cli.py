# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar
from unittest.mock import patch

import click
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
from nemo_helix_plugin.client.client import NemoClient
from typer.testing import CliRunner

ClientT = TypeVar("ClientT", bound=NemoClient)


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
    uploaded: set[str] = set()

    def _capture_upload(local_dir: Path, *, fileset: str, workspace: str, sdk: Any) -> None:
        assert fileset == "fabric-agent-ethos"
        assert workspace == "default"
        uploaded.update(path.relative_to(local_dir).as_posix() for path in local_dir.rglob("*") if path.is_file())

    with (
        _shared_cli_context(),
        patch("nemo_agents_plugin.cli.FilesClient.from_client", return_value=files),
        patch("nemo_agents_plugin.jobs.fileset_io.upload_to_fileset", _capture_upload),
    ):
        _upload_ethos_fileset(
            agent_name="fabric-agent",
            workspace="default",
            agent_root=agent_root,
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

    return patch("nemo_helix_plugin.client.client.Client", _Client)


class _CLIState:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        workspace: str | None = "default",
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.workspace = workspace
        self.default_headers = default_headers
        self.timeout: float | httpx.Timeout | None = None

    def typed_client(self, client_cls: type[ClientT], timeout: float | httpx.Timeout | None = None) -> ClientT:
        self.timeout = timeout
        return client_cls(
            base_url=self.base_url,
            workspace=self.workspace,
            default_headers=self.default_headers,
            timeout=timeout,
        )

    def get_base_url(self, default: str | None = None) -> str | None:
        return self.base_url

    def get_workspace(self) -> str | None:
        return self.workspace

    def get_output_format(self, override: str | None = None, *, apply_non_tty_default: bool = True) -> str:
        return override or "json"

    def get_no_truncate(self, override: bool | None = None) -> bool:
        return bool(override)

    def get_timestamp_format(self, override: str | None = None) -> str:
        return override or "iso8601"


def _shared_cli_context(base_url: str = "http://test") -> click.Context:
    return click.Context(click.Command("nemo"), obj=_CLIState(base_url=base_url))


@pytest.fixture(autouse=True)
def _default_cli_state_for_direct_invocations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct Typer invocations in these tests still run through a CLI state."""
    original_invoke = CliRunner.invoke

    def invoke(self: CliRunner, app: Any, args: Any | None = None, *pargs: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("obj", _CLIState())
        return original_invoke(self, app, args, *pargs, **kwargs)

    monkeypatch.setattr(CliRunner, "invoke", invoke)


def test_no_args_prints_help_successfully() -> None:
    app = AgentsCLI().get_cli()
    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "Agent lifecycle management" in result.stdout


def test_agents_client_uses_sdk_constructor_timeout_defaults() -> None:
    state = _CLIState()
    ctx = click.Context(click.Command("nemo"), obj=state)
    with ctx:
        client = _agents_client("http://test", "default")

    assert state.timeout is None
    assert client._timeout is None
    assert client._client.timeout == httpx.Timeout(60)


def test_run_starts_nat_server_for_nat_config(tmp_path: Path) -> None:
    config = tmp_path / "workflow.yaml"
    config.write_text("workflow:\n  _type: chat_completion\n")

    app = AgentsCLI().get_cli()
    with patch("nemo_agents_plugin.cli.run_command") as run:
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
    with patch("nemo_agents_plugin.cli.run_command") as run:
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
    with patch("nemo_agents_plugin.cli.run_command") as run:
        result = CliRunner().invoke(app, ["run", "--agent-config", str(config)])

    assert result.exit_code == 1
    assert f"agent config {config} root must be a YAML mapping" in result.stderr
    run.assert_not_called()


def test_list_404_prints_request_context_and_hint() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["list"])

    assert result.exit_code == 1
    assert "Error: GET agent API failed: HTTP 404 Not Found" in result.stderr
    # The test CLI state uses localhost; long URLs are elided mid-way.
    assert "Request: GET http://localhost:8080/" in result.stderr
    assert "Target: agents API route /apis/agents/v2/workspaces/default/agents" in result.stderr
    assert "route may not be deployed" in result.stderr


def test_agent_jobs_do_not_register_legacy_run_submit_verbs() -> None:
    import click
    from nemo_agents_plugin.jobs.evaluate_agent import EvaluateAgentJob
    from nemo_agents_plugin.jobs.execute import ExecuteAgentJob
    from nemo_agents_plugin.jobs.package_agent import PackageAgentJob
    from nemo_helix_plugin.commands import add_job_commands
    from nemo_helix_plugin.job import NemoJob
    from typer.main import get_command

    jobs: dict[str, type[NemoJob]] = {
        "agents.evaluate": EvaluateAgentJob,
        "agents.execute": ExecuteAgentJob,
        "agents.package-agent": PackageAgentJob,
    }

    agents_cli = AgentsCLI()
    app = agents_cli.get_cli()
    add_job_commands(app, jobs, cli=agents_cli)
    command = get_command(app)

    assert isinstance(command, click.Group)
    for job_name in {job_cls.name for job_cls in jobs.values()}:
        job_command = command.commands[job_name]
        # Non-legacy jobs submit from the group callback and keep only `explain` as a subcommand.
        assert isinstance(job_command, click.Group)
        assert set(job_command.commands) == {"explain"}
        for legacy_verb in ("run", "submit"):
            legacy_result = CliRunner().invoke(app, [job_name, legacy_verb])
            assert legacy_result.exit_code == 2
            assert f"No such command '{legacy_verb}'" in legacy_result.output


def test_agents_cli_has_no_create_command() -> None:
    result = CliRunner().invoke(AgentsCLI().get_cli(), ["create"])

    assert result.exit_code == 2
    assert "No such command 'create'" in result.output


def test_register_sends_labels() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert json.loads(req.content) == {
            "name": "calc",
            "labels": {"helix-agents/managed-by": "insights", "team": "search"},
        }
        return httpx.Response(201, json={"name": "calc"})

    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            AgentsCLI().get_cli(),
            ["register", "--name", "calc", "--label", "helix-agents/managed-by=insights", "-l", "team=search"],
        )

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("label", "error"),
    [("team", "must be KEY=VALUE"), ("helix.agents.managed-by=x", "invalid --label")],
)
def test_register_rejects_bad_label_before_post(label: str, error: str) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not POST with an invalid label")

    with _install_mock_transport(handler):
        result = CliRunner().invoke(AgentsCLI().get_cli(), ["register", "--name", "calc", "--label", label])

    assert result.exit_code == 2
    assert error in result.stderr


def test_list_filters_by_label() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert json.loads(req.url.params["filter"]) == {"labels.helix-compass/target-agent": "default/bot"}
        return httpx.Response(
            200,
            json={
                "data": [],
                "pagination": {
                    "page": 1,
                    "page_size": 20,
                    "current_page_size": 0,
                    "total_pages": 1,
                    "total_results": 0,
                },
            },
        )

    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            AgentsCLI().get_cli(), ["list", "--label", "helix-compass/target-agent=default/bot", "-f", "json"]
        )

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("config_text", [None, "{}", "name: calc\ndescription: Calculator\n"])
def test_register_name_only(tmp_path: Path, config_text: str | None) -> None:
    args = ["register", "--name", "calc", "--workspace", "team-a"]
    expected: dict[str, Any] = {"name": "calc"}
    if config_text is not None:
        config = tmp_path / "agent.yaml"
        config.write_text(config_text)
        args += ["--agent-config", str(config)]
        expected.update(config={}, config_format="nemo-agents-spec-v1")

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/apis/agents/v2/workspaces/team-a/agents"
        assert json.loads(req.content) == expected
        return httpx.Response(201, json={"name": "calc"})

    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.cli._upload_ethos_fileset") as upload,
        patch("nemo_agents_plugin.utils.inject_default_model") as resolve_model,
    ):
        result = CliRunner().invoke(AgentsCLI().get_cli(), args)

    assert result.exit_code == 0, result.output
    upload.assert_not_called()
    resolve_model.assert_not_called()


@pytest.mark.parametrize("config_format", [None, "nemo-agents-spec-v1", "nat-workflow-v1"])
def test_register_references_without_runtime_config(tmp_path: Path, config_format: str | None) -> None:
    config = tmp_path / "agent.yaml"
    text = (
        "name: calc\nhelix_registration:\n  reference:\n"
        "    fileset: team-a/source\n    container_image: nvcr.io/my-org/agent:1.0\n"
    )
    if config_format is not None:
        text += f"config_format: {config_format}\n"
    config.write_text(text)

    def handler(req: httpx.Request) -> httpx.Response:
        assert json.loads(req.content) == {
            "name": "calc",
            "config": {},
            "config_format": config_format or "nemo-agents-spec-v1",
            "reference": {"fileset": "team-a/source", "container_image": "nvcr.io/my-org/agent:1.0"},
        }
        return httpx.Response(201, json={"name": "calc"})

    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.cli._upload_ethos_fileset") as upload,
        patch("nemo_agents_plugin.utils.inject_default_model") as resolve_model,
    ):
        result = CliRunner().invoke(
            AgentsCLI().get_cli(), ["register", "--name", "calc", "--agent-config", str(config)]
        )

    assert result.exit_code == 0, result.output
    upload.assert_not_called()
    resolve_model.assert_not_called()


@pytest.mark.parametrize("placeholder", ["${NEMO_DEFAULT_MODEL}", "$NEMO_DEFAULT_MODEL"])
def test_register_resolves_default_model_placeholder(tmp_path, placeholder: str) -> None:
    """`nemo agents register` resolves NEMO_DEFAULT_MODEL before POST.

    Regression test: the agents service has no user context at
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
        result = CliRunner().invoke(app, ["register", "--name", "calc", "--agent-config", str(config)])

    assert result.exit_code == 0, result.stderr
    sent = _json.loads(captured["body"])
    assert sent["config"]["llms"]["llm"]["model_name"] == "nvidia-nemotron-3-super-v3"
    assert sent["config_format"] == "nat-workflow-v1"


@pytest.mark.parametrize(
    ("config_text", "error"),
    [
        ("llms:\n  - [unterminated\n", "invalid YAML in agent config"),
        ("", "root must be a YAML mapping"),
        ("null", "root must be a YAML mapping"),
        ("[]", "root must be a YAML mapping"),
        ("false", "root must be a YAML mapping"),
        ("config_format: []", "Input should be a valid string"),
        ("helix_registration:\n  reference: []", "Input should be a valid dictionary"),
        ("helix_registration:\n  default_inptu: hi", "Extra inputs are not permitted"),
    ],
)
def test_register_reports_invalid_agent_config_without_traceback(tmp_path, config_text: str, error: str) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text(config_text, encoding="utf-8")

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid config should fail before any API request")

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["register", "--name", "calc", "--agent-config", str(config)])

    assert result.exit_code == 1
    assert error in result.stderr
    assert "Traceback" not in result.stderr
    assert ".venv" not in result.stderr


@pytest.mark.parametrize("with_registration", [False, True])
def test_register_validates_platform_agent_config_before_post(tmp_path, with_registration: bool) -> None:
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
    if with_registration:
        config.write_text(
            config.read_text()
            + "\n".join(
                [
                    "helix_registration:",
                    "  reference:",
                    "    fileset: default/agent-source",
                    "  context_schema:",
                    "    type: object",
                    "    additionalProperties: false",
                    "  default_input: Analyze recent runs.",
                    "  post_run_actions: [insights.analysis, agents.noop]",
                    "",
                ]
            )
        )
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
            ["register", "--name", "fabric-agent", "--agent-config", str(config)],
        )

    assert result.exit_code == 0, result.stderr
    sent = _json.loads(captured["body"])
    assert captured["base_dir"] == tmp_path
    assert captured["validated_config"]["config_format"] == "nemo-agents-spec-v1"
    assert captured["validated_config"]["models"]["default"]["model"] == "user-default-model"
    # The block is lifted out before validation and sent as the agent's own fields.
    assert "helix_registration" not in captured["validated_config"]
    registration_fields = ("reference", "context_schema", "default_input", "post_run_actions")
    if with_registration:
        assert {key: sent[key] for key in registration_fields} == {
            "reference": {"fileset": "default/agent-source"},
            "context_schema": {"type": "object", "additionalProperties": False},
            "default_input": "Analyze recent runs.",
            "post_run_actions": ["agents.noop", "insights.analysis"],
        }
    else:
        assert not set(registration_fields) & sent.keys()
    assert sent["config"] == normalized_config
    assert sent["config_format"] == "nemo-agents-spec-v1"
    mock_upload.assert_called_once_with(
        agent_name="fabric-agent",
        workspace="default",
        agent_root=tmp_path,
    )


@pytest.mark.parametrize(
    "config_path", ["absolute", "agent.yaml", "./agent.yaml", "../fabric-agent-spec/agent.yaml", "../agent.yaml"]
)
def test_register_fabric_uploads_ethos_fileset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: str
) -> None:
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
    (package / "skills").mkdir()
    (package / "skills" / "SKILL.md").write_text("# Bundle skill\n")
    if config_path == "../agent.yaml":
        workdir = package / "work"
        workdir.mkdir()
    else:
        workdir = tmp_path if config_path == "absolute" else package
    monkeypatch.chdir(workdir)
    config_argument = str(config) if config_path == "absolute" else config_path
    normalized_config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes", "settings": {}}},
        "environment": {"provider": "local"},
    }

    async def _validate_platform_agent_config(config_dict: dict[str, Any], *, base_dir: Path):
        assert (base_dir / "skills" / "SKILL.md").read_text() == "# Bundle skill\n"
        return type("ValidationResult", (), {"agent_config": _ValidatedAgentConfig(normalized_config)})()

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert json.loads(req.content)["config"] == normalized_config
        return httpx.Response(200, json={"name": "fabric-agent"})

    uploaded: dict[str, Any] = {}

    def fake_upload(local_dir: Path, *, fileset: str, workspace: str, sdk: Any) -> None:
        uploaded["files"] = {
            path.relative_to(local_dir).as_posix(): path.read_text() for path in local_dir.rglob("*") if path.is_file()
        }
        uploaded["fileset"] = fileset
        uploaded["workspace"] = workspace
        uploaded["sdk_base_url"] = sdk.base_url

    files = _FakeEthosFiles([AGENT_SPEC_FILENAME], delete_error=FileNotFoundError())

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.fabric.validation.validate_platform_agent_config", _validate_platform_agent_config),
        patch("nemo_agents_plugin.jobs.fileset_io.upload_to_fileset", fake_upload),
        patch("nemo_agents_plugin.cli.FilesClient.from_client", return_value=files),
    ):
        result = CliRunner().invoke(
            app,
            ["register", "--name", "fabric-agent", "--agent-config", config_argument],
            obj=_CLIState(base_url="http://test"),
        )

    assert result.exit_code == 0, result.stderr
    if config_path == "absolute":
        warning = _spec_package_warning("fabric-agent", config)
        assert result.stderr.splitlines()[-len(warning) :] == list(warning)
    assert uploaded["files"] == {"agent.yaml": config.read_text(), "skills/SKILL.md": "# Bundle skill\n"}
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
        "Provide an existing ETHOS.md, then delete the acme-bot-spec package.",
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

    def _capture_upload(local_dir: Path, **_: Any) -> None:
        remote.update(
            (path.relative_to(local_dir).as_posix(), path.read_bytes())
            for path in local_dir.rglob("*")
            if path.is_file()
        )

    with (
        _shared_cli_context(),
        patch("nemo_agents_plugin.cli.FilesClient.from_client", return_value=files),
        patch("nemo_agents_plugin.jobs.fileset_io.upload_to_fileset", _capture_upload),
    ):
        _upload_ethos_fileset(
            agent_name="fabric-agent",
            workspace="default",
            agent_root=agent_root,
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


def test_register_fabric_rolls_back_agent_when_fileset_upload_fails(tmp_path) -> None:
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
    ):
        result = CliRunner().invoke(
            app,
            ["register", "--name", "fabric-agent", "--agent-config", str(config)],
        )

    assert result.exit_code == 1
    assert "failed to upload Ethos fileset" in result.stderr
    # Rollback removes the agent entity only; the Ethos fileset is durable and may
    # already hold an ETHOS.md written before this agent existed.
    assert methods == ["POST", "DELETE"]


def test_register_fabric_reports_rollback_failure(tmp_path) -> None:
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
            ["register", "--name", "fabric-agent", "--agent-config", str(config)],
        )

    assert result.exit_code == 1
    assert "failed to roll back agent 'fabric-agent'" in result.stderr
    assert "nemo agents delete fabric-agent" in result.stderr


@pytest.mark.parametrize("with_reference", [False, True])
def test_register_nat_does_not_upload_ethos_fileset(tmp_path, with_reference: bool) -> None:
    config = tmp_path / "agent.yml"
    config.write_text("llms:\n  llm:\n    _type: openai\n    model_name: nvidia-nemotron-3-super-v3\n")
    if with_reference:
        config.write_text(
            config.read_text() + "helix_registration:\n  reference:\n    container_image: nvcr.io/my-org/agent:1.0\n"
        )

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        body = json.loads(req.content)
        assert body["config_format"] == "nat-workflow-v1"
        assert "reference" not in body["config"]
        if with_reference:
            assert body["reference"] == {"container_image": "nvcr.io/my-org/agent:1.0"}
        else:
            assert "reference" not in body
        return httpx.Response(200, json={"name": "calc"})

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.cli._upload_ethos_fileset") as mock_upload,
        patch("nemo_agents_plugin.utils.get_default_model", return_value="nvidia-nemotron-3-super-v3"),
    ):
        result = CliRunner().invoke(app, ["register", "--name", "calc", "--agent-config", str(config)])

    assert result.exit_code == 0, result.stderr
    mock_upload.assert_not_called()


def test_register_rejects_unsupported_config_format(tmp_path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text("config_format: custom-v2\nname: custom-agent\n")

    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not POST unsupported config_format")

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["register", "--name", "custom-agent", "--agent-config", str(config)],
        )

    assert result.exit_code == 1
    assert "unsupported config_format 'custom-v2'" in result.stderr


@pytest.mark.parametrize("placeholder", ["${NEMO_DEFAULT_MODEL}", "$NEMO_DEFAULT_MODEL"])
def test_register_aborts_when_default_model_missing(tmp_path, placeholder: str) -> None:
    """If no default model is selected, refuse to POST a config with an unresolved
    NEMO_DEFAULT_MODEL placeholder (braced or bare). Regression test."""
    config = tmp_path / "agent.yml"
    config.write_text(f"llms:\n  llm:\n    _type: openai\n    model_name: {placeholder}\n")

    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not POST when placeholder is unresolved")

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(handler),
        patch("nemo_agents_plugin.utils.get_default_model", return_value=None),
    ):
        result = CliRunner().invoke(app, ["register", "--name", "calc", "--agent-config", str(config)])

    assert result.exit_code == 1
    assert "${NEMO_DEFAULT_MODEL}" in result.stderr
    assert "nemo setup" in result.stderr


def test_invoke_with_custom_timeout() -> None:
    """--timeout is threaded through to the HTTP request."""
    captured_read_timeout: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_read_timeout.append(req.extensions["timeout"]["read"])
        return httpx.Response(200, json={"model": "test", "choices": []})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["invoke", "--agent", "calc", "--input", "hi", "--timeout", "42"],
        )

    assert result.exit_code == 0, result.stderr
    assert captured_read_timeout == [42.0]


def test_invoke_timeout_error_message() -> None:
    """Timeout errors print actionable guidance mentioning --timeout."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=req)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["invoke", "--agent", "calc", "--input", "hi", "--timeout", "5"])

    assert result.exit_code == 1
    assert "timed out" in result.stderr.lower()
    assert "--timeout" in result.stderr


@pytest.mark.parametrize("with_reference", [False, True])
def test_local_invoke_runs_fabric_config_once(tmp_path: Path, with_reference: bool) -> None:
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
    if with_reference:
        config.write_text(
            config.read_text()
            + "helix_registration:\n  reference:\n    fileset: default/source\n    container_image: agent:1\n"
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
    assert "reference" not in captured["agent_config"].model_dump()
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

    The spinner must render only on stderr so consumers can pipe
    stdout to `jq`. CliRunner's non-TTY stderr auto-disables the spinner, so
    here we just verify the response JSON is intact on stdout.
    """
    import json as _json

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["invoke", "--agent", "calc", "--input", "ping"])

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
            ["invoke", "--agent", "calc", "--input", "ping", "--no-progress"],
        )

    assert result.exit_code == 0, result.stderr
    assert _json.loads(result.stdout)["choices"][0]["message"]["content"] == "hi"


def test_list_connection_error_prints_request_context_and_hint() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=req)

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["list"])

    assert result.exit_code == 1
    assert "Error: GET agent API failed: connection refused" in result.stderr
    # The test CLI state uses localhost; long URLs are elided mid-way.
    assert "Request: GET http://localhost:8080/" in result.stderr
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
                "--mode",
                "subprocess",
                "--environment",
                "default/env1",
                "--no-wait",
            ],
        )

    assert result.exit_code == 0, result.stderr
    import json as _j

    body = _j.loads(captured["body"])
    assert body["agent"] == "a1"
    assert body["environment"] == "default/env1"


def _deployment_modes(subprocess_enabled: bool, *, default_image: bool) -> dict[str, Any]:
    return {
        "data": [
            {"mode": "subprocess", "enabled": subprocess_enabled, "requires_image": False},
            {"mode": "docker", "enabled": False, "requires_image": not default_image},
            {"mode": "k8s", "enabled": True, "requires_image": not default_image},
            {"mode": "openshell", "enabled": False, "requires_image": not default_image},
        ]
    }


@pytest.mark.parametrize(
    ("subprocess_enabled", "expected_mode"),
    [(True, "subprocess"), (False, "k8s")],
)
def test_deploy_without_mode_uses_the_platform_default(subprocess_enabled: bool, expected_mode: str) -> None:
    posted: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/deployment-modes"):
            return httpx.Response(200, json=_deployment_modes(subprocess_enabled, default_image=True))
        posted.append(json.loads(req.content))
        return httpx.Response(201, json={"name": "d1", "status": "pending"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["deploy", "--agent", "a1", "--no-wait"])

    assert result.exit_code == 0, result.stderr
    assert [body["deployment_mode"] for body in posted] == [expected_mode]


def test_deploy_without_mode_fails_when_every_usable_mode_needs_an_image() -> None:
    posted: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/deployment-modes"):
            return httpx.Response(200, json=_deployment_modes(False, default_image=False))
        posted.append(req)
        return httpx.Response(201, json={"name": "d1", "status": "pending"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(app, ["deploy", "--agent", "a1", "--no-wait"])

    assert result.exit_code == 1
    assert "Pass --mode with --image" in result.stderr
    assert posted == []


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
            ["deploy", "--agent", "a1", "--use-image-entrypoint", "--no-wait"],
        )

    assert result.exit_code == 2
    assert "--use-image-entrypoint requires --mode docker or k8s." in result.stderr
    assert not called


def test_deploy_rejects_image_entrypoint_for_openshell() -> None:
    called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
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
                "openshell",
                "--use-image-entrypoint",
                "--no-wait",
            ],
        )

    assert result.exit_code == 2
    assert "--mode openshell needs the platform-injected serve command" in result.stderr
    assert not called


def test_deploy_accepts_openshell_mode() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(201, json={"name": "d1", "status": "pending"})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["deploy", "--agent", "a1", "--mode", "openshell", "--no-wait"],
        )

    assert result.exit_code == 0, result.output
    assert captured["body"]["deployment_mode"] == "openshell"


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
            ["deploy", "--agent", "a1", "--environment", "   ", "--no-wait"],
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
            ["environment-specs", "create", "ben", "--spec-file", str(spec)],
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
            ["environment-specs", "create", "prod", "--spec", '{"provider": "local"}'],
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
            ["environment-specs", "create", "bad", "--spec", '{"env":{"PORT":1}}'],
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
        ["environment-specs", "create", "x", "--spec-file", str(spec), "--spec", "{}"],
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
            ],
        )
        listing = CliRunner().invoke(app, ["compute-specs", "list", "--output-format", "json"])

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
        result = CliRunner().invoke(app, ["environments", "delete", "env1", "--yes"])

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
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            [
                "environment-specs",
                "create",
                "ben",
                "--spec",
                '{"provider": "local"}',
            ],
            obj=_CLIState(default_headers={"Authorization": "Bearer tok-xyz"}),
        )

    assert result.exit_code == 0, result.stderr
    assert captured["auth"] == "Bearer tok-xyz"


_GIT_URL = "git@gitlab.example.com:org/repo.git"
_RSA_LINE = "gitlab.example.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQRsaKeyBlob"
_HOST_KEYS = [{"key_type": "ssh-rsa", "fingerprint": "SHA256:rsafingerprint", "known_hosts_line": _RSA_LINE}]
_GIT_AGENT_FILES = {
    "agent.yaml": "config_format: nemo-agents-spec-v1\nname: git-agent\ndefault_harness: hermes\n"
    "harnesses:\n  hermes:\n    kind: hermes\n",
    "skills/search.md": "# Search\n",
}
_FILES_API = "/apis/files/v2/workspaces/default"
_GIT_FILESET = {
    "id": "fileset-1",
    "name": "git-agent-ethos",
    "workspace": "default",
    "description": "",
    "purpose": "generic",
    "storage": {"type": "local", "path": "/data/git-agent-ethos"},
    "metadata": {},
    "custom_fields": {},
    "project": "",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}


class _GitPlatform:
    """Answers the files and agents calls a git registration makes, recording each."""

    def __init__(self, *, paths: Sequence[str] = ("agent.yaml",), register_status: int = 200) -> None:
        self.paths = list(paths)
        self.register_status = register_status
        self.calls: list[tuple[str, str]] = []
        self.find_bodies: list[dict[str, Any]] = []
        self.fileset_body: dict[str, Any] | None = None
        self.agent_body: dict[str, Any] | None = None

    def __call__(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        self.calls.append((req.method, path))
        if path == f"{_FILES_API}/git-repositories/find-files":
            body = json.loads(req.content)
            self.find_bodies.append(body)
            if not body["storage"]["known_hosts"]:
                return httpx.Response(200, json={"host_keys": _HOST_KEYS})
            return httpx.Response(200, json={"revision": "a" * 40, "paths": self.paths})
        if req.method == "POST" and path == f"{_FILES_API}/filesets":
            self.fileset_body = json.loads(req.content)
            return httpx.Response(201, json=_GIT_FILESET)
        if req.method == "GET" and path == f"{_FILES_API}/filesets/git-agent-ethos/files":
            files = [
                {"file_ref": f"default/git-agent-ethos#{name}", "file_url": "", "path": name, "size": len(content)}
                for name, content in _GIT_AGENT_FILES.items()
            ]
            return httpx.Response(200, json={"data": files})
        if req.method == "GET" and path.startswith(f"{_FILES_API}/filesets/git-agent-ethos/-/"):
            name = path.removeprefix(f"{_FILES_API}/filesets/git-agent-ethos/-/")
            return httpx.Response(200, content=_GIT_AGENT_FILES[name].encode())
        if req.method == "DELETE" and path == f"{_FILES_API}/filesets/git-agent-ethos":
            return httpx.Response(200, json=_GIT_FILESET)
        if req.method == "POST" and path.endswith("/agents"):
            self.agent_body = json.loads(req.content)
            if self.register_status != 200:
                return httpx.Response(self.register_status, json={"detail": "Agent 'git-agent' already exists"})
            return httpx.Response(200, json={"name": "git-agent"})
        raise AssertionError(f"unexpected {req.method} {path}")


def _register_from_git(platform: _GitPlatform, known_hosts: Path, *extra: str) -> tuple[Any, list[set[str]]]:
    staged: list[set[str]] = []

    async def _validate_platform_agent_config(config_dict: dict[str, Any], *, base_dir: Path):
        staged.append({path.relative_to(base_dir).as_posix() for path in Path(base_dir).rglob("*") if path.is_file()})
        return type("ValidationResult", (), {"agent_config": _ValidatedAgentConfig(config_dict)})()

    app = AgentsCLI().get_cli()
    with (
        _install_mock_transport(platform),
        patch("nemo_agents_plugin.fabric.validation.validate_platform_agent_config", _validate_platform_agent_config),
        patch(
            "nemo_agents_plugin.cli._upload_ethos_fileset", side_effect=AssertionError("git agents are not uploaded")
        ),
    ):
        result = CliRunner().invoke(
            app,
            [
                "register",
                "--name",
                "git-agent",
                "--from-git",
                _GIT_URL,
                "--ssh-key-secret",
                "deploy-key",
                "--known-hosts-file",
                str(known_hosts),
                *extra,
            ],
            obj=_CLIState(base_url="http://test"),
        )
    return result, staged


def _known_hosts_file(tmp_path: Path) -> Path:
    path = tmp_path / "known_hosts"
    path.write_text(_RSA_LINE + "\n")
    return path


def test_register_from_git_creates_a_tracking_fileset_and_validates_its_files(tmp_path: Path) -> None:
    platform = _GitPlatform()

    result, staged = _register_from_git(platform, _known_hosts_file(tmp_path), "--revision", "main")

    assert result.exit_code == 0, result.stderr
    assert [body["storage"]["known_hosts"] for body in platform.find_bodies] == ["", _RSA_LINE]
    assert platform.fileset_body is not None
    assert platform.fileset_body["name"] == "git-agent-ethos"
    assert platform.fileset_body["storage"]["known_hosts"] == _RSA_LINE
    assert platform.fileset_body["storage"]["revision"] == "main"
    assert staged == [set(_GIT_AGENT_FILES)]
    assert platform.agent_body is not None
    assert platform.agent_body["name"] == "git-agent"
    assert ("DELETE", f"{_FILES_API}/filesets/git-agent-ethos") not in platform.calls


def test_register_from_git_lists_agent_directories_without_a_terminal(tmp_path: Path) -> None:
    platform = _GitPlatform(paths=["agents/a/agent.yaml", "agents/b/agent.yaml"])

    result, _ = _register_from_git(platform, _known_hosts_file(tmp_path), "--path", "src")

    assert result.exit_code == 2
    assert "src/agents/a\n" in result.stderr
    assert "src/agents/b" in result.stderr
    assert platform.fileset_body is None


def test_register_from_git_removes_the_fileset_when_registration_fails(tmp_path: Path) -> None:
    platform = _GitPlatform(register_status=409)

    result, _ = _register_from_git(platform, _known_hosts_file(tmp_path))

    assert result.exit_code == 1
    assert platform.calls[-1] == ("DELETE", f"{_FILES_API}/filesets/git-agent-ethos")


def test_register_from_git_stops_at_an_unknown_host_without_a_terminal(tmp_path: Path) -> None:
    platform = _GitPlatform()

    result, _ = _register_from_git(platform, tmp_path / "missing")

    assert result.exit_code == 2
    assert "--host-key" in result.stderr
    assert len(platform.calls) == 1


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--from-git", _GIT_URL], "--from-git needs --ssh-key-secret"),
        (["--from-git", _GIT_URL, "--ssh-key-secret", "k", "--agent-config", "{config}"], "cannot be combined"),
        (["--revision", "main"], "need --from-git"),
    ],
)
def test_register_from_git_option_errors(tmp_path: Path, args: list[str], error: str) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text("name: git-agent\n")
    app = AgentsCLI().get_cli()

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected {req.method} {req.url.path}")

    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app, ["register", "--name", "git-agent", *(arg.format(config=config) for arg in args)]
        )

    assert result.exit_code == 2
    assert error in result.stderr
