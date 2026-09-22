# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands resolve the workspace from a real config file.

The unit coverage in ``packages/nemo_platform_plugin/tests/test_commands.py``
injects a hand-written stand-in for the state object, so it pins the resolver
but never exercises the link that actually broke in production::

    config.yaml -> Config.load -> Context.workspace -> CLIContext.get_workspace()

That link is worth an integration test because
:meth:`CLIContext.get_workspace` swallows *every* exception and returns
``None``, which the resolver then turns into ``"default"``. Any future
breakage in config loading therefore degrades silently back to the original
bug -- wrong workspace, no error, no failing unit test. These tests drive the
commands with a real :class:`CLIContext` over a real config file on disk so
that regression surfaces.

Both command surfaces are covered: the generated verbs and the hand-written
plugin commands. They are built differently -- programmatic signatures versus
an ``Annotated`` option alias -- so exercising only one would leave the
other's real-config path untested.
"""

import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
import typer
import yaml
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_plugin.commands import add_function_commands, add_job_commands
from nemo_platform_plugin.function import NemoFunction
from nemo_platform_plugin.function_context import FunctionContext
from nemo_platform_plugin.job import NemoJob
from pydantic import BaseModel
from typer.testing import CliRunner

CONTEXT_WORKSPACE = "my-team-ws"
OTHER_WORKSPACE = "other-team-ws"

runner = CliRunner()


class _Spec(BaseModel):
    name: str = "Ada"


class _EchoFunction(NemoFunction[_Spec]):
    """Echoes the workspace the CLI resolved into ``ctx.workspace``."""

    name: ClassVar[str] = "echo-workspace"
    spec_schema: ClassVar[type[BaseModel]] = _Spec

    async def run(self, spec: _Spec, *, ctx: FunctionContext) -> dict:
        del spec
        return {"workspace": ctx.workspace}


class _EchoJob(NemoJob):
    name = "echo-job"
    description = "Job used to capture the workspace passed to submit_remote."

    def run(self, config: dict) -> dict:  # pragma: no cover - never run locally
        return config


def _write_config(path: Path, *, current_context: str = "team") -> None:
    """Write a two-context config file so context switching is observable."""
    path.write_text(
        yaml.safe_dump(
            {
                "current_context": current_context,
                "clusters": [{"name": "test-cluster", "base_url": "http://platform.test"}],
                "users": [{"name": "me", "type": "no-auth"}],
                "contexts": [
                    {
                        "name": "team",
                        "cluster": "test-cluster",
                        "user": "me",
                        "workspace": CONTEXT_WORKSPACE,
                    },
                    {
                        "name": "other-team",
                        "cluster": "test-cluster",
                        "user": "me",
                        "workspace": OTHER_WORKSPACE,
                    },
                ],
            }
        )
    )


@pytest.fixture
def config_file(isolated_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real config file on disk, replacing ``isolated_config``'s empty one.

    Depends on ``isolated_config`` explicitly so it runs *after* that autouse
    fixture has cleared ``NMP_*`` env vars and pointed ``NMP_CONFIG_FILE`` at
    an empty file -- otherwise ordering would be undefined and the empty file
    could win.
    """
    del isolated_config
    path = tmp_path / "nemo-config.yaml"
    _write_config(path)
    monkeypatch.setenv("NMP_CONFIG_FILE", str(path))
    return path


def _app(register) -> typer.Typer:
    """Build a Typer app whose callback seeds a real, config-backed CLIContext.

    Mirrors what ``nemo_platform_ext.cli.app.main`` does for plugin commands:
    no workspace override, so ``CLIContext`` must resolve it from the config
    file the same way a real invocation would.
    """
    app = typer.Typer()

    @app.callback()
    def _root(ctx: typer.Context) -> None:
        ctx.obj = CLIContext(overrides={})

    register(app)
    return app


def _function_app() -> typer.Typer:
    return _app(lambda a: add_function_commands(a, {"plugin.echo-workspace": _EchoFunction}))


def _job_app() -> typer.Typer:
    return _app(lambda a: add_job_commands(a, {"plugin.echo-job": _EchoJob}))


@pytest.fixture
def captured_submit_url(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the URL a function ``submit`` would POST to, without a server."""
    captured: list[str] = []

    def _fake_post(url: str, body: dict, *, headers: dict, timeout: float = 30.0, **_kwargs) -> None:
        del body, headers, timeout
        captured.append(url)
        typer.echo(json.dumps({"ok": True}))

    monkeypatch.setattr("nemo_platform_plugin.commands._post_function_submit", _fake_post)
    return captured


@pytest.fixture
def captured_submit_workspace(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Capture the workspace a job ``submit`` would hand to the scheduler."""
    captured: list[str | None] = []

    def _fake_submit(self, job_cls, spec, **kwargs):
        del self, job_cls, spec
        captured.append(kwargs.get("workspace"))
        return {"id": "job-1"}

    monkeypatch.setattr("nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote", _fake_submit)
    return captured


def test_config_file_workspace_reaches_function_run(config_file: Path) -> None:
    del config_file
    result = runner.invoke(_function_app(), ["echo-workspace", "run", "--spec", "{}"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"workspace": CONTEXT_WORKSPACE}


def test_config_file_workspace_reaches_function_submit_url(config_file: Path, captured_submit_url: list[str]) -> None:
    del config_file
    result = runner.invoke(_function_app(), ["echo-workspace", "submit", "--spec", "{}"])
    assert result.exit_code == 0, result.output
    assert captured_submit_url[0].endswith(f"/v2/workspaces/{CONTEXT_WORKSPACE}/echo-workspace")


def test_config_file_workspace_reaches_job_submit(
    config_file: Path, captured_submit_workspace: list[str | None]
) -> None:
    del config_file
    result = runner.invoke(_job_app(), ["echo-job", "submit", "--spec", "{}"])
    assert result.exit_code == 0, result.output
    assert captured_submit_workspace[0] == CONTEXT_WORKSPACE


def test_switching_current_context_switches_the_workspace(config_file: Path) -> None:
    """The verb follows ``current_context``, not a baked-in name."""
    _write_config(config_file, current_context="other-team")
    result = runner.invoke(_function_app(), ["echo-workspace", "run", "--spec", "{}"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"workspace": OTHER_WORKSPACE}


def test_env_var_overrides_config_file_workspace(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del config_file
    monkeypatch.setenv("NMP_WORKSPACE", "env-ws")
    result = runner.invoke(_function_app(), ["echo-workspace", "run", "--spec", "{}"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"workspace": "env-ws"}


def test_explicit_flag_overrides_config_file_workspace(config_file: Path) -> None:
    del config_file
    result = runner.invoke(
        _function_app(),
        ["echo-workspace", "run", "--spec", "{}", "--workspace", "flag-ws"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"workspace": "flag-ws"}


def test_falls_back_to_default_when_config_has_no_workspace(
    isolated_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A context with no ``workspace:`` key still yields the platform default."""
    del isolated_config
    path = tmp_path / "no-workspace.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "current_context": "team",
                "clusters": [{"name": "test-cluster", "base_url": "http://platform.test"}],
                "users": [{"name": "me", "type": "no-auth"}],
                "contexts": [{"name": "team", "cluster": "test-cluster", "user": "me"}],
            }
        )
    )
    monkeypatch.setenv("NMP_CONFIG_FILE", str(path))

    result = runner.invoke(_function_app(), ["echo-workspace", "run", "--spec", "{}"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"workspace": "default"}


# ---------------------------------------------------------------------------
# Hand-written plugin commands
# ---------------------------------------------------------------------------


def test_hand_written_plugin_command_uses_config_file_workspace(config_file: Path) -> None:
    """A real hand-written plugin command resolves from the config file.

    ``nemo auditor configs list`` is hand-written (not a generated verb): it
    declares the flag with the shared ``WorkspaceOption`` alias rather than a
    programmatic signature, so it exercises a different construction path from
    the generated verbs above against the same state object the top-level
    ``nemo`` callback installs.
    """
    del config_file
    import httpx
    from nemo_platform_ext.cli.app import app

    captured: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(_handler)
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    with patch.object(httpx, "Client", _factory):
        result = runner.invoke(
            app,
            ["auditor", "configs", "list"],
            obj=CLIContext(overrides={}),
        )

    assert result.exit_code == 0, result.output
    assert captured, "no HTTP request was issued"
    assert f"/workspaces/{CONTEXT_WORKSPACE}/" in str(captured[0].url), captured[0].url
