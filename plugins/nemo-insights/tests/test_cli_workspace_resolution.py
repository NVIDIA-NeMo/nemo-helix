# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo insights`` verbs resolve the workspace from the active CLI context.

Regression guard: every ``--workspace`` option on these commands used to be
declared with a literal ``"default"`` Typer default, so the command body could
never tell an omitted flag from an explicit ``--workspace default``. The
workspace the operator selected (``nemo config use-context``, a ``workspace:``
key in their context, or ``$NMP_WORKSPACE``) was silently discarded and the
command acted on ``default`` -- potentially the wrong tenant, with no warning.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import typer
from nemo_insights_plugin import cli
from nemo_insights_plugin.entities import AnalysisConfig, AnalysisRun
from nemo_insights_plugin.schema import AnalysisRunPage, AnalysisRunResponse
from nemo_platform_plugin.nooa_model_client import ConfiguredModelRefs
from typer.testing import CliRunner

runner = CliRunner()

CONTEXT_WORKSPACE = "my-team-ws"
EXPLICIT_WORKSPACE = "team-alpha"
RUN_NAME = "insights-run-0123456789abcdef0123456789abcdef"


class _ContextState:
    """Minimal stand-in for ``CLIContext`` with a non-'default' workspace."""

    def __init__(self, workspace: str | None = CONTEXT_WORKSPACE) -> None:
        self._workspace = workspace

    def get_workspace(self) -> str | None:
        return self._workspace


class _Page:
    """Stands in for a paged list response; only ``model_dump`` is exercised."""

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return {"data": []}


class _StubAnalysisConfigs:
    """Records the workspace each ``analysis`` verb resolved."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _config(self, workspace: str, agent: str) -> AnalysisConfig:
        return AnalysisConfig(name=agent, workspace=workspace, agent=agent, enabled=True)

    async def enable(self, **kwargs: Any) -> AnalysisConfig:
        self.calls.append(kwargs)
        return self._config(kwargs["workspace"], kwargs["agent"])

    async def disable(self, **kwargs: Any) -> AnalysisConfig:
        self.calls.append(kwargs)
        return self._config(kwargs["workspace"], kwargs["agent"])

    async def get(self, **kwargs: Any) -> AnalysisConfig:
        self.calls.append(kwargs)
        return self._config(kwargs["workspace"], kwargs["agent"])

    async def list_configs(self, **kwargs: Any) -> _Page:
        self.calls.append(kwargs)
        return _Page()


class _StubAnalysisRuns:
    """Records the workspace each ``analysis-runs`` verb resolved."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _response(self, workspace: str) -> AnalysisRunResponse:
        run = AnalysisRun(name=RUN_NAME, workspace=workspace, agent="demo-agent")
        return AnalysisRunResponse(run=run, job={"name": RUN_NAME, "status": "completed"})

    async def create(self, **kwargs: Any) -> AnalysisRunResponse:
        self.calls.append(kwargs)
        return self._response(kwargs["workspace"])

    async def get(self, **kwargs: Any) -> AnalysisRunResponse:
        self.calls.append(kwargs)
        return self._response(kwargs["workspace"])

    async def list_runs(self, **kwargs: Any) -> AnalysisRunPage:
        self.calls.append(kwargs)
        return AnalysisRunPage(data=[], pagination=None, sort="-created_at", filter=None)


class _StubClient:
    def __init__(self, configs: _StubAnalysisConfigs, runs: _StubAnalysisRuns) -> None:
        self.insights = SimpleNamespace(analysis_configs=configs, analysis_runs=runs)

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _no_ambient_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """``$NMP_WORKSPACE`` is a real resolver input; keep it out of these tests."""
    monkeypatch.delenv("NMP_WORKSPACE", raising=False)


@pytest.fixture(autouse=True)
def _configured_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "configured_model_refs",
        lambda: ConfiguredModelRefs(default="default/big", fast="default/small"),
    )


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> tuple[_StubAnalysisConfigs, _StubAnalysisRuns]:
    configs = _StubAnalysisConfigs()
    runs = _StubAnalysisRuns()
    client = _StubClient(configs, runs)
    monkeypatch.setattr(cli, "make_client", lambda base_url: client)
    return configs, runs


def _app(state: object | None) -> typer.Typer:
    """Wrap the insights app in a parent group that seeds ``ctx.obj`` like ``nemo`` does."""
    parent = typer.Typer()

    @parent.callback()
    def _root(ctx: typer.Context) -> None:
        ctx.obj = state

    parent.add_typer(cli.InsightsCLI().get_cli(), name="insights")
    return parent


# Every command that takes "the workspace this command acts on", with the
# arguments needed to reach the client call.
COMMANDS: list[tuple[str, list[str]]] = [
    ("analysis enable", ["insights", "analysis", "enable", "--agent", "demo-agent"]),
    ("analysis disable", ["insights", "analysis", "disable", "--agent", "demo-agent"]),
    ("analysis status (one agent)", ["insights", "analysis", "status", "--agent", "demo-agent"]),
    ("analysis status (all)", ["insights", "analysis", "status"]),
    ("analysis-runs create", ["insights", "analysis-runs", "create", "--agent", "demo-agent"]),
    ("analysis-runs list", ["insights", "analysis-runs", "list"]),
    ("analysis-runs get", ["insights", "analysis-runs", "get", RUN_NAME]),
]


def _recorded_workspaces(stubs: tuple[_StubAnalysisConfigs, _StubAnalysisRuns]) -> list[str]:
    configs, runs = stubs
    return [call["workspace"] for call in configs.calls + runs.calls]


@pytest.mark.parametrize(("label", "argv"), COMMANDS, ids=[label for label, _ in COMMANDS])
def test_omitted_workspace_uses_the_active_context(
    label: str,
    argv: list[str],
    stubs: tuple[_StubAnalysisConfigs, _StubAnalysisRuns],
) -> None:
    del label
    result = runner.invoke(_app(_ContextState()), argv)

    assert result.exit_code == 0, result.output
    assert _recorded_workspaces(stubs) == [CONTEXT_WORKSPACE]


@pytest.mark.parametrize(("label", "argv"), COMMANDS, ids=[label for label, _ in COMMANDS])
def test_explicit_workspace_wins_over_the_active_context(
    label: str,
    argv: list[str],
    stubs: tuple[_StubAnalysisConfigs, _StubAnalysisRuns],
) -> None:
    del label
    result = runner.invoke(_app(_ContextState()), [*argv, "--workspace", EXPLICIT_WORKSPACE])

    assert result.exit_code == 0, result.output
    assert _recorded_workspaces(stubs) == [EXPLICIT_WORKSPACE]


@pytest.mark.parametrize(("label", "argv"), COMMANDS, ids=[label for label, _ in COMMANDS])
def test_without_cli_state_the_workspace_falls_back_to_default(
    label: str,
    argv: list[str],
    stubs: tuple[_StubAnalysisConfigs, _StubAnalysisRuns],
) -> None:
    """Direct/unit invocation with no ``ctx.obj`` keeps the pre-existing behavior."""
    del label
    result = runner.invoke(_app(None), argv)

    assert result.exit_code == 0, result.output
    assert _recorded_workspaces(stubs) == ["default"]


@pytest.mark.parametrize(("label", "argv"), COMMANDS, ids=[label for label, _ in COMMANDS])
def test_nmp_workspace_env_is_used_when_no_state_is_installed(
    label: str,
    argv: list[str],
    stubs: tuple[_StubAnalysisConfigs, _StubAnalysisRuns],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del label
    monkeypatch.setenv("NMP_WORKSPACE", "env-ws")

    result = runner.invoke(_app(None), argv)

    assert result.exit_code == 0, result.output
    assert _recorded_workspaces(stubs) == ["env-ws"]
