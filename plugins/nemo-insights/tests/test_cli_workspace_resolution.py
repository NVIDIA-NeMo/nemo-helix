# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo insights`` verbs resolve the workspace from the active CLI context.

Regression guard: every ``--workspace`` option on these commands used to be
declared with a literal ``"default"`` Typer default, so the command body could
never tell an omitted flag from an explicit ``--workspace default``. The
workspace the operator selected (``nemo config use-context``, a ``workspace:``
key in their context, or ``$NHX_WORKSPACE``) was silently discarded and the
command acted on ``default`` -- potentially the wrong tenant, with no warning.
"""

from __future__ import annotations

import pytest
from nemo_insights_plugin.testing.cli import (
    RUN_NAME,
    FakeInsightsAPI,
    WireState,
    app_with_state,
    config_json,
    page_json,
    run_json,
    run_response_json,
)
from typer.testing import CliRunner

runner = CliRunner()

CONTEXT_WORKSPACE = "my-team-ws"
EXPLICIT_WORKSPACE = "team-alpha"

pytestmark = pytest.mark.usefixtures("configured_models")


@pytest.fixture
def api() -> FakeInsightsAPI:
    """Every route these commands call, answering for any workspace."""
    api = FakeInsightsAPI()
    api.on("POST", "/analysis-configs/demo-agent/enable", config_json())
    api.on("POST", "/analysis-configs/demo-agent/disable", config_json(enabled=False))
    api.on("GET", "/analysis-configs/demo-agent", config_json())
    api.on("GET", "/analysis-configs", page_json([config_json()]))
    api.on("POST", "/analysis-runs", run_response_json())
    api.on("GET", "/analysis-runs", page_json([run_json()]))
    api.on("GET", f"/analysis-runs/{RUN_NAME}", run_response_json("completed"))
    return api


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
IDS = [label for label, _ in COMMANDS]


def _workspaces_for(api: FakeInsightsAPI, state: WireState, argv: list[str]) -> list[str]:
    result = runner.invoke(app_with_state(state), argv)
    assert result.exit_code == 0, result.output
    return [request.workspace for request in api.requests]


@pytest.mark.parametrize(("label", "argv"), COMMANDS, ids=IDS)
def test_omitted_workspace_uses_the_active_context(label: str, argv: list[str], api: FakeInsightsAPI) -> None:
    del label
    assert _workspaces_for(api, WireState(api, workspace=CONTEXT_WORKSPACE), argv) == [CONTEXT_WORKSPACE]


@pytest.mark.parametrize(("label", "argv"), COMMANDS, ids=IDS)
def test_explicit_workspace_wins_over_the_active_context(label: str, argv: list[str], api: FakeInsightsAPI) -> None:
    del label
    state = WireState(api, workspace=CONTEXT_WORKSPACE)
    assert _workspaces_for(api, state, [*argv, "--workspace", EXPLICIT_WORKSPACE]) == [EXPLICIT_WORKSPACE]


@pytest.mark.parametrize(("label", "argv"), COMMANDS, ids=IDS)
def test_a_context_without_a_workspace_falls_back_to_default(label: str, argv: list[str], api: FakeInsightsAPI) -> None:
    del label
    assert _workspaces_for(api, WireState(api, workspace=None), argv) == ["default"]


@pytest.mark.parametrize(("label", "argv"), COMMANDS, ids=IDS)
def test_nhx_workspace_env_is_used_when_the_context_has_no_workspace(
    label: str, argv: list[str], api: FakeInsightsAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    del label
    monkeypatch.setenv("NHX_WORKSPACE", "env-ws")
    assert _workspaces_for(api, WireState(api, workspace=None), argv) == ["env-ws"]
