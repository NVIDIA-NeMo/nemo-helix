# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents optimize`` — one group holding the router job and every contributed verb."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import click
import httpx
import pytest
import typer
from nemo_agent_optimization_plugin import cli as cli_module
from nemo_agent_optimization_plugin.cli import AgentOptimizeCLI
from nemo_agent_optimization_plugin.schemas.strategies import (
    STRATEGIES_PATH,
    OptimizationStrategy,
    OptimizationStrategyList,
)
from nemo_helix_plugin.client.errors import AuthenticationError, NemoTransportError
from typer.main import get_command
from typer.testing import CliRunner


def _register_a_verb(group: typer.Typer) -> None:
    @group.command("fake-prepare")
    def _prepare() -> None:
        """Stage something."""


def _register_badly(group: typer.Typer) -> None:
    raise RuntimeError("this contribution is broken")


_LISTING = {"data": [{"name": "legacy", "description": "Numeric HPO."}]}


def install(monkeypatch: pytest.MonkeyPatch, contributions: dict[str, Any] | None = None) -> None:
    """Pin verb discovery, so the group is built from these contributions and nothing installed."""
    monkeypatch.setattr(cli_module, "discover", lambda group: dict(contributions or {}))


def optimize_group(monkeypatch: pytest.MonkeyPatch, contributions: dict[str, Any] | None = None) -> click.Group:
    install(monkeypatch, contributions)
    command = get_command(AgentOptimizeCLI().get_cli())
    assert isinstance(command, click.Group)
    return command


def test_the_group_carries_the_router_job_and_list_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    group = optimize_group(monkeypatch)
    assert {"run-strategy", "list-strategies"} <= set(group.commands)


def test_run_strategy_submits_without_a_legacy_verb(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run-strategy`` submits on its own; there is no legacy ``run`` / ``submit`` beneath it.

    callback, so the command may still carry ``explain``.  What must not come
    back is a nested ``submit`` the caller has to type.
    """
    run_strategy = optimize_group(monkeypatch).commands["run-strategy"]
    nested = getattr(run_strategy, "commands", {})
    assert "submit" not in nested
    assert "run" not in nested


def test_run_strategy_takes_a_strategy_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["run-strategy", "--help"])

    assert result.exit_code == 0, result.output
    assert "--strategy" in result.output
    assert "--optimize-config" in result.output


def test_run_strategy_carries_fields_it_has_no_flag_for_through_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """A strategy's own inputs have no generated flag; ``--spec`` must deliver them to the platform intact."""
    install(monkeypatch)
    submitted: dict[str, Any] = {}

    def _submit_remote(_self: object, _job_cls: type, spec: dict, **_kwargs: Any) -> dict[str, Any]:
        submitted.update(spec)
        return {"id": "job-1"}

    monkeypatch.setattr("nemo_helix_plugin.scheduler.NemoJobScheduler.submit_remote", _submit_remote)

    result = CliRunner().invoke(
        AgentOptimizeCLI().get_cli(),
        [
            "run-strategy",
            "--strategy",
            "custom",
            "--spec",
            '{"dataset": "my-dataset", "objective": "accuracy"}',
        ],
        # The platform comes from the global ``nemo --base-url`` / active context.
        obj=SimpleNamespace(get_base_url=lambda default=None: "http://platform.test"),
    )

    assert result.exit_code == 0, result.output
    assert submitted == {"strategy": "custom", "dataset": "my-dataset", "objective": "accuracy"}


def test_a_contributing_plugin_hangs_its_own_verb_off_the_group(monkeypatch: pytest.MonkeyPatch) -> None:
    group = optimize_group(monkeypatch, {"fake-prepare": _register_a_verb})
    assert "fake-prepare" in group.commands


def test_a_contribution_that_fails_to_register_does_not_take_the_group_down(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        group = optimize_group(monkeypatch, {"broken": _register_badly, "fake-prepare": _register_a_verb})

    assert "fake-prepare" in group.commands
    assert "run-strategy" in group.commands
    assert "'broken' failed to register" in caplog.text


def _remote(monkeypatch: pytest.MonkeyPatch, strategies: list[dict[str, str]] | Exception) -> None:
    """Stand in for the platform's `GET /strategies`, or make reaching it fail."""

    def _fake() -> tuple[list[OptimizationStrategy], str]:
        if isinstance(strategies, Exception):
            raise strategies
        return [OptimizationStrategy.model_validate(s) for s in strategies], "http://platform"

    monkeypatch.setattr(cli_module, "_remote_strategies", _fake)


_STRATEGIES = [
    {"name": "legacy", "description": "Numeric HPO."},
    {"name": "acme", "description": "Prompt rewriting."},
]


def test_list_strategies_reports_the_platforms_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The platform runs the job, so it is the only authority on what `--strategy` takes."""
    install(monkeypatch)
    _remote(monkeypatch, _STRATEGIES)

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies", "-f", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _STRATEGIES


def test_list_strategies_table_shows_each_description(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    _remote(monkeypatch, _STRATEGIES)

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies", "-f", "table"])

    assert result.exit_code == 0, result.output
    assert "Description" in result.stdout
    assert "Numeric HPO." in result.stdout
    assert "Prompt rewriting." in result.stdout


def test_piped_list_strategies_is_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Like every `nemo` list command: CliRunner is not a TTY, so no flag means JSON."""
    install(monkeypatch)
    _remote(monkeypatch, _STRATEGIES)

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies"])

    assert result.exit_code == 0, result.output
    assert [s["name"] for s in json.loads(result.stdout)] == ["legacy", "acme"]


def test_list_strategies_code_prints_the_client_call(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    _remote(monkeypatch, AssertionError("-f code must not call the platform"))
    state = SimpleNamespace(get_base_url=lambda default=None: "http://platform", get_workspace=lambda: "default")

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert "AgentOptimizationClient" in result.stdout
    assert "list_strategies()" in result.stdout


def test_an_unreachable_platform_is_an_error_not_a_local_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This environment's installs describe a different machine, so they are never the answer."""
    install(monkeypatch)
    _remote(monkeypatch, NemoTransportError(httpx.ConnectError("connection refused")))

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies"])

    assert result.exit_code == 1
    assert "connection refused" in result.stderr
    assert not result.stdout.strip()


def test_list_strategies_trusts_an_empty_answer_from_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform that answers "none" is authoritative; it must not fall back."""
    install(monkeypatch)
    _remote(monkeypatch, [])

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies"])

    assert result.exit_code == 0, result.output
    assert "No optimization strategies are installed on http://platform." in result.stderr
    # The note stays on stderr, so piped output is still a parseable (empty) list.
    assert json.loads(result.stdout) == []


def test_a_rejected_request_is_reported_and_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    rejection = httpx.Response(
        401,
        json={"detail": "token expired"},
        request=httpx.Request("GET", f"http://platform{STRATEGIES_PATH}"),
    )
    _remote(monkeypatch, AuthenticationError(rejection))

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies"])

    assert result.exit_code == 1
    assert "401" in result.stderr
    assert "token expired" in result.stderr


def test_the_listing_requires_cli_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI state owns platform client construction; this helper has no local fallback."""
    monkeypatch.setattr(cli_module, "resolve_base_url", lambda: "http://platform")

    with pytest.raises(RuntimeError, match="No NeMo Helix CLI state"):
        cli_module._remote_strategies()


def test_under_nemo_the_listing_uses_the_cli_shared_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared client carries the global base URL and the context's auth; the CLI owns its lifetime."""
    install(monkeypatch)
    requested: list[type] = []

    class _Shared:
        def list_strategies(self) -> Any:
            return SimpleNamespace(data=lambda: OptimizationStrategyList.model_validate(_LISTING))

    class _State:
        def get_base_url(self, default: str | None = None) -> str | None:
            return "http://shared-platform"

        def typed_client(self, client_cls: type, timeout: float = 60.0) -> Any:
            requested.append(client_cls)
            return _Shared()

        def get_no_truncate(self, override: bool | None = None) -> bool:
            return bool(override)

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies", "-f", "json"], obj=_State())

    assert result.exit_code == 0, result.output
    assert [s["name"] for s in json.loads(result.stdout)] == ["legacy"]
    assert requested == [cli_module.AgentOptimizationClient]
    assert "Targeting http://shared-platform" in result.stderr


def test_list_strategies_has_no_base_url_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies", "--base-url", "http://x"])

    assert result.exit_code == 2


def test_the_cli_name_matches_its_entry_point_key() -> None:
    """``nemo.cli.agents`` mounts this group by entry-point key; a mismatch warns at discovery."""
    assert AgentOptimizeCLI.name == "optimize"
