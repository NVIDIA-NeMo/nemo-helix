# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents optimize`` — one group holding the router job and every contributed verb."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import click
import httpx
import pytest
import typer
from nemo_agent_optimization_plugin import cli as cli_module
from nemo_agent_optimization_plugin.cli import AgentOptimizeCLI
from nemo_agent_optimization_plugin.schemas.strategies import STRATEGIES_PATH, OptimizationStrategyList
from nemo_helix_plugin.client.errors import AuthenticationError, NemoTransportError
from typer.main import get_command
from typer.testing import CliRunner


def _register_a_verb(group: typer.Typer) -> None:
    @group.command("fake-prepare")
    def _prepare() -> None:
        """Stage something."""


def _register_badly(group: typer.Typer) -> None:
    raise RuntimeError("this contribution is broken")


_LISTING = {"data": [{"name": "nat", "description": "Numeric HPO."}]}


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

    ``generate_legacy_verbs = False`` hangs submission off the command's own
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


def _remote(monkeypatch: pytest.MonkeyPatch, names: list[str] | Exception) -> None:
    """Stand in for the platform's `GET /strategies`, or make reaching it fail."""

    def _fake(base_url: str | None) -> tuple[list[str], str]:
        if isinstance(names, Exception):
            raise names
        return names, "http://platform"

    monkeypatch.setattr(cli_module, "_remote_strategy_names", _fake)


def _stdout(result: Any) -> list[str]:
    """Only the names: the source note is deliberately on stderr."""
    return result.stdout.split()


def test_list_strategies_reports_the_platforms_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The platform runs the job, so it is the only authority on what `--strategy` takes."""
    install(monkeypatch)
    _remote(monkeypatch, ["nat", "acme"])

    result = CliRunner().invoke(AgentOptimizeCLI().get_cli(), ["list-strategies"])

    assert result.exit_code == 0, result.output
    assert _stdout(result) == ["nat", "acme"]
    assert "installed on http://platform" in result.stderr


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
    assert "No optimization strategies are installed." in result.stdout


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


def test_the_listing_builds_a_client_for_the_resolved_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the plumbing the stubbed tests skip; the request itself is tested in test_client.py."""
    captured: dict[str, Any] = {}

    class _StubClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def list_strategies(self) -> Any:
            return SimpleNamespace(data=lambda: OptimizationStrategyList.model_validate(_LISTING))

    monkeypatch.setattr(cli_module, "AgentOptimizationClient", _StubClient)
    monkeypatch.setattr(cli_module, "_resolve_base_url", lambda _base_url: "http://platform")
    monkeypatch.setattr(cli_module, "_context_headers", lambda: {"Authorization": "Bearer token"})

    names, target = cli_module._remote_strategy_names(None)

    assert names == ["nat"]
    assert target == "http://platform"
    assert captured["base_url"] == "http://platform"
    assert captured["default_headers"] == {"Authorization": "Bearer token"}


def test_the_cli_name_matches_its_entry_point_key() -> None:
    """``nemo.cli.agents`` mounts this group by entry-point key; a mismatch warns at discovery."""
    assert AgentOptimizeCLI.name == "optimize"
