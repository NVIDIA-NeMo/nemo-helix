# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for shared-context base-URL, workspace, and auth-token resolution.

These pin the three behaviours that make ``nemo agents`` usable against a
remote, secured, multi-workspace platform:

- **Base URL** resolves through the shared CLI context the rest of the CLI
  uses (``nemo config set --base-url`` / ``NHX_BASE_URL``), with an explicit
  ``--base-url`` / ``NEMO_BASE_URL`` still taking precedence, and the resolved
  target echoed to stderr so a mis-pointed command is visible instead of
  silently hitting localhost.
- **Workspace** resolves through the active CLI context (``nemo config
  use-context`` / ``$NHX_WORKSPACE``) when ``--workspace`` is omitted, instead
  of silently acting on ``default`` — with an explicit ``--workspace`` still
  winning and ``default`` preserved when no CLI state is installed.
- **Auth** headers from the shared context (the ``Authorization: Bearer``
  token behind ``nemo auth login``) are attached to every platform HTTP call,
  so agents commands are not rejected 401/403 on a secured cluster.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import patch

import httpx
from nemo_agents_plugin.cli import AgentsCLI
from typer.testing import CliRunner


def _install_mock_transport(handler) -> AbstractContextManager[Any]:
    """Patch ``httpx.Client`` in the CLI module to use a ``MockTransport``."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    class _Client(real_client):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    return patch("nemo_agents_plugin.cli.httpx.Client", _Client)


def _empty_page() -> dict[str, Any]:
    return {
        "data": [],
        "pagination": {
            "page": 1,
            "page_size": 0,
            "current_page_size": 0,
            "total_pages": 1,
            "total_results": 0,
        },
    }


def _capturing(captured: list[httpx.Request], *, json_body: Any = None):
    """Return a handler that records every request and replies 200."""

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, request=req, json=json_body if json_body is not None else _empty_page())

    return handler


class _FakeUser:
    def __init__(self, token: str | None) -> None:
        self._token = token

    def get_client_config(self) -> dict[str, object]:
        if self._token is None:
            return {}
        return {"default_headers": {"Authorization": f"Bearer {self._token}"}}


class _FakeCluster:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


class _FakeSDKContext:
    def __init__(self, base_url: str, token: str | None) -> None:
        self.user = _FakeUser(token)
        self.cluster = _FakeCluster(base_url)


class _FakeCLIContext:
    """Minimal stand-in for ``CLIContext`` (typer.Context.obj)."""

    def __init__(
        self,
        base_url: str = "http://config-host:9999",
        token: str | None = "cfg-token",
        workspace: str | None = None,
    ) -> None:
        self._sdk = _FakeSDKContext(base_url, token)
        self._workspace = workspace

    def get_workspace(self) -> str | None:
        return self._workspace

    def get_sdk_context(self) -> _FakeSDKContext:
        return self._sdk

    def get_base_url(self, default: str | None = None) -> str | None:
        return str(self._sdk.cluster.base_url)


# ---------------------------------------------------------------------------
# Base URL resolution
# ---------------------------------------------------------------------------


def test_base_url_flag_overrides_configured_context() -> None:
    """An explicit ``--base-url`` wins over the configured context base URL."""
    captured: list[httpx.Request] = []
    app = AgentsCLI().get_cli()
    with _install_mock_transport(_capturing(captured)):
        result = CliRunner().invoke(
            app,
            ["list", "--base-url", "http://flag-host:1111"],
            obj=_FakeCLIContext(base_url="http://config-host:9999"),
        )

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert captured, "expected a request to be issued"
    assert captured[0].url.host == "flag-host"
    assert captured[0].url.port == 1111


def test_base_url_falls_back_to_configured_context() -> None:
    """With no flag/env, agents commands target the configured context base URL.

    This is the P0 regression: previously agents ignored the shared config
    and silently hit localhost:8080.
    """
    captured: list[httpx.Request] = []
    app = AgentsCLI().get_cli()
    with _install_mock_transport(_capturing(captured)):
        result = CliRunner().invoke(
            app,
            ["list"],
            obj=_FakeCLIContext(base_url="http://config-host:9999"),
        )

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert captured[0].url.host == "config-host"
    assert captured[0].url.port == 9999


def test_base_url_env_overrides_configured_context() -> None:
    """``NEMO_BASE_URL`` (command-level env) still takes precedence over config."""
    captured: list[httpx.Request] = []
    app = AgentsCLI().get_cli()
    with _install_mock_transport(_capturing(captured)):
        result = CliRunner().invoke(
            app,
            ["list"],
            obj=_FakeCLIContext(base_url="http://config-host:9999"),
            env={"NEMO_BASE_URL": "http://env-host:2222"},
        )

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert captured[0].url.host == "env-host"
    assert captured[0].url.port == 2222


def test_base_url_defaults_to_localhost_without_context() -> None:
    """Backwards compatibility: no context and no flag -> localhost:8080."""
    captured: list[httpx.Request] = []
    app = AgentsCLI().get_cli()
    with _install_mock_transport(_capturing(captured)):
        result = CliRunner().invoke(app, ["list"])

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert captured[0].url.host == "localhost"
    assert captured[0].url.port == 8080


def test_resolved_target_is_echoed_to_stderr_only() -> None:
    """The resolved target is announced on stderr, keeping stdout clean for pipes."""
    app = AgentsCLI().get_cli()
    with _install_mock_transport(_capturing([])):
        result = CliRunner().invoke(
            app,
            ["list", "--base-url", "http://flag-host:1234", "-o", "json"],
            obj=_FakeCLIContext(),
        )

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "Targeting http://flag-host:1234" in (result.stderr or "")
    assert "Targeting" not in result.stdout


# ---------------------------------------------------------------------------
# Auth token attachment
# ---------------------------------------------------------------------------


def test_auth_header_attached_from_context() -> None:
    """The bearer token from the shared context is attached to platform calls."""
    captured: list[httpx.Request] = []
    app = AgentsCLI().get_cli()
    with _install_mock_transport(_capturing(captured)):
        result = CliRunner().invoke(
            app,
            ["list", "--base-url", "http://h:1"],
            obj=_FakeCLIContext(token="secret-token"),
        )

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert captured[0].headers.get("authorization") == "Bearer secret-token"


def test_no_auth_header_without_context() -> None:
    """No context -> no auth header (unauthenticated local dev keeps working)."""
    captured: list[httpx.Request] = []
    app = AgentsCLI().get_cli()
    with _install_mock_transport(_capturing(captured)):
        result = CliRunner().invoke(app, ["list", "--base-url", "http://h:1"])

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "authorization" not in captured[0].headers


def test_platform_invoke_attaches_auth_and_targets_context_base_url() -> None:
    """``invoke --agent`` routes through the gateway with the context token+URL."""
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "96"}}]})

    app = AgentsCLI().get_cli()
    with _install_mock_transport(handler):
        result = CliRunner().invoke(
            app,
            ["invoke", "--agent", "calc", "--input", "12*8", "--no-progress"],
            obj=_FakeCLIContext(base_url="http://config-host:9999", token="tkn"),
        )

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert captured[0].url.host == "config-host"
    assert captured[0].url.port == 9999
    assert captured[0].headers.get("authorization") == "Bearer tkn"


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------
#
# The bug these pin: declaring ``--workspace`` with a literal ``"default"``
# Typer default makes an omitted flag indistinguishable from an explicit
# ``--workspace default``, so the workspace the user selected (via
# ``nemo config use-context`` / ``$NHX_WORKSPACE``) was silently discarded and
# the command acted on ``default`` — potentially the wrong tenant.


def _workspace_from(req: httpx.Request) -> str:
    """Pull the workspace segment out of an agents API URL."""
    parts = req.url.path.strip("/").split("/")
    return parts[parts.index("workspaces") + 1]


def _invoke_capturing(args: list[str], **kwargs: Any) -> tuple[Any, list[httpx.Request]]:
    captured: list[httpx.Request] = []
    app = AgentsCLI().get_cli()
    with _install_mock_transport(_capturing(captured)):
        result = CliRunner().invoke(app, [*args, "--base-url", "http://h:1"], **kwargs)
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert captured, "expected a request to be issued"
    return result, captured


def test_workspace_falls_back_to_active_context_workspace(monkeypatch) -> None:
    """With no ``--workspace``, commands act on the active CLI context's workspace."""
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)
    _, captured = _invoke_capturing(["list"], obj=_FakeCLIContext(workspace="team-a"))
    assert _workspace_from(captured[0]) == "team-a"


def test_workspace_flag_overrides_active_context_workspace(monkeypatch) -> None:
    """An explicit ``--workspace`` still wins over the active context."""
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)
    _, captured = _invoke_capturing(
        ["list", "--workspace", "flag-ws"],
        obj=_FakeCLIContext(workspace="team-a"),
    )
    assert _workspace_from(captured[0]) == "flag-ws"


def test_workspace_defaults_without_cli_state(monkeypatch) -> None:
    """No CLI state and no flag -> ``default``, preserving direct/unit invocation."""
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)
    _, captured = _invoke_capturing(["list"])
    assert _workspace_from(captured[0]) == "default"


def test_workspace_falls_back_to_env_without_cli_state(monkeypatch) -> None:
    """``$NHX_WORKSPACE`` applies when no CLI state object is installed."""
    monkeypatch.setenv("NHX_WORKSPACE", "env-ws")
    _, captured = _invoke_capturing(["list"])
    assert _workspace_from(captured[0]) == "env-ws"


def test_state_workspace_wins_over_the_env_fallback(monkeypatch) -> None:
    """A reporting CLI state short-circuits the resolver's bare env lookup.

    Internal ordering, not user-facing precedence: a real ``CLIContext``
    applies ``$NHX_WORKSPACE`` before ``get_workspace()`` answers, so the env
    var still wins in practice. ``_FakeCLIContext`` ignores the environment
    on purpose so this isolates the fallback path.
    """
    monkeypatch.setenv("NHX_WORKSPACE", "env-ws")
    _, captured = _invoke_capturing(["list"], obj=_FakeCLIContext(workspace="team-a"))
    assert _workspace_from(captured[0]) == "team-a"


def test_workspace_resolution_applies_to_subgroup_commands(monkeypatch) -> None:
    """Resolution is not list-only: nested groups honour the context too."""
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)
    _, captured = _invoke_capturing(
        ["deployments", "list"],
        obj=_FakeCLIContext(workspace="team-a"),
    )
    assert _workspace_from(captured[0]) == "team-a"

    _, captured = _invoke_capturing(
        ["environments", "list", "-w", "flag-ws"],
        obj=_FakeCLIContext(workspace="team-a"),
    )
    assert _workspace_from(captured[0]) == "flag-ws"
