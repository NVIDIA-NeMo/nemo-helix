# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import TypeVar, cast

import click
import httpx
import pytest
import typer
from nemo_helix_plugin.cli_state import (
    base_url_from_context,
    cli_state,
    current_cli_state,
    resolve_base_url,
    resolve_cli_workspace,
    resolve_context_headers,
    resolve_local_cli_sdks,
    resolve_output_format,
    shared_cli_client,
    shared_cli_client_context,
)
from nemo_helix_plugin.client.client import NemoClient

ClientT = TypeVar("ClientT", bound=NemoClient)


def _typer_context_with_obj(obj: object | None) -> typer.Context:
    return cast(typer.Context, SimpleNamespace(obj=obj))


class TestResolveLocalCliSdks:
    def test_returns_none_without_context_obj(self) -> None:
        assert resolve_local_cli_sdks(_typer_context_with_obj(None)) == (None, None)

    def test_uses_cli_context_client_getters(self) -> None:
        sdk = object()
        async_sdk = object()

        # Stand in for either nemo_helix_ext.cli.core.context.CLIContext
        # or its vendored nemo_helix.cli.core.context.CLIContext copy.
        class _State:
            def get_client(self) -> object:
                return sdk

            def get_async_client(self) -> object:
                return async_sdk

        assert resolve_local_cli_sdks(_typer_context_with_obj(_State())) == (sdk, async_sdk)

    def test_falls_back_to_none_when_only_one_getter_is_defined(self) -> None:
        """A state object that exposes only one getter still resolves the side it provides."""
        sdk = object()

        class _SyncOnlyState:
            def get_client(self) -> object:
                return sdk

        resolved_sdk, resolved_async_sdk = resolve_local_cli_sdks(_typer_context_with_obj(_SyncOnlyState()))
        assert resolved_sdk is sdk
        assert resolved_async_sdk is None


class _WorkspaceState:
    """Stand-in for ``CLIContext`` exposing an active workspace."""

    def __init__(self, workspace: str | None = "my-team-ws") -> None:
        self._workspace = workspace

    def get_workspace(self) -> str | None:
        return self._workspace


class _ExplodingState:
    """A state object whose workspace lookup fails (e.g. unreadable config)."""

    def get_workspace(self) -> str:
        raise RuntimeError("config is unreadable")


@pytest.fixture(autouse=True)
def _clear_workspace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these tests independent of the developer's own environment."""
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)


class TestResolveCliWorkspace:
    """Explicit-context twin: precedence and failure handling."""

    def test_explicit_wins_over_context(self) -> None:
        ctx = _typer_context_with_obj(_WorkspaceState())
        assert resolve_cli_workspace(ctx, "flag-ws") == "flag-ws"

    def test_uses_context_workspace_when_no_explicit_value(self) -> None:
        assert resolve_cli_workspace(_typer_context_with_obj(_WorkspaceState())) == "my-team-ws"

    def test_falls_back_to_default_without_state(self) -> None:
        assert resolve_cli_workspace(_typer_context_with_obj(None)) == "default"

    def test_falls_back_to_env_without_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NHX_WORKSPACE", "env-ws")
        assert resolve_cli_workspace(_typer_context_with_obj(None)) == "env-ws"

    def test_state_without_getter_falls_back(self) -> None:
        assert resolve_cli_workspace(_typer_context_with_obj(SimpleNamespace())) == "default"

    def test_empty_workspace_is_treated_as_unset(self) -> None:
        assert resolve_cli_workspace(_typer_context_with_obj(_WorkspaceState(""))) == "default"

    def test_failing_lookup_falls_back_instead_of_raising(self) -> None:
        """A broken config must not take down an otherwise-valid command."""
        assert resolve_cli_workspace(_typer_context_with_obj(_ExplodingState())) == "default"

    def test_env_still_applies_when_the_state_reports_no_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Why the resolver reads ``$NHX_WORKSPACE`` even though the state does.

        With no ``config.yaml`` on disk, ``CLIContext.get_workspace()`` swallows
        the ``FileNotFoundError`` and returns ``None`` — so nothing else has
        consulted the environment. Dropping this lookup as "redundant" would
        silently ignore ``$NHX_WORKSPACE`` in a fresh container or CI job.
        """
        monkeypatch.setenv("NHX_WORKSPACE", "env-ws")
        assert resolve_cli_workspace(_typer_context_with_obj(_WorkspaceState(None))) == "env-ws"


class _FormatState:
    """Stand-in for ``CLIContext`` whose resolved preference is fixed."""

    def __init__(self, resolved: str) -> None:
        self._resolved = resolved
        self.calls = 0

    def get_output_format(self) -> str:
        self.calls += 1
        return self._resolved


class TestCliState:
    def test_returns_the_context_obj(self) -> None:
        state = _FormatState("json")
        assert cli_state(_typer_context_with_obj(state)) is state

    def test_raises_outside_the_cli(self) -> None:
        with pytest.raises(RuntimeError, match="run the command through `nemo`"):
            cli_state(_typer_context_with_obj(None))


class _AmbientState:
    def __init__(self) -> None:
        self.client = NemoClient(
            base_url="https://platform",
            default_headers={"Authorization": "Bearer token"},
            http_client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, request=req))),
        )
        self.requests: list[tuple[type[NemoClient], float | httpx.Timeout | None]] = []

    def get_base_url(self, default: str | None = None) -> str:
        return self.client.base_url

    def typed_client(
        self,
        client_cls: type[ClientT],
        timeout: float | httpx.Timeout | None = None,
    ) -> ClientT:
        self.requests.append((client_cls, timeout))
        return client_cls.from_client(self.client)


class TestAmbientCliState:
    def test_resolves_state_from_click_context(self) -> None:
        state = _AmbientState()
        with click.Context(click.Command("cmd"), obj=state):
            assert current_cli_state() is state

    def test_resolves_and_announces_base_url_from_ambient_context(self) -> None:
        state = _AmbientState()
        with click.Context(click.Command("cmd"), obj=state):
            assert base_url_from_context() == "https://platform"
            assert resolve_base_url() == "https://platform"

    def test_shared_cli_client_uses_state_typed_client(self) -> None:
        state = _AmbientState()
        with click.Context(click.Command("cmd"), obj=state):
            client = shared_cli_client(NemoClient, timeout=12.0)

        assert client.base_url == "https://platform"
        assert state.requests == [(NemoClient, 12.0)]

    def test_shared_cli_client_requires_ambient_state(self) -> None:
        with pytest.raises(RuntimeError, match="No NeMo Helix CLI state"):
            shared_cli_client(NemoClient)

    def test_shared_cli_client_context_yields_without_owning_transport(self) -> None:
        state = _AmbientState()
        with click.Context(click.Command("cmd"), obj=state):
            with shared_cli_client_context(NemoClient) as client:
                assert client.base_url == "https://platform"

        assert state.requests == [(NemoClient, None)]

    def test_resolve_context_headers_uses_shared_client(self) -> None:
        state = _AmbientState()
        with click.Context(click.Command("cmd"), obj=state):
            assert resolve_context_headers({"X-Request-ID": "req-1"}, url="https://platform/apis") == {
                "Authorization": "Bearer token",
                "X-Request-ID": "req-1",
            }


class TestResolveOutputFormat:
    def test_explicit_wins_without_consulting_state(self) -> None:
        state = _FormatState("yaml")
        assert resolve_output_format(_typer_context_with_obj(state), "csv") == "csv"
        assert state.calls == 0

    def test_delegates_to_state_when_flag_omitted(self) -> None:
        """Agent mode, the global flag, preferences, and the non-TTY rule all live in the state."""
        assert resolve_output_format(_typer_context_with_obj(_FormatState("markdown"))) == "markdown"

    def test_without_state_uses_table_on_a_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        assert resolve_output_format(_typer_context_with_obj(None)) == "table"

    def test_without_state_uses_json_when_piped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        assert resolve_output_format(_typer_context_with_obj(None)) == "json"
