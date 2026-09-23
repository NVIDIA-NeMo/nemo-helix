# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import typer
from nemo_helix_plugin.cli_state import resolve_cli_workspace, resolve_local_cli_sdks


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
