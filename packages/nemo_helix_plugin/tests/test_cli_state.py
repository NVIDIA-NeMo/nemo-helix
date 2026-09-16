# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import typer
from nemo_helix_plugin.cli_state import resolve_local_cli_sdks


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
