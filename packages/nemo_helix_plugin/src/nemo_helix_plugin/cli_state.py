# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for plugin CLI commands that need access to the CLI's state object.

The top-level ``nemo`` CLI populates ``typer.Context.obj`` with a state object
that exposes per-invocation handles to the platform SDK clients (sync and
async). The auto-generated ``run`` / ``submit`` verbs in
:mod:`nemo_helix_plugin.commands` consume that state through these helpers,
and **plugin-authored** Typer commands (i.e. anything a plugin registers via
:meth:`~nemo_helix_plugin.cli.NemoCLI.get_cli` rather than the
auto-generated verbs) should use the same surface so they participate in the
same protocol.

Example::

    import typer
    from nemo_helix_plugin.cli_state import resolve_local_cli_sdks

    def my_command(typer_ctx: typer.Context) -> None:
        sdk, async_sdk = resolve_local_cli_sdks(typer_ctx)
        if sdk is None and async_sdk is None:
            typer.echo("No NeMo Helix SDK is available.", err=True)
            raise typer.Exit(code=1)
        ...
"""

import typer


def resolve_local_cli_sdks(
    typer_ctx: typer.Context,
) -> tuple[object | None, object | None]:
    """Pull ``(sdk, async_sdk)`` out of the CLI state object on ``typer_ctx.obj``.

    Returns ``(None, None)`` when no state object is set (e.g. plugin tests
    that exercise a Typer app directly without populating ``ctx.obj``). When
    a state object is set but does not implement one of the getter methods,
    that side returns ``None`` while the other still resolves — letting
    callers decide which handle they actually need.
    """
    state = typer_ctx.obj
    if not state:
        return None, None
    sdk = state.get_client() if hasattr(state, "get_client") else None
    async_sdk = state.get_async_client() if hasattr(state, "get_async_client") else None
    return sdk, async_sdk
