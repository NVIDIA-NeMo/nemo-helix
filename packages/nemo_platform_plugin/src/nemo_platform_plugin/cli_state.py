# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for plugin CLI commands that need access to the CLI's state object.

The top-level ``nemo`` CLI populates ``typer.Context.obj`` with a state object
that exposes per-invocation handles to the platform SDK clients (sync and
async). The auto-generated ``run`` / ``submit`` verbs in
:mod:`nemo_platform_plugin.commands` consume that state through these helpers,
and **plugin-authored** Typer commands (i.e. anything a plugin registers via
:meth:`~nemo_platform_plugin.cli.NemoCLI.get_cli` rather than the
auto-generated verbs) should use the same surface so they participate in the
same protocol.

Example::

    import typer
    from nemo_platform_plugin.cli_state import resolve_local_cli_sdks

    def my_command(typer_ctx: typer.Context) -> None:
        sdk, async_sdk = resolve_local_cli_sdks(typer_ctx)
        if sdk is None and async_sdk is None:
            typer.echo("No NeMo Platform SDK is available.", err=True)
            raise typer.Exit(code=1)
        ...
"""

import logging
import os
from typing import Any, cast

import click
import typer
from nemo_platform_plugin.entities import DEFAULT_WORKSPACE

logger = logging.getLogger(__name__)


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


def _workspace_from_state(state: Any) -> str | None:
    """Read the active workspace off a CLI state object, if it exposes one."""
    if state is None or not hasattr(state, "get_workspace"):
        return None
    try:
        resolved = state.get_workspace()
    except Exception:
        logger.debug("Failed to resolve workspace from CLI state", exc_info=True)
        return None
    return cast(str, resolved) if resolved else None


def resolve_cli_workspace(typer_ctx: typer.Context, explicit: str | None = None) -> str:
    """Resolve the workspace a command should act on, given an explicit context.

    An explicit ``--workspace`` wins. Otherwise the active CLI context
    decides, and it applies ``$NMP_WORKSPACE`` over its own configured
    workspace. If there is no usable context — no config file yet, or a
    plugin CLI driven outside ``nemo`` — fall back to ``$NMP_WORKSPACE``,
    then ``"default"``.

    Net user-visible order: ``--workspace`` > ``$NMP_WORKSPACE`` > the
    context's configured workspace > ``"default"``.

    The environment is therefore read in two places. That is deliberate, not
    redundant: with no config file on disk ``get_workspace()`` returns
    ``None``, and the lookup below is the only thing that honors
    ``$NMP_WORKSPACE`` — the documented way to pick a workspace in a fresh
    container or CI job.

    Use :func:`resolve_workspace` instead when the command does not already
    take a :class:`typer.Context`.
    """
    if explicit is not None:
        return explicit
    return _workspace_from_state(typer_ctx.obj) or os.environ.get("NMP_WORKSPACE") or DEFAULT_WORKSPACE


def resolve_workspace(explicit: str | None = None) -> str:
    """Ambient twin of :func:`resolve_cli_workspace`.

    Reads the *current* Click context rather than one passed in, so
    plugin-authored commands can honor the active workspace without threading
    ``typer.Context`` through every signature. ``Context.obj`` is inherited
    from parent contexts, so this resolves the same state object the top-level
    ``nemo`` callback installed.

    Same resolution as its twin. Outside a Click invocation (e.g. a direct
    unit test) there is no state object, so it falls back to
    ``$NMP_WORKSPACE`` then ``"default"`` — matching the pre-existing
    behavior of the commands that call it.

    Declare the option as ``typer.Option(None, "--workspace", ...)`` and pass
    the flag value here. A literal ``"default"`` as the Typer default is the
    bug this exists to prevent: it makes an omitted flag indistinguishable
    from an explicit one, so the active context can never win.
    """
    if explicit is not None:
        return explicit
    ctx = click.get_current_context(silent=True)
    state = ctx.obj if ctx is not None else None
    return _workspace_from_state(state) or os.environ.get("NMP_WORKSPACE") or DEFAULT_WORKSPACE
