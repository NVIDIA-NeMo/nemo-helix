# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI-context resolution for the ``nemo agents`` command group.

The agents plugin makes platform calls from two places — the platform
commands in :mod:`nemo_agents_plugin.cli` and ``nemo agents usage show`` in
:mod:`nemo_agents_plugin.usage.cli`.  Both must resolve the platform base URL
and the auth token the same way every other ``nemo`` command does: through the
shared CLI context object stored on ``typer.Context.obj`` (the global
``nemo --base-url`` flag, ``NHX_BASE_URL``, and the active context).

These helpers read the *ambient* Click context so callers deep in a command's
call stack can resolve configuration without threading the context object
through every function signature.  They live in their own module (rather than
in ``cli.py``) because ``cli.py`` imports the usage CLI at module load, so a
back-import would create a cycle.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import click
from nemo_helix_plugin.client.client import NemoClient

logger = logging.getLogger(__name__)

ClientT = TypeVar("ClientT", bound=NemoClient)

DEFAULT_BASE_URL = "http://localhost:8080"


def current_cli_state() -> Any:
    """Return the shared CLI context object (``typer.Context.obj``) if present.

    Returns ``None`` when the plugin is exercised outside a Click invocation
    (e.g. a direct unit test), so callers fall back to their own defaults.
    """
    ctx = click.get_current_context(silent=True)
    return ctx.obj if ctx is not None else None


def base_url_from_context() -> str | None:
    """Return the base URL configured in the shared CLI context, if any."""
    state = current_cli_state()
    if state is None or not hasattr(state, "get_base_url"):
        return None
    try:
        return state.get_base_url(default=None)
    except Exception:
        logger.debug("Failed to resolve base URL from CLI context", exc_info=True)
        return None


def resolve_base_url() -> str:
    """Resolve the platform base URL and announce the target on stderr.

    The shared CLI context decides (``nemo --base-url``, ``NHX_BASE_URL``,
    ``nemo config set --base-url``), so ``nemo agents`` targets the same
    platform as every other ``nemo`` command. Outside ``nemo`` (no context),
    the built-in localhost default applies.

    The resolved target is echoed to stderr (never stdout, so piped/JSON
    output stays clean) so a mis-pointed command is visible instead of
    silently hitting the wrong platform.
    """
    resolved = base_url_from_context() or DEFAULT_BASE_URL
    click.echo(f"Targeting {resolved}", err=True)
    return resolved


def shared_cli_client(client_cls: type[ClientT], *, timeout: float = 60.0) -> ClientT | None:
    """Return *client_cls* built on the ``nemo`` CLI's shared client, or ``None`` outside ``nemo``.

    The shared client carries the resolved base URL and the context's auth
    (with token refresh), exactly as core commands use it.
    """
    state = current_cli_state()
    if state is None or not hasattr(state, "typed_client"):
        return None
    return state.typed_client(client_cls, timeout=timeout)


def resolve_context_headers() -> dict[str, str]:
    """Return auth (and other) default headers from the shared CLI context.

    Mirrors ``nemo_helix_plugin.commands.resolve_submit_auth_headers``:
    reads the SDK client config off the shared context so ``nemo agents``
    attaches the same ``Authorization: Bearer`` token as the rest of the CLI
    (i.e. the token established by ``nemo auth login``).  Returns an empty
    mapping when no context or token is available — leaving requests
    unauthenticated exactly as before, so local unauthenticated dev keeps
    working.
    """
    state = current_cli_state()
    if state is None or not hasattr(state, "get_sdk_context"):
        return {}
    try:
        client_config = state.get_sdk_context().user.get_client_config()
    except Exception:
        logger.debug("Failed to resolve auth headers from CLI context", exc_info=True)
        return {}
    headers = client_config.get("default_headers") if isinstance(client_config, dict) else None
    if isinstance(headers, dict):
        return {str(key): str(value) for key, value in headers.items()}
    return {}
