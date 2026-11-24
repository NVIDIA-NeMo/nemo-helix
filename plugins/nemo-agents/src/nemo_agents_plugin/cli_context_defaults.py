# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fill ``x-helix-default`` context hints on ``nemo agents execute``.

A plugin-managed agent type can mark a context property with an
``x-helix-default`` hint, such as ``configured_default_model``. The server never
fills these in, so a run's context is always what its caller sent. The CLI
does, from its own configuration, for any hinted property the caller left out,
so a ``nemo agents execute`` whose ``--spec`` context omits a hinted model runs
with the model chosen at ``nemo setup``.

The generated ``execute`` command is kept as is; only its ``--spec`` input is
amended before the command builds the job spec from it.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from typing import Any

import typer
from nemo_helix_plugin._spec_flags import deep_merge
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.plugin_managed_agents import (
    CONFIGURED_DEFAULT_MODEL,
    CONFIGURED_FAST_MODEL,
    HELIX_DEFAULT_SCHEMA_KEY,
)
from nemo_helix_plugin.cli_state import resolve_cli_workspace, shared_cli_client
from nemo_helix_plugin.client.config.config import get_context
from nemo_helix_plugin.jobs._cli_options import load_spec_file
from nemo_helix_plugin.refs import parse_entity_ref

logger = logging.getLogger(__name__)

_HINT_DESCRIPTIONS = {
    CONFIGURED_DEFAULT_MODEL: "the configured default model",
    CONFIGURED_FAST_MODEL: "the configured fast model",
}


def fill_context_defaults_on_submit(group: typer.Typer) -> None:
    """Wrap *group*'s generated submit callback so it fills hinted context defaults first.

    The wrapper keeps the generated callback's signature (``functools.wraps``),
    so Typer builds exactly the same flags.
    """
    callback_info = group.registered_callback
    original = callback_info.callback if callback_info is not None else None
    if callback_info is None or original is None:
        raise RuntimeError("missing generated execute submit callback to wrap")

    @functools.wraps(original)
    def submit(*args: Any, **kwargs: Any) -> Any:
        typer_ctx = kwargs.get("typer_ctx", args[0] if args else None)
        if isinstance(typer_ctx, typer.Context) and typer_ctx.invoked_subcommand is None:
            kwargs["spec"] = spec_with_context_defaults(typer_ctx, kwargs, get_agent=_get_agent_context_schema)
        return original(*args, **kwargs)

    callback_info.callback = submit


def spec_with_context_defaults(
    typer_ctx: typer.Context,
    kwargs: dict[str, Any],
    *,
    get_agent: Callable[[str, str], dict[str, Any] | None],
) -> str:
    """Return the submit's ``--spec`` JSON with hinted context defaults added.

    Only properties the caller left out of both ``--spec-file`` and ``--spec``
    are added, so an explicit value always wins. Anything this cannot work out
    (no agent ref, an agent it cannot read, malformed input) leaves ``--spec``
    as it was: the generated command and the server report those properly.
    *get_agent* returns an agent's ``context_schema`` given ``(workspace, name)``.
    """
    spec_str = kwargs.get("spec") or "{}"
    # The deprecated ``--config`` / ``--config-file`` aliases replace --spec outright; leave them alone.
    if kwargs.get("config") is not None or kwargs.get("config_file") is not None:
        return spec_str
    try:
        overlay = json.loads(spec_str)
        base = load_spec_file(kwargs["spec_file"]) if kwargs.get("spec_file") is not None else {}
    except (OSError, ValueError):
        return spec_str
    if not isinstance(overlay, dict) or not isinstance(base, dict):
        return spec_str
    merged = deep_merge(base, overlay)

    agent = kwargs.get("agent") if isinstance(kwargs.get("agent"), str) else merged.get("agent")
    if not isinstance(agent, str) or agent.lstrip().startswith("{"):
        return spec_str
    context = merged.get("context", {})
    if not isinstance(context, dict):
        return spec_str

    workspace = resolve_cli_workspace(typer_ctx, kwargs.get("workspace"))
    try:
        ref = parse_entity_ref(agent, default_workspace=workspace)
    except ValueError:
        return spec_str
    schema = get_agent(ref.workspace, ref.name)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return spec_str

    filled: dict[str, str] = {}
    for key, prop in properties.items():
        hint = prop.get(HELIX_DEFAULT_SCHEMA_KEY) if isinstance(prop, dict) else None
        if key in context or not isinstance(hint, str):
            continue
        value = _hint_value(hint)
        if value is None:
            continue
        filled[key] = value
        typer.echo(f"Using {_HINT_DESCRIPTIONS[hint]} for context.{key}: {value}", err=True)
    if not filled:
        return spec_str
    return json.dumps(deep_merge(overlay, {"context": filled}))


def _hint_value(hint: str) -> str | None:
    """Resolve a hint from the CLI's configuration, or ``None`` when it has no value for it."""
    if hint not in _HINT_DESCRIPTIONS:
        return None
    try:
        context = get_context()
    except Exception:
        logger.debug("No CLI context to fill %s from.", hint, exc_info=True)
        return None
    if hint == CONFIGURED_DEFAULT_MODEL:
        return context.default_model or None
    return context.fast_model or context.default_model or None


def _get_agent_context_schema(workspace: str, name: str) -> dict[str, Any] | None:
    try:
        agent = shared_cli_client(AgentsClient).get_agent(workspace=workspace, name=name).data()
    except Exception:
        logger.debug("Could not read agent %s/%s to fill its context defaults.", workspace, name, exc_info=True)
        return None
    return agent.context_schema
