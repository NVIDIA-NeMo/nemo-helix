# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chooses the rails engine (LLMRails or IORails) for a config.

LLMRails is the default. With ``NEMO_GUARDRAILS_IORAILS_ENGINE`` set, IORails is used for the configs
it supports. The library can also swap ``LLMRails`` for IORails at import time when that variable is
set; choosing here instead keeps the engine independent of import order and lets unsupported configs
stay on LLMRails.
"""

import asyncio
import logging
import os
from typing import TypeGuard, Union

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.rails.llm.llmrails import LLMRails
from nhx.guardrails.app.constants import NIM_LLM
from nhx.guardrails.app.utils.context_utils import get_x_model_auth_token_from_context

logger = logging.getLogger(__name__)

IORAILS_ENGINE_ENV = "NEMO_GUARDRAILS_IORAILS_ENGINE"
"""Same variable nemoguardrails reads, so one switch controls both the service and the library."""

_TRUE_VALUES = frozenset({"1", "true", "yes"})

RailsEngine = Union[LLMRails, Guardrails]
"""Either engine. ``Guardrails`` fronts IORails and exposes the LLMRails surface the service uses."""


def iorails_enabled() -> bool:
    """Whether the operator opted in to IORails for configs it can serve."""
    return os.environ.get(IORAILS_ENGINE_ENV, "").strip().lower() in _TRUE_VALUES


def _iorails_unsupported_reason(config: RailsConfig) -> str | None:
    """Why IORails cannot serve ``config``, or None when it can.

    Beyond the library's own flow checks, IORails only speaks the chat-completions API, so a
    config whose models use the text-completions engine must stay on LLMRails.
    """
    if any(model.engine == NIM_LLM for model in config.models):
        return "config uses a text-completions model; IORails supports chat completions only"

    return IORails.unsupported_reason(config)


def uses_iorails(engine: RailsEngine) -> TypeGuard[Guardrails]:
    """Whether ``engine`` is backed by IORails rather than LLMRails."""
    return isinstance(engine, Guardrails) and engine.use_iorails_engine


_retiring: set[asyncio.Task] = set()
"""Strong references so shutdown tasks are not garbage collected before they finish."""


def retire_rails_engine(engine: RailsEngine) -> None:
    """Stop an IORails engine the registry no longer serves.

    Cached engines are replaced when a config or its credentials change. IORails holds connection
    pools and queue workers, so dropping the reference would leak them. Shutdown drains queued work
    first, so in-flight requests still complete. Must run on the server event loop; outside one the
    engine is left to garbage collection. LLMRails has nothing to release.
    """
    if not uses_iorails(engine):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("No running event loop to stop a retired IORails engine; leaving it to garbage collection")
        return
    task = loop.create_task(_shutdown_quietly(engine))
    _retiring.add(task)
    task.add_done_callback(_retiring.discard)


async def _shutdown_quietly(engine: Guardrails) -> None:
    try:
        await engine.shutdown()
    except Exception:
        logger.warning("Failed to stop a retired IORails engine", exc_info=True)


def _with_caller_main_model_key(config: RailsConfig) -> RailsConfig:
    """Send the caller's ``X-Model-Authorization`` key to the main model, as LLMRails does.

    IORails reads an API key only from an environment variable, while LLMRails takes this one from the
    request. The key goes in the main model's ``default_headers``, which IORails sends with every call.
    The registry's cache key includes the key, so each caller gets its own engine. A model that already
    sets ``Authorization`` (platform auth for Inference Gateway models) keeps it.
    """
    token = get_x_model_auth_token_from_context()
    if not token:
        return config

    models = []
    for model in config.models:
        parameters = dict(model.parameters or {})
        headers = dict(parameters.get("default_headers") or {})
        if model.type == "main" and not any(name.lower() == "authorization" for name in headers):
            parameters["default_headers"] = {**headers, "Authorization": f"Bearer {token}"}
            model = model.model_copy(update={"parameters": parameters})
        models.append(model)

    return config.model_copy(update={"models": models})


def create_rails_engine(config: RailsConfig) -> RailsEngine:
    """Build the rails engine for ``config``.

    IORails when it is enabled and supports the config; otherwise LLMRails, with a warning if IORails
    was enabled but can't serve it.

    Blocking: LLMRails construction does file I/O, so run this in a worker thread from an event loop.
    """
    if not iorails_enabled():
        return LLMRails(config)

    reason = _iorails_unsupported_reason(config)
    if reason is not None:
        logger.warning("IORails is enabled but this config stays on LLMRails: %s", reason)
        return LLMRails(config)

    # IORails starts its HTTP clients lazily on the first request, so nothing to await here.
    return Guardrails(_with_caller_main_model_key(config), require_iorails=True)
