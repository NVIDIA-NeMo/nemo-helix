# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass request headers to the IORails main model on each call.

Engines are shared across requests, so request headers can't be built into them. If they were, a
unique header like Envoy's ``x-request-id`` would force a new engine on every request. The LLMRails
main model (``ChatNIM``) gets around this by reading headers from the request context each time it
makes a call. IORails only sets a model's headers once, when the engine is built, so we add two
hooks:

- Wrap ``ModelEngine._prepare_request`` to add the request's headers to main-model calls. Task
  models still only send their configured headers.
- Wrap each engine's admission queue so the headers reach IORails' worker tasks, which don't
  inherit the request's context. Streaming doesn't use the queue and already sees the request's
  context.

This is adapted from the IGW guardrails plugin (``nemo_guardrails_plugin.iorails_runtime``). It reads
the service's request context instead of the plugin's, and only touches the main model. Both can
wrap ``_prepare_request`` in the same process without interfering, since each only sees its own
request context.
"""

import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from nemoguardrails import Guardrails
from nemoguardrails.guardrails._http import merge_headers_case_insensitive
from nemoguardrails.guardrails.model_engine import ModelEngine
from nhx.guardrails.app.utils.context_utils import (
    get_request_default_headers_from_context,
    request_default_headers_var,
)

_T = TypeVar("_T")

_install_lock = threading.Lock()
_installed = False


def install_main_model_header_merge() -> None:
    """Add the current request's headers to IORails main-model calls. Only installs once per process."""
    global _installed
    with _install_lock:  # engines are built in worker threads
        if _installed:
            return

        original_prepare_request = getattr(ModelEngine, "_prepare_request", None)
        if original_prepare_request is None:
            raise RuntimeError(
                "Cannot send request headers to the IORails main model: ModelEngine._prepare_request is "
                "missing from the installed nemoguardrails version."
            )

        def _prepare_request(self: ModelEngine, messages: Any, **kwargs: Any) -> Any:
            params = original_prepare_request(self, messages, **kwargs)
            if getattr(self.model_config, "type", None) != "main":
                return params

            # Outside a request there's nothing to add.
            request_headers = get_request_default_headers_from_context()
            if not request_headers:
                return params

            # Request headers win over configured ones. Names are compared case-insensitively.
            return params._replace(headers=merge_headers_case_insensitive(params.headers, request_headers))

        ModelEngine._prepare_request = _prepare_request  # type: ignore[method-assign]
        _installed = True


def carry_request_headers_through_queue(engine: Guardrails) -> None:
    """Pass the caller's request headers along to work this engine runs on its worker tasks."""
    queue: Any = getattr(engine.rails_engine, "_generate_async_queue", None)
    original_submit = getattr(queue, "submit", None)
    if original_submit is None:
        raise RuntimeError(
            "Cannot send request headers to the IORails main model: the engine's admission queue is "
            "missing from the installed nemoguardrails version."
        )

    async def submit(func: Callable[..., Awaitable[_T]], *args: Any, **kwargs: Any) -> _T:
        # Grab the headers now, while we're still in the caller's context.
        request_headers = get_request_default_headers_from_context()

        async def run_with_headers(*run_args: Any, **run_kwargs: Any) -> _T:
            # Workers are long-lived and handle many requests, so set the headers for this call only
            # and reset them afterwards.
            token = request_default_headers_var.set(request_headers)
            try:
                return await func(*run_args, **run_kwargs)
            finally:
                request_default_headers_var.reset(token)

        return await original_submit(run_with_headers, *args, **kwargs)

    # Only this engine's queue is wrapped. Each engine wraps its own when it's built.
    queue.submit = submit
