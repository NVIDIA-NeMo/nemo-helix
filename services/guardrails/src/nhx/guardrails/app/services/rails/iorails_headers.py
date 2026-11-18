# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send the current request's headers to the IORails main model on every call.

Engines are cached and shared across requests, so a request's headers can't be built into them: a
unique header per request (such as the ``x-request-id`` Envoy adds) would rebuild the engine every
time. The LLMRails main model (``ChatNIM``) reads them from the request context at call time
instead. IORails fixes a model's headers when the engine is built and has no per-call hook, so two
shims give its main model the same per-call behavior:

- ``ModelEngine._prepare_request`` is wrapped to merge the request's headers into main-model calls.
  Task models keep only their configured headers, as documented for task model calls.
- IORails runs non-streaming work on worker tasks that don't inherit the request's context, so each
  engine's admission queue is wrapped to carry the headers across. Streaming starts its work with
  ``create_task``, which already inherits the context.

The IGW guardrails plugin does the same for its engines
(``nemo_guardrails_plugin.iorails_runtime``); this copy reads the service's request context and
applies to the main model only. Both wrappers can sit on ``_prepare_request`` in one process: each
reads its own request context, which is empty during the other's requests.
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
    """Merge the in-scope request headers into every IORails main-model call (once per process)."""
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

            # Nothing in scope (a call outside a request): leave the model's own headers as they are.
            request_headers = get_request_default_headers_from_context()
            if not request_headers:
                return params

            # The request's headers override the model's configured ones, matched case-insensitively.
            return params._replace(headers=merge_headers_case_insensitive(params.headers, request_headers))

        ModelEngine._prepare_request = _prepare_request  # type: ignore[method-assign]
        _installed = True


def carry_request_headers_through_queue(engine: Guardrails) -> None:
    """Make the submitter's request headers visible to work the engine runs on its own tasks."""
    queue: Any = getattr(engine.rails_engine, "_generate_async_queue", None)
    original_submit = getattr(queue, "submit", None)
    if original_submit is None:
        raise RuntimeError(
            "Cannot send request headers to the IORails main model: the engine's admission queue is "
            "missing from the installed nemoguardrails version."
        )

    async def submit(func: Callable[..., Awaitable[_T]], *args: Any, **kwargs: Any) -> _T:
        # Read the headers here, in the submitter's context: the queue's workers run in their own.
        request_headers = get_request_default_headers_from_context()

        async def run_with_headers(*run_args: Any, **run_kwargs: Any) -> _T:
            # Runs on a long-lived worker task that serves many requests in turn: put the submitter's
            # headers in scope for this call only, so they never reach the worker's next request.
            token = request_default_headers_var.set(request_headers)
            try:
                return await func(*run_args, **run_kwargs)
            finally:
                request_default_headers_var.reset(token)

        return await original_submit(run_with_headers, *args, **kwargs)

    # Wrap this engine's queue only; other engines get their own wrapper when they are built.
    queue.submit = submit
