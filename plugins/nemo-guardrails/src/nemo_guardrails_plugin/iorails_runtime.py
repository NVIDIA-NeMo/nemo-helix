# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in IORails engine for the input and output rail checks, streaming or not.

IORails is a lighter alternative to LLMRails: it supports a subset of rails and calls models over
plain HTTP, without the per-request Colang runtime. A request is routed here only when
``NEMO_GUARDRAILS_IORAILS_ENGINE`` is set and IORails can serve it; everything else (unsupported
flows, requests for detailed logs) stays on LLMRails.

Request-scoped platform headers
-------------------------------
LLMRails clients read the caller's service-principal and tracing headers from a ``ContextVar`` at
call time (see :mod:`nemo_guardrails_plugin.llm_clients`). IORails fixes a model's headers when its
engine is built, and engines are cached per config, so they can't carry per-request identity. Two
shims close that gap:

- ``ModelEngine._prepare_request`` is wrapped to merge the context's headers into every outbound
  rail-model call.
- IORails runs each check on a worker task that doesn't inherit the submitter's context, so the
  engine's admission queue is wrapped to carry the headers across.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, TypeVar

from nemo_guardrails_plugin.llm_clients import _request_headers_ctx, get_request_headers
from nemo_guardrails_plugin.llmrails_cache import StableRailsConfig
from nemoguardrails import Guardrails
from nemoguardrails.guardrails._http import merge_headers_case_insensitive
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    GenerationLog,
    GenerationResponse,
    RailsResult,
    RailStatus,
    RailType,
)

logger = logging.getLogger(__name__)

IORAILS_ENGINE_ENV = "NEMO_GUARDRAILS_IORAILS_ENGINE"
"""Same variable nemoguardrails reads, so one switch controls both the library and this plugin."""

_TRUE_VALUES = frozenset({"1", "true", "yes"})

_T = TypeVar("_T")

_HEADER_PROPAGATION_INSTALLED = False


def iorails_enabled() -> bool:
    """Whether the operator opted in to IORails for requests it can serve."""
    return os.environ.get(IORAILS_ENGINE_ENV, "").strip().lower() in _TRUE_VALUES


def _install_model_engine_header_merge() -> None:
    """Merge the in-scope request headers into every IORails model call (once per process)."""
    global _HEADER_PROPAGATION_INSTALLED
    if _HEADER_PROPAGATION_INSTALLED:
        return

    original_prepare_request = getattr(ModelEngine, "_prepare_request", None)
    if original_prepare_request is None:
        raise RuntimeError(
            "Cannot propagate request headers through IORails: ModelEngine._prepare_request is missing "
            "from the installed nemoguardrails version."
        )

    def _prepare_request(self: ModelEngine, messages: Any, **kwargs: Any) -> Any:
        params = original_prepare_request(self, messages, **kwargs)

        # Nothing in scope (e.g. a call outside a request): leave the model's own headers as they are.
        request_headers = get_request_headers()
        if not request_headers:
            return params

        # The request's headers override the model's static ones, matched case-insensitively.
        return params._replace(headers=merge_headers_case_insensitive(params.headers, request_headers))

    # Patch the class once so every engine built afterwards picks it up.
    ModelEngine._prepare_request = _prepare_request  # type: ignore[method-assign]
    _HEADER_PROPAGATION_INSTALLED = True


def _carry_request_headers_through_queue(engine: Guardrails) -> None:
    """Make the submitter's request headers visible to work the engine runs on its own tasks."""
    queue: Any = getattr(engine.rails_engine, "_generate_async_queue", None)
    original_submit = getattr(queue, "submit", None)
    if original_submit is None:
        raise RuntimeError(
            "Cannot propagate request headers through IORails: the engine's admission queue is missing "
            "from the installed nemoguardrails version."
        )

    async def submit(func: Callable[..., Awaitable[_T]], *args: Any, **kwargs: Any) -> _T:
        # Read the headers here, in the submitter's context: the queue's workers run in their own.
        request_headers = get_request_headers()

        async def run_with_headers(*run_args: Any, **run_kwargs: Any) -> _T:
            # Runs on a worker task: put the submitter's headers in scope for the duration of the call.
            token = _request_headers_ctx.set(request_headers)
            try:
                return await func(*run_args, **run_kwargs)
            finally:
                _request_headers_ctx.reset(token)

        return await original_submit(run_with_headers, *args, **kwargs)

    # Wrap this engine's queue only; other engines get their own wrapper when they are built.
    queue.submit = submit


def _to_generation_response(result: RailsResult, rail_type: RailType) -> GenerationResponse:
    """Shape an IORails verdict like the LLMRails result the middleware already consumes.

    The middleware treats ``activated_rails[*].stop`` as "blocked" and the assistant message as the
    post-rail text (the refusal when blocked, the possibly rewritten text otherwise). IORails only
    reports which rail blocked, so a pass carries an empty rail list rather than an invented rail.
    """
    activated_rails = (
        [ActivatedRail(type=rail_type.value, name=result.rail or rail_type.value, stop=True)]
        if result.status is RailStatus.BLOCKED
        else []
    )
    return GenerationResponse(
        response=[{"role": "assistant", "content": result.content}],
        log=GenerationLog(activated_rails=activated_rails),
    )


class IORailsRuntime:
    """Owns one started IORails engine per stabilized config.

    Engines hold HTTP clients and an admission queue, so they are built once per config content hash,
    shared across concurrent requests, and shut down with the middleware.
    """

    def __init__(self) -> None:
        self._engines: dict[str, Guardrails] = {}
        self._unsupported_reasons: dict[str, str | None] = {}
        self._build_lock = asyncio.Lock()
        self._closed = False

    def unsupported_reason(self, stable: StableRailsConfig) -> str | None:
        """Why IORails cannot serve ``stable``, or None when it can.

        Memoized per content hash: the answer depends only on the config, and the check compiles
        the configured rails.
        """
        # First time this config is seen: ask IORails once, and warn once if it can't serve it.
        if stable.content_hash not in self._unsupported_reasons:
            reason = IORails.unsupported_reason(stable.rails)
            self._unsupported_reasons[stable.content_hash] = reason
            if reason is not None:
                logger.warning(
                    "IORails is enabled but config %s stays on LLMRails: %s", stable.content_hash[:12], reason
                )

        return self._unsupported_reasons[stable.content_hash]

    async def check(
        self,
        stable: StableRailsConfig,
        *,
        messages: list[dict[str, Any]],
        rail_types: Sequence[str],
    ) -> GenerationResponse:
        """Run exactly one rail family (input or output) without any main-model generation.

        Callers must have confirmed :meth:`unsupported_reason` is None and must have the request's
        platform headers in scope (``platform_headers_context``).
        """
        if len(rail_types) != 1:
            raise ValueError("IORails checks run exactly one rail type at a time.")
        rail_type = RailType(rail_types[0])

        engine = await self._get_or_build(stable)
        result = await engine.check_async(messages, rail_types=[rail_type])
        return _to_generation_response(result, rail_type)

    async def stream_output_rails(
        self,
        stable: StableRailsConfig,
        token_stream: AsyncIterator[str],
        *,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Guard a token stream that is already running with the config's output rails.

        IGW relays the response from the upstream model, so IORails doesn't generate it. Buffering,
        ``stream_first`` and block behavior match a response IORails generated itself; a block arrives
        as a JSON string with a top-level ``error``.

        The caller's platform headers must be in scope while this is iterated.
        """
        from nemoguardrails.guardrails.iorails import _TurnConversation

        engine = await self._get_or_build(stable)
        run_output_rails = getattr(engine.rails_engine, "_run_output_rails_in_streaming", None)
        if run_output_rails is None:
            raise RuntimeError(
                "Cannot guard a streamed response with IORails: its output-rail stage is missing from the "
                "installed nemoguardrails version."
            )

        # Run all of the config's output rails over the stream IGW is relaying.
        guarded = run_output_rails(token_stream, _TurnConversation(messages=messages), enabled=True)
        try:
            async for chunk in guarded:
                yield chunk
        finally:
            # Release the guarded stream even if the consumer stops early.
            await guarded.aclose()

    async def _get_or_build(self, stable: StableRailsConfig) -> Guardrails:
        # Fast path: this config already has a running engine.
        engine = self._engines.get(stable.content_hash)
        if engine is not None:
            return engine

        async with self._build_lock:
            # Another request may have built it while this one waited for the lock.
            engine = self._engines.get(stable.content_hash)
            if engine is not None:
                return engine
            if self._closed:
                raise RuntimeError("IORails runtime is closed.")

            # The header shims must be in place before the engine serves its first request.
            _install_model_engine_header_merge()
            engine = Guardrails(stable.rails, require_iorails=True)
            try:
                _carry_request_headers_through_queue(engine)
                await engine.startup()
            except BaseException:
                # Don't leave a half-started engine behind: nothing is cached, so a retry starts clean.
                await engine.shutdown()
                raise

            self._engines[stable.content_hash] = engine
            logger.info("Built IORails engine for config %s", stable.content_hash[:12])

            return engine

    async def close(self) -> None:
        """Stop every engine and reject further checks."""
        # Under the lock, so no engine can be built while the runtime is shutting down.
        async with self._build_lock:
            if self._closed:
                return
            self._closed = True
            engines = list(self._engines.values())
            self._engines.clear()

        # Stop engines concurrently and report every failure, not just the first.
        results = await asyncio.gather(*(engine.shutdown() for engine in engines), return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]

        if errors:
            raise BaseExceptionGroup("Failed to stop IORails engines", errors)
