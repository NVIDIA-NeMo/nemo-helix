# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request glue between the IGW source pipeline and the cached LLMRails pool.

Owns:

- :func:`build_main_llm` — the per-request main LLM build. The only piece
  that can't be cached, since the main model name comes from the request body.
- :func:`run_generate_in_new_loop` — runs :meth:`LLMRails.generate_async`
  on a worker thread with its own event loop.
- :class:`GenerationResponse` → :class:`GenerationLog` munging for the
  response builders.

Source introspection, stabilization, and the pool itself live in
:mod:`nemo_guardrails_plugin.llmrails_cache`.
"""

import asyncio
import logging
from typing import Any

from nemo_guardrails_plugin.constants import DEFAULT_MAIN_ENGINE, W3C_TRACE_CONTEXT_HEADERS
from nemo_guardrails_plugin.llmrails_cache import InferenceTargetResolver
from nemo_guardrails_plugin.transforms import GenerationResponseMapper
from nemo_platform_plugin.guardrail.types import GenerationLog, GenerationLogOptionsParam, GuardrailsData
from nemo_platform_plugin.guardrail.types import (
    GenerationStats as PlatformGenerationStats,
)
from nemoguardrails.llm.models.initializer import init_llm_model
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse
from nemoguardrails.types import LLMModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-request execution
# ---------------------------------------------------------------------------


def run_generate_in_new_loop(
    llm_rails: LLMRails,
    messages: list[dict] | None = None,
    prompt: str | None = None,
    options: dict | GenerationOptions | None = None,
    state: dict | None = None,
):
    """Run :meth:`LLMRails.generate_async` in a new event loop on the current thread.

    Synchronous on purpose: invoke via :func:`asyncio.to_thread` so the
    CPU-heavy Colang walk runs off the IGW main loop (holding the main loop
    for this long can stall liveness probes and get the pod killed).

    ``shutdown_default_executor`` is load-bearing: without it each request
    leaves idle worker threads behind that then compete with the next
    request's ``LLMRails.__init__`` for the GIL.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            llm_rails.generate_async(
                messages=messages,
                prompt=prompt,
                options=options,
                state=state,
            )
        )
    finally:
        try:
            # Cancel aiohttp/httpx cleanup tasks created during generate_async.
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def build_generate_async_options(
    rail_types: list[str], user_log_options: GenerationLogOptionsParam | None
) -> dict[str, Any]:
    """Build the options passed to :meth:`LLMRails.generate_async`.

    Forces ``activated_rails`` on so the response middleware can detect
    blocked responses; adds any user-requested log flags on top.
    """
    log_options = dict(user_log_options or {})
    log_options["activated_rails"] = True
    return {"rails": rail_types, "log": log_options}


# ---------------------------------------------------------------------------
# Generation log shaping
# ---------------------------------------------------------------------------


def _build_generation_stats(
    input_stats: PlatformGenerationStats | None,
    output_stats: PlatformGenerationStats | None,
) -> PlatformGenerationStats:
    input_stats_dict = input_stats.model_dump() if input_stats else {}
    output_stats_dict = output_stats.model_dump() if output_stats else {}

    def _sum_float(key: str) -> float | None:
        input_value, output_value = input_stats_dict.get(key), output_stats_dict.get(key)
        if input_value is None and output_value is None:
            return None
        return float(input_value or 0) + float(output_value or 0)

    def _sum_int(key: str) -> int | None:
        input_value, output_value = input_stats_dict.get(key), output_stats_dict.get(key)
        if input_value is None and output_value is None:
            return None
        return int(input_value or 0) + int(output_value or 0)

    return PlatformGenerationStats(
        input_rails_duration=input_stats_dict.get("input_rails_duration"),
        output_rails_duration=output_stats_dict.get("output_rails_duration"),
        total_duration=_sum_float("total_duration"),
        llm_calls_duration=_sum_float("llm_calls_duration"),
        llm_calls_count=_sum_int("llm_calls_count"),
        llm_calls_total_prompt_tokens=_sum_int("llm_calls_total_prompt_tokens"),
        llm_calls_total_completion_tokens=_sum_int("llm_calls_total_completion_tokens"),
        llm_calls_total_tokens=_sum_int("llm_calls_total_tokens"),
    )


def _build_generation_log(
    user_log_options: GenerationLogOptionsParam | None,
    input_generation_response: GenerationResponse | None = None,
    output_generation_response: GenerationResponse | None = None,
) -> GenerationLog | None:
    """Merge input and output rail logs into a :class:`GenerationLog`.

    Only includes fields the user asked for. Returns ``None`` when no log
    fields were requested.
    """
    if not user_log_options:
        return None

    input_log = GenerationResponseMapper.to_platform_generation_log(
        input_generation_response.log if input_generation_response else None
    )
    output_log = GenerationResponseMapper.to_platform_generation_log(
        output_generation_response.log if output_generation_response else None
    )

    activated_rails = None
    if user_log_options.get("activated_rails"):
        input_activated_rails = input_log.activated_rails if input_log and input_log.activated_rails else []
        output_activated_rails = output_log.activated_rails if output_log and output_log.activated_rails else []
        activated_rails = input_activated_rails + output_activated_rails

    llm_calls = None
    if user_log_options.get("llm_calls"):
        input_llm_calls = input_log.llm_calls if input_log and input_log.llm_calls else []
        output_llm_calls = output_log.llm_calls if output_log and output_log.llm_calls else []
        llm_calls = input_llm_calls + output_llm_calls

    internal_events = None
    if user_log_options.get("internal_events"):
        input_internal_events = input_log.internal_events if input_log and input_log.internal_events else []
        output_internal_events = output_log.internal_events if output_log and output_log.internal_events else []
        internal_events = input_internal_events + output_internal_events

    stats = None
    if user_log_options.get("stats"):
        input_stats = input_log.stats if input_log else None
        output_stats = output_log.stats if output_log else None
        stats = _build_generation_stats(input_stats, output_stats)

    colang_history = output_log.colang_history if user_log_options.get("colang_history") and output_log else None

    return GenerationLog(
        activated_rails=activated_rails,
        llm_calls=llm_calls,
        internal_events=internal_events,
        stats=stats,
        colang_history=colang_history,
    )


def build_guardrails_data(
    config_id: str,
    input_generation_response: GenerationResponse | None = None,
    output_generation_response: GenerationResponse | None = None,
    user_log_options: GenerationLogOptionsParam | None = None,
) -> GuardrailsData:
    """Build the :class:`GuardrailsData` to inject into the response body.

    ``log`` is only populated when the caller requested it and there is
    at least one rail response to report.
    """
    if not input_generation_response and not output_generation_response:
        return GuardrailsData(config_ids=[config_id])

    if not user_log_options or not any(user_log_options.values()):
        return GuardrailsData(config_ids=[config_id])

    output_log = _build_generation_log(user_log_options, input_generation_response, output_generation_response)

    return GuardrailsData(config_ids=[config_id], log=output_log)


# ---------------------------------------------------------------------------
# Per-request main LLM construction
# ---------------------------------------------------------------------------


def build_main_llm(
    request_body: dict[str, Any],
    request_headers: dict[str, str],
    resolve_inference_target: InferenceTargetResolver,
    main_model_template: Model | None = None,
) -> LLMModel:
    """Construct the per-request main LangChain LLM client.

    Injected into the cached :class:`LLMRails` via ``rails.update_llm``.
    ``main_model_template=None`` is production-normal — IGW owns
    main-LLM routing and configs typically declare only task LLMs
    (content-safety, topic-control, embeddings).

    Precedence:

    - engine: ``main_model_template.engine`` else :data:`DEFAULT_MAIN_ENGINE`.
    - model name: always ``request_body["model"]``.
    - ``base_url``: config ``parameters.base_url`` else the IGW gateway URL.
    - ``default_headers``: static config headers ∪ allowlisted request
      headers — ``x-*`` (NeMo Platform principal, ``x-otel-*``, custom) plus W3C
      Trace Context (``traceparent``, ``tracestate``, ``baggage``).

    Synchronous on purpose (``init_llm_model`` does a blocking LangChain
    import); callers should run it on a worker thread. Raises
    :class:`ValueError` when ``request_body`` lacks a non-empty string
    ``model`` — the middleware maps this to a 400 at the plugin boundary.
    """
    request_model_name = request_body.get("model")
    # Reject non-strings here so they 4xx instead of failing deep inside
    # init_llm_model's provider lookup.
    if not request_model_name or not isinstance(request_model_name, str):
        raise ValueError("request body must include a non-empty string 'model'")

    engine = DEFAULT_MAIN_ENGINE
    static_params: dict[str, Any] = {}
    if main_model_template is not None:
        engine = main_model_template.engine
        # IGW only routes chat-completions; surface the mismatch instead
        # of silently overriding so misconfigured deployments don't
        # debug a phantom ``mode`` value. ``init_llm_model`` is still
        # invoked with ``mode='chat'`` below.
        if main_model_template.mode != "chat":
            logger.warning(
                "Main model in guardrails config sets mode=%r; IGW only "
                "supports chat completions — forcing mode='chat'.",
                main_model_template.mode,
            )
        # Shallow copy so a future ``static_params["foo"] = ...`` edit
        # can't mutate the cached template across requests.
        static_params = dict(main_model_template.parameters)
    else:
        # Flag the implicit default so deployments targeting non-NIM
        # providers notice before they hit a confusing 4xx.
        logger.debug(
            "No 'main' model entry in guardrails config; defaulting engine=%r for request model %r",
            DEFAULT_MAIN_ENGINE,
            request_model_name,
        )
    base_url = static_params.get("base_url")
    if not base_url or not isinstance(base_url, str):
        target = resolve_inference_target(request_model_name)
        base_url = target.openai_base_url

    static_headers = dict(static_params.get("default_headers") or {})
    # Forward ``x-*`` (NeMo Platform principal, ``x-otel-*``, custom) and W3C
    # Trace Context (``traceparent``, ``tracestate``, ``baggage``) so
    # tracing survives even when the upstream emits standard-form
    # headers. ``Authorization`` and other non-allowlisted headers are
    # intentionally dropped — IGW handles auth.
    dynamic_headers = {
        k: v for k, v in request_headers.items() if k.lower().startswith("x-") or k.lower() in W3C_TRACE_CONTEXT_HEADERS
    }

    kwargs: dict[str, Any] = {k: v for k, v in static_params.items() if k not in ("base_url", "default_headers")}
    kwargs["base_url"] = base_url
    kwargs["default_headers"] = {**static_headers, **dynamic_headers}

    return init_llm_model(
        model_name=request_model_name,
        provider_name=engine,
        mode="chat",
        kwargs=kwargs,
    )
