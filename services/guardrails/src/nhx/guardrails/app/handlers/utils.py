# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional, Type, Union

from nemoguardrails import RailsConfig as InternalSdkRailsConfig
from nemoguardrails.rails.llm.config import _get_flow_name
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    GenerationLog,
    GenerationResponse,
    RailsResult,
    RailStatus,
    RailType,
)
from nhx.guardrails.app.constants import NIM_CHAT, NIM_LLM
from nhx.guardrails.app.services.rails.engine import RailsEngine, uses_iorails
from nhx.guardrails.app.utils.context_utils import (
    get_request_default_headers_from_context,
    set_request_default_headers_into_context,
)
from nhx.guardrails.app.utils.platform_request_headers import get_platform_auth_headers_from_context
from nhx.guardrails.config import settings
from nhx.guardrails.entities.values._private import Model, ModelParameters, RailsConfig
from nhx.guardrails.entities.values.chat import GuardrailChatCompletionRequest
from nhx.guardrails.entities.values.check import GuardrailCheckRequest
from nhx.guardrails.entities.values.completions import GuardrailCompletionRequest


def to_internal_rails_config(config: RailsConfig) -> InternalSdkRailsConfig:
    """Convert our RailsConfig model to the nemoguardrails' RailsConfig model.
    This is needed to instantiate the LLMRails class using the expected Pydantic model."""
    return InternalSdkRailsConfig.parse_obj(config.model_dump(exclude_none=True))


def get_model_config_object(schema_type: Union[str, Type], model_name: str, type: str, **kwargs):
    """Return the model config object based on the schema type.

    For NIM, we need to use different LLM engines depending on the schema of the request.

    Mapped schema types:
    - GuardrailChatCompletionRequest
    - GuardrailCheckRequest
    - GuardrailCompletionRequest

    Args:
        schema_type: The schema type to use.
        model_name: The model name.
        type: The type of the model (e.g., main, llama_guard).
        kwargs: Additional keyword arguments.
    """

    engine = settings.default_llm_provider

    if engine == "nim":
        if schema_type in [GuardrailChatCompletionRequest, GuardrailCheckRequest]:
            engine = NIM_CHAT

        elif schema_type == GuardrailCompletionRequest:
            engine = NIM_LLM

    return Model(model=model_name, type=type, engine=engine, parameters=kwargs)


def get_main_model_from_config(models: List[Model]) -> Optional[Model]:
    """Get the main model from the config."""
    for model in models:
        if model.type == "main":
            return model
    return None


def get_merged_custom_headers_token(token: str | None) -> str:
    """Return the caller-specific part of the engine cache key.

    Only values that get built into the engine belong here: the caller's model credential and the
    platform auth headers. Request ``x-*`` headers aren't built into the engine, so they're left out.
    Including them would rebuild the engine on every request whenever a proxy like Envoy adds a
    unique ``x-request-id``.
    """
    headers = get_platform_auth_headers_from_context()
    headers_str = json.dumps(headers, sort_keys=True)
    merged_token_headers = f"{token}{headers_str}" if token else headers_str
    return merged_token_headers


def get_merged_custom_headers(static_headers: dict[str, str], req_custom_headers: dict[str, str]) -> dict[str, str]:
    """Merge custom headers from the existing static headers from the config and the request custom headers."""
    static_headers = {k.lower(): v for k, v in static_headers.items()}
    return {**static_headers, **req_custom_headers}


def model_with_req_scoped_custom_headers(model: Model, req_custom_headers: dict[str, str] | None):
    """Add custom headers from request context to the model."""
    if not req_custom_headers:
        req_custom_headers = {}

    params = model.parameters or {}
    default_headers = params.get("default_headers") or {}
    # Merge config's default headers with incoming request's custom headers
    merged_default_headers = get_merged_custom_headers(default_headers, req_custom_headers)
    # Add merged default headers to model's `parameters`
    parameters = {**params, "default_headers": merged_default_headers}

    # `model_copy` skips validation, so build `ModelParameters` explicitly. Downstream code
    # (e.g. `apply_platform_auth_headers`) relies on `parameters` being a model, not a dict.
    return model.model_copy(update={"parameters": ModelParameters.model_validate(parameters)})


def set_main_model_merged_custom_headers_into_context(model: Optional[Model]) -> None:
    """Set the headers the main model sends for this request.

    That's the model's configured ``default_headers`` with the request's headers on top (header
    names are compared case-insensitively). The engine is shared across requests, so this has to
    happen per request instead of when the engine is built.
    """
    # for /v1/checks, main model is currently set, but not required technically and might be removed in the future
    # so we don't raise an error on no main model here
    if model is None:
        return
    if model.type != "main":
        raise ValueError(f"Expected main model. Got: {model.type}")

    default_headers = (model.parameters or {}).get("default_headers", None)
    if not default_headers:
        return
    set_request_default_headers_into_context(
        get_merged_custom_headers(default_headers, get_request_default_headers_from_context())
    )


def update_models_in_config(config: RailsConfig, main_model: Model):
    """Update a model config in a rail configuration.

    Looks if a model with the same type exists, and if so, it replaces it.
    If not, it adds it.
    """
    models = config.models.copy()
    main_model_index = None

    for index, model in enumerate(models):
        if model.type == main_model.type:
            main_model_index = index
        # The engine built from this config is shared across requests, so only the configured
        # headers go into its models. The main model gets request headers per call (see
        # `set_main_model_merged_custom_headers_into_context`). Task models don't get them at all.
        models[index] = model_with_req_scoped_custom_headers(model, None)

    main_model_parameters = {k: v for k, v in (main_model.parameters or {}).items() if v is not None}

    if main_model_index is not None:
        merged_params = {**(models[main_model_index].parameters or {}), **main_model_parameters}
        models[main_model_index] = main_model.model_copy(
            update={"parameters": ModelParameters.model_validate(merged_params)}
        )
    else:
        # If there's no main model, we add the current one.
        models.append(
            main_model.model_copy(update={"parameters": ModelParameters.model_validate(main_model_parameters)})
        )

    updated_config = config.model_copy(update={"models": models})

    return updated_config


def get_rail_types_from_config(config: RailsConfig) -> Dict[str, List[str]]:
    """Extract the type of rails used in a config.

    Return:
        dict: {"rails": ["input|output", ...]}
    """
    update_options = {"rails": []}

    if config.rails.input and config.rails.input.flows:
        update_options["rails"].append("input")

    if config.rails.output and config.rails.output.flows:
        update_options["rails"].append("output")

    return update_options


def get_rails_name_from_config(config: RailsConfig) -> List[str]:
    """Extract the names of the rails used in a config."""
    rails = []
    if config.rails.input and config.rails.input.flows:
        rails.extend(config.rails.input.flows)
    if config.rails.output and config.rails.output.flows:
        rails.extend(config.rails.output.flows)
    return rails


def get_main_model_from_rails_config(rails_config: RailsConfig) -> Model | None:
    """Returns the main model from given rails config's list of models."""
    return next((model for model in rails_config.models if model.type == "main"), None)


async def run_generate_async(
    llm_rails: RailsEngine,
    messages: Optional[List[dict]] = None,
    prompt: Optional[str] = None,
    options: Optional[Any] = None,
    state: Optional[Any] = None,
) -> Any:
    """Run the rails on the server event loop, like the standalone NeMo Guardrails server does.

    Model HTTP clients are tied to the event loop that first used them. NeMo Guardrails shares one
    client per endpoint across the whole process, and IORails keeps its own. We used to run each
    request on a new loop, which left those clients attached to a closed loop. Later requests then
    had to wait and retry, or hung under load. Building an engine is the slow part, and that still
    happens in a worker thread.
    """
    return await llm_rails.generate_async(messages=messages, prompt=prompt, options=options, state=state)


def _text_of_content_parts(parts: List[Any]) -> str:
    """Join the text parts of OpenAI-style message content with newlines, dropping non-text parts."""
    texts = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts)


def _text_only_messages(messages: List[dict]) -> List[dict]:
    """Copy of ``messages`` with list-valued ``content`` flattened to its text.

    IORails reads a message only when ``content`` is a string, so content parts would look empty and the
    rail would be skipped, passing the request unchecked. Parts a text rail cannot inspect (images,
    audio, files) are dropped; a message made only of those is therefore not checked.
    """
    return [
        {**message, "content": _text_of_content_parts(message["content"])}
        if isinstance(message.get("content"), list)
        else message
        for message in messages
    ]


def _last_text_by_role(messages: List[dict], role: str) -> str:
    """Text of the last message with ``role``, or "" when there is none or it is not plain text."""
    for message in reversed(messages):
        if message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _rails_result_to_generation_response(
    result: RailsResult,
    flows_by_type: Dict[RailType, List[str]],
) -> GenerationResponse:
    """Shape an IORails verdict like the LLMRails generation result the checks response is built from.

    The checks response reads ``log.activated_rails`` to report each rail's status, with ``stop``
    meaning blocked. IORails reports only the verdict and which rail blocked, so every rail that ran is
    reported as passed, up to the one that blocked. The blocking rail is reported even when it is not
    one of the config's flows (a rail that failed to run), so a block is never reported as success.
    """
    activated_rails: List[ActivatedRail] = []
    blocked = result.status is RailStatus.BLOCKED
    for rail_type, flows in flows_by_type.items():
        for flow in flows:
            # IORails names the blocking rail without the ``$model=...`` / ``(...)`` suffix its flow
            # carries, using this same helper.
            is_blocking_rail = blocked and _get_flow_name(flow) == result.rail
            activated_rails.append(ActivatedRail(type=rail_type.value, name=flow, stop=is_blocking_rail))
            if is_blocking_rail:
                return GenerationResponse(
                    response=[{"role": "assistant", "content": result.content}],
                    log=GenerationLog(activated_rails=activated_rails),
                )

    if blocked:
        rail_type = next(iter(flows_by_type), RailType.OUTPUT)
        activated_rails.append(ActivatedRail(type=rail_type.value, name=result.rail or rail_type.value, stop=True))

    return GenerationResponse(
        response=[{"role": "assistant", "content": result.content}],
        log=GenerationLog(activated_rails=activated_rails),
    )


async def run_check_async(
    rails_engine: RailsEngine,
    messages: List[dict],
    rail_types: List[str],
    state: Optional[Any] = None,
) -> GenerationResponse:
    """Run the config's rails over ``messages`` as supplied, without generating a response.

    A check audits text the caller already has, so the main model must not run: IORails'
    ``generate_async`` always calls it and then judges its reply, which would report a verdict on text
    nobody supplied. IORails' ``check_async`` judges the supplied messages. Runs on the server loop for
    the same reason as :func:`run_generate_async`.

    Only IORails needs this. LLMRails treats a trailing assistant message as the text to check and
    skips generation, so its checks stay on :func:`run_generate_async`.
    """
    if not uses_iorails(rails_engine):
        raise ValueError("run_check_async only serves IORails engines; use run_generate_async for LLMRails")
    if state is not None:
        raise ValueError("state is not supported by IORails; it is a stateless input/output rails engine")

    messages = _text_only_messages(messages)
    rails_config = rails_engine.config
    flows_by_type: Dict[RailType, List[str]] = {}
    for rail_type in (RailType.INPUT, RailType.OUTPUT):
        if rail_type.value not in rail_types:
            continue
        flows = list(getattr(rails_config.rails, rail_type.value).flows or [])
        # A rail that has nothing to read does not run, so it is not reported as having passed.
        checked_role = "user" if rail_type is RailType.INPUT else "assistant"
        if flows and _last_text_by_role(messages, checked_role):
            flows_by_type[rail_type] = flows

    result = await rails_engine.check_async(messages, rail_types=list(flows_by_type))
    return _rails_result_to_generation_response(result, flows_by_type)


def stream_rails_async(
    llm_rails: RailsEngine,
    prompt: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    options: Optional[Any] = None,
    state: Optional[Any] = None,
) -> AsyncIterator[Union[str, dict]]:
    """Stream a guarded response from either rails engine.

    IORails is stateless and only honors ``options``, so ``state`` would be silently dropped. Reject
    it instead, as the non-streaming IORails path does.
    """
    if uses_iorails(llm_rails):
        if state is not None:
            raise ValueError("state is not supported by IORails; it is a stateless input/output rails engine")
        return llm_rails.stream_async(prompt=prompt, messages=messages, options=options)

    return llm_rails.stream_async(prompt=prompt, messages=messages, options=options, state=state)
