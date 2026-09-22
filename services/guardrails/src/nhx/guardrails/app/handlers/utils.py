# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from typing import Any, Dict, List, Optional, Type, Union

from nemoguardrails import LLMRails
from nemoguardrails import RailsConfig as InternalSdkRailsConfig
from nhx.guardrails.app.constants import NIM_CHAT, NIM_LLM
from nhx.guardrails.app.utils.context_utils import (
    get_request_default_headers_from_context,
    set_request_default_headers_into_context,
)
from nhx.guardrails.config import settings
from nhx.guardrails.entities.values._private import Model, RailsConfig
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
    """Merge the custom headers from the request context into the token."""
    headers = get_request_default_headers_from_context()
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

    return model.model_copy(update={"parameters": parameters})


def set_main_model_merged_custom_headers_into_context(model: Optional[Model]) -> None:
    """Set the merged custom headers for the main model."""
    # for /v1/checks, main model is currently set, but not required technically and might be removed in the future
    # so we don't raise an error on no main model here
    if model is None:
        return
    if model.type != "main":
        raise ValueError(f"Expected main model. Got: {model.type}")

    default_headers = (model.parameters or {}).get("default_headers", None)
    if not default_headers:
        return
    set_request_default_headers_into_context(default_headers)


def update_models_in_config(config: RailsConfig, main_model: Model):
    """Update a model config in a rail configuration.

    Looks if a model with the same type exists, and if so, it replaces it.
    If not, it adds it.
    """
    models = config.models.copy()
    main_model_index = None

    # Custom headers defined in the guardrail config
    req_custom_headers = get_request_default_headers_from_context()

    for index, model in enumerate(models):
        if model.type == main_model.type:
            main_model_index = index
        # Merge the custom headers from the request context into the model
        # The model.parameters["default_headers"] is a source of truth for the request custom headers
        # Although this object is cached, the cache key includes the custom headers,
        # so it will be invalidated when any request custom header changes
        models[index] = model_with_req_scoped_custom_headers(model, req_custom_headers)

    main_model_parameters = {k: v for k, v in (main_model.parameters or {}).items() if v is not None}

    if main_model_index is not None:
        merged_params = {**(models[main_model_index].parameters or {}), **main_model_parameters}
        models[main_model_index] = main_model.model_copy(update={"parameters": merged_params})
    else:
        # If there's no main model, we add the current one.
        models.append(main_model.model_copy(update={"parameters": main_model_parameters}))

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


def _run_generate_in_new_loop(
    llm_rails: LLMRails,
    messages: Optional[List[dict]] = None,
    prompt: Optional[str] = None,
    options: Optional[Any] = None,
    state: Optional[Any] = None,
) -> Any:
    """Run LLMRails.generate_async in a new event loop in the current thread.

    Called via asyncio.to_thread, which copies the current context so all
    context variables (auth token, headers, main model, etc.) are available.

    NemoGuardrails' generate_async has CPU-intensive Colang flow processing
    between LLM calls that can hold the asyncio event loop for multiple seconds at
    a time. This could block the event loop for an extended period, causing Kubernetes
    liveness probes to time out and kill the pod. Running the function in a
    dedicated thread with its own event loop keeps the main event loop free to
    respond to health checks and other concurrent requests.
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
            # Cancel any tasks that are still pending (ex. aiohttp/httpx cleanup tasks
            # that were created during generate_async but not yet completed).
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            # Exit the worker threads created for each LLM call in this request.
            # Without this, they linger as idle pool threads into the next request.
            # When the next request initializes a new LLMRails object, those idle
            # threads compete for the Python GIL with LLMRails' internal setup thread.
            # The main event loop is starved of the GIL for that duration and cannot
            # respond to Kubernetes liveness probes, which kills the pod.
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()
            asyncio.set_event_loop(None)


async def run_generate_async(
    llm_rails: LLMRails,
    messages: Optional[List[dict]] = None,
    prompt: Optional[str] = None,
    options: Optional[Any] = None,
    state: Optional[Any] = None,
) -> Any:
    """Run LLMRails.generate_async without blocking the FastAPI event loop.

    asyncio.to_thread copies the current contextvars context before entering
    the thread, so context variables set prior to this call remain accessible
    inside the guardrails pipeline.
    """
    return await asyncio.to_thread(
        _run_generate_in_new_loop,
        llm_rails,
        messages=messages,
        prompt=prompt,
        options=options,
        state=state,
    )
