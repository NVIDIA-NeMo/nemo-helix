# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host loop for native Switchyard ``Algorithm.run_stream`` (CallModel / Done)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from nemo_platform_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareError,
    InferenceRequest,
    NemoInferenceMiddleware,
)
from nemo_switchyard._native_availability import load_libsy, native_rust_available
from nemo_switchyard._native_ir import (
    apply_llm_request_to_openai_body,
    llm_request_to_openai_chat,
    openai_chat_to_agg,
    openai_chat_to_llm_request,
)

JUDGE_HTTP_TIMEOUT_SECONDS = 30.0
NATIVE_STREAM_TIMEOUT_SECONDS = 60.0
_SESSION_HEADER = "x-switchyard-session-id"
_REQUEST_ID_HEADER = "x-request-id"
_DEFAULT_AUTH_HEADER = "Authorization: Bearer {secret}"


class JudgeTransport(Protocol):
    async def complete(
        self,
        model_entity_id: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]: ...


@dataclass
class NativeBinding:
    """Per-VM native Algorithm plus the category→model map passed to run_stream."""

    algorithm: Any
    models: dict[str, list[str]]
    config_type: str


def native_request_dict(body: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an OpenAI Chat body into a libsy ``LlmRequest`` dict."""
    return openai_chat_to_llm_request(dict(body))


def routing_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Headers libsy may read (session affinity). Never forwards caller credentials."""
    allowed = {_SESSION_HEADER, _REQUEST_ID_HEADER}
    return {key: value for key, value in headers.items() if key.lower() in allowed}


def wrap_llm_response(payload: Mapping[str, Any]) -> Any:
    agg = openai_chat_to_agg(dict(payload))
    if native_rust_available():
        return load_libsy().LlmResponse.Agg(agg)
    return agg


def apply_outcome_to_request(request: InferenceRequest, outcome: Any) -> None:
    selected = list(getattr(outcome, "selected_model_ids", ()) or ())
    if not selected:
        raise InferenceMiddlewareError(
            "Switchyard Done outcome had no selected_model_ids",
            status_code=500,
        )
    rewritten = getattr(outcome, "request", None)
    if isinstance(rewritten, dict):
        request.body = apply_llm_request_to_openai_body(request.body, rewritten)
    request.body["model"] = selected[0]
    request.typed_body = request.body


class IgwJudgeTransport:
    """Provider-direct judge HTTP with provider auth, never caller credentials."""

    def __init__(
        self,
        middleware: NemoInferenceMiddleware,
        *,
        timeout: float = JUDGE_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._middleware = middleware
        self._timeout = timeout

    async def complete(
        self,
        model_entity_id: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        del headers
        target = self._middleware.get_inference_url_and_model(model_entity_id)
        url = f"{target.model_provider_gateway_url.rstrip('/')}/chat/completions"
        payload = {**body, "model": target.served_model_name}
        outbound = _provider_outbound_headers(self._middleware, model_entity_id)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
                response = await client.post(url, json=payload, headers=outbound)
        except httpx.TimeoutException as exc:
            raise InferenceMiddlewareError(
                f"Switchyard judge timed out after {self._timeout}s",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceMiddlewareError(f"Switchyard judge request failed: {exc}", status_code=502) from exc
        if response.status_code >= 400:
            detail = " ".join((response.text or "").split())[:500]
            suffix = f": {detail}" if detail else ""
            raise InferenceMiddlewareError(
                f"Switchyard judge returned HTTP {response.status_code}{suffix}",
                status_code=502,
            )
        data = response.json()
        if not isinstance(data, dict):
            raise InferenceMiddlewareError("Switchyard judge returned a non-object JSON body", status_code=502)
        return data


async def _serve_call(call: Any, transport: JudgeTransport, headers: dict[str, str]) -> None:
    models = list(getattr(call, "models", ()) or ())
    if not models:
        error = InferenceMiddlewareError("Switchyard CallModel listed no models", status_code=500)
        call.fail(error)
        raise error
    chat_body = llm_request_to_openai_chat(dict(call.request))
    last_error: Exception | None = None
    for model_id in models:
        try:
            payload = await transport.complete(model_id, chat_body, headers)
            call.respond(wrap_llm_response(payload))
            return
        except InferenceMiddlewareError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = InferenceMiddlewareError(str(exc), status_code=502)
            continue
    assert last_error is not None
    call.fail(last_error)
    raise last_error


def _immediate_not_wired() -> None:
    raise InferenceMiddlewareError(
        "Switchyard Done.response is set; capability and stage_router do not fill it. "
        "Failing closed rather than returning ImmediateResponse.",
        status_code=500,
    )


async def run_native_stream(
    *,
    algorithm: Any,
    request: InferenceRequest,
    models: Mapping[str, Sequence[str]],
    headers: dict[str, str],
    transport: JudgeTransport,
    timeout: float = NATIVE_STREAM_TIMEOUT_SECONDS,
) -> InferenceRequest | ImmediateResponse:
    """Drive ``run_stream`` until Done. Does not call the user model on empty response."""
    try:
        return await asyncio.wait_for(
            _run_native_stream(
                algorithm=algorithm,
                request=request,
                models=models,
                headers=headers,
                transport=transport,
            ),
            timeout=timeout,
        )
    except TimeoutError as exc:
        raise InferenceMiddlewareError(
            f"Switchyard run_stream timed out after {timeout}s",
            status_code=504,
        ) from exc


async def _run_native_stream(
    *,
    algorithm: Any,
    request: InferenceRequest,
    models: Mapping[str, Sequence[str]],
    headers: dict[str, str],
    transport: JudgeTransport,
) -> InferenceRequest | ImmediateResponse:
    request_dict = native_request_dict(request.body)
    categories = {key: list(value) for key, value in models.items()}
    libsy_headers = routing_headers(headers)
    outcome: Any = None
    async for step in algorithm.run_stream(request_dict, categories, headers=libsy_headers or None):
        call = getattr(step, "call", None)
        if call is not None:
            await _serve_call(call, transport, libsy_headers)
            continue
        done = getattr(step, "outcome", None)
        if done is not None:
            outcome = done
            continue
        raise InferenceMiddlewareError(
            f"Unknown Switchyard step {type(step).__name__}",
            status_code=500,
        )
    if outcome is None:
        raise InferenceMiddlewareError("Switchyard run_stream ended without Done", status_code=500)
    if getattr(outcome, "response", None) is not None:
        _immediate_not_wired()
    apply_outcome_to_request(request, outcome)
    return request


def _provider_outbound_headers(middleware: NemoInferenceMiddleware, model_entity_id: str) -> dict[str, str]:
    """Build provider auth and extra headers; never copies the inbound caller map."""
    headers: dict[str, str] = {}
    entity = middleware.get_model_entity(model_entity_id)
    if entity is None:
        return headers
    pairs = getattr(entity, "model_providers", None)
    info = None
    provider = None
    if pairs:
        _served, info = pairs[0]
        provider = getattr(info, "model_provider", None)
    if provider is None:
        providers = getattr(entity, "providers", None) or []
        provider = providers[0] if providers else None
    if provider is None:
        return headers
    defaults = getattr(provider, "default_extra_headers", None) or {}
    required = getattr(provider, "required_extra_headers", None) or {}
    headers.update(dict(defaults))
    headers.update(dict(required))
    secret = getattr(info, "secret_value", None) if info is not None else None
    secret_name = getattr(provider, "api_key_secret_name", None)
    if secret_name and not secret:
        raise InferenceMiddlewareError(
            f"Switchyard judge provider secret {secret_name!r} is not cached; "
            "IGW must refresh ModelCache before native judge calls.",
            status_code=502,
        )
    if secret:
        header_name, header_value = _render_auth_header(str(secret), getattr(provider, "auth_header_format", None))
        headers[header_name] = header_value
    return headers


def _render_auth_header(secret_value: str, auth_header_format: str | None) -> tuple[str, str]:
    template = auth_header_format or _DEFAULT_AUTH_HEADER
    rendered = template.replace("{{ auth_secret }}", secret_value).replace("{secret}", secret_value)
    name, _, value = rendered.partition(": ")
    if not value:
        return "Authorization", f"Bearer {secret_value}"
    return name, value
