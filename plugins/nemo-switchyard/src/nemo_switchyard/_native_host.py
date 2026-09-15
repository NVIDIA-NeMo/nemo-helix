# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host loop for native Switchyard ``Algorithm.run_stream`` (CallModel / Done)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
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

JUDGE_HTTP_TIMEOUT_SECONDS = 30.0


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


def coerce_message_content(content: Any) -> Any:
    """Coerce OpenAI string content to Switchyard content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content


def native_request_dict(body: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the IGW body into a dict libsy can parse (content blocks, not strings)."""
    request = deepcopy(dict(body))
    messages = request.get("messages")
    if isinstance(messages, list):
        rewritten: list[Any] = []
        for message in messages:
            if not isinstance(message, dict):
                rewritten.append(message)
                continue
            item = dict(message)
            if "content" in item:
                item["content"] = coerce_message_content(item["content"])
            rewritten.append(item)
        request["messages"] = rewritten
    return request


def wrap_llm_response(payload: Mapping[str, Any]) -> Any:
    if native_rust_available():
        return load_libsy().LlmResponse.Agg(payload)
    return dict(payload)


def apply_outcome_to_request(request: InferenceRequest, outcome: Any) -> None:
    selected = list(getattr(outcome, "selected_model_ids", ()) or ())
    if not selected:
        raise InferenceMiddlewareError(
            "Switchyard Done outcome had no selected_model_ids",
            status_code=500,
        )
    request.body["model"] = selected[0]
    request.typed_body = request.body


class IgwJudgeTransport:
    """Judge HTTP via ``get_inference_url_and_model`` (provider-direct, never NMP_BASE_URL)."""

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
        target = self._middleware.get_inference_url_and_model(model_entity_id)
        url = f"{target.model_provider_gateway_url.rstrip('/')}/chat/completions"
        payload = {**body, "model": target.served_model_name}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
                response = await client.post(url, json=payload, headers=dict(headers))
        except httpx.TimeoutException as exc:
            raise InferenceMiddlewareError(
                f"Switchyard judge timed out after {self._timeout}s",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceMiddlewareError(f"Switchyard judge request failed: {exc}", status_code=502) from exc
        if response.status_code >= 400:
            raise InferenceMiddlewareError(
                f"Switchyard judge returned HTTP {response.status_code}",
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
    try:
        payload = await transport.complete(models[0], dict(call.request), headers)
        call.respond(wrap_llm_response(payload))
    except Exception as exc:
        call.fail(exc)
        if isinstance(exc, InferenceMiddlewareError):
            raise
        raise InferenceMiddlewareError(str(exc), status_code=502) from exc


def _immediate_not_wired() -> None:
    raise InferenceMiddlewareError(
        "Switchyard Done.response is set; ImmediateResponse is not wired until RC2 "
        "documents which algorithms fill it. Failing closed.",
        status_code=500,
    )


async def run_native_stream(
    *,
    algorithm: Any,
    request: InferenceRequest,
    models: Mapping[str, Sequence[str]],
    headers: dict[str, str],
    transport: JudgeTransport,
) -> InferenceRequest | ImmediateResponse:
    """Drive ``run_stream`` until Done. Does not call the user model on empty response."""
    request_dict = native_request_dict(request.body)
    categories = {key: list(value) for key, value in models.items()}
    outcome: Any = None
    async for step in algorithm.run_stream(request_dict, categories, headers=headers or None):
        call = getattr(step, "call", None)
        if call is not None:
            await _serve_call(call, transport, headers)
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
