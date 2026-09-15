# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed sync and async clients for the Inference Gateway proxy routes.

Usage::

    client = InferenceGatewayClient(base_url="...", workspace="default")
    client.provider_ready(name="nvidia-build")
    completion = client.openai_post(
        trailing_uri="v1/chat/completions",
        body=JsonBody({"model": "default/my-model", "messages": [...]}),
    ).data()
"""

import json

import httpx
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method
from nemo_platform_plugin.inference_gateway import endpoints


def _decode_provider_proxy_body(content: bytes) -> object:
    """Decode a provider proxy body.

    Provider proxy responses are intentionally dynamic because the gateway
    forwards upstream provider payloads. Decode JSON when possible and return
    text otherwise so callers can validate the small shape they need.
    """
    text = content.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class _InferenceGatewayMethods:
    provider_get = method(endpoints.provider_get)
    provider_post = method(endpoints.provider_post)
    provider_put = method(endpoints.provider_put)
    provider_patch = method(endpoints.provider_patch)
    provider_delete = method(endpoints.provider_delete)
    provider_ready = method(endpoints.provider_ready)
    get_provider_models_raw = method(endpoints.get_provider_models_raw)
    stream_provider = method(endpoints.stream_provider)
    model_get = method(endpoints.model_get)
    model_post = method(endpoints.model_post)
    model_put = method(endpoints.model_put)
    model_patch = method(endpoints.model_patch)
    model_delete = method(endpoints.model_delete)
    stream_model = method(endpoints.stream_model)
    openai_get = method(endpoints.openai_get)
    openai_post = method(endpoints.openai_post)
    stream_openai = method(endpoints.stream_openai)
    list_openai_models = method(endpoints.list_openai_models)
    get_openai_model = method(endpoints.get_openai_model)


class InferenceGatewayClient(_InferenceGatewayMethods, NemoClient):
    """Sync client for the Inference Gateway proxy routes."""

    def get_provider_models(
        self,
        *,
        workspace: str | None = None,
        name: str,
        timeout: float | httpx.Timeout | None = None,
    ) -> object:
        """Return the decoded ``GET /v1/models`` payload for a provider."""
        client = self.with_options(timeout=timeout) if timeout is not None else self
        response = client.get_provider_models_raw(workspace=workspace, name=name)
        return _decode_provider_proxy_body(response.read())


class AsyncInferenceGatewayClient(_InferenceGatewayMethods, AsyncNemoClient):
    """Async client for the Inference Gateway proxy routes."""

    async def get_provider_models(
        self,
        *,
        workspace: str | None = None,
        name: str,
        timeout: float | httpx.Timeout | None = None,
    ) -> object:
        """Return the decoded ``GET /v1/models`` payload for a provider."""
        client = self.with_options(timeout=timeout) if timeout is not None else self
        response = await client.get_provider_models_raw(workspace=workspace, name=name)
        return _decode_provider_proxy_body(await response.read())


class InferenceGatewayProviderClient(InferenceGatewayClient):
    """Sync client for the narrow provider proxy reads the Models reconciler makes."""


class AsyncInferenceGatewayProviderClient(AsyncInferenceGatewayClient):
    """Async client for the narrow provider proxy reads the Models reconciler makes."""
