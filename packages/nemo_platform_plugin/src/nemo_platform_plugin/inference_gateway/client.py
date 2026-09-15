# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed clients for narrow Inference Gateway provider proxy calls."""

from __future__ import annotations

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


class _InferenceGatewayProviderMethods:
    get_provider_models_raw = method(endpoints.get_provider_models_raw)


class InferenceGatewayProviderClient(_InferenceGatewayProviderMethods, NemoClient):
    """Sync client for Inference Gateway provider proxy reads."""

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


class AsyncInferenceGatewayProviderClient(_InferenceGatewayProviderMethods, AsyncNemoClient):
    """Async client for Inference Gateway provider proxy reads."""

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
