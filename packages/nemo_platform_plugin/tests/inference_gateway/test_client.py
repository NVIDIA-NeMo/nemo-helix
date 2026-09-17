# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway provider client tests."""

from __future__ import annotations

import httpx
from nemo_platform_plugin.inference_gateway.client import (
    AsyncInferenceGatewayProviderClient,
    InferenceGatewayProviderClient,
)

BASE = "http://test:8000"


def test_get_provider_models_decodes_json_and_uses_provider_route() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, request=request, json={"object": "list", "data": [{"id": "model-a"}]})

    client = InferenceGatewayProviderClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.get_provider_models(workspace="team-a", name="provider-a")

    assert result == {"object": "list", "data": [{"id": "model-a"}]}
    assert seen[0].url.path == "/apis/inference-gateway/v2/workspaces/team-a/provider/provider-a/-/v1/models"


async def test_async_get_provider_models_returns_text_for_non_json_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"not-json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncInferenceGatewayProviderClient(
            base_url=BASE,
            workspace="default",
            http_client=http_client,
        )

        result = await client.get_provider_models(name="provider-a")

    assert result == "not-json"
