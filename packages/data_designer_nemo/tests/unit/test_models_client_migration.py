# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider lookup through the typed Models client, over a mocked httpx transport.

Driving a real ``NemoClient`` -> ``client_from_platform`` -> ``ModelsClient``
chain asserts the wire contract (method, path, parsed model) rather than
restating the call the implementation happens to make.
"""

from __future__ import annotations

import httpx
import pytest
from data_designer_nemo.model_provider import get_nhx_provider, get_nhx_provider_async
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import ModelProvider

BASE = "http://test:8000"


def _provider_json(name: str = "my-provider", workspace: str = "other", **extra: object) -> dict:
    base = {
        "id": f"provider-{name}",
        "name": name,
        "workspace": workspace,
        "host_url": "https://api.example.com",
        "status": "READY",
        "status_message": "",
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
    }
    base.update(extra)
    return base


def _recording_transport(payload: dict) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, request=request, json=payload)

    return httpx.MockTransport(handler), seen


def test_get_nhx_provider_hits_the_provider_endpoint() -> None:
    transport, seen = _recording_transport(_provider_json())
    sdk = NemoClient(base_url=BASE, workspace="default", http_client=httpx.Client(transport=transport))

    result = get_nhx_provider(sdk, "other", "my-provider")

    assert isinstance(result, ModelProvider)
    assert (result.workspace, result.name) == ("other", "my-provider")
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert str(seen[0].url) == f"{BASE}/apis/models/v2/workspaces/other/providers/my-provider"


@pytest.mark.asyncio
async def test_get_nhx_provider_async_hits_the_provider_endpoint() -> None:
    transport, seen = _recording_transport(_provider_json())
    sdk = AsyncNemoClient(base_url=BASE, workspace="default", http_client=httpx.AsyncClient(transport=transport))

    result = await get_nhx_provider_async(sdk, "other", "my-provider")

    assert isinstance(result, ModelProvider)
    assert (result.workspace, result.name) == ("other", "my-provider")
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert str(seen[0].url) == f"{BASE}/apis/models/v2/workspaces/other/providers/my-provider"


@pytest.mark.parametrize(
    ("host_url", "expected_suffix"),
    [("https://api.example.com", "/-/v1"), ("https://api.example.com/v1", "/-")],
)
def test_provider_route_url_conditionally_appends_v1(host_url: str, expected_suffix: str) -> None:
    """The endpoint handed to Data Designer and the anonymizer, built from a fetched provider."""
    transport, _ = _recording_transport(_provider_json(host_url=host_url))
    sdk = NemoClient(base_url=BASE, workspace="default", http_client=httpx.Client(transport=transport))

    provider = get_nhx_provider(sdk, "other", "my-provider")
    url = client_from_platform(sdk, ModelsClient).get_provider_route_openai_url(provider)

    assert url == f"{BASE}/apis/inference-gateway/v2/workspaces/other/provider/my-provider{expected_suffix}"
