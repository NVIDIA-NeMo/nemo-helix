# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest
from nemo_evaluator.sdk.resources import AsyncEvaluator, Evaluator, evaluator_sdk_resources
from nemo_platform_plugin.sdk import AsyncNeMoPlatform, NeMoPlatform

BASE = "http://test"


def test_evaluator_sdk_sync_resource_accepts_generated_platform_client() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    platform = NeMoPlatform(
        base_url=BASE,
        workspace="default",
        default_headers={"X-NMP-Principal-Id": "service:evaluator"},
        http_client=http_client,
    )

    resource_factory = evaluator_sdk_resources.sync_resource
    assert resource_factory is not None
    resource = resource_factory(platform)

    assert isinstance(resource, Evaluator)
    assert resource._client._http is http_client
    assert resource._client.default_headers["X-NMP-Principal-Id"] == "service:evaluator"


@pytest.mark.asyncio
async def test_evaluator_sdk_async_resource_accepts_generated_platform_client() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    ) as http_client:
        platform = AsyncNeMoPlatform(
            base_url=BASE,
            workspace="default",
            default_headers={"X-NMP-Principal-Id": "service:evaluator"},
            http_client=http_client,
        )

        resource_factory = evaluator_sdk_resources.async_resource
        assert resource_factory is not None
        resource = resource_factory(platform)

        assert isinstance(resource, AsyncEvaluator)
        assert resource._client._http is http_client
        assert resource._client.default_headers["X-NMP-Principal-Id"] == "service:evaluator"
