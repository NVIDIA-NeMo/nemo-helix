# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest
from nmp.core.inference_gateway.api.middleware_registry import MiddlewareRegistry
from nmp.core.inference_gateway.config import DebugModelProvider, config
from nmp.core.inference_gateway.service import InferenceGatewayService


@pytest.mark.asyncio
async def test_debug_startup_hydrates_model_entity_metadata(mocker):
    sdk = Mock()
    model_entity_getter = Mock()
    refresh_model_cache = AsyncMock()
    http_client = Mock()
    http_client.close = AsyncMock()

    mocker.patch(
        "nmp.core.inference_gateway.api.middleware_registry.load_middleware_plugins",
        AsyncMock(return_value=MiddlewareRegistry()),
    )
    mocker.patch(
        "nmp.core.inference_gateway.api.model_cache.model_entity_getter_from_sdk",
        return_value=model_entity_getter,
    )
    mocker.patch(
        "nmp.core.inference_gateway.api.model_cache.refresh_model_cache",
        refresh_model_cache,
    )
    mocker.patch("nmp.core.inference_gateway.service.aiohttp.ClientSession", return_value=http_client)
    mocker.patch.object(
        config,
        "debug_model_providers",
        [DebugModelProvider(workspace="default", name="debug", host_url="http://debug.local")],
    )

    service = InferenceGatewayService()
    mocker.patch.object(service.dependency_provider, "get_sdk_client", return_value=sdk)
    await service.on_startup()
    await service.on_shutdown()

    refresh_model_cache.assert_awaited_once()
    await_args = refresh_model_cache.await_args
    assert await_args is not None
    assert await_args.kwargs["model_entity_getter"] is model_entity_getter
