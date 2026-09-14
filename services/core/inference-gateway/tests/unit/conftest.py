# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for Inference Gateway service."""

from datetime import datetime
from typing import Iterator
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from nemo_platform.types.inference import ModelProvider, ServedModelMapping
from nemo_platform.types.inference.virtual_model import VirtualModel
from nmp.core.inference_gateway.api.dependencies import (
    global_http_client,
    global_middleware_registry,
    global_model_cache,
    global_virtual_model_cache,
)
from nmp.core.inference_gateway.api.middleware_registry import MiddlewareRegistry
from nmp.core.inference_gateway.api.model_cache import ModelCache, ModelProviderInfo
from nmp.core.inference_gateway.api.virtual_model_cache import VirtualModelCache
from nmp.core.inference_gateway.config import DebugModelProvider, config
from nmp.core.inference_gateway.service import InferenceGatewayService


def default_model_infos() -> list[ModelProviderInfo]:
    return [
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="ollama",
                host_url="http://localhost:11434",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id="e2e-test/meta_llama-3.2-1b-instruct",
                        served_model_name="meta/llama-3.2-1b-instruct",
                    )
                ],
                status="READY",
            ),
        ),
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="tot",
                host_url="https://mock-nim.example.invalid",
                api_key_secret_name="fake_api_key_secret_name",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id="e2e-test/meta_llama-3.2-1b-instruct",
                        served_model_name="meta/llama-3.2-1b-instruct",
                    )
                ],
                status="READY",
            ),
            secret_value="fake_secret_value",
        ),
    ]


def new_model_infos() -> list[ModelProviderInfo]:
    return [
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="new",
                host_url="http://localhost:8080",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status="READY",
            ),
        ),
    ]


@pytest.fixture
def model_cache():
    ret = ModelCache()
    for model_info in default_model_infos():
        ret.update_model_info(model_info)
    ret.rebuild_model_entity_map()
    return ret


def autoprovisioned_vms_for_cache(model_cache: ModelCache) -> list[VirtualModel]:
    """Return implicit ``autoprovisioned`` VirtualModels mirroring every served entity in *model_cache*.

    In production the provider reconciler creates one autoprovisioned VirtualModel per served
    model entity (named after the entity, with ``default_model_entity = "{workspace}/{name}"``).
    The IGW's ``openai_proxy`` and ``model_entity_proxy`` routes both require a VirtualModel to
    exist for any inference request to succeed, so unit tests that exercise those routes need
    a VirtualModel cache populated with the same default. Tests that need a custom or absent
    VirtualModel can override the ``virtual_model_cache`` fixture.

    LoRA composite served-model ids (``"&adapters/" in model_entity_id``) are intentionally
    excluded here, mirroring the production reconciler's skip at
    ``provider_reconciler.py:440`` — the IGW relies on ``ModelCache``'s split-on-first-``/``
    keying plus the request-side rewrite of ``parse_model_entity_ref`` to surface composite
    LoRA ids through whichever VirtualModel the operator manually associated with them.
    """
    now = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    return [
        VirtualModel(
            id=f"{workspace}/{name}",
            entity_id=f"{workspace}/{name}",
            workspace=workspace,
            name=name,
            parent=workspace,
            db_version=1,
            default_model_entity=f"{workspace}/{name}",
            autoprovisioned=True,
            created_at=now,
            updated_at=now,
        )
        for (workspace, name) in model_cache.model_entity_info_map.keys()
        if "&adapters/" not in name
    ]


@pytest.fixture
def virtual_model_cache(model_cache: ModelCache) -> VirtualModelCache:
    """VirtualModelCache pre-populated with implicit ``autoprovisioned`` VMs.

    Mirrors the production behavior of ``provider_reconciler._ensure_passthrough_virtual_model``
    so tests don't need to manually create a VM for every fixture entity. Tests that want to
    exercise "no VirtualModel" behavior should override this fixture with an empty cache.
    """
    cache = VirtualModelCache()
    cache.rebuild(autoprovisioned_vms_for_cache(model_cache))
    return cache


@pytest.fixture
def middleware_registry() -> MiddlewareRegistry:
    """Empty MiddlewareRegistry — no plugins loaded. Proxy tests exercise pass-through."""
    return MiddlewareRegistry()


@pytest.fixture
def app_and_client(
    mocker, model_cache, virtual_model_cache, middleware_registry, mock_proxy_client, mock_nmp_sdk
) -> Iterator[tuple[FastAPI, TestClient]]:
    """
    This is a joint fixture for both a fastapi app and client. The reason they are combined
    is because we want an app fixture that has already had its `lifespan` function
    called. This doesn't happen until a TestClient is created from that app, and used in a
    context manager. See more details: https://fastapi.tiangolo.com/advanced/testing-events/.
    If you want only one or the other, you can use the more specific fixtures below.
    """

    # use a debug list of ModelProviders
    mocker.patch.object(
        config,
        "debug_model_providers",
        [
            DebugModelProvider(
                workspace="default",
                name="ollama",
                host_url="http://localhost:11434",
                served_models=[],
            )
        ],
    )

    service = InferenceGatewayService()
    mocker.patch.object(service.dependency_provider, "get_sdk_client", return_value=mock_nmp_sdk)
    app = service.app
    app.dependency_overrides[global_http_client] = lambda: mock_proxy_client
    app.dependency_overrides[global_model_cache] = lambda: model_cache
    app.dependency_overrides[global_virtual_model_cache] = lambda: virtual_model_cache
    app.dependency_overrides[global_middleware_registry] = lambda: middleware_registry
    with TestClient(app) as test_client:
        yield app, test_client


@pytest.fixture
def app(app_and_client: tuple[FastAPI, TestClient]) -> FastAPI:
    return app_and_client[0]


@pytest.fixture
def client(app_and_client: tuple[FastAPI, TestClient]) -> TestClient:
    return app_and_client[1]


@pytest.fixture
def mock_proxy_response():
    m = Mock(spec=aiohttp.ClientResponse)
    m.status = 200
    m.headers = CIMultiDictProxy(CIMultiDict([("content-type", "application/json")]))
    m.closed = False

    # In tests, you can modify this field to easily change what gets returned
    # by the request. Defaults to a minimal JSON object so the post-VM middleware
    # path (which buffers and json.loads the upstream response) works out of the box.
    m._body = [b'{"id":"test","choices":[{"message":{"content":"ok"}}]}']

    async def _async_chunk_iterator():
        for chunk in m._body:
            yield chunk

    m.content.iter_chunked = Mock(return_value=_async_chunk_iterator())
    m.content.iter_any = Mock(return_value=_async_chunk_iterator())

    # ``fetch_proxy_response`` (used by the VirtualModel middleware path that all
    # inference now flows through) calls ``read()`` to buffer the full body and
    # parse it as JSON. Concatenate ``_body`` so tests that override it still
    # control what the upstream "returns".
    async def _read():
        return b"".join(m._body)

    m.read = _read
    return m


@pytest.fixture
def mock_proxy_client(mock_proxy_response):
    """Create a mock HTTP client session."""
    m = Mock(spec=aiohttp.ClientSession)
    m.request = AsyncMock(return_value=mock_proxy_response)
    return m


@pytest.fixture
def mock_nmp_sdk():
    """Create a mock async NeMo Platform SDK client.

    This mocks AsyncNeMoPlatform for use with the inference gateway.
    """
    m = AsyncMock()
    return m
