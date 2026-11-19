# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.config import clear_nemo_config_override, set_nemo_config_override
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_observability_plugin.config import ObservabilityConfig, ObservabilityConnection
from nemo_observability_plugin.service import ObservabilityService

MLFLOW = ObservabilityConnection(
    name="mlflow", provider="mlflow", base_url="http://mlflow.test:5000", default_project="7"
)
PHOENIX = ObservabilityConnection(name="phoenix", provider="phoenix", base_url="http://phoenix.test:6006")


@pytest.fixture
def config() -> Iterator[ObservabilityConfig]:
    cfg = ObservabilityConfig(connections=[MLFLOW, PHOENIX])
    set_nemo_config_override(cfg)
    yield cfg
    clear_nemo_config_override(ObservabilityConfig)


UpstreamHandler = Callable[[httpx.Request], httpx.Response]


class _Stream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body


@pytest.fixture
def upstream_requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def make_client(config: ObservabilityConfig, upstream_requests: list[httpx.Request]):
    """TestClient for the mounted plugin whose upstream is answered by *handler*."""

    def _make(handler: UpstreamHandler | None = None) -> TestClient:
        def record(request: httpx.Request) -> httpx.Response:
            upstream_requests.append(request)
            canned = (handler or (lambda _: httpx.Response(200, text="ok")))(request)
            # A Response built from content= is pre-read; re-wrap it so the proxy can stream it like a real one.
            return httpx.Response(canned.status_code, headers=canned.headers, stream=_Stream(canned.content))

        service = ObservabilityService()
        service._client = httpx.AsyncClient(transport=httpx.MockTransport(record), follow_redirects=False)
        app = FastAPI()
        for spec in service.get_routers():
            app.include_router(spec.router, prefix=f"/apis/observability{spec.prefix}")
        app.dependency_overrides[get_effective_principal_id] = lambda: "user-1"
        return TestClient(app)

    return _make
