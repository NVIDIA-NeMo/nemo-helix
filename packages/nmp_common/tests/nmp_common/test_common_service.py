# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nmp.common.service module."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.dependencies import get_nemo_client as plugin_get_nemo_client
from nmp.common.config import PlatformConfig
from nmp.common.observability.otel import scoped_otel_headers
from nmp.common.service import DependencyProvider, RouterConfig, Service
from nmp.common.service import __all__ as service_exports
from nmp.common.service import get_nemo_client as facade_get_nemo_client
from nmp.common.service.dependencies import get_nemo_client


class MockService(Service):
    """Mock implementation of Service for testing."""

    def __init__(self, dependency_provider: DependencyProvider | None = None):
        super().__init__(name="test-service", module_name="nmp.test", dependency_provider=dependency_provider)

    def get_routers(self) -> List[RouterConfig]:
        router = APIRouter()

        @router.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        return [RouterConfig(router, tag="Test", description="Test endpoints")]


class NestedRouterService(Service):
    """Service with an included router nested under its configured router."""

    def __init__(self):
        super().__init__(name="nested-router", module_name="nmp.test")

    def get_routers(self) -> List[RouterConfig]:
        child_router = APIRouter()

        @child_router.get("/child")
        async def child_endpoint():
            return {"message": "nested"}

        parent_router = APIRouter()
        parent_router.include_router(child_router, prefix="/nested")
        return [RouterConfig(parent_router, tag="Nested", description="Nested endpoints")]


class TestRouterConfig:
    """Tests for RouterConfig dataclass."""

    def test_router_config_creation(self):
        """Test creating a RouterConfig."""
        router = APIRouter()
        config = RouterConfig(router=router, tag="Test", description="Test description")

        assert config.router is router
        assert config.tag == "Test"
        assert config.description == "Test description"


class TestServiceBase:
    """Tests for Service base class."""

    def test_service_init(self):
        """Test Service initialization."""
        service = MockService()

        assert service.name == "test-service"
        assert service.module_name == "nmp.test"

    def test_service_title(self):
        """Test Service title property."""
        service = MockService()
        assert service.title == "Test Service Service"

    def test_service_description(self):
        """Test Service description property."""
        service = MockService()
        assert "Test Service Service" in service.description

    def test_service_version(self):
        """Test Service version property."""
        service = MockService()
        assert service.version == "0.0.1"

    @pytest.mark.asyncio
    async def test_service_is_ready_default(self):
        """Test is_ready() default returns True."""
        service = MockService()
        assert await service.is_ready() is True

    def test_service_repr(self):
        """Test Service __repr__."""
        service = MockService()
        assert "MockService" in repr(service)
        assert "test-service" in repr(service)

    def test_service_get_routers(self):
        """Test get_routers returns RouterConfig list."""
        service = MockService()
        routers = service.get_routers()

        assert len(routers) == 1
        assert isinstance(routers[0], RouterConfig)
        assert routers[0].tag == "Test"

    def test_service_create_app(self):
        """Test Service creates FastAPI app."""
        service = MockService()
        app = service.create_app()

        assert app is not None
        assert app.title == service.title
        assert app.version == service.version

    def test_service_app_property_caches(self):
        """Test app property returns cached instance."""
        service = MockService()
        app1 = service.app
        app2 = service.app

        assert app1 is app2

    def test_service_custom_router_included(self):
        """Test custom routers are included in app."""
        service = MockService()
        app = service.app

        with TestClient(app) as client:
            response = client.get("/test")

        assert response.status_code == 200
        assert response.json() == {"message": "test"}

    def test_service_tags_untagged_nested_router_routes_in_openapi(self):
        service = NestedRouterService()

        openapi = service.app.openapi()

        assert openapi["paths"]["/nested/child"]["get"]["tags"] == ["Nested"]

    def test_service_openapi_schema_is_generated_lazily(self, monkeypatch: pytest.MonkeyPatch):
        import nmp.common.service.base as service_base

        calls = 0

        def fake_get_openapi(**_kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {"components": {"schemas": {}}, "paths": {}}

        monkeypatch.setattr(service_base, "get_openapi", fake_get_openapi)

        service = MockService()
        app = service.app

        assert calls == 0
        assert app.openapi_schema is None

        first_schema = app.openapi()
        second_schema = app.openapi()

        assert calls == 1
        assert first_schema is second_schema


class TestServiceAsync:
    """Async tests for Service class."""

    @pytest.mark.asyncio
    async def test_service_startup(self):
        """Test startup runs without error; is_ready() remains True."""
        service = MockService()
        assert await service.is_ready() is True

        await service.startup()

        assert await service.is_ready() is True

    @pytest.mark.asyncio
    async def test_service_on_startup_default(self):
        """Test on_startup default implementation does nothing."""
        service = MockService()
        # Should not raise
        await service.on_startup()

    @pytest.mark.asyncio
    async def test_service_on_shutdown_default(self):
        """Test on_shutdown default implementation does nothing."""
        service = MockService()
        # Should not raise
        await service.on_shutdown()

    @pytest.mark.asyncio
    async def test_service_is_ready_default(self):
        """Test is_ready() default returns True."""
        service = MockService()
        assert await service.is_ready() is True

    @pytest.mark.asyncio
    async def test_wait_for_service_ready_retries_malformed_status_payloads(self):
        """Test malformed 200 /status payloads are retried."""
        requests: list[httpx.Request] = []
        responses = [
            httpx.Response(status_code=200, content=b"{"),
            httpx.Response(status_code=200, json=["not", "a", "mapping"]),
            httpx.Response(status_code=200, json={"services": {"ready": ["entities"]}}),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return responses.pop(0)

        provider = DependencyProvider()
        provider._platform_config = PlatformConfig(base_url="http://platform.local")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider._http_client = client
            service = MockService(dependency_provider=provider)

            ready = await service.wait_for_service_ready("entities", timeout=1.0, poll_interval=0)

        assert ready is True
        assert len(requests) == 3

    @pytest.mark.asyncio
    async def test_wait_for_service_ready_skips_service_absent_from_status(self):
        """Test dependencies absent from /status are treated as not part of this deployment."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"services": {"ready": ["entities"], "not_ready": []}})

        provider = DependencyProvider()
        provider._platform_config = PlatformConfig(base_url="http://platform.local")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider._http_client = client
            service = MockService(dependency_provider=provider)

            ready = await service.wait_for_service_ready("models", timeout=1.0, poll_interval=0)

        assert ready is True
        assert len(requests) == 1


class CloseCountingAsyncClient(httpx.AsyncClient):
    """Async transport that records lifecycle closure while retaining real HTTPX behavior."""

    def __init__(self) -> None:
        super().__init__(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1
        await super().aclose()


class TestDependencyProvider:
    """Tests for DependencyProvider class."""

    def test_init(self):
        """Test DependencyProvider initialization."""
        provider = DependencyProvider()
        assert provider._sdk_client is None
        assert provider._http_client is None

    def test_nemo_client_dependency_is_exported_with_exact_plugin_identity(self):
        assert get_nemo_client is plugin_get_nemo_client
        assert facade_get_nemo_client is plugin_get_nemo_client
        assert "get_nemo_client" in service_exports

    def test_setup_dependencies_registers_nemo_client_override(self):
        provider = DependencyProvider()
        app = FastAPI()

        provider.setup_dependencies(app, MockService())

        assert app.dependency_overrides[get_nemo_client] == provider.get_request_scoped_nemo_client

    @pytest.mark.asyncio
    @pytest.mark.parametrize("first_client", ["sdk", "nemo"], ids=["sdk-first", "nemo-first"])
    async def test_sdk_and_nemo_clients_share_provider_transport_regardless_of_order(self, first_client: str):
        provider = DependencyProvider()

        if first_client == "sdk":
            sdk = provider.get_request_scoped_sdk()
            nemo = provider.get_request_scoped_nemo_client()
        else:
            nemo = provider.get_request_scoped_nemo_client()
            sdk = provider.get_request_scoped_sdk()

        assert sdk._client is provider.get_http_client()
        assert nemo._http is provider.get_http_client()

        await provider.close()

    def test_request_scoped_nemo_clients_are_distinct_and_share_transport(self):
        provider = DependencyProvider()
        transport = AsyncMock(spec=httpx.AsyncClient)
        provider._http_client = transport

        with patch(
            "nmp.common.sdk_factory.get_principal_auth_headers",
            return_value={
                "X-NMP-Principal-Id": "user-one@example.com",
                "X-NMP-Principal-On-Behalf-Of": "delegate-one@example.com",
            },
        ):
            with scoped_otel_headers({"traceparent": "00-trace-one-span-one-01"}):
                first = provider.get_request_scoped_nemo_client()
        with patch(
            "nmp.common.sdk_factory.get_principal_auth_headers",
            return_value={"X-NMP-Principal-Id": "user-two@example.com"},
        ):
            with scoped_otel_headers({"traceparent": "00-trace-two-span-two-01"}):
                second = provider.get_request_scoped_nemo_client()

        assert first is not second
        assert first._http is transport
        assert second._http is transport
        assert first._default_headers["X-NMP-Principal-Id"] == "user-one@example.com"
        assert first._default_headers["X-NMP-Principal-On-Behalf-Of"] == "delegate-one@example.com"
        assert first._default_headers["traceparent"] == "00-trace-one-span-one-01"
        assert second._default_headers["X-NMP-Principal-Id"] == "user-two@example.com"
        assert second._default_headers["traceparent"] == "00-trace-two-span-two-01"

    @pytest.mark.asyncio
    async def test_close_closes_shared_sdk_and_nemo_transport_exactly_once(self):
        provider = DependencyProvider()
        transport = CloseCountingAsyncClient()
        provider._http_client = transport
        sdk = provider.get_sdk_client()
        nemo = provider.get_request_scoped_nemo_client()

        await provider.close()
        await provider.close()

        assert sdk._client is transport
        assert nemo._http is transport
        assert transport.close_count == 1
        assert provider._http_client is None
        assert provider._sdk_client is None

    @pytest.mark.asyncio
    async def test_concurrent_first_dependency_resolution_creates_one_transport_and_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from nmp.common import sdk_factory
        from nmp.common.service import base as service_base

        provider = DependencyProvider()
        resolution_ready = Barrier(13)
        factory_started = Event()
        release_factory = Event()
        created: list[CloseCountingAsyncClient] = []
        created_lock = Lock()

        def resolve_dependency(index: int) -> AsyncNeMoPlatform | AsyncNemoClient:
            resolution_ready.wait(timeout=5)
            factory = provider.get_request_scoped_sdk if index % 2 == 0 else provider.get_request_scoped_nemo_client
            return factory()

        def create_transport() -> CloseCountingAsyncClient:
            transport = CloseCountingAsyncClient()
            with created_lock:
                created.append(transport)
            factory_started.set()
            assert release_factory.wait(timeout=5)
            return transport

        endpoint = SimpleNamespace(async_sdk_http_client=lambda: create_transport())
        monkeypatch.setattr(service_base, "resolve_platform_endpoint", lambda: endpoint)

        with patch.object(
            sdk_factory, "get_async_platform_sdk", wraps=sdk_factory.get_async_platform_sdk
        ) as sdk_factory_call:
            with ThreadPoolExecutor(max_workers=12) as executor:
                futures = [executor.submit(resolve_dependency, index) for index in range(12)]
                resolution_ready.wait(timeout=5)
                assert factory_started.wait(timeout=5)
                time.sleep(0.05)
                release_factory.set()
                clients = [future.result(timeout=5) for future in futures]

        assert sdk_factory_call.call_count == 1

        transport = provider.get_http_client()
        sdk_clients = [client for client in clients if isinstance(client, AsyncNeMoPlatform)]
        nemo_clients = [client for client in clients if isinstance(client, AsyncNemoClient)]

        assert created == [transport]
        assert len({id(client) for client in sdk_clients}) == 1
        assert all(client._client is transport for client in sdk_clients)
        assert all(client._http is transport for client in nemo_clients)

        await provider.close()
        assert created[0].close_count == 1

    @pytest.mark.asyncio
    async def test_fastapi_caches_nemo_client_within_request_and_isolates_requests(self):
        provider = DependencyProvider()
        app = FastAPI()
        provider.setup_dependencies(app, MockService())
        resolved: list[tuple[AsyncNemoClient, AsyncNemoClient]] = []

        @app.get("/clients")
        async def clients(
            first: AsyncNemoClient = Depends(get_nemo_client),
            second: AsyncNemoClient = Depends(get_nemo_client),
        ) -> dict[str, bool]:
            resolved.append((first, second))
            return {"same": first is second}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first_response = await client.get("/clients")
            second_response = await client.get("/clients")

        assert first_response.json() == {"same": True}
        assert second_response.json() == {"same": True}
        assert resolved[0][0] is resolved[0][1]
        assert resolved[1][0] is resolved[1][1]
        assert resolved[0][0] is not resolved[1][0]
        assert resolved[0][0]._http is resolved[1][0]._http

        await provider.close()

    @pytest.mark.asyncio
    async def test_service_principal_sdk_shares_provider_transport(self):
        provider = DependencyProvider()
        cached_sdk = provider.get_sdk_client()
        service_sdk = provider.get_sdk_client(as_service="entities")

        assert service_sdk is not cached_sdk
        assert service_sdk._client is provider.get_http_client()
        assert cached_sdk._client is provider.get_http_client()

        await provider.close()

    @pytest.mark.asyncio
    async def test_close_without_clients(self):
        """Test close when no clients were created."""
        provider = DependencyProvider()
        await provider.close()  # Should not raise


class TestServiceWithProvider:
    """Tests for Service with DependencyProvider."""

    def test_service_has_provider(self):
        """Test Service creates DependencyProvider by default."""
        service = MockService()
        assert service.dependency_provider is not None
        assert isinstance(service.dependency_provider, DependencyProvider)


class LifecycleService(MockService):
    def __init__(self):
        super().__init__()
        self.events: list[str] = []
        self.started = threading.Event()

    async def on_startup(self) -> None:
        self.events.append("on_startup")

    async def startup(self) -> None:
        self.events.append("startup")
        self.started.set()
        await asyncio.Event().wait()

    async def on_shutdown(self) -> None:
        self.events.append("on_shutdown")
        await super().on_shutdown()


def test_service_lifespan_runs_startup_task_and_shutdown_cleanup() -> None:
    service = LifecycleService()

    with TestClient(service.app) as client:
        assert service.started.wait(timeout=1.0)
        assert client.get("/test").json() == {"message": "test"}

    assert service.events == ["on_startup", "startup", "on_shutdown"]
    assert service._startup_background_tasks
    assert service._startup_background_tasks[0].cancelled()


@pytest.mark.asyncio
async def test_wait_for_dependencies_returns_false_when_dependency_times_out() -> None:
    service = MockService()
    service._dependencies = ["entities", "auth"]

    with patch.object(service, "wait_for_service_ready", new=AsyncMock(side_effect=[True, False])) as wait:
        ready = await service._wait_for_dependencies(timeout=0.01)

    assert ready is False
    assert [call.args[0] for call in wait.await_args_list] == ["entities", "auth"]
