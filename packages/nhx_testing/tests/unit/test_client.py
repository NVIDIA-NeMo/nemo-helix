# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nhx_testing client utilities."""

import asyncio
import threading
from collections.abc import Mapping

import httpx
import pytest
from fastapi import Depends, FastAPI, Request, Response
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.config import Runtime
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient, WorkspacesClient
from nhx.common.auth import Principal
from nhx.common.config import AuthConfig, Configuration
from nhx.common.entities.client import EntityClient
from nhx.common.observability.otel import scoped_otel_headers
from nhx.common.service.dependencies import get_nemo_client, get_sync_nemo_client
from nhx.core.entities.service import EntitiesService
from nhx.core.inference_gateway.config import InferenceGatewayConfig
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.testing import (
    ClientContext,
    MockProviderResponse,
    TestClientHttpAdapter,
    as_user,
    async_nemo_client_for_app,
    create_test_client,
    mock_async_nemo_client,
    mock_nemo_client,
    nemo_client_for_test_client,
    request_scoped_nemo_client_overrides,
    short_unique_name,
    unique_email,
)


@pytest.fixture(autouse=True)
def clear_config_overrides():
    """Clear configuration overrides before and after each test."""
    Configuration.clear_overrides()
    yield
    Configuration.clear_overrides()


# =============================================================================
# short_unique_name tests
# =============================================================================


def test_short_unique_name_basic():
    """Test that short_unique_name generates unique names with prefix."""
    name1 = short_unique_name("test")
    name2 = short_unique_name("test")

    assert name1.startswith("test-")
    assert name2.startswith("test-")
    assert name1 != name2  # Should be unique


def test_short_unique_name_respects_max_length():
    """Test that short_unique_name respects max_length constraint."""
    name = short_unique_name("verylongprefix", max_length=20)
    assert len(name) <= 20


def test_short_unique_name_truncates_long_prefix():
    """Test that long prefixes are truncated to fit max_length."""
    # With max_length=15 and 8-char suffix + hyphen, prefix can be max 6 chars
    name = short_unique_name("abcdefghij", max_length=15)
    assert len(name) == 15
    assert name.startswith("abcdef-")


# =============================================================================
# unique_email tests
# =============================================================================


def test_unique_email_basic():
    """Test that unique_email generates unique emails."""
    email1 = unique_email()
    email2 = unique_email()

    assert email1.endswith("@example.com")
    assert email2.endswith("@example.com")
    assert email1 != email2  # Should be unique


def test_unique_email_custom_prefix():
    """Test that unique_email respects custom prefix."""
    email = unique_email("admin")
    assert email.startswith("admin-")
    assert email.endswith("@example.com")


# =============================================================================
# MockProviderResponse tests
# =============================================================================


def test_mock_provider_response_defaults():
    """Test that MockProviderResponse defaults response_code to 200."""
    response = MockProviderResponse(response_body={"ok": True})
    assert response.response_code == 200
    assert response.response_body == {"ok": True}


# =============================================================================
# create_test_client tests
# =============================================================================


def test_create_test_client_returns_typed_sync_client():
    """Test that client_type=NemoClient yields a typed client routed to the test app."""
    with create_test_client(EntitiesService, client_type=NemoClient, workspace="default") as client:
        assert isinstance(client, NemoClient)
        assert client.workspace == "default"
        assert WorkspacesClient.from_client(client).get_workspace(name="default").data().name == "default"


def test_create_test_client_returns_typed_async_client():
    """Test that client_type=AsyncNemoClient yields a typed client routed over ASGI."""
    with create_test_client(EntitiesService, client_type=AsyncNemoClient) as client:
        assert isinstance(client, AsyncNemoClient)


def test_create_test_client_returns_entity_client_backed_by_typed_client():
    """Test that client_type=EntityClient yields an EntityClient over the typed entities client."""
    with create_test_client(EntitiesService, client_type=EntityClient) as entity_client:
        assert isinstance(entity_client, EntityClient)
        assert isinstance(entity_client._client, AsyncEntitiesClient)


def test_create_test_client_returns_client_context():
    """Test that create_test_client with ClientContext returns ClientContext."""
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        assert isinstance(ctx, ClientContext)
        assert isinstance(ctx.client, NemoClient)
        assert isinstance(ctx.async_client, AsyncNemoClient)
        assert ctx.test_client is not None


def test_create_test_client_uses_loopback_platform_base_url():
    """Test that in-process platform clients use a loopback base URL."""
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        assert Configuration.get_platform_config().base_url == "http://127.0.0.1"
        assert ctx.client.base_url == "http://127.0.0.1"
        assert ctx.async_client.base_url == "http://127.0.0.1"
        assert str(ctx.test_client.base_url).rstrip("/") == "http://127.0.0.1"


def test_create_test_client_auth_defaults_use_platform_base_url():
    """Test that auth callouts target the same in-process platform URL."""
    with create_test_client(EntitiesService, client_type=ClientContext, auth_enabled=True):
        platform_base_url = Configuration.get_platform_config().base_url
        assert platform_base_url == "http://127.0.0.1"
        assert Configuration.get_service_config(AuthConfig).policy_decision_point_base_url == platform_base_url


def test_create_test_client_uses_docker_runtime_without_probing_docker(monkeypatch, mocker):
    """Test that the default platform config keeps the docker runtime without pinging the daemon."""
    monkeypatch.delenv("NHX_RUNTIME", raising=False)
    probe = mocker.patch("nemo_helix_plugin.capabilities._probe_docker_uncached")
    with create_test_client(EntitiesService, client_type=ClientContext):
        assert Configuration.get_platform_config().runtime == Runtime.DOCKER
    probe.assert_not_called()


def test_create_test_client_creates_default_workspace():
    """Test that create_test_client creates default workspace."""
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        # Default workspace should exist
        workspace = WorkspacesClient.from_client(ctx.client).get_workspace(name="default").data()
        assert workspace.name == "default"


def test_create_test_client_async_client_reaches_app():
    """Async callers and nested requests share the application's lifespan loop."""
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        app = ctx.test_client.app
        assert isinstance(app, FastAPI)
        assert ctx.test_client.portal is not None
        service_loop = ctx.test_client.portal.call(asyncio.get_running_loop)

        @app.get("/test-loop")
        async def get_loop():
            assert asyncio.get_running_loop() is service_loop
            workspace = await AsyncWorkspacesClient.from_client(ctx.async_client).get_workspace(name="default")
            return {"name": workspace.data().name}

        async def get_concurrently():
            responses = await asyncio.gather(*(ctx.async_client._http.get("/test-loop") for _ in range(4)))
            assert [response.json() for response in responses] == [{"name": "default"}] * 4

        asyncio.run(get_concurrently())
        asyncio.run(get_concurrently())


@pytest.mark.asyncio
async def test_async_request_stream_is_consumed_on_the_callers_loop():
    caller_loop = asyncio.get_running_loop()
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        app = ctx.test_client.app
        assert isinstance(app, FastAPI)

        @app.post("/test-body")
        async def echo_body(request: Request):
            return {"body": (await request.body()).decode()}

        async def body():
            assert asyncio.get_running_loop() is caller_loop
            yield b"hello"
            await asyncio.sleep(0)
            yield b" world"

        response = await ctx.async_client._http.post("/test-body", content=body())
        assert response.json() == {"body": "hello world"}


@pytest.mark.asyncio
async def test_async_upload_stops_reading_after_early_response():
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        app = ctx.test_client.app
        assert isinstance(app, FastAPI)

        @app.post("/test-body-limit")
        async def reject_large_body(request: Request):
            async for chunk in request.stream():
                assert len(chunk) > 8
                return Response(status_code=413)
            raise AssertionError("The application did not receive the first chunk")

        async def body():
            yield b"123456789"
            raise AssertionError("The transport read beyond the rejected chunk")

        response = await ctx.async_client._http.post("/test-body-limit", content=body())
        assert response.status_code == 413


@pytest.mark.asyncio
async def test_server_upload_timeout_cancels_body_on_callers_loop():
    caller_loop = asyncio.get_running_loop()
    body_stalled = threading.Event()
    body_closed = asyncio.Event()
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        app = ctx.test_client.app
        assert isinstance(app, FastAPI)

        @app.post("/test-body-timeout")
        async def read_with_timeout(request: Request):
            pending_read = asyncio.create_task(request.body())
            try:
                assert await asyncio.to_thread(body_stalled.wait, 5)
                await asyncio.wait_for(pending_read, 0.05)
            except TimeoutError:
                return Response(status_code=408)
            finally:
                pending_read.cancel()
                await asyncio.gather(pending_read, return_exceptions=True)
            raise AssertionError("The stalled upload unexpectedly completed")

        async def body():
            try:
                yield b"first chunk"
                body_stalled.set()
                await asyncio.Event().wait()
            finally:
                assert asyncio.get_running_loop() is caller_loop
                body_closed.set()

        response = await asyncio.wait_for(ctx.async_client._http.post("/test-body-timeout", content=body()), 5)
        assert response.status_code == 408
        await asyncio.wait_for(body_closed.wait(), 5)


@pytest.mark.asyncio
async def test_cancelling_async_upload_cancels_handler_and_body_read():
    caller_loop = asyncio.get_running_loop()
    body_started = asyncio.Event()
    body_closed = asyncio.Event()
    handler_cancelled = threading.Event()
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        app = ctx.test_client.app
        assert isinstance(app, FastAPI)

        @app.post("/test-body-cancellation")
        async def read_body(request: Request):
            try:
                await request.body()
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise

        async def body():
            try:
                yield b"first chunk"
                body_started.set()
                await asyncio.Event().wait()
            finally:
                assert asyncio.get_running_loop() is caller_loop
                body_closed.set()

        request = asyncio.create_task(ctx.async_client._http.post("/test-body-cancellation", content=body()))
        try:
            await asyncio.wait_for(body_started.wait(), 5)
        finally:
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        await asyncio.wait_for(body_closed.wait(), 5)
        assert await asyncio.to_thread(handler_cancelled.wait, 5)


def test_create_test_client_injects_provider_transport_for_typed_service_clients():
    """Test that DependencyProvider typed clients route through the in-process transport."""
    service = EntitiesService()
    with create_test_client(lambda: service, client_type=ClientContext) as ctx:
        app = ctx.test_client.app
        assert isinstance(app, FastAPI)
        provider = service.dependency_provider
        service_client = provider.get_service_nemo_client("entities")
        assert service_client._http is ctx.async_client._http

        async def _get() -> str:
            return (await AsyncWorkspacesClient.from_client(service_client).get_workspace(name="default")).data().name

        assert asyncio.run(_get()) == "default"
        assert provider.get_service_sync_nemo_client("entities")._http is ctx.client._http


# =============================================================================
# as_user tests
# =============================================================================


def test_typed_client_errors_surface_from_test_app():
    """Test that typed clients raise nemo_helix_plugin errors for app responses."""
    with create_test_client(EntitiesService, client_type=NemoClient) as client:
        with pytest.raises(NotFoundError):
            WorkspacesClient.from_client(client).get_workspace(name="does-not-exist")


def test_as_user_returns_new_client():
    """Test that as_user returns a new typed client carrying the principal headers."""
    with create_test_client(EntitiesService, client_type=ClientContext) as ctx:
        user_client = as_user(ctx.client, "test@example.com")
        assert user_client is not ctx.client
        assert user_client.default_headers["X-NHX-Principal-Id"] == "test@example.com"
        assert "X-NHX-Principal-Id" not in ctx.client.default_headers


# =============================================================================
# igw_mock_provider_mode tests
# =============================================================================


def test_mock_provider_mode_enabled_sets_prefix():
    """Test that igw_mock_provider_mode=True sets the prefix to 'igw-mock-'."""
    with create_test_client(
        InferenceGatewayService,
        client_type=ClientContext,
        igw_mock_provider_mode=True,
    ) as ctx:
        # Verify the config override is set correctly
        config = Configuration.get_service_config(InferenceGatewayConfig)
        assert config.mock_provider_prefix == "igw-mock-"
        assert ctx.client is not None


def test_mock_provider_mode_merges_with_existing_config():
    """Test that igw_mock_provider_mode merges with existing service_configs."""
    custom_refresh_interval = 999

    with create_test_client(
        InferenceGatewayService,
        client_type=ClientContext,
        igw_mock_provider_mode=True,
        service_configs={
            InferenceGatewayService: InferenceGatewayConfig(
                refresh_model_cache_interval_sec=custom_refresh_interval,
            ),
        },
    ) as ctx:
        # Verify both the custom config and mock_provider_prefix are set
        config = Configuration.get_service_config(InferenceGatewayConfig)
        assert config.mock_provider_prefix == "igw-mock-"
        assert config.refresh_model_cache_interval_sec == custom_refresh_interval
        assert ctx.client is not None


def test_mock_provider_mode_false_does_not_add_prefix():
    """Test that igw_mock_provider_mode=False doesn't add the igw-mock- prefix."""
    with create_test_client(
        InferenceGatewayService,
        client_type=ClientContext,
        igw_mock_provider_mode=False,
        service_configs={
            InferenceGatewayService: InferenceGatewayConfig(
                refresh_model_cache_interval_sec=123,
            ),
        },
    ) as ctx:
        # Verify that when igw_mock_provider_mode=False, the prefix is not set
        config = Configuration.get_service_config(InferenceGatewayConfig)
        # Our custom config is there
        assert config.refresh_model_cache_interval_sec == 123
        # But mock_provider_prefix should be None (the default)
        assert config.mock_provider_prefix is None
        assert ctx.client is not None


def test_config_overrides_cleared_after_context_exit():
    """Test that config overrides are properly cleared after context exits."""
    # First, enable mock provider mode
    with create_test_client(
        InferenceGatewayService,
        client_type=ClientContext,
        igw_mock_provider_mode=True,
    ):
        config = Configuration.get_service_config(InferenceGatewayConfig)
        assert config.mock_provider_prefix == "igw-mock-"

    # After context exit, overrides should be cleared
    assert InferenceGatewayConfig not in Configuration._overrides


# =============================================================================
# Typed client helpers
# =============================================================================


def _workspace_json(name: str) -> dict[str, str]:
    return {
        "id": name,
        "name": name,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


def test_mock_nemo_client_routes_requests_to_handler():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_workspace_json("default"))

    client = mock_nemo_client(handler, WorkspacesClient, headers={"X-Test": "1"})

    assert isinstance(client, WorkspacesClient)
    assert client.get_workspace(name="default").data().name == "default"
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/apis/entities/v2/workspaces/default"
    assert seen[0].headers["X-Test"] == "1"
    client.close()


@pytest.mark.asyncio
async def test_mock_async_nemo_client_routes_requests_to_handler():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "missing"})

    client = mock_async_nemo_client(handler, AsyncWorkspacesClient)

    with pytest.raises(NotFoundError):
        await client.get_workspace(name="missing")
    await client.close()


def _principal_echo_app() -> FastAPI:
    app = FastAPI()

    @app.get("/echo")
    async def echo(client: AsyncNemoClient = Depends(get_nemo_client)) -> dict[str, str]:
        return dict(client.default_headers)

    @app.get("/echo-sync")
    def echo_sync(client: NemoClient = Depends(get_sync_nemo_client)) -> dict[str, str]:
        return dict(client.default_headers)

    return app


@pytest.mark.asyncio
async def test_async_nemo_client_for_app_reaches_app():
    app = FastAPI()

    @app.get("/apis/entities/v2/workspaces/{name}")
    async def get_workspace(name: str) -> dict[str, str]:
        return _workspace_json(name)

    async with async_nemo_client_for_app(app, AsyncWorkspacesClient) as client:
        assert (await client.get_workspace(name="ws-a")).data().name == "ws-a"


def test_nemo_client_for_test_client_reaches_app():
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/apis/entities/v2/workspaces/{name}")
    def get_workspace(name: str) -> dict[str, str]:
        return _workspace_json(name)

    with TestClient(app) as test_client:
        client = nemo_client_for_test_client(test_client, WorkspacesClient)
        assert isinstance(client._http, TestClientHttpAdapter)
        assert client.get_workspace(name="ws-b").data().name == "ws-b"


def test_request_scoped_nemo_client_overrides_carry_request_headers():
    from fastapi.testclient import TestClient
    from nhx.common.auth import auth_client_context
    from nhx.common.auth.client import AuthClient

    app = _principal_echo_app()
    base_async = AsyncNemoClient(base_url="http://nemo.test", default_headers={"X-Base": "1"})
    base_sync = NemoClient(base_url="http://nemo.test", default_headers={"X-Base": "1"})
    app.dependency_overrides.update(request_scoped_nemo_client_overrides(base_async, base_sync))

    auth_client = AuthClient(
        config=AuthConfig(enabled=False),
        principal=Principal(id="user@example.com", email="user@example.com"),
    )
    token = auth_client_context.set(auth_client)
    try:
        with scoped_otel_headers({"traceparent": "00-trace-span-01"}):
            with TestClient(app) as test_client:
                async_headers = test_client.get("/echo").json()
                sync_headers = test_client.get("/echo-sync").json()
    finally:
        auth_client_context.reset(token)

    for headers in (async_headers, sync_headers):
        assert headers["X-Base"] == "1"
        assert headers["X-NHX-Principal-Id"] == "user@example.com"
        assert headers["traceparent"] == "00-trace-span-01"
    assert "X-NHX-Principal-Id" not in base_async.default_headers


class _ScriptedPostClient:
    def __init__(self, status_codes: list[int]) -> None:
        self.status_codes = list(status_codes)
        self.calls = 0

    def post(self, url: str, *, json: Mapping[str, str], headers: Mapping[str, str]) -> httpx.Response:
        self.calls += 1
        self.last = (url, json, headers)
        return httpx.Response(self.status_codes.pop(0), text="denied")


def test_post_after_auth_warmup_retries_the_first_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    from nhx.testing.client import _post_after_auth_warmup

    monkeypatch.setattr("nhx.testing.client.time.sleep", lambda _seconds: None)
    client = _ScriptedPostClient([403, 201])

    _post_after_auth_warmup(
        client,
        "/apis/entities/v2/workspaces",
        json={"name": "destination"},
        headers={"X-NHX-Principal-Id": "user@example.com"},
    )

    assert client.calls == 2


def test_post_after_auth_warmup_raises_when_the_denial_is_not_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    from nhx.testing.client import _post_after_auth_warmup

    monkeypatch.setattr("nhx.testing.client.time.sleep", lambda _seconds: None)
    client = _ScriptedPostClient([422])

    with pytest.raises(RuntimeError, match="422"):
        _post_after_auth_warmup(
            client,
            "/apis/entities/v2/workspaces",
            json={"name": "destination"},
            headers={"X-NHX-Principal-Id": "user@example.com"},
        )
