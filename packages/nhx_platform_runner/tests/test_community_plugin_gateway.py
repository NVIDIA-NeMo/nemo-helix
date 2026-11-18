# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlunsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.authz_discovery import _derive_service_contribution
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.community_plugins.registry import EndpointMapping, ExternalPluginRegistry, RegistryStore
from nemo_helix_plugin.dependencies import get_nemo_client
from nhx.platform_runner.community_plugins import gateway
from nhx.platform_runner.community_plugins.gateway import ExternalPluginsService
from pydantic import ValidationError


def registration(**overrides: object) -> dict:
    return {
        "name": "sample",
        "title": "Sample API",
        "endpoints": [
            {
                "method": "GET",
                "path": "/items/{item}",
                "access": "read",
                "upstream_url": "https://external.example/api/items/{item}",
            },
            {
                "method": "POST",
                "path": "/query",
                "access": "write",
                "upstream_url": "https://external.example/api/query",
            },
            {
                "method": "GET",
                "path": "/v2/workspaces/{workspace}/status",
                "access": "read",
                "upstream_url": "http://127.0.0.1:8081/apis/legacy/v2/workspaces/{workspace}/status",
            },
        ],
        **overrides,
    }


def application(service: ExternalPluginsService, platform_client: AsyncNemoClient | None = None) -> FastAPI:
    request_client = platform_client or AsyncNemoClient(base_url="http://platform.invalid")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            await service.on_shutdown()
            await request_client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.dependency_overrides[get_nemo_client] = lambda: request_client
    app.include_router(service.get_routers()[0].router, prefix="/apis/plugins")
    return app


@pytest.fixture
def state(tmp_path: Path) -> Path:
    path = tmp_path / "installed.json"
    path.write_text(json.dumps({"schema_version": 1, "plugins": [registration()]}))
    return path


@pytest.fixture
def upstream_calls() -> list[httpx.Request]:
    return []


@pytest.fixture
def client(state: Path, upstream_calls: list[httpx.Request]) -> Iterator[TestClient]:
    def respond(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request)
        return httpx.Response(
            200,
            json={"result": "ok"},
            headers={
                "set-cookie": "external_session=private",
                "x-nhx-principal": "forged",
                "access-control-allow-origin": "*",
                "mcp-session-id": "opaque-session",
                "connection": "keep-alive",
                "x-secret": "do-not-forward",
            },
        )

    service = ExternalPluginsService(state)
    service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with TestClient(application(service)) as test_client:
        yield test_client


def test_proxy_strips_platform_credentials_and_unapproved_headers(
    client: TestClient,
    upstream_calls: list[httpx.Request],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_PLUGIN_SAMPLE_AUTHORIZATION", "Bearer must-not-implicitly-forward")
    response = client.get(
        "/apis/plugins/sample/items/hello?a=1&a=2",
        headers={
            "authorization": "Bearer platform-secret",
            "cookie": "platform_session=secret",
            "x-nhx-principal": "admin",
            "x-nhx-internal": "true",
            "x-forwarded-for": "spoofed",
            "proxy-authorization": "secret",
            "accept": "application/json",
            "mcp-session-id": "caller-session",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"result": "ok"}
    request = upstream_calls[0]
    assert str(request.url) == "https://external.example/api/items/hello?a=1&a=2"
    assert "authorization" not in request.headers
    assert request.headers["accept"] == "application/json"
    assert request.headers["mcp-session-id"] == "caller-session"
    for name in ["cookie", "x-nhx-principal", "x-nhx-internal", "x-forwarded-for", "proxy-authorization"]:
        assert name not in request.headers
    for name in ["set-cookie", "x-nhx-principal", "access-control-allow-origin", "x-secret", "connection"]:
        assert name not in response.headers
    assert response.headers["mcp-session-id"] == "opaque-session"
    assert response.headers["cache-control"] == "no-store"
    assert "sandbox" in response.headers["content-security-policy"]


def test_no_ambient_cookie_or_platform_auth_forwarding(client: TestClient, upstream_calls: list[httpx.Request]) -> None:
    for _ in range(2):
        assert (
            client.get("/apis/plugins/sample/items/id", headers={"authorization": "Bearer platform"}).status_code == 200
        )
    for request in upstream_calls:
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers


def test_proxy_body_and_workspace_mapping(client: TestClient, upstream_calls: list[httpx.Request]) -> None:
    assert client.post("/apis/plugins/sample/query", json={"query": "hello"}).status_code == 200
    assert json.loads(upstream_calls[-1].content) == {"query": "hello"}
    assert client.get("/apis/plugins/sample/v2/workspaces/default/status").status_code == 200
    assert str(upstream_calls[-1].url) == "http://127.0.0.1:8081/apis/legacy/v2/workspaces/default/status"


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/apis/plugins/missing/items/id"),
        ("GET", "/apis/plugins/sample/undeclared"),
        ("POST", "/apis/plugins/sample/items/id"),
        ("DELETE", "/apis/plugins/sample/query"),
        ("GET", "/apis/plugins/sample/items/%252fescape"),
        ("GET", "/apis/plugins/sample/items/%2fescape"),
    ],
)
def test_undeclared_requests_never_contact_upstream(
    client: TestClient,
    upstream_calls: list[httpx.Request],
    method: str,
    path: str,
) -> None:
    assert client.request(method, path).status_code == 404
    assert upstream_calls == []


def test_disabled_plugin_and_live_registry_reload(
    client: TestClient,
    state: Path,
    upstream_calls: list[httpx.Request],
) -> None:
    assert client.get("/apis/plugins/sample/items/id").status_code == 200
    state.write_text(json.dumps({"plugins": [registration(enabled=False)]}))
    assert client.get("/apis/plugins/sample/items/id").status_code == 404
    assert len(upstream_calls) == 1


def test_installed_metadata_and_invalid_registry_fail_closed(
    client: TestClient,
    state: Path,
    upstream_calls: list[httpx.Request],
) -> None:
    metadata = client.get("/apis/plugins/-/installed")
    assert metadata.status_code == 200
    assert metadata.json()["plugins"][0]["name"] == "sample"
    state.write_text("not-json")
    assert client.get("/apis/plugins/-/installed").status_code == 503
    assert client.get("/apis/plugins/sample/items/id").status_code == 503
    assert upstream_calls == []


def test_request_size_limit(
    client: TestClient, upstream_calls: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gateway, "_MAX_REQUEST_BYTES", 4)
    assert client.post("/apis/plugins/sample/query", content=b"12345").status_code == 413
    assert upstream_calls == []


@pytest.mark.parametrize("result,expected", [(302, 502), (401, 401), (500, 500), (204, 204)])
def test_upstream_status_and_no_redirects(state: Path, result: int, expected: int) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(result, headers={"location": "https://other.example/stolen", "set-cookie": "secret=1"})

    service = ExternalPluginsService(state)
    service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True)
    with TestClient(application(service)) as client:
        response = client.get("/apis/plugins/sample/items/id")
    assert response.status_code == expected
    assert "location" not in response.headers
    assert len(calls) == 1


@pytest.mark.parametrize("exception,expected", [(httpx.ReadTimeout, 504), (httpx.ConnectError, 502)])
def test_network_errors(state: Path, exception: type[httpx.RequestError], expected: int) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise exception("upstream details must not leak", request=request)

    service = ExternalPluginsService(state)
    service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    with TestClient(application(service)) as client:
        response = client.get("/apis/plugins/sample/items/id")
    assert response.status_code == expected
    assert "must not leak" not in response.text


def test_decoded_response_does_not_forward_old_encoding_or_length(state: Path) -> None:
    data = b'{"result":"compressed"}'
    service = ExternalPluginsService(state)
    service.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=gzip.compress(data),
                headers={"content-encoding": "gzip", "content-type": "application/json"},
            )
        )
    )
    with TestClient(application(service)) as client:
        response = client.get("/apis/plugins/sample/items/id")
    assert response.content == data
    assert "content-encoding" not in response.headers
    assert "content-length" not in response.headers


def test_openapi_has_unique_operation_ids_and_workspace_path_parameter(state: Path) -> None:
    schema = application(ExternalPluginsService(state)).openapi()
    operations = [operation for path in schema["paths"].values() for operation in path.values()]
    ids = [operation["operationId"] for operation in operations]
    assert len(ids) == len(set(ids))
    for namespace in ("workspace", "platform"):
        for method in ("get", "head", "post", "put", "patch", "delete"):
            assert f"plugins_proxy_{namespace}_{method}" in ids
    parameters = schema["paths"]["/apis/plugins/{name}/v2/workspaces/{workspace}/{plugin_endpoint}"]["get"][
        "parameters"
    ]
    assert any(parameter["name"] == "workspace" and parameter["in"] == "path" for parameter in parameters)


def test_gateway_authz_is_static_and_workspace_aware(state: Path) -> None:
    contribution, problems, warnings = _derive_service_contribution(ExternalPluginsService(state))
    assert problems == []
    paths = contribution.endpoints
    generic = paths["/apis/plugins/{name}/{plugin_endpoint:path}"]
    workspace = paths["/apis/plugins/{name}/v2/workspaces/{workspace}/{plugin_endpoint:path}"]
    assert generic["get"].permissions == ["plugins.gateway.read"]
    assert generic["post"].permissions == ["plugins.gateway.write"]
    assert generic["get"].scopes == ["plugins:read", "platform:read"]
    assert generic["post"].scopes == ["plugins:write", "platform:write"]
    assert workspace["get"].permissions == generic["get"].permissions
    assert workspace["post"].permissions == generic["post"].permissions


@pytest.mark.parametrize(
    "updates",
    [
        # Construct dummy userinfo separately to avoid a secret-shaped URI literal.
        {"upstream_url": urlunsplit(("https", "user:secret@example.org", "/api/{item}", "", ""))},
        {"upstream_url": "https://example.org/api/{item}?token=secret"},
        {"upstream_url": "https://example.org/api/{item}#fragment"},
        {"upstream_url": "http://external.example/api/{item}"},
        {"upstream_url": "file:///etc/passwd"},
        {"upstream_url": "https://{item}.example.org/api/item"},
        {"upstream_url": "https://example.org/api/{undeclared}"},
        {"upstream_url": "https://example.org:99999/api/item"},
        {"path": "/items/../escape"},
        {"path": "/items/%2e%2e"},
        {"path": "/items/{item:path}"},
        {"path": "/-/installed"},
        {"access": "write"},
    ],
)
def test_invalid_mappings_rejected(updates: dict) -> None:
    values = registration()["endpoints"][0] | updates
    with pytest.raises(ValidationError):
        EndpointMapping.model_validate(values)


def test_duplicate_and_ambiguous_state_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="names must be unique"):
        ExternalPluginRegistry.model_validate({"plugins": [registration(), registration()]})
    endpoints = registration()["endpoints"]
    with pytest.raises(ValidationError, match="Overlapping"):
        ExternalPluginRegistry.model_validate({"plugins": [registration(endpoints=[endpoints[0], endpoints[0]])]})
    assert RegistryStore(tmp_path / "missing.json").load().plugins == ()


def test_wildcard_workspace_is_not_forwarded(client: TestClient, upstream_calls: list[httpx.Request]) -> None:
    response = client.get("/apis/plugins/sample/v2/workspaces/-/status")
    assert response.status_code == 400
    assert upstream_calls == []


@pytest.mark.parametrize("content_type", ["text/html", "text/javascript", "application/javascript", "image/svg+xml"])
def test_api_gateway_is_not_an_unpinned_script_host(state: Path, content_type: str) -> None:
    service = ExternalPluginsService(state)
    service.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b"untrusted active content",
                headers={"content-type": content_type},
            )
        )
    )
    with TestClient(application(service)) as client:
        response = client.get("/apis/plugins/sample/items/id")
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"] == "attachment"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_sse_stream_is_relayed_and_closed(state: Path) -> None:
    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self) -> AsyncGenerator[bytes]:
            yield b"data: one\n\n"
            yield b"data: two\n\n"

        async def aclose(self) -> None:
            self.closed = True

    stream = Stream()
    service = ExternalPluginsService(state)
    service.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                stream=stream,
                headers={"content-type": "text/event-stream"},
            )
        )
    )
    with TestClient(application(service)) as client:
        response = client.get("/apis/plugins/sample/items/id")
    assert response.content == b"data: one\n\ndata: two\n\n"
    assert response.headers["content-type"] == "text/event-stream"
    assert stream.closed
    assert service.http_client is None


def test_connection_nominated_headers_are_removed(state: Path) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={}, headers={"connection": "mcp-session-id", "mcp-session-id": "private"})

    service = ExternalPluginsService(state)
    service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with TestClient(application(service)) as client:
        response = client.get(
            "/apis/plugins/sample/items/id",
            headers={
                "connection": "mcp-protocol-version",
                "mcp-protocol-version": "private",
            },
        )
    assert "mcp-protocol-version" not in requests[0].headers
    assert "mcp-session-id" not in response.headers


def test_default_store_is_in_platform_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_DATA_DIR", str(tmp_path))
    assert ExternalPluginsService().store.path == tmp_path / "plugins" / "installed.json"


async def test_client_lifecycle_has_no_ambient_credentials(state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://not-a-real-proxy.invalid")
    service = ExternalPluginsService(state)
    await service.on_startup()
    client = service.http_client
    assert client is not None
    assert not client.trust_env
    assert not client.follow_redirects
    assert client.auth is None
    await service.on_shutdown()
    assert client.is_closed
    assert service.http_client is None
