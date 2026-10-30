# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real SDK must authenticate proxy requests without exposing tenant keys to direct endpoints.

The SDK is optional in the shared package. Runner CI requires it explicitly before running this
module; other consumers can run the package's SDK-free tests without installing it.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from nhx_sandbox.harbor_opensandbox_driver import Connection, OpenSandboxDriver

pytest.importorskip("opensandbox")

TENANT_KEY = "test-tenant-key"
ENDPOINT_HEADERS = {
    "X-Test-Route": "sandbox-route",
    "OpenSandbox-Secure-Access": "test-sandbox-access",
    "X-EXECD-ACCESS-TOKEN": "test-execd-access",
    "OPENSANDBOX-EGRESS-AUTH": "test-egress-access",
}


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's environment must not turn a no-key connection into an authenticated one."""
    monkeypatch.delenv("OPEN_SANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("OPENSANDBOX_API_KEY", raising=False)


@pytest.fixture
def captured_requests() -> list[httpx.Request]:
    """Requests captured below the real SDK's HTTP and streaming clients."""
    return []


@pytest.fixture
def transport(captured_requests: list[httpx.Request]) -> httpx.MockTransport:
    """Answer the SDK's requests without a server, sockets, or real credentials."""

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path.endswith("/command"):
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b'data: {"type":"execution_complete","timestamp":0,"execution_time":0}\n\n',
            )
        if request.url.path.endswith("/policy"):
            return httpx.Response(
                200,
                json={"status": "ok", "mode": "deny_all", "policy": {"defaultAction": "deny", "egress": []}},
            )
        if request.url.path == "/v1/sandboxes":
            return httpx.Response(
                200,
                json={
                    "items": [],
                    "pagination": {"page": 1, "pageSize": 20, "totalItems": 0, "totalPages": 0, "hasNextPage": False},
                },
            )
        return httpx.Response(200, json={})

    return httpx.MockTransport(respond)


@pytest.fixture
async def adapter_clients() -> AsyncIterator[list[Any]]:
    """Release every HTTP client, including the command adapter's separate SSE client."""
    clients: list[Any] = []
    yield clients
    for adapter in clients:
        await adapter._httpx_client.aclose()
        sse_client = getattr(adapter, "_sse_client", None)
        if sse_client is not None:
            await sse_client.aclose()


@pytest.mark.parametrize("service", ["health", "command", "filesystem", "egress"])
@pytest.mark.parametrize(
    ("use_server_proxy", "api_key", "expected_key"),
    [(True, TENANT_KEY, TENANT_KEY), (False, TENANT_KEY, None), (True, None, None), (True, "", None)],
    ids=["proxy", "direct", "no-key", "empty-key"],
)
async def test_sdk_requests_authenticate_only_the_server_proxy(
    service: str,
    use_server_proxy: bool,
    api_key: str | None,
    expected_key: str | None,
    transport: httpx.MockTransport,
    captured_requests: list[httpx.Request],
    adapter_clients: list[Any],
) -> None:
    """A dependency regression here recreates the multi-tenant readiness 401, including on SSE calls."""
    driver = OpenSandboxDriver(
        Connection(domain="server.example", api_key=api_key, protocol="http", use_server_proxy=use_server_proxy)
    )
    config = driver.connection_config()
    config.transport = transport
    models = importlib.import_module("opensandbox.models.sandboxes")
    endpoint = models.SandboxEndpoint(
        endpoint="server.example/v1/sandboxes/test-sandbox/proxy/44772" if use_server_proxy else "10.0.0.1:44772",
        headers=ENDPOINT_HEADERS,
    )
    adapter_module = importlib.import_module(f"opensandbox.adapters.{service}_adapter")
    adapter_name = {
        "health": "HealthAdapter",
        "command": "CommandsAdapter",
        "filesystem": "FilesystemAdapter",
        "egress": "EgressAdapter",
    }[service]
    adapter_type = getattr(adapter_module, adapter_name)
    adapter = adapter_type(config, endpoint)
    adapter_clients.append(adapter)

    if service == "health":
        assert await adapter.ping("test-sandbox")
    elif service == "command":
        result = await adapter.run("true")
        assert result.exit_code == 0
    elif service == "filesystem":
        entry = driver.sdk["WriteEntry"](path="/tmp/test.txt", data="hello")
        await adapter.write_files([entry])
    else:
        policy = await adapter.get_policy()
        assert policy.default_action == "deny"

    assert len(captured_requests) == 1
    headers = captured_requests[0].headers
    assert headers.get_list("OPEN-SANDBOX-API-KEY") == ([expected_key] if expected_key is not None else [])
    for name, value in ENDPOINT_HEADERS.items():
        assert headers[name] == value


async def test_sdk_lifecycle_requests_keep_the_tenant_key(
    transport: httpx.MockTransport, captured_requests: list[httpx.Request]
) -> None:
    """Removing the local proxy workaround must preserve the manager's lifecycle authentication."""
    driver = OpenSandboxDriver(Connection(domain="server.example", api_key=TENANT_KEY, protocol="http"))
    config = driver.connection_config()
    config.transport = transport
    manager = await driver.sdk["SandboxManager"].create(connection_config=config)
    try:
        result = await manager.list_sandbox_infos(driver.sdk["SandboxFilter"](metadata={"owner": "test"}))
        assert result.sandbox_infos == []
    finally:
        await manager.close()

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.url.path == "/v1/sandboxes"
    assert request.headers.get_list("OPEN-SANDBOX-API-KEY") == [TENANT_KEY]
