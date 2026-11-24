# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the OpenSandbox endpoint transport."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urljoin

import httpx
import pytest
from nemo_deployments_plugin.backends.opensandbox import endpoint as endpoint_module
from nemo_deployments_plugin.backends.opensandbox.endpoint import (
    OpenSandboxEndpointTransport,
    sandbox_endpoint_url,
)
from nemo_deployments_plugin.config import DeploymentsConfig, ExecutorConfigEntry
from nemo_deployments_plugin.endpoint_transport import endpoint_transport

pytest.importorskip("opensandbox")  # optional extra

SID = "7920fbcd-4e18-4753-bb15-41957b6f12e0"
OTHER = "11111111-2222-3333-4444-555555555555"
ADVERTISED = f"http://{SID}-8000.osb.helix.internal"
GATEWAY = "http://gateway.opensandbox-system.svc.cluster.local"
TOKEN_HEADER = "OpenSandbox-Secure-Access"
API_KEY_HEADER = "OPEN-SANDBOX-API-KEY"


class _FakeOpenSandbox:
    """OpenSandbox server endpoint lookup plus a uri-mode gateway with secure access on."""

    def __init__(self) -> None:
        self.lookups = 0
        self.lookup_status = 200
        self.generation = 0
        self.gateway_requests: list[tuple[str, dict[str, list[str]]]] = []

    def token(self, sandbox_id: str, port: str) -> str:
        return f"token-{sandbox_id}-{port}-{self.generation}"


class _Handler(BaseHTTPRequestHandler):
    state: _FakeOpenSandbox

    def _send(self, status: int, body: dict[str, Any] | None = None) -> None:
        payload = json.dumps(body or {}).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parts = self.path.split("?")[0].split("/")
        if len(parts) == 6 and parts[1:3] == ["v1", "sandboxes"] and parts[4] == "endpoints":
            self.state.lookups += 1
            if self.state.lookup_status != 200 or self.headers.get(API_KEY_HEADER) != "osb-key":
                return self._send(self.state.lookup_status if self.state.lookup_status != 200 else 401)
            return self._send(
                200, {"endpoint": "unused", "headers": {TOKEN_HEADER: self.state.token(parts[3], parts[5])}}
            )
        headers = {name: self.headers.get_all(name) or [] for name in (TOKEN_HEADER, API_KEY_HEADER)}
        self.state.gateway_requests.append((self.path, headers))
        if headers[TOKEN_HEADER] != [self.state.token(parts[1], parts[2])]:
            return self._send(401)
        return self._send(200, {"path": "/" + "/".join(parts[3:])})

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def osb(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[_FakeOpenSandbox, str]]:
    state = _FakeOpenSandbox()
    server = ThreadingHTTPServer(("127.0.0.1", 0), type("Handler", (_Handler,), {"state": state}))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "osb-key")
    endpoint_module._endpoint_headers.clear()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        endpoint_module._endpoint_headers.clear()


def _transport(base: str) -> OpenSandboxEndpointTransport:
    return OpenSandboxEndpointTransport(connect_base=base, server_base=base)


async def _get(
    transport: OpenSandboxEndpointTransport, url: str, headers: dict[str, str] | None = None
) -> httpx.Response:
    """GET ``url`` via ``target()`` and a client built from ``client_kwargs()``."""
    dial, routing_headers = transport.target(url)
    async with httpx.AsyncClient(timeout=5, **transport.client_kwargs()) as client:
        return await client.get(dial, headers={**(headers or {}), **routing_headers})


def test_advertised_url_names_the_sandbox_and_port() -> None:
    assert sandbox_endpoint_url(SID, 8000) == ADVERTISED


@pytest.mark.parametrize("sandbox_id", ["", "Upper-Case", "has/slash", "has.dot", "x" * 60])
def test_advertised_url_rejects_ids_that_are_not_a_host_label(sandbox_id: str) -> None:
    with pytest.raises(ValueError, match="host name"):
        sandbox_endpoint_url(sandbox_id, 8000)


def test_target_routes_an_advertised_url_through_the_gateway() -> None:
    transport = OpenSandboxEndpointTransport(connect_base=GATEWAY)
    assert transport.target(f"{ADVERTISED}/v1/chat/completions?stream=true") == (
        f"{GATEWAY}/{SID}/8000/v1/chat/completions?stream=true",
        {},
    )
    assert transport.target(ADVERTISED) == (f"{GATEWAY}/{SID}/8000/", {})


@pytest.mark.parametrize(
    "url",
    [
        f"{GATEWAY}/{SID}/8000/health",
        f"http://{SID}-8000.example.com/health",
        f"http://{SID}.osb.helix.internal/health",
    ],
)
def test_target_leaves_other_urls_alone(url: str) -> None:
    assert OpenSandboxEndpointTransport(connect_base=GATEWAY).target(url) == (url, {})


@pytest.mark.parametrize("trailing", ["../44772/command", f"../../{OTHER}/8000/command", "/44772/command"])
def test_a_path_joined_to_an_advertised_url_stays_on_its_sandbox_and_port(trailing: str) -> None:
    joined = urljoin(ADVERTISED + "/", trailing)  # the agents gateway's join
    dial, _ = OpenSandboxEndpointTransport(connect_base=GATEWAY).target(joined)
    assert dial.startswith(f"{GATEWAY}/{SID}/8000/")
    assert "/../" not in dial


async def test_request_carries_the_looked_up_headers_and_not_the_callers(osb: tuple[_FakeOpenSandbox, str]) -> None:
    state, base = osb
    response = await _get(_transport(base), f"{ADVERTISED}/health", headers={TOKEN_HEADER.lower(): "from-caller"})
    assert response.status_code == 200
    assert response.json() == {"path": "/health"}
    assert state.gateway_requests == [
        (f"/{SID}/8000/health", {TOKEN_HEADER: [state.token(SID, "8000")], API_KEY_HEADER: []})
    ]


async def test_headers_are_looked_up_once_per_sandbox_port(osb: tuple[_FakeOpenSandbox, str]) -> None:
    state, base = osb
    routes = [(SID, 8000), (SID, 8001), (OTHER, 8000)]
    for _ in range(2):
        for sandbox_id, port in routes:
            response = await _get(_transport(base), f"{sandbox_endpoint_url(sandbox_id, port)}/health")
            assert response.status_code == 200
    assert state.lookups == len(routes)


async def test_a_failed_lookup_fails_the_request_before_it_is_sent(osb: tuple[_FakeOpenSandbox, str]) -> None:
    state, base = osb
    state.lookup_status = 503
    with pytest.raises(httpx.ConnectError) as raised:
        await _get(_transport(base), f"{ADVERTISED}/health")
    assert str(raised.value) == "sandbox endpoint authorization is unavailable"
    assert state.gateway_requests == []


async def test_a_missing_api_key_fails_the_request(
    osb: tuple[_FakeOpenSandbox, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    state, base = osb
    monkeypatch.delenv("OPEN_SANDBOX_API_KEY")
    with pytest.raises(httpx.ConnectError):
        await _get(_transport(base), f"{ADVERTISED}/health")
    assert (state.lookups, state.gateway_requests) == (0, [])


async def test_requests_that_are_not_a_gateway_sandbox_route_are_refused(osb: tuple[_FakeOpenSandbox, str]) -> None:
    state, base = osb
    transport = _transport(base)
    for url in (f"{base}/health", f"{base}/{SID}/not-a-port/health", f"http://127.0.0.1:1/{SID}/8000/health"):
        with pytest.raises(httpx.ConnectError):
            await _get(transport, url)
    assert (state.lookups, state.gateway_requests) == (0, [])


async def test_a_401_drops_the_cached_headers(osb: tuple[_FakeOpenSandbox, str]) -> None:
    state, base = osb
    assert (await _get(_transport(base), f"{ADVERTISED}/health")).status_code == 200
    state.generation += 1  # rotate the token
    assert (await _get(_transport(base), f"{ADVERTISED}/health")).status_code == 401
    assert (await _get(_transport(base), f"{ADVERTISED}/health")).status_code == 200
    assert state.lookups == 2


def test_opensandbox_executor_resolves_to_its_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DeploymentsConfig(
        executors=[
            ExecutorConfigEntry(
                name="osb",
                backend="opensandbox",
                config={
                    "domain": "opensandbox-server.opensandbox-system.svc.cluster.local",
                    "gateway_address": GATEWAY.removeprefix("http://"),
                },
            )
        ]
    )
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: config))
    assert endpoint_transport("osb") == OpenSandboxEndpointTransport(
        connect_base=GATEWAY,
        server_base="http://opensandbox-server.opensandbox-system.svc.cluster.local",
        api_key_env="OPEN_SANDBOX_API_KEY",
    )
