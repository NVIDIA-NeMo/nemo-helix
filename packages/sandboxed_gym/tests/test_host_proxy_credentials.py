# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox API key on requests to the Gym host, and nowhere it could leak.

An OpenSandbox server with tenants checks the API key on proxied requests too, not only on its own
API. Every request this package sends to the Gym host goes through that proxy, so without the key
each one is refused with 401 and the host never reports ready. The key can create, delete and exec
into any sandbox of the tenant, so it must stay in this process:

* out of the session descriptor, which reaches sandboxed job code through the proxy's
  unauthenticated ``/session`` route, other jobs through session files, and NeMo-RL through the Ray
  actor;
* off any request that follows a redirect: the host runs untrusted code, its responses come back
  through the proxy, and urllib copies every header onto a redirected request wherever it points.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import sandboxed_gym.host.models as host_models
from fastapi.testclient import TestClient
from sandboxed_gym.config import BrokerEndpoint
from sandboxed_gym.host.models import GymHostHandle, GymHostSpec, GymHostVolumeMount, open_host
from sandboxed_gym.host.opensandbox import OPEN_SANDBOX_API_KEY_HEADER, OpenSandboxGymHostProvider
from sandboxed_gym.orchestrator import SandboxedGymSession
from sandboxed_gym.proxy_app import AUTH_HEADER, build_proxy_app
from sandboxed_gym.serve import write_session_file
from sandboxed_gym.serve_config import SandboxedGymServeConfig

KEY = "tenant-key-0123456789"
CREDENTIALS = {OPEN_SANDBOX_API_KEY_HEADER: KEY}
#: What the server hands back for a proxied endpoint: routing headers, never the API key.
ENDPOINT_HEADERS = {"OpenSandbox-Secure-Access": "per-sandbox-token"}
HEALTH_URL = "http://server/v1/sandboxes/gym-1/proxy/8080/health"
ROLLOUT_URL = "http://server/v1/sandboxes/gym-1/proxy/8080/rollouts/run"


def _host(health_url: str = HEALTH_URL, rollout_url: str = ROLLOUT_URL) -> GymHostHandle:
    return GymHostHandle(
        host_id="gym-1",
        health_url=health_url,
        rollout_url=rollout_url,
        headers=ENDPOINT_HEADERS,
        _credentials=CREDENTIALS,
    )


# ---------------------------------------------------------------------------------------------
# The provider: when the key is attached
# ---------------------------------------------------------------------------------------------


class _Routes:
    def __init__(self, headers: dict[str, str]) -> None:
        self.health_url = HEALTH_URL
        self.rollout_url = ROLLOUT_URL
        self.headers = headers


class _ResourceHandle:
    sandbox_id = "gym-1"
    raw = object()


class _StubDriver:
    async def create(self, spec: GymHostSpec) -> _ResourceHandle:
        return _ResourceHandle()


def _spec() -> GymHostSpec:
    return GymHostSpec(
        job_id="job-1",
        runtime_image="example/gym:latest",
        environment_mount=GymHostVolumeMount(pvc_claim="env", mount_path="/job/environment", read_only=True),
        workspace_mount=GymHostVolumeMount(pvc_claim="work", mount_path="/job/work"),
    )


def _created_host(monkeypatch: pytest.MonkeyPatch, connection: dict[str, Any]) -> GymHostHandle:
    provider = OpenSandboxGymHostProvider(connection=connection)

    async def _resolve(raw: Any, port: int) -> _Routes:
        return _Routes(dict(ENDPOINT_HEADERS))

    monkeypatch.setattr(provider, "_provider_for_spec", lambda spec: _StubDriver())
    monkeypatch.setattr(provider, "_to_sandbox_spec", lambda spec: spec)
    monkeypatch.setattr(provider, "_resolve_routes", _resolve)
    return asyncio.run(provider.create_host(_spec()))


def test_a_host_behind_the_server_proxy_carries_the_key_as_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server proxy is the default route to the host, and a server with tenants requires the key on it."""
    monkeypatch.delenv("OPEN_SANDBOX_API_KEY", raising=False)

    host = _created_host(monkeypatch, {"domain": "server", "api_key": KEY})

    assert host._credentials == CREDENTIALS
    # The shareable headers stay exactly what the server returned for the address.
    assert host.headers == ENDPOINT_HEADERS


def test_the_key_comes_from_the_sdks_environment_variable_when_the_connection_names_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Jobs get the key as OPEN_SANDBOX_API_KEY, which the SDK reads when the connection has none."""
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", KEY)

    host = _created_host(monkeypatch, {"domain": "server"})

    assert host._credentials == CREDENTIALS


def test_a_host_reached_directly_never_gets_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the proxy, requests go straight to the sandbox, which runs untrusted environment code."""
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", KEY)

    host = _created_host(monkeypatch, {"domain": "server", "api_key": KEY, "use_server_proxy": False})

    assert host._credentials == {}


def test_no_key_configured_adds_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty header would be refused just as a missing one is, and reads as a configured key."""
    monkeypatch.delenv("OPEN_SANDBOX_API_KEY", raising=False)

    host = _created_host(monkeypatch, {"domain": "server"})

    assert host._credentials == {}


def test_the_handle_exposes_the_key_only_to_open_host() -> None:
    """No public field, method or ``repr`` hands the key to a caller who could send it some other way."""
    host = _host()

    public = {f.name: getattr(host, f.name) for f in fields(host) if not f.name.startswith("_")}
    assert KEY not in repr(public)
    assert KEY not in repr(host)
    assert not any(callable(getattr(host, name)) for name in dir(host) if not name.startswith("_"))


# ---------------------------------------------------------------------------------------------
# Every request to the host sends the key
# ---------------------------------------------------------------------------------------------


class _Response:
    """A finished upstream response, for both ``with open_host(...)`` and the proxy's streaming reads."""

    status = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        body, self._body = self._body, b""
        return body

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    def close(self) -> None:
        return None


def _record_requests(monkeypatch: pytest.MonkeyPatch, body: bytes) -> list[dict[str, str]]:
    """Answer every request ``open_host`` sends with ``body``, recording each request's headers."""
    sent: list[dict[str, str]] = []

    def _open(request: Request, timeout: float | None = None) -> _Response:
        # urllib capitalizes header names; compare case-insensitively.
        sent.append({name.lower(): value for name, value in request.header_items()})
        return _Response(body)

    monkeypatch.setattr(host_models, "build_opener", lambda *handlers: SimpleNamespace(open=_open))
    return sent


def _session(host: GymHostHandle | None = None, rollout_auth_token: str | None = None) -> SandboxedGymSession:
    cfg = SandboxedGymServeConfig.model_validate(
        {
            "job_id": "job-1",
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
                "rollout_retry_backoff_s": 0,
            },
            "rollout_auth_token": rollout_auth_token,
        }
    )
    return SandboxedGymSession(
        cfg=cfg,
        broker_server=MagicMock(),
        broker=BrokerEndpoint(url="http://broker.svc:1", host="broker.svc", port=1, token="btok"),
        host_provider=MagicMock(),
        host=host or _host(),
        orchestrator_url="http://orch.svc:8090",
    )


def test_the_readiness_poll_sends_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without it the server answers 401 to every poll and the job fails waiting for a healthy host."""
    sent = _record_requests(monkeypatch, json.dumps({"status": "ready"}).encode())
    provider = OpenSandboxGymHostProvider(connection={"domain": "server", "api_key": KEY})

    asyncio.run(provider.wait_ready(_host(), timeout_s=5))

    assert sent[0][OPEN_SANDBOX_API_KEY_HEADER.lower()] == KEY
    assert sent[0]["opensandbox-secure-access"] == "per-sandbox-token"


def test_rollouts_send_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluator collects through ``arun_rollouts``, which posts each chunk to the host."""
    sent = _record_requests(monkeypatch, json.dumps({"results": [{"reward": 1}]}).encode())

    _session()._post_chunk([{"id": 1}])

    assert sent[0][OPEN_SANDBOX_API_KEY_HEADER.lower()] == KEY
    assert sent[0]["content-type"] == "application/json"


def test_listing_the_hosts_entries_sends_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """NeMo-RL asks for the host's routes after spin-up, through the same proxy."""
    sent = _record_requests(monkeypatch, json.dumps({"entries": {"env": ["http"]}}).encode())

    assert _session().list_entries() == {"env": ["http"]}
    assert sent[0][OPEN_SANDBOX_API_KEY_HEADER.lower()] == KEY


def test_the_proxy_forwards_health_checks_with_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _record_requests(monkeypatch, json.dumps({"status": "ready"}).encode())
    client = TestClient(build_proxy_app(_session(rollout_auth_token="secret")))

    response = client.get("/health", headers={AUTH_HEADER: "secret"})

    assert response.json() == {"status": "ready"}
    assert sent[0][OPEN_SANDBOX_API_KEY_HEADER.lower()] == KEY


def test_the_proxy_forwards_rollouts_with_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _record_requests(monkeypatch, json.dumps({"results": []}).encode())
    client = TestClient(build_proxy_app(_session(rollout_auth_token="secret")))

    response = client.post("/rollouts/run", headers={AUTH_HEADER: "secret"}, json={"examples": []})

    assert response.status_code == 200
    assert sent[0][OPEN_SANDBOX_API_KEY_HEADER.lower()] == KEY


# ---------------------------------------------------------------------------------------------
# Nothing the host can redirect us to receives the key, or any request at all
# ---------------------------------------------------------------------------------------------


class _Servers(SimpleNamespace):
    bait: str  # base URL of a "host" that redirects every request
    trap: str  # base URL the redirects point at
    trapped: list[dict[str, str]]  # headers of every request that reached the trap


@pytest.fixture
def servers() -> Iterator[_Servers]:
    """A host that answers every request with ``302`` to a second server, which records what reaches it."""
    trapped: list[dict[str, str]] = []

    class _Trap(BaseHTTPRequestHandler):
        def _record(self) -> None:
            trapped.append({name.lower(): value for name, value in self.headers.items()})
            body = json.dumps({"status": "ready", "entries": {}, "results": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _record

        def log_message(self, format: str, *args: Any) -> None:
            return None

    trap = ThreadingHTTPServer(("127.0.0.1", 0), _Trap)
    trap_url = f"http://127.0.0.1:{trap.server_address[1]}"

    class _Bait(BaseHTTPRequestHandler):
        def _redirect(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self.send_response(302)
            self.send_header("Location", f"{trap_url}/stolen")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST = _redirect

        def log_message(self, format: str, *args: Any) -> None:
            return None

    bait = ThreadingHTTPServer(("127.0.0.1", 0), _Bait)
    for server in (trap, bait):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield _Servers(bait=f"http://127.0.0.1:{bait.server_address[1]}", trap=trap_url, trapped=trapped)
    finally:
        for server in (trap, bait):
            server.shutdown()
            server.server_close()


def _baiting_host(servers: _Servers) -> GymHostHandle:
    return _host(health_url=f"{servers.bait}/health", rollout_url=f"{servers.bait}/rollouts/run")


def test_plain_urllib_would_hand_the_key_to_the_redirect_target(servers: _Servers) -> None:
    """Why ``open_host`` refuses redirects: urllib's default opener copies every header onto the new request."""
    with urlopen(Request(f"{servers.bait}/health", headers=CREDENTIALS), timeout=5):
        pass

    assert servers.trapped[0][OPEN_SANDBOX_API_KEY_HEADER.lower()] == KEY


def test_open_host_refuses_a_redirect(servers: _Servers) -> None:
    with pytest.raises(HTTPError) as excinfo:
        open_host(_baiting_host(servers), f"{servers.bait}/health", timeout=5)

    assert excinfo.value.code == 302
    assert servers.trapped == []


def test_the_readiness_poll_never_follows_a_redirect(servers: _Servers) -> None:
    provider = OpenSandboxGymHostProvider(connection={"domain": "server", "api_key": KEY})

    with pytest.raises(TimeoutError, match="302"):
        asyncio.run(provider.wait_ready(_baiting_host(servers), timeout_s=0.5))

    assert servers.trapped == []


def test_rollouts_never_follow_a_redirect(servers: _Servers) -> None:
    with pytest.raises(Exception, match="302"):
        _session(_baiting_host(servers))._post_chunk([{"id": 1}])

    assert servers.trapped == []


def test_listing_entries_never_follows_a_redirect(servers: _Servers) -> None:
    with pytest.raises(RuntimeError, match="302"):
        _session(_baiting_host(servers)).list_entries()

    assert servers.trapped == []


def test_the_proxy_never_follows_a_redirect(servers: _Servers) -> None:
    """Nor does it pass one on: its caller gets the status but no ``Location`` to follow."""
    client = TestClient(build_proxy_app(_session(_baiting_host(servers), rollout_auth_token="secret")))

    health = client.get("/health", headers={AUTH_HEADER: "secret"})
    rollouts = client.post("/rollouts/run", headers={AUTH_HEADER: "secret"}, json={"examples": []})

    assert health.status_code == 302
    assert rollouts.status_code == 302
    assert "location" not in health.headers
    assert "location" not in rollouts.headers
    assert servers.trapped == []


# ---------------------------------------------------------------------------------------------
# What the session hands on never contains the key
# ---------------------------------------------------------------------------------------------


def test_the_descriptor_carries_the_shareable_headers_without_the_key() -> None:
    """The descriptor is the hand-off to other jobs and, through ``/session``, to sandboxed job code."""
    descriptor = _session().descriptor()

    assert descriptor.headers == ENDPOINT_HEADERS
    assert KEY not in descriptor.model_dump_json()


def test_a_session_file_never_contains_the_key(tmp_path: Path) -> None:
    """Session files are written for a later job to read, often on shared storage."""
    path = tmp_path / "session.json"

    write_session_file(path, _session(), mode="orchestrator")

    assert KEY not in path.read_text()


@pytest.mark.parametrize(
    ("rollout_auth_token", "caller_token"),
    [(None, None), ("secret", "secret")],
    ids=["unauthenticated-route", "authenticated-caller"],
)
def test_the_session_route_never_returns_the_key(rollout_auth_token: str | None, caller_token: str | None) -> None:
    """Even an authenticated caller gets the descriptor's headers only: the key never leaves this process."""
    client = TestClient(build_proxy_app(_session(rollout_auth_token=rollout_auth_token)))
    headers = {AUTH_HEADER: caller_token} if caller_token else {}

    response = client.get("/session", headers=headers)

    assert response.status_code == 200
    assert response.json()["headers"] == ENDPOINT_HEADERS
    assert KEY not in response.text
