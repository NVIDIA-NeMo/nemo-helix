# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the service-principal auth-proxy sidecar forwarder."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.auth import StaticToken, TokenProviderAuth
from nhx.common.auth.principal_identifier import InvalidPrincipalIdentifier
from nhx.common.auth.workload_proxy import main as workload_proxy_main
from nhx.common.auth.workload_proxy.main import build_app
from nhx.common.controller import ControllerManager, Loop


@pytest.mark.parametrize("status", [200, 401, 403, 307])
@respx.mock
def test_transport_auth_replaces_inbound_identity_and_preserves_response(status: int) -> None:
    route = respx.post(
        "https://platform.test/apis/inference-gateway/v2/workspaces/test/openai/-/v1/responses?stream=1"
    ).mock(
        return_value=httpx.Response(
            status,
            content=b'data: {"output":"hello"}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream", "location": "https://other.test/"},
        )
    )
    app = build_app(base_url="https://platform.test", auth=TokenProviderAuth(StaticToken("workload-token")))
    with TestClient(app, follow_redirects=False) as client:
        response = client.post(
            "/apis/inference-gateway/v2/workspaces/test/openai/-/v1/responses?stream=1",
            headers={
                "authorization": "Bearer not-used",
                "x-nhx-principal-id": "service:admin",
                "x-nhx-principal-on-behalf-of": "attacker",
                "x-nhx-subject-aliases": "admin",
            },
            json={"stream": True},
        )
    assert route.call_count == 1
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer workload-token"
    assert not any(name.startswith("x-nhx-") for name in sent.headers)
    assert response.status_code == status
    assert response.content == b'data: {"output":"hello"}\n\ndata: [DONE]\n\n'
    assert response.headers["content-type"] == "text/event-stream"


def test_proxy_requires_exactly_one_auth_mode() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_app(base_url="https://platform.test")
    with pytest.raises(ValueError, match="exactly one"):
        build_app(base_url="https://platform.test", principal="agents", auth=httpx.BasicAuth("a", "b"))
    with pytest.raises(ValueError, match="on_behalf_of"):
        build_app(base_url="https://platform.test", auth=httpx.BasicAuth("a", "b"), on_behalf_of="user")


@respx.mock
def test_forward_stamps_service_principal_and_preserves_path() -> None:
    upstream = "http://nemo-helix-api:8080"
    route = respx.post(f"{upstream}/apis/inference-gateway/v2/workspaces/default/openai/-/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"ok": True},
            headers={"connection": "x-upstream-hop", "x-upstream-hop": "secret"},
        )
    )
    app = build_app(base_url=upstream, principal="agents")
    client = TestClient(app)

    resp = client.post(
        "/apis/inference-gateway/v2/workspaces/default/openai/-/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={
            "authorization": "Bearer not-used",
            "connection": "x-client-hop",
            "x-client-hop": "secret",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert route.called
    sent = route.calls.last.request
    # The proxy sets the service-principal identity and drops the placeholder auth.
    assert sent.headers["x-nhx-principal-id"] == "service:agents"
    assert sent.headers["x-nhx-actor-aliases"] == "service:agents"
    assert "authorization" not in {k.lower() for k in sent.headers}
    assert "x-client-hop" not in {k.lower() for k in sent.headers}
    assert "connection" not in resp.headers
    assert "x-upstream-hop" not in resp.headers
    assert sent.content == b'{"model":"m","messages":[]}'


@pytest.mark.parametrize("method", ["GET", "HEAD", "DELETE", "OPTIONS"])
@respx.mock
def test_forward_keeps_bodyless_requests_bodyless(method: str) -> None:
    route = respx.request(method, "http://platform.test/inference").mock(return_value=httpx.Response(200))
    app = build_app(base_url="http://platform.test", principal="agents")
    with TestClient(app) as client:
        response = client.request(method, "/inference")
    assert response.status_code == 200
    sent = route.calls.last.request
    assert sent.content == b""
    assert "transfer-encoding" not in sent.headers
    assert "content-length" not in sent.headers


@pytest.mark.parametrize("chunked", [False, True])
@respx.mock
def test_forward_preserves_request_body_framing(chunked: bool) -> None:
    body = b'{"model":"m","messages":[]}'
    route = respx.post("http://platform.test/inference").mock(return_value=httpx.Response(200, json={}))
    app = build_app(base_url="http://platform.test", principal="agents")
    with TestClient(app) as client:
        response = client.post(
            "/inference",
            content=iter([body[:10], body[10:]]) if chunked else body,
            headers={"transfer-encoding": "chunked", "content-length": "999"} if chunked else {},
        )
    assert response.status_code == 200
    sent = route.calls.last.request
    assert sent.content == body
    if chunked:
        assert sent.headers["transfer-encoding"] == "chunked"
        assert "content-length" not in sent.headers
    else:
        assert sent.headers["content-length"] == str(len(body))
        assert "transfer-encoding" not in sent.headers


@pytest.mark.parametrize("principal", ["service:", "service:/path", "service:*", "has spaces"])
def test_build_app_rejects_malformed_service_principal(principal: str) -> None:
    with pytest.raises(InvalidPrincipalIdentifier):
        build_app(base_url="http://nemo-helix-api:8080", principal=principal)


@respx.mock
def test_forward_stamps_on_behalf_of_when_configured() -> None:
    upstream = "http://nemo-helix-api:8080"
    route = respx.get(f"{upstream}/apis/entities/v2/workspaces").mock(return_value=httpx.Response(200, json={}))
    app = build_app(base_url=upstream, principal="agents", on_behalf_of="user:alice")
    client = TestClient(app)

    client.get("/apis/entities/v2/workspaces")

    sent = route.calls.last.request
    # Service principal clears the route gate; on-behalf-of narrows access to the creator.
    assert sent.headers["x-nhx-principal-id"] == "service:agents"
    assert sent.headers["x-nhx-actor-aliases"] == "service:agents"
    assert sent.headers["x-nhx-principal-on-behalf-of"] == "user:alice"
    assert sent.headers["x-nhx-subject-aliases"] == "user:alice"


@respx.mock
def test_forward_omits_on_behalf_of_when_not_configured() -> None:
    upstream = "http://nemo-helix-api:8080"
    route = respx.get(f"{upstream}/apis/entities/v2/workspaces").mock(return_value=httpx.Response(200, json={}))
    app = build_app(base_url=upstream, principal="agents")
    client = TestClient(app)

    client.get("/apis/entities/v2/workspaces")

    sent = route.calls.last.request
    assert "x-nhx-principal-on-behalf-of" not in {k.lower() for k in sent.headers}


@respx.mock
def test_forward_strips_inbound_on_behalf_of_to_prevent_spoofing() -> None:
    upstream = "http://nemo-helix-api:8080"
    route = respx.get(f"{upstream}/apis/entities/v2/workspaces").mock(return_value=httpx.Response(200, json={}))
    # The delegated identity is baked in at deploy time; a co-located workload must
    # not be able to override it (or inject one when none is configured) via headers.
    app = build_app(base_url=upstream, principal="agents", on_behalf_of="user:alice")
    client = TestClient(app)

    client.get(
        "/apis/entities/v2/workspaces",
        headers={
            "x-nhx-principal-id": "service:platform",
            "x-nhx-actor-account-id": "account-attacker",
            "x-nhx-principal-email": "attacker@evil.test",
            "x-nhx-principal-groups": "platform-admins",
            "x-nhx-actor-aliases": "attacker-alias",
            "x-nhx-principal-on-behalf-of": "user:attacker",
            # Companion metadata must not be smuggled onto our stamped OBO id:
            # the platform derives effective groups/email from these and feeds
            # them to the PDP, so attacker-chosen values would escalate.
            "x-nhx-principal-on-behalf-of-email": "attacker@evil.test",
            "x-nhx-principal-on-behalf-of-groups": "platform-admins",
            "x-nhx-subject-account-id": "account-attacker",
            "x-nhx-subject-aliases": "attacker-alias",
        },
    )

    sent = route.calls.last.request
    sent_keys = {k.lower() for k in sent.headers}
    assert sent.headers["x-nhx-principal-id"] == "service:agents"
    assert sent.headers["x-nhx-actor-aliases"] == "service:agents"
    assert sent.headers["x-nhx-principal-on-behalf-of"] == "user:alice"
    assert sent.headers["x-nhx-subject-aliases"] == "user:alice"
    # Inbound companion metadata cannot be attached to either stamped identity.
    assert "x-nhx-actor-account-id" not in sent_keys
    assert "x-nhx-principal-email" not in sent_keys
    assert "x-nhx-principal-groups" not in sent_keys
    assert "x-nhx-subject-account-id" not in sent_keys
    assert "x-nhx-principal-on-behalf-of-email" not in sent_keys
    assert "x-nhx-principal-on-behalf-of-groups" not in sent_keys


@respx.mock
def test_forward_strips_inbound_on_behalf_of_when_none_configured() -> None:
    upstream = "http://nemo-helix-api:8080"
    route = respx.get(f"{upstream}/apis/entities/v2/workspaces").mock(return_value=httpx.Response(200, json={}))
    app = build_app(base_url=upstream, principal="agents")
    client = TestClient(app)

    client.get(
        "/apis/entities/v2/workspaces",
        headers={
            "x-nhx-principal-on-behalf-of": "user:attacker",
            "x-nhx-principal-on-behalf-of-email": "attacker@evil.test",
            "x-nhx-principal-on-behalf-of-groups": "platform-admins",
            "x-nhx-subject-account-id": "account-attacker",
            "x-nhx-subject-aliases": "attacker-alias",
        },
    )

    sent = route.calls.last.request
    sent_keys = {k.lower() for k in sent.headers}
    assert "x-nhx-principal-on-behalf-of" not in sent_keys
    assert "x-nhx-subject-account-id" not in sent_keys
    assert "x-nhx-subject-aliases" not in sent_keys
    assert "x-nhx-principal-on-behalf-of-email" not in sent_keys
    assert "x-nhx-principal-on-behalf-of-groups" not in sent_keys


@respx.mock
def test_forward_normalizes_bare_principal_name() -> None:
    upstream = "http://nemo-helix-api:8080"
    route = respx.get(f"{upstream}/apis/entities/v2/workspaces").mock(return_value=httpx.Response(200, json={}))
    # Already-prefixed principal is passed through unchanged.
    app = build_app(base_url=upstream, principal="service:models")
    client = TestClient(app)
    client.get("/apis/entities/v2/workspaces")
    assert route.calls.last.request.headers["x-nhx-principal-id"] == "service:models"


@respx.mock
def test_forward_passes_through_upstream_status() -> None:
    upstream = "http://nemo-helix-api:8080"
    respx.get(f"{upstream}/apis/entities/v2/workspaces").mock(return_value=httpx.Response(403, json={"detail": "no"}))
    app = build_app(base_url=upstream, principal="agents")
    client = TestClient(app)

    resp = client.get("/apis/entities/v2/workspaces")
    assert resp.status_code == 403


@respx.mock
def test_forward_preserves_separate_set_cookie_headers() -> None:
    upstream = "http://nemo-helix-api:8080"
    respx.get(f"{upstream}/session").mock(
        return_value=httpx.Response(
            200,
            headers=[
                ("set-cookie", "session=one; Path=/; HttpOnly"),
                ("set-cookie", "csrf=two; Path=/"),
            ],
        )
    )
    app = build_app(base_url=upstream, principal="agents")

    with TestClient(app) as client:
        response = client.get("/session")

    assert response.headers.get_list("set-cookie") == [
        "session=one; Path=/; HttpOnly",
        "csrf=two; Path=/",
    ]


def test_healthz_does_not_require_upstream() -> None:
    app = build_app(base_url="http://nemo-helix-api:8080", principal="agents")
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_lifespan_closes_upstream_client() -> None:
    # Entering TestClient as a context manager runs the lifespan; the shared
    # httpx client's connection pool must be closed on shutdown.
    created: list[httpx.AsyncClient] = []
    real_async_client = httpx.AsyncClient

    def _tracking_client(*args, **kwargs) -> httpx.AsyncClient:
        client = real_async_client(*args, **kwargs)
        created.append(client)
        return client

    with patch("nhx.common.auth.workload_proxy.main.httpx.AsyncClient", side_effect=_tracking_client):
        app = build_app(base_url="http://nemo-helix-api:8080", principal="agents")

    assert len(created) == 1
    upstream_client = created[0]
    with TestClient(app) as test_client:
        assert test_client.get("/healthz").status_code == 200
        assert upstream_client.is_closed is False
    assert upstream_client.is_closed is True


@pytest.fixture(autouse=True)
def _reset_controller_manager() -> Iterator[None]:
    ControllerManager._instance = None
    yield
    ControllerManager._instance = None


def test_run_registers_and_unregisters_healthy_loop_for_health_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose a running auth-proxy as healthy and remove it on shutdown."""
    monkeypatch.setenv("NHX_AUTH_PROXY_PRINCIPAL", "agents")
    monkeypatch.setenv("NEMO_BASE_URL", "http://nemo-helix-api:8080")

    server_started = threading.Event()

    class _FakeServer:
        def __init__(self, config: object) -> None:
            self.should_exit = False

        def run(self) -> None:
            server_started.set()
            while not self.should_exit:
                threading.Event().wait(timeout=0.05)

    monkeypatch.setattr(workload_proxy_main.uvicorn, "Server", _FakeServer)

    manager = ControllerManager.get_instance()
    for _ in range(2):
        server_started.clear()
        stop_signal = threading.Event()
        run_thread = threading.Thread(target=workload_proxy_main.run, args=(stop_signal,), daemon=True)
        run_thread.start()
        try:
            assert server_started.wait(timeout=2)

            def _registered() -> bool:
                return "auth-proxy" in manager.get_all_loops()

            assert _wait_until(_registered)
            assert _wait_until(lambda: manager.validate_all_healthy() == (True, {"auth-proxy": True}))
        finally:
            stop_signal.set()
            run_thread.join(timeout=5)
        assert not run_thread.is_alive()
        assert manager.validate_all_healthy() == (True, {})


def test_run_leaves_controller_tracked_when_uvicorn_thread_outlives_shutdown(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep a hung Uvicorn server tracked until it exits."""
    monkeypatch.setenv("NHX_AUTH_PROXY_PRINCIPAL", "agents")
    monkeypatch.setenv("NEMO_BASE_URL", "http://nemo-helix-api:8080")
    monkeypatch.setattr(workload_proxy_main, "_UVICORN_JOIN_TIMEOUT_SECONDS", 0.05)

    server_started = threading.Event()
    hang_forever = threading.Event()

    class _HangingFakeServer:
        def __init__(self, config: object) -> None:
            self.should_exit = False

        def run(self) -> None:
            server_started.set()
            hang_forever.wait(timeout=5)

    monkeypatch.setattr(workload_proxy_main.uvicorn, "Server", _HangingFakeServer)

    manager = ControllerManager.get_instance()
    stop_signal = threading.Event()
    run_thread = threading.Thread(target=workload_proxy_main.run, args=(stop_signal,), daemon=True)
    run_thread.start()
    try:
        assert server_started.wait(timeout=2)
        assert _wait_until(lambda: "auth-proxy" in manager.get_all_loops())

        with caplog.at_level(logging.WARNING, logger="nhx.common.controller.controller_manager"):
            stop_signal.set()
            run_thread.join(timeout=5)
        assert not run_thread.is_alive()

        assert "auth-proxy" in manager.get_all_loops()
        assert "Leaving health tracking in place for 'auth-proxy'" in caplog.text
    finally:
        hang_forever.set()

    assert _wait_until(lambda: "auth-proxy" not in manager.get_all_loops())

    restarted_server = threading.Event()

    class _RestartedFakeServer:
        def __init__(self, config: object) -> None:
            self.should_exit = False

        def run(self) -> None:
            restarted_server.set()
            while not self.should_exit:
                threading.Event().wait(timeout=0.001)

    monkeypatch.setattr(workload_proxy_main.uvicorn, "Server", _RestartedFakeServer)

    second_stop_signal = threading.Event()
    second_run_thread = threading.Thread(
        target=workload_proxy_main.run,
        args=(second_stop_signal,),
        daemon=True,
    )
    second_run_thread.start()
    try:
        assert restarted_server.wait(timeout=2)
        assert _wait_until(lambda: manager.validate_all_healthy() == (True, {"auth-proxy": True}))
    finally:
        second_stop_signal.set()
        second_run_thread.join(timeout=5)

    assert not second_run_thread.is_alive()
    assert manager.validate_all_healthy() == (True, {})


def test_run_setup_failure_before_server_start_is_visible_as_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report auth-proxy setup failures through controller health."""
    monkeypatch.delenv("NHX_AUTH_PROXY_PRINCIPAL", raising=False)

    manager = ControllerManager.get_instance()
    stop_signal = threading.Event()

    with pytest.raises(RuntimeError, match="NHX_AUTH_PROXY_PRINCIPAL"):
        workload_proxy_main.run(stop_signal)

    all_healthy, status = manager.validate_all_healthy()
    assert all_healthy is False
    assert status.get("auth-proxy") is False


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "proxy.example.com"])
def test_build_server_rejects_non_loopback_bind_by_default(
    monkeypatch: pytest.MonkeyPatch,
    host: str,
) -> None:
    monkeypatch.setenv("NHX_AUTH_PROXY_PRINCIPAL", "agents")
    monkeypatch.setenv("NEMO_BASE_URL", "http://nemo-helix-api:8080")
    monkeypatch.setenv("NHX_AUTH_PROXY_HOST", host)

    with pytest.raises(RuntimeError, match="must be a loopback address"):
        workload_proxy_main._build_server()


@pytest.mark.parametrize("host", ["127.0.0.1", "127.10.20.30", "::1", "[::1]", "localhost"])
def test_loopback_host_validation_accepts_explicit_loopback(host: str) -> None:
    assert workload_proxy_main._is_loopback_host(host)


def test_startup_log_redacts_upstream_url_credentials(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="nhx.common.auth.workload_proxy.main"):
        workload_proxy_main._log_startup(
            host="127.0.0.1",
            port=8090,
            base_url="https://sensitive-user:sensitive-password@platform.example.test:8443/base",  # trufflehog:ignore
            principal="agents",
            on_behalf_of=None,
        )

    assert "sensitive-user" not in caplog.text
    assert "sensitive-password" not in caplog.text
    assert "https://platform.example.test:8443/base" in caplog.text


def test_build_server_allows_explicit_non_loopback_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_AUTH_PROXY_PRINCIPAL", "agents")
    monkeypatch.setenv("NEMO_BASE_URL", "http://nemo-helix-api:8080")
    monkeypatch.setenv("NHX_AUTH_PROXY_HOST", "0.0.0.0")
    monkeypatch.setenv("NHX_AUTH_PROXY_ALLOW_NON_LOOPBACK", "true")
    monkeypatch.setattr(workload_proxy_main, "build_app", lambda **_kwargs: object())
    sentinel_server = object()
    monkeypatch.setattr(workload_proxy_main.uvicorn, "Server", lambda _config: sentinel_server)

    _, _, _, host, _, server = workload_proxy_main._build_server()

    assert host == "0.0.0.0"
    assert server is sentinel_server


def test_run_registration_failure_never_starts_server_and_remains_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject duplicate tracking before starting Uvicorn."""
    monkeypatch.setenv("NHX_AUTH_PROXY_PRINCIPAL", "agents")
    monkeypatch.setenv("NEMO_BASE_URL", "http://nemo-helix-api:8080")
    server_started = threading.Event()

    class _FakeServer:
        def __init__(self, config: object) -> None:
            self.should_exit = False

        def run(self) -> None:
            server_started.set()
            while not self.should_exit:
                threading.Event().wait(timeout=0.01)

    monkeypatch.setattr(workload_proxy_main.uvicorn, "Server", _FakeServer)
    manager = ControllerManager.get_instance()

    def _fail_registration(_name: str, _loop: object) -> None:
        raise ValueError("duplicate")

    monkeypatch.setattr(manager, "register", _fail_registration)

    with pytest.raises(ValueError, match="duplicate"):
        workload_proxy_main.run(threading.Event())

    assert not server_started.wait(timeout=0.5)
    assert manager.validate_all_healthy() == (False, {"auth-proxy": False})


def test_run_uvicorn_thread_start_failure_removes_registered_loop_and_remains_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NHX_AUTH_PROXY_PRINCIPAL", "agents")
    monkeypatch.setenv("NEMO_BASE_URL", "http://nemo-helix-api:8080")

    real_start = threading.Thread.start

    def _start(thread: threading.Thread) -> None:
        if thread.name != "auth-proxy-uvicorn":
            real_start(thread)
        else:
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", _start)
    manager = ControllerManager.get_instance()

    with pytest.raises(RuntimeError, match="thread start failed"):
        workload_proxy_main.run(threading.Event())

    assert manager.get_all_loops() == {}
    assert manager.validate_all_healthy() == (False, {"auth-proxy": False})


def test_run_health_loop_start_failure_tracks_uvicorn_until_delayed_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NHX_AUTH_PROXY_PRINCIPAL", "agents")
    monkeypatch.setenv("NEMO_BASE_URL", "http://nemo-helix-api:8080")
    monkeypatch.setattr(workload_proxy_main, "_UVICORN_JOIN_TIMEOUT_SECONDS", 0.05)

    server_started = threading.Event()
    release_server = threading.Event()

    class _HangingFakeServer:
        def __init__(self, config: object) -> None:
            self.should_exit = False

        def run(self) -> None:
            server_started.set()
            release_server.wait(timeout=5)

    def _fail_health_loop_start(_loop: Loop) -> None:
        raise RuntimeError("health loop start failed")

    monkeypatch.setattr(workload_proxy_main.uvicorn, "Server", _HangingFakeServer)
    monkeypatch.setattr(Loop, "start", _fail_health_loop_start)
    manager = ControllerManager.get_instance()

    try:
        with pytest.raises(RuntimeError, match="health loop start failed"):
            workload_proxy_main.run(threading.Event())

        assert server_started.wait(timeout=2)
        assert "auth-proxy" in manager.get_all_loops()
        assert manager.validate_all_healthy() == (False, {"auth-proxy": False})
        with pytest.raises(RuntimeError, match="still stopping"):
            manager.await_controller_registration("auth-proxy")
    finally:
        release_server.set()

    assert _wait_until(lambda: "auth-proxy" not in manager.get_all_loops())
    assert manager.validate_all_healthy() == (False, {"auth-proxy": False})


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
