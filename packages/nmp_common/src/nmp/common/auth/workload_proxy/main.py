# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-principal auth-proxy sidecar.

Runs inside a deployed workload's pod as a loopback forwarder. A co-located
workload whose HTTP client we do not control (e.g. a NAT agent calling the
Inference Gateway) points its platform base URL at this proxy
(``http://127.0.0.1:<port>``) and sends no credentials of its own. The proxy
stamps a service-principal identity header (``X-NMP-Principal-Id: service:<name>``)
on every forwarded request, which the platform authorizes via the ServiceSystem
role. This is the same static service-identity the platform's own SDK clients
use (``get_platform_sdk(as_service=...)``); the proxy exists only for workloads
that cannot set the header themselves.

When ``NMP_AUTH_PROXY_ON_BEHALF_OF`` is set, the proxy additionally stamps
``X-NMP-Principal-On-Behalf-Of`` so the platform authorizes the request as that
delegated principal rather than granting the service principal's full
(ServiceSystem) reach. This scopes a deployed workload's platform access to the
identity that created it (e.g. an agent deployment acting as its creator). The
delegated identity is baked in at deploy time and is *not* taken from the
incoming request — the inbound principal/OBO headers are stripped so a co-located
workload cannot spoof a different identity.

Started via ``nemo services run --sidecars auth-proxy``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from ipaddress import ip_address

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from nmp.common.auth import Principal
from nmp.common.controller import Controller, ControllerManager, Loop, TimedLoopWaiter

logger = logging.getLogger(__name__)

# Loopback host + port the proxy listens on. The workload targets this address.
AUTH_PROXY_HOST_ENVVAR = "NMP_AUTH_PROXY_HOST"
AUTH_PROXY_PORT_ENVVAR = "NMP_AUTH_PROXY_PORT"
# Service-principal name stamped on forwarded requests (e.g. "agents").
AUTH_PROXY_PRINCIPAL_ENVVAR = "NMP_AUTH_PROXY_PRINCIPAL"
# Optional principal id to delegate to via on-behalf-of (e.g. the workload's
# creator). When set, the service principal acts on behalf of this identity so
# the platform scopes access to what that principal can reach.
AUTH_PROXY_ON_BEHALF_OF_ENVVAR = "NMP_AUTH_PROXY_ON_BEHALF_OF"
AUTH_PROXY_ALLOW_NON_LOOPBACK_ENVVAR = "NMP_AUTH_PROXY_ALLOW_NON_LOOPBACK"
DEFAULT_AUTH_PROXY_HOST = "127.0.0.1"
DEFAULT_AUTH_PROXY_PORT = 8090

_READ_TIMEOUT_ENVVAR = "NMP_AUTH_PROXY_READ_TIMEOUT"
_PRINCIPAL_ID_HEADER = "x-nmp-principal-id"
_PRINCIPAL_ACCOUNT_ID_HEADER = "x-nmp-actor-account-id"
_PRINCIPAL_EMAIL_HEADER = "x-nmp-principal-email"
_PRINCIPAL_GROUPS_HEADER = "x-nmp-principal-groups"
_PRINCIPAL_ALIASES_HEADER = "x-nmp-actor-aliases"
_ON_BEHALF_OF_HEADER = "x-nmp-principal-on-behalf-of"
# Companion metadata for the on-behalf-of principal. The platform derives the
# delegated user's groups/email from these (Principal.from_headers -> effective_*),
# and effective_groups/effective_email feed the PDP authorization input. We stamp
# only the OBO id here, so any inbound companion headers are untrusted and must be
# dropped — otherwise a colocated workload could pair our stamped OBO id with
# attacker-chosen groups/email and be evaluated with those, defeating the scoping.
_ON_BEHALF_OF_EMAIL_HEADER = "x-nmp-principal-on-behalf-of-email"
_ON_BEHALF_OF_GROUPS_HEADER = "x-nmp-principal-on-behalf-of-groups"
_ON_BEHALF_OF_ACCOUNT_ID_HEADER = "x-nmp-subject-account-id"
_ON_BEHALF_OF_ALIASES_HEADER = "x-nmp-subject-aliases"

# Request-header sanitization drops identity, framing, and hop-by-hop metadata:
# - the workload's own credential / principal / on-behalf-of headers (we set the
#   identity), so they can't be spoofed or conflict with what we stamp;
# - host and content-length, which httpx recomputes for the upstream request
#   (a stale value corrupts routing / the body).
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_STRIP_REQUEST_HEADERS = _HOP_BY_HOP_HEADERS | frozenset(
    {
        "host",
        "content-length",
        "authorization",
        _PRINCIPAL_ID_HEADER,
        _PRINCIPAL_ACCOUNT_ID_HEADER,
        _PRINCIPAL_EMAIL_HEADER,
        _PRINCIPAL_GROUPS_HEADER,
        _PRINCIPAL_ALIASES_HEADER,
        _ON_BEHALF_OF_HEADER,
        _ON_BEHALF_OF_EMAIL_HEADER,
        _ON_BEHALF_OF_GROUPS_HEADER,
        _ON_BEHALF_OF_ACCOUNT_ID_HEADER,
        _ON_BEHALF_OF_ALIASES_HEADER,
    }
)
# We stream the response, so the upstream's framing headers no longer apply.
_STRIP_RESPONSE_HEADERS = _HOP_BY_HOP_HEADERS | frozenset({"content-length"})
_MULTI_VALUE_RESPONSE_HEADERS = frozenset({"set-cookie"})


def _sanitized_headers(headers: Mapping[str, str], *, strip: frozenset[str]) -> dict[str, str]:
    """Remove fixed and Connection-nominated hop-by-hop headers."""
    connection_tokens = {
        token.strip().lower()
        for key, value in headers.items()
        if key.lower() == "connection"
        for token in value.split(",")
        if token.strip()
    }
    blocked = strip | connection_tokens
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def _upstream_base_url() -> str:
    """Return the platform base URL to forward to (env override or platform config)."""
    from nemo_platform_plugin.config import get_platform_config

    return (os.environ.get("NEMO_BASE_URL") or os.environ.get("NMP_BASE_URL") or get_platform_config().base_url).rstrip(
        "/"
    )


def build_app(*, base_url: str, principal: str, on_behalf_of: str | None = None) -> FastAPI:
    """Build the forwarding FastAPI app for the given upstream and service principal.

    When *on_behalf_of* is provided, every forwarded request also carries
    ``X-NMP-Principal-On-Behalf-Of``, delegating to that principal so the
    platform scopes access to what it can reach rather than the service
    principal's full ServiceSystem reach.
    """
    principal_id = principal if principal.startswith("service:") else f"service:{principal}"
    principal_context = Principal(
        id=principal_id,
        authz_aliases=[principal_id],
        on_behalf_of=on_behalf_of,
        on_behalf_of_authz_aliases=[on_behalf_of] if on_behalf_of else [],
    )
    read_timeout = float(os.environ.get(_READ_TIMEOUT_ENVVAR, "300"))
    timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=60.0, pool=10.0)
    client = httpx.AsyncClient(base_url=base_url, timeout=timeout, follow_redirects=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(title="nmp-auth-proxy", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def forward(request: Request, path: str) -> StreamingResponse:
        headers = _sanitized_headers(request.headers, strip=_STRIP_REQUEST_HEADERS)
        headers.update({name.lower(): value for name, value in principal_context.get_headers().items()})
        url = httpx.URL(path="/" + path, query=request.url.query.encode("utf-8"))
        # Stream request bodies to keep a co-located caller from forcing the
        # privileged proxy to buffer an unbounded payload in memory.
        upstream = client.build_request(request.method, url, headers=headers, content=request.stream())
        response = await client.send(upstream, stream=True)

        async def _body() -> AsyncIterator[bytes]:
            # The finally runs on normal completion, exception, and client
            # disconnect (Starlette closes the generator), so this is the only
            # cleanup the response needs.
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        resp_headers = _sanitized_headers(
            response.headers,
            strip=_STRIP_RESPONSE_HEADERS | _MULTI_VALUE_RESPONSE_HEADERS,
        )
        proxy_response = StreamingResponse(
            _body(),
            status_code=response.status_code,
            headers=resp_headers,
        )
        for cookie in response.headers.get_list("set-cookie"):
            proxy_response.headers.append("set-cookie", cookie)
        return proxy_response

    return app


_UVICORN_JOIN_TIMEOUT_SECONDS = 10.0


def _join_uvicorn_thread(server: uvicorn.Server, thread: threading.Thread, *, context: str) -> bool:
    """Signal the uvicorn server to stop and join its thread. Returns whether it's still alive."""
    server.should_exit = True
    thread.join(timeout=_UVICORN_JOIN_TIMEOUT_SECONDS)
    still_alive = thread.is_alive()
    if still_alive:
        logger.warning("auth-proxy uvicorn thread did not finish %s", context)
    return still_alive


class _ServerThreadController(Controller):
    """Reports unhealthy if the auth-proxy's uvicorn thread has died."""

    def __init__(self, thread: threading.Thread) -> None:
        self._thread = thread

    def step(self) -> None:
        pass

    @property
    def is_healthy(self) -> bool:
        return self._thread.is_alive()

    @property
    def unhealthy_reason(self) -> str | None:
        if self._thread.is_alive():
            return None
        return "auth-proxy uvicorn thread is not running"


def _unregister_quietly(manager: ControllerManager, name: str, *, context: str) -> None:
    """Best-effort unregister that never masks a caller's in-flight exception.

    Called from inside ``except Exception:`` blocks below. A bare ``manager.unregister(...)``
    there would let a second exception (e.g. a ``KeyError`` from racing with
    ``ControllerManager.watch_delayed_exit``'s background cleanup thread over the same
    loop name) replace the original failure on the bare ``raise`` that follows,
    hiding the real cause from logs/health status.
    """
    try:
        manager.unregister(name)
    except Exception:
        logger.exception("auth-proxy failed to unregister %r %s", name, context)


def run(parent_stop_signal: threading.Event | None = None) -> None:
    """Serve the auth proxy and report its lifecycle through ControllerManager."""
    if parent_stop_signal is None:
        _run_untracked()
        return

    manager = ControllerManager.get_instance()
    generation = manager.await_controller_registration("auth-proxy")
    with manager.controller_registration_context("auth-proxy", generation):
        try:
            base_url, principal, on_behalf_of, host, port, server = _build_server()
        except Exception:
            manager.mark_controller_failed("auth-proxy", generation, reason="auth-proxy server setup failed")
            raise
        _log_startup(host=host, port=port, base_url=base_url, principal=principal, on_behalf_of=on_behalf_of)

        thread = threading.Thread(target=server.run, name="auth-proxy-uvicorn", daemon=True)

        health_loop = Loop(
            waiter=TimedLoopWaiter(sleep_secs=1.0, stop_signal=parent_stop_signal),
            controller=_ServerThreadController(thread),
            stop_signal=parent_stop_signal,
        )
        health_loop.name = "auth-proxy"
        try:
            # Register first so stale tracking blocks a second server bind.
            manager.register(health_loop.name, health_loop)
        except Exception:
            manager.mark_controller_failed("auth-proxy", generation, reason="auth-proxy health registration failed")
            raise

        try:
            thread.start()
        except Exception:
            _unregister_quietly(manager, health_loop.name, context="after Uvicorn thread failed to start")
            manager.mark_controller_failed("auth-proxy", generation, reason="auth-proxy Uvicorn thread failed to start")
            raise

        try:
            health_loop.start()
        except Exception:
            uvicorn_still_alive = _join_uvicorn_thread(server, thread, context="after startup failed")
            manager.mark_controller_failed("auth-proxy", generation, reason="auth-proxy health loop failed to start")
            if uvicorn_still_alive:
                # Do not allow another generation to bind while the failed
                # generation's Uvicorn thread still owns the listen socket.
                manager.mark_controller_stopping("auth-proxy", generation)
                manager.watch_delayed_exit(
                    thread, "auth-proxy", generation, clear_state=False, thread_name="auth-proxy-uvicorn-cleanup"
                )
            else:
                _unregister_quietly(manager, health_loop.name, context="after health loop failed to start")
            raise

        try:
            while not parent_stop_signal.is_set():
                parent_stop_signal.wait(timeout=1)
        finally:
            # The health loop may stop before Uvicorn, so check Uvicorn directly.
            uvicorn_still_alive = _join_uvicorn_thread(server, thread, context="in time")
            health_loop.join(timeout=5)
            if uvicorn_still_alive:
                logger.warning(
                    "Leaving health tracking in place for %r; its uvicorn thread did not finish in time", "auth-proxy"
                )
                manager.mark_controller_stopping("auth-proxy", generation)
                manager.watch_delayed_exit(
                    thread, "auth-proxy", generation, clear_state=True, thread_name="auth-proxy-uvicorn-cleanup"
                )
            else:
                manager.stop_tracking_controller("auth-proxy", generation)
            logger.info("auth-proxy sidecar stopped")


def _build_server() -> tuple[str, str, str | None, str, int, uvicorn.Server]:
    base_url = _upstream_base_url()
    principal = os.environ.get(AUTH_PROXY_PRINCIPAL_ENVVAR)
    if not principal:
        raise RuntimeError(f"{AUTH_PROXY_PRINCIPAL_ENVVAR} is required for the auth-proxy sidecar")
    on_behalf_of = os.environ.get(AUTH_PROXY_ON_BEHALF_OF_ENVVAR) or None
    host = os.environ.get(AUTH_PROXY_HOST_ENVVAR, DEFAULT_AUTH_PROXY_HOST)
    if not _is_loopback_host(host) and os.environ.get(AUTH_PROXY_ALLOW_NON_LOOPBACK_ENVVAR, "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        raise RuntimeError(
            f"{AUTH_PROXY_HOST_ENVVAR} must be a loopback address; set "
            f"{AUTH_PROXY_ALLOW_NON_LOOPBACK_ENVVAR}=true only if external exposure is intentional"
        )
    port = int(os.environ.get(AUTH_PROXY_PORT_ENVVAR, str(DEFAULT_AUTH_PROXY_PORT)))
    app = build_app(base_url=base_url, principal=principal, on_behalf_of=on_behalf_of)
    config = uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)
    return base_url, principal, on_behalf_of, host, port, uvicorn.Server(config)


def _is_loopback_host(host: str) -> bool:
    """Return whether a bind host is explicitly loopback-only."""
    normalized = host.strip().strip("[]")
    if normalized.lower() == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _log_startup(*, host: str, port: int, base_url: str, principal: str, on_behalf_of: str | None) -> None:
    logger.info(
        "Starting auth-proxy sidecar on %s:%s -> %s (principal=service:%s, delegated=%s)",
        host,
        port,
        _redact_url_credentials(base_url),
        principal,
        on_behalf_of is not None,
    )


def _redact_url_credentials(url: str) -> str:
    """Remove URL userinfo before writing an upstream address to logs."""
    try:
        parsed = httpx.URL(url)
        return str(parsed.copy_with(username=None, password=None))
    except Exception:
        return "<invalid upstream URL>"


def _run_untracked() -> None:
    base_url, principal, on_behalf_of, host, port, server = _build_server()
    _log_startup(host=host, port=port, base_url=base_url, principal=principal, on_behalf_of=on_behalf_of)
    server.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run()
