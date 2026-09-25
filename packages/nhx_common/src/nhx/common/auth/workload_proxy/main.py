# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-principal auth-proxy sidecar.

Runs inside a deployed workload's pod as a loopback forwarder. A co-located
workload whose HTTP client we do not control (e.g. a NAT agent calling the
Inference Gateway) points its platform base URL at this proxy
(``http://127.0.0.1:<port>``) and sends no credentials of its own. The proxy
owns the platform credentialing mode on behalf of that client.

When ``NHX_WORKLOAD_IDENTITY_TOKEN_FILE`` is set, the proxy exchanges that
subject-token file for NeMo Helix access tokens and forwards requests with
``Authorization: Bearer <token>``. This is the preferred production path because
the token exchange endpoint can mint delegated workload tokens and refresh them
without exposing long-lived credentials to the colocated workload.

When workload token exchange is not configured, the proxy falls back to
trusted-header mode and stamps a service-principal identity header
(``X-NHX-Principal-Id: service:<name>``) on forwarded requests. This is the same
static service-identity the platform's own SDK clients use
(``get_platform_sdk(as_service=...)``).

In trusted-header mode, when ``NHX_AUTH_PROXY_ON_BEHALF_OF`` is set, the proxy
additionally stamps ``X-NHX-Principal-On-Behalf-Of`` so the platform authorizes
the request as that delegated principal rather than granting the service
principal's full (ServiceSystem) reach. This scopes a deployed workload's
platform access to the identity that created it (e.g. an agent deployment acting
as its creator). The delegated identity is baked in at deploy time and is *not*
taken from the incoming request — the inbound principal/OBO headers are stripped
so a co-located workload cannot spoof a different identity.

Started via ``nemo services run --sidecars auth-proxy``.
"""

from __future__ import annotations

import logging
import os
import threading
from ipaddress import ip_address
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from nemo_helix_plugin.client.auth import TokenProviderAuth
from nemo_helix_plugin.client.auth_proxy import build_auth_proxy_app
from nemo_helix_plugin.client.constants import (
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    is_workload_identity_token_file_set,
)
from nemo_helix_plugin.client.oidc_factory import resolve_workload_exchange_provider
from nhx.common.auth import Principal
from nhx.common.auth.principal_identifier import normalize_service_principal_identifier
from nhx.common.controller import Controller, ControllerManager, Loop, TimedLoopWaiter

logger = logging.getLogger(__name__)

# Loopback host + port the proxy listens on. The workload targets this address.
AUTH_PROXY_HOST_ENVVAR = "NHX_AUTH_PROXY_HOST"
AUTH_PROXY_PORT_ENVVAR = "NHX_AUTH_PROXY_PORT"
# Service-principal name stamped on forwarded requests (e.g. "agents").
AUTH_PROXY_PRINCIPAL_ENVVAR = "NHX_AUTH_PROXY_PRINCIPAL"
# Optional principal id to delegate to via on-behalf-of (e.g. the workload's
# creator). When set, the service principal acts on behalf of this identity so
# the platform scopes access to what that principal can reach.
AUTH_PROXY_ON_BEHALF_OF_ENVVAR = "NHX_AUTH_PROXY_ON_BEHALF_OF"
AUTH_PROXY_ALLOW_NON_LOOPBACK_ENVVAR = "NHX_AUTH_PROXY_ALLOW_NON_LOOPBACK"
DEFAULT_AUTH_PROXY_HOST = "127.0.0.1"
DEFAULT_AUTH_PROXY_PORT = 8090


def _upstream_base_url() -> str:
    """Return the platform base URL to forward to (env override or platform config)."""
    from nemo_helix_plugin.config import get_platform_config

    return (os.environ.get("NEMO_BASE_URL") or os.environ.get("NHX_BASE_URL") or get_platform_config().base_url).rstrip(
        "/"
    )


def _workload_token_auth(base_url: str) -> httpx.Auth | None:
    if not is_workload_identity_token_file_set():
        return None
    provider = resolve_workload_exchange_provider(
        base_url=base_url,
        subject_token_file=Path(os.environ[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR]),
    )
    return TokenProviderAuth(provider)


def build_app(
    *,
    base_url: str,
    principal: str | None = None,
    on_behalf_of: str | None = None,
    auth: httpx.Auth | None = None,
) -> FastAPI:
    """Build a forwarder with either service identity or transport authentication.

    When workload identity token exchange is configured via
    ``NHX_WORKLOAD_IDENTITY_TOKEN_FILE``, the proxy uses transport authentication
    and forwards exchanged bearer tokens. Trusted identity headers and
    *on_behalf_of* are not forwarded in that mode; workload tokens carry their
    own delegation.

    When *on_behalf_of* is provided in trusted-header mode, every forwarded
    request also carries ``X-NHX-Principal-On-Behalf-Of``, delegating to that
    principal so the platform scopes access to what it can reach rather than the
    service principal's full ServiceSystem reach.

    With explicit *auth*, the transport resolves credentials for each request.
    No service identity headers are added.
    """
    if auth is not None:
        if principal is not None:
            raise ValueError("Provide exactly one of principal or auth")
        if on_behalf_of is not None:
            raise ValueError("on_behalf_of requires principal authentication")
        return build_auth_proxy_app(base_url=base_url, auth=auth)

    workload_auth = _workload_token_auth(base_url)
    if workload_auth is not None:
        return build_auth_proxy_app(base_url=base_url, auth=workload_auth)

    if principal is None:
        raise ValueError("Provide exactly one of principal or auth")

    principal_id = normalize_service_principal_identifier(principal)
    principal_context = Principal(
        id=principal_id,
        authz_aliases=[principal_id],
        on_behalf_of=on_behalf_of,
        on_behalf_of_authz_aliases=[on_behalf_of] if on_behalf_of else [],
    )
    return build_auth_proxy_app(base_url=base_url, headers=principal_context.get_headers())


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
