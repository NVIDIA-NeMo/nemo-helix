# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for :func:`~nhx.core.inference_gateway.testing.fixtures.igw_loopback_harness`.

- :func:`serve_app_in_thread` — run an ASGI app on a freshly bound
  ``127.0.0.1:<port>`` so plugin code making outbound HTTP back into IGW
  can reach it over a real socket.
- :func:`override_platform_base_url` — point IGW's resolver at the
  loopback URL instead of the production default.
- :func:`per_request_http_client` — yield-style FastAPI dependency that
  creates an ``aiohttp.ClientSession`` per request. Necessary because the
  loopback fixture serves the app from two event loops simultaneously,
  and ``ClientSession`` instances are loop-bound — a singleton would
  raise ``RuntimeError: ... attached to a different loop`` from whichever
  loop didn't originate it. The default (non-loopback) fixture doesn't
  need this since only one loop is involved.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from unittest.mock import patch

import aiohttp
from fastapi import FastAPI

# Generous because uvicorn startup on a cold interpreter under load can
# take a few seconds; tests fail fast and loud if these are exceeded.
_STARTUP_TIMEOUT_SECONDS = 10.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0


@contextmanager
def serve_app_in_thread(app: FastAPI, *, host: str = "127.0.0.1") -> Generator[str, None, None]:
    """Run *app* on a freshly bound ``host:<random-port>`` and yield the base URL.

    Uses the pre-bound-socket trick (``bind((host, 0))`` then hand the
    listening socket to uvicorn) so the OS atomically picks a free port —
    avoids check-then-bind races under parallel test runs.

    Yields ``"http://<host>:<port>"`` (no path).

    Raises:
        RuntimeError: If the uvicorn thread exits before signalling startup,
            or fails to stop within :data:`_SHUTDOWN_TIMEOUT_SECONDS`.
        TimeoutError: If startup doesn't complete within
            :data:`_STARTUP_TIMEOUT_SECONDS`.
    """
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    sock.listen(128)

    bound_host, port = sock.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            # Already started by the TestClient context that owns this app;
            # firing lifespan again would double-fire startup hooks.
            lifespan="off",
            log_level="warning",
        )
    )

    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()

    try:
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("Test app server stopped before startup completed.")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out after {_STARTUP_TIMEOUT_SECONDS:.0f}s waiting for test app server to start."
                )
            time.sleep(0.01)

        yield f"http://{bound_host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        if thread.is_alive():
            raise RuntimeError(f"Test app server did not stop within {_SHUTDOWN_TIMEOUT_SECONDS:.0f} seconds.")


@contextmanager
def override_platform_base_url(base_url: str) -> Generator[None, None, None]:
    """Patch ``get_platform_config`` to advertise *base_url* as the platform URL.

    The loopback fixture binds IGW on a real ``127.0.0.1:<port>`` that
    the production-default ``HelixConfig.base_url`` knows nothing
    about. The plugin resolver
    (:meth:`InferenceMiddlewareCacheAccessor.get_openai_compatible_inference_url_and_model`)
    reads ``get_platform_config().base_url`` to build the URL it returns,
    so without this patch it hands callers a URL pointing at the
    production default — unreachable from the test process.

    Patches the import site
    (``nhx.core.inference_gateway.api.middleware_registry``) rather than
    the source so unrelated readers of :func:`get_platform_config` keep
    their production defaults; the override is scoped to IGW's resolver.
    """
    from nhx.common.config import get_platform_config

    # ``model_copy(update=...)`` works on frozen models and avoids
    # mutating the singleton (which would leak across tests).
    overridden = get_platform_config().model_copy(update={"base_url": base_url})
    with patch(
        "nhx.core.inference_gateway.api.middleware_registry.get_platform_config",
        return_value=overridden,
    ):
        yield


async def per_request_http_client() -> AsyncGenerator[aiohttp.ClientSession, None]:
    """FastAPI yield-style dependency: one ``ClientSession`` per request.

    Trades a session-creation per request for a guaranteed-correct close
    on the same loop that created it. Wrong tradeoff in production (the
    real :func:`global_http_client` is a singleton tied to service
    lifespan) but right for the loopback fixture, where two loops drive
    the same app and a long-lived session would be bound to whichever
    one created it first.

    Constructor flags mirror the production client so headers and
    decompression behave identically: ``DummyCookieJar`` (no cross-domain
    cookie persistence), ``auto_decompress=False`` (middleware requests
    override it per request), and ``Accept-Encoding`` left to the inbound
    request.
    """
    session = aiohttp.ClientSession(
        cookie_jar=aiohttp.DummyCookieJar(),
        auto_decompress=False,
        skip_auto_headers=["Accept-Encoding"],
    )
    try:
        yield session
    finally:
        await session.close()


__all__ = ["override_platform_base_url", "per_request_http_client", "serve_app_in_thread"]
