# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the auth proxy on loopback so a job reaches the platform with its own identity."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.client.auth import EnvAccessToken, TokenProviderAuth
from nemo_helix_plugin.client.auth_proxy import build_auth_proxy_app
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, get_env_access_token
from nemo_helix_plugin.client.oidc_factory import resolve_workload_exchange_provider
from nemo_helix_plugin.client_provider import get_forwarding_headers, get_nemo_client
from uvicorn import Config, Server

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT_SECONDS = 10.0
_SHUTDOWN_TIMEOUT_SECONDS = 5


@contextmanager
def platform_auth_proxy() -> Generator[str | None, None, None]:
    """Serve a loopback origin that forwards to the platform as this job.

    Yields the origin to route platform calls through, or ``None`` when the job
    has no identity to forward -- an auth-disabled platform serializes an
    anonymous principal into its jobs -- in which case callers reach the
    platform directly, exactly as they did before any of this existed.

    The identity is resolved here and never enters the agent's config or
    environment. Under workload identity the token provider stays in this
    process and re-exchanges as tokens expire, so nothing pins a run to the
    lifetime of one token.
    """
    token_file = os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR)
    if not (token_file or get_env_access_token() or os.environ.get("NHX_PRINCIPAL")):
        yield None
        return

    base_url = os.environ.get("NHX_BASE_URL", "")
    parsed_base = urlsplit(base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.hostname:
        raise ValueError("NHX_BASE_URL must be set to the Platform URL for authenticated agent jobs")
    if token_file:
        provider = resolve_workload_exchange_provider(base_url=base_url, subject_token_file=Path(token_file))
        # Fail before starting Fabric if the job cannot authenticate. Subsequent
        # requests use the same provider, which exchanges again as tokens expire.
        provider.get_access_token()
        app = build_auth_proxy_app(base_url=base_url, auth=TokenProviderAuth(provider))
    elif get_env_access_token():
        # A subprocess Job under token exchange: the supervisor rotates the delegated token file.
        app = build_auth_proxy_app(base_url=base_url, auth=TokenProviderAuth(EnvAccessToken()))
    else:
        with get_nemo_client() as client:
            headers = get_forwarding_headers(client)
        if not any(name.lower() == "x-nhx-principal-id" and value.strip() for name, value in headers.items()):
            # Auth-disabled platforms serialize an anonymous Principal into
            # jobs too. Preserve that existing unauthenticated execution mode.
            if not platform_auth_enabled():
                yield None
                return
            raise ValueError("NHX_PRINCIPAL must provide a principal ID for authenticated agent jobs")
        app = build_auth_proxy_app(base_url=base_url, headers=headers)

    with serve_auth_proxy(app) as origin:
        yield origin


@contextmanager
def serve_auth_proxy(app: FastAPI) -> Generator[str, None, None]:
    """Serve ``app`` on a loopback port in a background thread, yielding its origin."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = Server(
            Config(
                app,
                host="127.0.0.1",
                port=port,
                access_log=False,
                log_config=None,
                timeout_graceful_shutdown=_SHUTDOWN_TIMEOUT_SECONDS,
            )
        )
        errors: list[BaseException] = []

        def serve() -> None:
            try:
                server.run(sockets=[listener])
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve, name="auth-proxy", daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
            while not server.started:
                if not thread.is_alive():
                    raise RuntimeError("Auth proxy failed to start") from (errors[0] if errors else None)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Auth proxy startup timed out")
                thread.join(timeout=0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            # Graceful shutdown drains requests already in flight, which is what
            # a trajectory posted as the agent exits depends on.
            server.should_exit = True
            thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS + 1)
            if thread.is_alive():
                server.force_exit = True
                thread.join(timeout=1)
            if thread.is_alive():
                logger.error("Auth proxy did not stop within its shutdown deadline")
