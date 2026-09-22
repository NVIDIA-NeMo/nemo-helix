# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reach the platform from inside a job with the executing job's identity.

The agent process an execute job runs reaches the platform two ways: its
inference goes to Inference Gateway, and Relay posts the run's trajectory to
Intake. Neither carries a platform credential, so both travel through one
loopback proxy, which authenticates every forwarded request with the job's own
identity. The task around it needs none of this -- it holds an SDK that
authenticates its own calls.

A proxy rather than credentials in the agent's environment is what makes a
long-running job work at all: the token provider exchanges on demand, so a run
that outlives any single access token keeps exporting. It also keeps the bearer
in this process, where the agent subprocess cannot read it back out.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from nemo_agents_plugin.agent_config import AgentConfig, ModelConfig
from nemo_platform_plugin.auth import platform_auth_enabled
from nemo_platform_plugin.client.auth import TokenProviderAuth
from nemo_platform_plugin.client.auth_proxy import build_auth_proxy_app
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_platform_plugin.client.oidc_factory import resolve_workload_exchange_provider
from nemo_platform_plugin.sdk_provider import get_forwarding_headers, get_platform_sdk

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT_SECONDS = 10.0
_SHUTDOWN_TIMEOUT_SECONDS = 5


def _gateway_models(config: AgentConfig) -> Iterator[tuple[ModelConfig, str]]:
    models = [*config.models.values(), *(h.model for h in config.harnesses.values() if h.model is not None)]
    for model in models:
        base_url = model.base_url or model.settings.get("base_url")
        if isinstance(base_url, str) and urlsplit(base_url).path.startswith("/apis/inference-gateway/"):
            yield model, base_url


def routes_inference_through_gateway(config: AgentConfig) -> bool:
    """Whether any of *config*'s models point at Inference Gateway.

    One of the two reasons to run a proxy; the caller ORs it with the other
    (a trajectory export) to decide whether to start one at all.
    """
    return any(_gateway_models(config))


def rewrite_gateway_models(config: AgentConfig, origin: str | None) -> AgentConfig:
    """Return *config* with its gateway models pointed at *origin*.

    Returns the config itself when there is no proxy to route through or no
    gateway model to route, so an unaffected run is handed exactly what it was
    given. Otherwise a deep copy is rewritten: the input belongs to the caller,
    and the proxy origin is true only for the life of this invocation.
    """
    if origin is None or not routes_inference_through_gateway(config):
        return config

    proxy = urlsplit(origin)
    runtime_config = config.model_copy(deep=True)
    for model, original_url in _gateway_models(runtime_config):
        model.base_url = urlsplit(original_url)._replace(scheme=proxy.scheme, netloc=proxy.netloc).geturl()
        model.settings.pop("base_url", None)
    return runtime_config


@contextmanager
def platform_auth_proxy() -> Iterator[str | None]:
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
    if not (token_file or os.environ.get("NMP_PRINCIPAL")):
        yield None
        return

    base_url = os.environ.get("NMP_BASE_URL", "")
    parsed_base = urlsplit(base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.hostname:
        raise ValueError("NMP_BASE_URL must be set to the Platform URL for authenticated agent jobs")
    if token_file:
        provider = resolve_workload_exchange_provider(base_url=base_url, subject_token_file=Path(token_file))
        # Fail before starting Fabric if the job cannot authenticate. Subsequent
        # requests use the same provider, which exchanges again as tokens expire.
        provider.get_access_token()
        app = build_auth_proxy_app(base_url=base_url, auth=TokenProviderAuth(provider))
    else:
        with get_platform_sdk() as sdk:
            headers = get_forwarding_headers(sdk)
        if not any(name.lower() == "x-nmp-principal-id" and value.strip() for name, value in headers.items()):
            # Auth-disabled platforms serialize an anonymous Principal into
            # jobs too. Preserve that existing unauthenticated execution mode.
            if not platform_auth_enabled():
                yield None
                return
            raise ValueError("NMP_PRINCIPAL must provide a principal ID for authenticated agent jobs")
        app = build_auth_proxy_app(base_url=base_url, headers=headers)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
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

        thread = threading.Thread(target=serve, name="agent-job-auth-proxy", daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
            while not server.started:
                if not thread.is_alive():
                    raise RuntimeError("Agent job auth proxy failed to start") from (errors[0] if errors else None)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Agent job auth proxy startup timed out")
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
                logger.error("Agent job auth proxy did not stop within its shutdown deadline")


@contextmanager
def optional_platform_auth_proxy() -> Iterator[str | None]:
    """:func:`platform_auth_proxy`, yielding ``None`` rather than failing the run.

    For a job whose only reason to reach the platform is its trajectory export.
    Telemetry is not worth failing an agent over, so a proxy that cannot resolve
    credentials or bind a port leaves the run untraced instead -- the same
    degradation a failed token exchange produced when the export carried its own
    headers. Inference has no such fallback and must keep using
    :func:`platform_auth_proxy` directly.
    """
    stack = ExitStack()
    try:
        origin = stack.enter_context(platform_auth_proxy())
    except Exception:
        logger.warning(
            "Could not start the platform auth proxy for telemetry; the trajectory will be posted without credentials.",
            exc_info=True,
        )
        yield None
        return
    # Entering the stack is a no-op; exiting it unwinds the proxy, so an error
    # raised by the body still reaches the proxy's own shutdown.
    with stack:
        yield origin
