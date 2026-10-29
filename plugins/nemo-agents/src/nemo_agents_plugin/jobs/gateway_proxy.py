# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reach the platform from inside a job with the executing job's identity.

The agent process an execute job runs reaches the platform three ways: its
inference goes to Inference Gateway, and Relay posts the run's trajectory and
its OpenTelemetry signals to Intake. None carries a platform credential, so all
travel through one loopback proxy, which authenticates every forwarded request
with the job's own identity. The task around it needs none of this -- it holds
an SDK that authenticates its own calls.

A proxy rather than credentials in the agent's environment is what makes a
long-running job work at all: the token provider exchanges on demand, so a run
that outlives any single access token keeps exporting. It also keeps the bearer
in this process, where the agent subprocess cannot read it back out.
"""

from __future__ import annotations

import copy
import logging
import os
import socket
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from nemo_agents_plugin.agent_config import AgentConfig, ModelConfig, TelemetryConfig
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.client.auth import EnvAccessToken, TokenProviderAuth
from nemo_helix_plugin.client.auth_proxy import build_auth_proxy_app
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, get_env_access_token
from nemo_helix_plugin.client.oidc_factory import resolve_workload_exchange_provider
from nemo_helix_plugin.client_provider import get_forwarding_headers, get_nemo_client

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT_SECONDS = 10.0
_SHUTDOWN_TIMEOUT_SECONDS = 5


def _is_gateway_url(base_url: object) -> bool:
    return isinstance(base_url, str) and urlsplit(base_url).path.startswith("/apis/inference-gateway/")


def _onto_proxy(url: str, origin: str) -> str:
    proxy = urlsplit(origin)
    return urlsplit(url)._replace(scheme=proxy.scheme, netloc=proxy.netloc).geturl()


def _gateway_models(config: AgentConfig) -> Iterator[ModelConfig]:
    models = [*config.models.values(), *(h.model for h in config.harnesses.values() if h.model is not None)]
    for model in models:
        if _is_gateway_url(model.base_url or model.settings.get("base_url")):
            yield model


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

    runtime_config = config.model_copy(deep=True)
    for model in _gateway_models(runtime_config):
        model.base_url = _onto_proxy(model.base_url or model.settings["base_url"], origin)
        model.settings.pop("base_url", None)
    return runtime_config


def _fabric_gateway_models(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for model in (config.get("models") or {}).values():
        if _is_gateway_url(model.get("base_url") or (model.get("settings") or {}).get("base_url")):
            yield model


def routes_fabric_inference_through_gateway(config: dict[str, Any]) -> bool:
    """:func:`routes_inference_through_gateway` for a translated Fabric config."""
    return any(_fabric_gateway_models(config))


def rewrite_fabric_gateway_models(config: dict[str, Any], origin: str | None) -> dict[str, Any]:
    """:func:`rewrite_gateway_models` for a translated Fabric config, with the same contract."""
    if origin is None or not routes_fabric_inference_through_gateway(config):
        return config

    runtime_config = copy.deepcopy(config)
    for model in _fabric_gateway_models(runtime_config):
        settings = model.get("settings") or {}
        model["base_url"] = _onto_proxy(model.get("base_url") or settings["base_url"], origin)
        settings.pop("base_url", None)
    return runtime_config


def _intake_otlp_endpoints(telemetry: TelemetryConfig) -> Iterator[dict[str, Any]]:
    opentelemetry = telemetry.opentelemetry
    if not isinstance(opentelemetry, dict):
        return
    for section in (opentelemetry, opentelemetry.get("logs"), opentelemetry.get("metrics")):
        endpoints = section.get("endpoints") if isinstance(section, dict) else None
        for endpoint_config in endpoints if isinstance(endpoints, list) else []:
            endpoint = endpoint_config.get("endpoint") if isinstance(endpoint_config, dict) else None
            if isinstance(endpoint, str) and urlsplit(endpoint).path.startswith("/apis/intake/"):
                yield endpoint_config


def exports_otlp_to_intake(config: AgentConfig) -> bool:
    """Whether any of *config*'s OpenTelemetry trace, log or metric endpoints point at Intake.

    The third reason to run a proxy, alongside gateway inference and the ATIF export.
    """
    return any(_intake_otlp_endpoints(config.telemetry))


def rewrite_intake_otlp_endpoints(config: AgentConfig, origin: str | None) -> AgentConfig:
    """Return *config* with its Intake OpenTelemetry endpoints pointed at *origin*.

    The same contract as :func:`rewrite_gateway_models`: the config itself when
    there is nothing to route or no proxy to route through, otherwise a deep
    copy. A collector the agent chose for itself is left alone.
    """
    if origin is None or not exports_otlp_to_intake(config):
        return config

    runtime_config = config.model_copy(deep=True)
    for endpoint_config in _intake_otlp_endpoints(runtime_config.telemetry):
        endpoint_config["endpoint"] = _onto_proxy(endpoint_config["endpoint"], origin)
    return runtime_config


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
def optional_platform_auth_proxy() -> Generator[str | None, None, None]:
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
