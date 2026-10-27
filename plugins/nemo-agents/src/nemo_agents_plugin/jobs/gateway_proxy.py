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
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, overload
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


def _gateway_models(config: AgentConfig | Mapping[str, Any]) -> Iterator[tuple[ModelConfig | dict[str, Any], str]]:
    if isinstance(config, AgentConfig):
        models: list[ModelConfig | dict[str, Any]] = [
            *config.models.values(),
            *(h.model for h in config.harnesses.values() if h.model is not None),
        ]
    else:
        model_entries = config.get("models")
        harness_entries = config.get("harnesses")
        models = (
            [model for model in model_entries.values() if isinstance(model, dict)]
            if isinstance(model_entries, Mapping)
            else []
        )
        if isinstance(harness_entries, Mapping):
            models.extend(
                model
                for harness in harness_entries.values()
                if isinstance(harness, Mapping) and isinstance((model := harness.get("model")), dict)
            )
    for model in models:
        if isinstance(model, ModelConfig):
            base_url = model.base_url or model.settings.get("base_url")
        else:
            base_url = model.get("base_url")
            settings = model.get("settings")
            if not base_url and isinstance(settings, Mapping):
                base_url = settings.get("base_url")
        if isinstance(base_url, str) and urlsplit(base_url).path.startswith("/apis/inference-gateway/"):
            yield model, base_url


def routes_inference_through_gateway(config: AgentConfig | Mapping[str, Any]) -> bool:
    """Whether any of *config*'s models point at Inference Gateway.

    One of the two reasons to run a proxy; the caller ORs it with the other
    (a trajectory export) to decide whether to start one at all.
    """
    return any(_gateway_models(config))


@overload
def rewrite_gateway_models(config: AgentConfig, origin: str | None) -> AgentConfig: ...


@overload
def rewrite_gateway_models(config: Mapping[str, Any], origin: str | None) -> Mapping[str, Any]: ...


def rewrite_gateway_models(
    config: AgentConfig | Mapping[str, Any], origin: str | None
) -> AgentConfig | Mapping[str, Any]:
    """Return *config* with its gateway models pointed at *origin*.

    Returns the config itself when there is no proxy to route through or no
    gateway model to route, so an unaffected run is handed exactly what it was
    given. Otherwise a deep copy is rewritten: the input belongs to the caller,
    and the proxy origin is true only for the life of this invocation.
    """
    if origin is None or not routes_inference_through_gateway(config):
        return config

    proxy = urlsplit(origin)
    runtime_config = config.model_copy(deep=True) if isinstance(config, AgentConfig) else copy.deepcopy(config)
    for model, original_url in _gateway_models(runtime_config):
        rewritten = urlsplit(original_url)._replace(scheme=proxy.scheme, netloc=proxy.netloc).geturl()
        if isinstance(model, ModelConfig):
            model.base_url = rewritten
            model.settings.pop("base_url", None)
        else:
            model["base_url"] = rewritten
            settings = model.get("settings")
            if isinstance(settings, dict):
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

    proxy = urlsplit(origin)
    runtime_config = config.model_copy(deep=True)
    for endpoint_config in _intake_otlp_endpoints(runtime_config.telemetry):
        endpoint = urlsplit(endpoint_config["endpoint"])
        endpoint_config["endpoint"] = endpoint._replace(scheme=proxy.scheme, netloc=proxy.netloc).geturl()
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
