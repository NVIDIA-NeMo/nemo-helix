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
from collections.abc import Generator, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from urllib.parse import urlsplit

from nemo_agents_plugin.agent_config import AgentConfig, ModelConfig, TelemetryConfig
from nemo_agents_plugin.fabric.gateway_models import gateway_models, is_gateway_url
from nemo_helix_plugin.client.auth_proxy_server import platform_auth_proxy as platform_auth_proxy
from nemo_helix_plugin.client.auth_proxy_server import serve_auth_proxy as serve_auth_proxy
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR

logger = logging.getLogger(__name__)


def _onto_proxy(url: str, origin: str) -> str:
    proxy = urlsplit(origin)
    return urlsplit(url)._replace(scheme=proxy.scheme, netloc=proxy.netloc).geturl()


def _gateway_models(config: AgentConfig) -> Iterator[ModelConfig]:
    models = [*config.models.values(), *(h.model for h in config.harnesses.values() if h.model is not None)]
    for model in models:
        if is_gateway_url(model.base_url or model.settings.get("base_url")):
            yield model


def job_has_platform_identity() -> bool:
    """Whether this job carries an identity the proxy could forward.

    True when the substrate serialized a non-anonymous identity into the job --
    a workload-identity token file, or a principal. It is the broadest reason to
    run a proxy: an authenticated job may reach the platform API at any time (to
    read its own record, list jobs, submit sub-jobs), not only for the inference
    and telemetry paths the config happens to name. An auth-disabled platform
    serializes an anonymous principal, for which :func:`platform_auth_proxy`
    still yields ``None`` -- so this only decides whether to *attempt* a proxy;
    the proxy itself remains the authority on whether there is anything to
    forward.
    """
    return bool(os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR) or os.environ.get("NHX_PRINCIPAL"))


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


def routes_fabric_inference_through_gateway(config: dict[str, Any]) -> bool:
    """:func:`routes_inference_through_gateway` for a translated Fabric config."""
    return bool(gateway_models(config))


def rewrite_fabric_gateway_models(config: dict[str, Any], origin: str | None) -> dict[str, Any]:
    """:func:`rewrite_gateway_models` for a translated Fabric config, with the same contract."""
    if origin is None or not routes_fabric_inference_through_gateway(config):
        return config

    runtime_config = copy.deepcopy(config)
    for model in gateway_models(runtime_config):
        if not isinstance(model.config, dict):
            continue
        model.config["base_url"] = _onto_proxy(model.base_url, origin)
        settings = model.config.get("settings")
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

    runtime_config = config.model_copy(deep=True)
    for endpoint_config in _intake_otlp_endpoints(runtime_config.telemetry):
        endpoint_config["endpoint"] = _onto_proxy(endpoint_config["endpoint"], origin)
    return runtime_config


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
