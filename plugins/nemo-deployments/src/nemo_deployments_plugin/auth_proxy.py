# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth-proxy sidecar compilation.

A DeploymentConfig with ``auth_proxy_sidecar=True`` gets a loopback auth-proxy
sidecar injected: the nhx-api image running ``nemo services run --sidecars
auth-proxy``, which stamps ``X-NHX-Principal-Id: service:<identity>`` on the
workload's platform calls. The workload targets the proxy on localhost.

Injection is a no-op when platform auth is disabled — the workload's calls are
already trusted on the internal network, so no identity header is needed.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from nemo_deployments_plugin.config import DeploymentsConfig
from nemo_deployments_plugin.entities import (
    Container,
    DeploymentConfig,
    EnvVar,
    ExecAction,
    Probe,
    RestartPolicy,
)
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.config import LOOPBACK_ADDRESSES, get_nemo_config
from nemo_helix_plugin.jobs.image import get_qualified_image

logger = logging.getLogger(__name__)

AUTH_PROXY_CONTAINER_NAME = "auth-proxy"
_NATIVE_SIDECAR_RESTART_POLICY: RestartPolicy = "Always"
_AUTH_PROXY_PRINCIPAL_ENVVAR = "NHX_AUTH_PROXY_PRINCIPAL"
_AUTH_PROXY_ON_BEHALF_OF_ENVVAR = "NHX_AUTH_PROXY_ON_BEHALF_OF"
_AUTH_PROXY_HOST_ENVVAR = "NHX_AUTH_PROXY_HOST"
_AUTH_PROXY_PORT_ENVVAR = "NHX_AUTH_PROXY_PORT"
_AUTH_PROXY_COMMAND = ["nemo", "services", "run", "--sidecars", "auth-proxy"]


def auth_proxy_port() -> int:
    """Return the loopback port the auth-proxy sidecar listens on."""
    return get_nemo_config(DeploymentsConfig).auth_proxy_port


def _upstream_base_url(*, docker: bool) -> str:
    """Return the platform base URL the sidecar forwards to, reachable from its container.

    In docker mode the platform base URL is often a host loopback the container
    cannot reach; substitute the docker-reachable host (e.g. host.docker.internal)
    the same way jobs do. In k8s the base URL is the in-cluster Service DNS and is
    used verbatim.
    """
    from nemo_helix_plugin.config import determine_loopback_override, get_platform_config

    base_url = get_platform_config().base_url.rstrip("/")
    if not docker:
        return base_url
    override = determine_loopback_override()
    if not override:
        return base_url
    parts = urlsplit(base_url)
    if (parts.hostname or "").lower() not in LOOPBACK_ADDRESSES:
        return base_url
    netloc = override if parts.port is None else f"{override}:{parts.port}"
    return parts._replace(netloc=netloc).geturl()


def build_auth_proxy_container(config: DeploymentConfig, *, docker: bool = False) -> Container | None:
    """Return the auth-proxy sidecar Container for *config*, or None.

    Returns None when the config does not request the sidecar, or when platform
    auth is disabled (the sidecar would be pointless — internal calls are already
    trusted). Pass ``docker=True`` so the sidecar's upstream is rewritten to a
    docker-reachable host.
    """
    if not config.auth_proxy_sidecar:
        return None
    if not platform_auth_enabled():
        logger.debug("auth_proxy_sidecar requested but platform auth is disabled; skipping sidecar injection")
        return None

    deployments_config = get_nemo_config(DeploymentsConfig)
    # Guaranteed present: DeploymentConfig validates identity when the sidecar is enabled.
    identity = config.auth_proxy_sidecar_identity
    on_behalf_of = config.auth_proxy_sidecar_on_behalf_of
    port = deployments_config.auth_proxy_port
    image = deployments_config.auth_proxy_image or get_qualified_image(deployments_config.auth_proxy_image_name)

    env = [
        EnvVar(name="NHX_BASE_URL", value=_upstream_base_url(docker=docker)),
        EnvVar(name=_AUTH_PROXY_PRINCIPAL_ENVVAR, value=identity),
        EnvVar(name=_AUTH_PROXY_HOST_ENVVAR, value="127.0.0.1"),
        EnvVar(name=_AUTH_PROXY_PORT_ENVVAR, value=str(port)),
    ]
    if on_behalf_of:
        env.append(EnvVar(name=_AUTH_PROXY_ON_BEHALF_OF_ENVVAR, value=on_behalf_of))

    return Container(
        name=AUTH_PROXY_CONTAINER_NAME,
        image=image,
        command=list(_AUTH_PROXY_COMMAND),
        env=env,
    ).model_copy(
        update={
            "restart_policy": _NATIVE_SIDECAR_RESTART_POLICY,
            # The proxy binds loopback only, so a pod-IP httpGet probe would be
            # refused; exec-probe curls localhost inside the container netns.
            "readiness_probe": Probe(
                exec=ExecAction(command=["sh", "-c", f"curl -sf http://127.0.0.1:{port}/healthz"]),
                initialDelaySeconds=1,
                periodSeconds=5,
                failureThreshold=12,
            ),
        }
    )


def is_auth_proxy_container(container: Container) -> bool:
    """Return whether a compiled container is the platform auth-proxy sidecar."""
    env_names = {env.name for env in container.env}
    return (
        container.name == AUTH_PROXY_CONTAINER_NAME
        and container.command == _AUTH_PROXY_COMMAND
        and _AUTH_PROXY_PRINCIPAL_ENVVAR in env_names
    )
