# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile DeploymentConfig into docker.containers.run kwargs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_deployments_plugin.auth_proxy import build_auth_proxy_container
from nemo_deployments_plugin.backends.labels import docker_volume_name
from nemo_deployments_plugin.entities import Container, DeploymentConfig, DockerDeploymentConfig, VolumeMount
from nemo_deployments_plugin.types import RestartPolicy
from nemo_helix_plugin.auth import AuthContext


class DeploymentConfigError(ValueError):
    """Invalid deployment config for docker backend."""


@dataclass(frozen=True)
class DockerDeploymentPlan:
    """The docker backend's view of a (possibly multi-container) DeploymentConfig.

    - ``init_containers`` run to completion, in order, before the group starts.
    - ``primary`` is the server container; it keeps the canonical deployment
      container name, publishes host ports, and owns any GPUs.
    - ``sidecars`` share the primary's network namespace (``network=container:``)
      and volumes; they do not publish host ports or own GPUs.
    """

    primary: Container
    init_containers: list[Container] = field(default_factory=list)
    sidecars: list[Container] = field(default_factory=list)

    @property
    def is_multi_container(self) -> bool:
        """True when the plan has any init containers or sidecars beyond the primary."""
        return bool(self.init_containers or self.sidecars)


def parse_docker_backend_config(backend_config: dict[str, Any]) -> DockerDeploymentConfig:
    docker_section = backend_config.get("docker") or {}
    return DockerDeploymentConfig.model_validate(docker_section)


def build_docker_plan(config: DeploymentConfig, *, auth_context: AuthContext | None = None) -> DockerDeploymentPlan:
    """Split a DeploymentConfig into a docker orchestration plan.

    The first container is the primary/server; any additional containers are
    sidecars sharing the primary's network namespace and volumes (e.g. the LoRA
    adapters sidecar). Init containers run to completion first.

    Raises DeploymentConfigError for shapes the docker backend cannot honor.
    """
    if not config.containers:
        raise DeploymentConfigError("docker backend requires at least one container")

    primary = config.containers[0]
    sidecars = list(config.containers[1:])

    # Auth-proxy sidecar (no-op unless requested and platform auth is enabled).
    # It shares the primary's netns, so the primary reaches it on localhost —
    # the same loopback address the workload targets in k8s. Declares no ports.
    auth_proxy = build_auth_proxy_container(config, docker=True, auth_context=auth_context)
    if auth_proxy is not None:
        sidecars.append(auth_proxy)

    # Sidecars share the primary's netns, so they cannot publish their own host
    # ports. (The primary owns the published ports for the whole group.)
    for sidecar in sidecars:
        if sidecar.ports:
            raise DeploymentConfigError(
                f"docker sidecar container '{sidecar.name}' may not declare ports; "
                "it shares the primary container's network namespace"
            )

    return DockerDeploymentPlan(
        primary=primary,
        init_containers=list(config.init_containers),
        sidecars=sidecars,
    )


def validate_config_for_docker(config: DeploymentConfig) -> Container:
    """Back-compat single-container accessor: returns the primary container.

    Retained for callers/tests that only need the primary; use
    :func:`build_docker_plan` for multi-container orchestration.
    """
    return build_docker_plan(config).primary


def restart_policy_kwargs(restart_policy: RestartPolicy, backoff_limit: int) -> dict[str, Any]:
    if restart_policy == "Always":
        return {"restart_policy": {"Name": "always"}}
    if restart_policy == "OnFailure":
        if backoff_limit < 1:
            raise DeploymentConfigError("backoff_limit must be at least 1 for OnFailure restart policy")
        return {"restart_policy": {"Name": "on-failure", "MaximumRetryCount": backoff_limit}}
    return {}


def env_dict(container: Container) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in container.env:
        if item.value is not None:
            result[item.name] = item.value
    return result


def merged_volume_mounts(config: DeploymentConfig, container: Container) -> list[VolumeMount]:
    by_name: dict[str, VolumeMount] = {}
    for mount in config.volume_mounts:
        by_name[mount.name] = mount
    for mount in container.volume_mounts:
        by_name[mount.name] = mount
    return list(by_name.values())


def build_volume_bindings(
    workspace: str,
    mounts: list[VolumeMount],
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for mount in mounts:
        vol_name = docker_volume_name(workspace, mount.name)
        bindings[vol_name] = {
            "bind": mount.mount_path,
            "mode": "ro" if mount.read_only else "rw",
        }
    return bindings


def build_port_bindings(
    container: Container,
    host_ports: dict[int, int],
) -> dict[str, int | list[tuple[str, int]] | None]:
    ports: dict[str, int | list[tuple[str, int]] | None] = {}
    for port_spec in container.ports:
        container_port = port_spec.container_port
        protocol = port_spec.protocol.lower()
        key = f"{container_port}/{protocol}"
        host_port = host_ports.get(container_port)
        if host_port is not None:
            ports[key] = host_port
        else:
            ports[key] = container_port
    return ports


def gpu_count_from_container(container: Container) -> int:
    limit = container.resources.limits.get("nvidia.com/gpu")
    if not limit:
        return 0
    try:
        return int(limit)
    except ValueError:
        return 0


def device_requests_for_gpus(gpu_ids: list[int]) -> list[dict[str, Any]]:
    if not gpu_ids:
        return []
    return [
        {
            "Driver": "nvidia",
            "Count": 0,
            "DeviceIDs": [str(gpu_id) for gpu_id in gpu_ids],
            "Capabilities": [["gpu"]],
        }
    ]
