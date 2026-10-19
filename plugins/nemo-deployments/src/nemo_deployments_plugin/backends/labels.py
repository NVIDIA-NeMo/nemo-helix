# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared deployment/volume identity labels and substrate-safe resource naming.

Naming uses ``nemo_helix_plugin.k8s_naming`` (plugins cannot import ``nhx_common``).
Identity labels drive orphan cleanup and idempotency across docker and k8s backends.

Label domain is ``nemo.nvidia.com/*`` (deployments plugin scope). Core services such as
jobs and models use ``nhx.nvidia.com/*`` today; converging prefixes is out of scope for 757.

Deployment and volume resources use separate workspace label keys
(``deployment-workspace`` vs ``volume-workspace``) so list/watch queries can target one
resource kind without ambiguous selectors, even though the workspace value is the same string.
"""

from typing import Any

from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_helix_plugin.k8s_naming import k8s_safe_name, workspace_name_identity

MANAGED_BY_KEY = "managed-by"
DEPLOYMENT_WORKSPACE_LABEL = "nemo.nvidia.com/deployment-workspace"
DEPLOYMENT_NAME_LABEL = "nemo.nvidia.com/deployment-name"
RESTART_POLICY_LABEL = "nemo.nvidia.com/restart-policy"
CONFIG_NAME_LABEL = "nemo.nvidia.com/deployment-config"
VOLUME_WORKSPACE_LABEL = "nemo.nvidia.com/volume-workspace"
VOLUME_NAME_LABEL = "nemo.nvidia.com/volume-name"
BACKOFF_LIMIT_LABEL = "nemo.nvidia.com/backoff-limit"
RESOURCE_SCOPE_LABEL = "nemo.nvidia.com/resource-scope"
DEFAULT_RESOURCE_SCOPE = "default"
# Role of a docker container within a multi-container deployment group
# (e.g. "server" or a sidecar container name). Used to discover companion
# containers (LoRA adapters sidecar) that share the primary server container.
CONTAINER_ROLE_LABEL = "nemo.nvidia.com/container-role"

# The primary container in a deployment group keeps the canonical deployment
# name (``container_name``); companions derive their names from it plus role.
CONTAINER_ROLE_SERVER = "server"


def deployment_key(workspace: str, name: str) -> str:
    """Return the canonical identity string used for hashing and label keys."""
    return workspace_name_identity(workspace, name)


def container_name(workspace: str, deployment_name: str) -> str:
    """Docker container name for a deployment (``dep-`` prefix, hashed identity).

    This is the *primary* (server) container of a deployment group. Companion
    containers (init/sidecar) derive their names from it via
    :func:`companion_container_name`.
    """
    return k8s_deployment_resource_name(workspace, deployment_name)


def companion_container_name(workspace: str, deployment_name: str, role: str) -> str:
    """Docker container name for a companion (sidecar/init) container in a group.

    Derived from the primary container name so the whole group shares the
    ``dep-<hash>`` stem and is easy to discover/tear down together.
    """
    return f"{container_name(workspace, deployment_name)}-{role}"


def docker_volume_name(workspace: str, volume_name: str) -> str:
    """Docker volume name for a deployment volume (``dep-vol-`` prefix, hashed identity)."""
    return k8s_volume_resource_name(workspace, volume_name)


def k8s_deployment_resource_name(workspace: str, deployment_name: str) -> str:
    """Kubernetes resource name for a deployment (Deployment, Job, Service, etc.)."""
    return k8s_safe_name(
        f"dep-{workspace}-{deployment_name}",
        hash_input=deployment_key(workspace, deployment_name),
    )


def k8s_volume_resource_name(workspace: str, volume_name: str) -> str:
    """Kubernetes PVC name for a deployment volume."""
    return k8s_safe_name(
        f"dep-vol-{workspace}-{volume_name}",
        hash_input=deployment_key(workspace, volume_name),
    )


def k8s_deployment_configmap_name(workspace: str, deployment_name: str) -> str:
    """Kubernetes ConfigMap name for deployment config files."""
    return k8s_safe_name(
        f"dep-cm-{workspace}-{deployment_name}",
        hash_input=f"{deployment_key(workspace, deployment_name)}/config",
    )


def k8s_deployment_secret_name(workspace: str, deployment_name: str) -> str:
    """Kubernetes Secret name for a deployment's resolved secret env vars."""
    return k8s_safe_name(
        f"dep-sec-{workspace}-{deployment_name}",
        hash_input=f"{deployment_key(workspace, deployment_name)}/secret",
    )


def deployment_identity_labels(
    workspace: str,
    name: str,
    restart_policy: str,
    *,
    config_name: str,
    backoff_limit: int = 6,
    resource_scope: str = DEFAULT_RESOURCE_SCOPE,
) -> dict[str, str]:
    """Return identity labels attached to deployment backend resources."""
    return {
        MANAGED_BY_KEY: MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: workspace,
        DEPLOYMENT_NAME_LABEL: name,
        RESTART_POLICY_LABEL: restart_policy,
        CONFIG_NAME_LABEL: config_name,
        BACKOFF_LIMIT_LABEL: str(backoff_limit),
        RESOURCE_SCOPE_LABEL: resource_scope,
    }


def volume_identity_labels(
    workspace: str, name: str, *, resource_scope: str = DEFAULT_RESOURCE_SCOPE
) -> dict[str, str]:
    """Return identity labels attached to volume backend resources."""
    return {
        MANAGED_BY_KEY: MANAGED_BY_LABEL,
        VOLUME_WORKSPACE_LABEL: workspace,
        VOLUME_NAME_LABEL: name,
        RESOURCE_SCOPE_LABEL: resource_scope,
    }


def managed_by_filter(*, resource_scope: str | None = None) -> dict[str, Any]:
    """Return a Docker SDK filter dict for plugin-managed resources."""
    labels = [f"{MANAGED_BY_KEY}={MANAGED_BY_LABEL}"]
    if resource_scope is not None:
        labels.append(f"{RESOURCE_SCOPE_LABEL}={resource_scope}")
    return {"label": labels if len(labels) > 1 else labels[0]}


def managed_by_label_selector(*, resource_scope: str | None = None) -> str:
    """Kubernetes label selector for plugin-managed resources."""
    selector = f"{MANAGED_BY_KEY}={MANAGED_BY_LABEL}"
    if resource_scope is not None:
        selector = f"{selector},{RESOURCE_SCOPE_LABEL}={resource_scope}"
    return selector
