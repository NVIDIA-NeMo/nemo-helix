# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Referential integrity checks before destructive operations."""

from __future__ import annotations

from nemo_deployments_plugin.entities import Container, Deployment, DeploymentConfig
from nemo_helix_plugin.entity_client import NemoEntitiesClient


async def deployment_names_using_config(
    entity_client: NemoEntitiesClient,
    *,
    workspace: str,
    config_name: str,
) -> list[str]:
    """Return deployment names in workspace that reference the given config."""
    page = 1
    names: list[str] = []
    while True:
        result = await entity_client.list(
            Deployment,
            workspace=workspace,
            page=page,
            page_size=100,
            filter_obj={"deployment_config": config_name},
        )
        names.extend(dep.name for dep in result.data)
        if result.pagination is None or page >= result.pagination.total_pages:
            break
        page += 1
    return names


def _container_mounts_volume(container: Container, volume_name: str) -> bool:
    return any(mount.name == volume_name for mount in container.volume_mounts)


def _config_references_volume(config: DeploymentConfig, volume_name: str) -> bool:
    if any(mount.name == volume_name for mount in config.volume_mounts):
        return True
    for container in (*config.containers, *config.init_containers):
        if _container_mounts_volume(container, volume_name):
            return True
    return False


async def deployment_config_names_referencing_volume(
    entity_client: NemoEntitiesClient,
    *,
    workspace: str,
    volume_name: str,
) -> list[str]:
    """Return deployment-config names in workspace whose mounts reference the volume."""
    page = 1
    names: list[str] = []
    while True:
        result = await entity_client.list(
            DeploymentConfig,
            workspace=workspace,
            page=page,
            page_size=100,
        )
        for config in result.data:
            if _config_references_volume(config, volume_name):
                names.append(config.name)
        if result.pagination is None or page >= result.pagination.total_pages:
            break
        page += 1
    return names
