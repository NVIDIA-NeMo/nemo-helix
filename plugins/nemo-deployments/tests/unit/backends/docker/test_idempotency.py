# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Idempotency tests for DockerDeploymentBackend.create_deployment."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from backends.docker.docker_helpers import container_attrs, sample_config
from docker.errors import APIError
from nemo_deployments_plugin.backends.docker.backend import DockerDeploymentBackend
from nemo_deployments_plugin.backends.labels import (
    CONFIG_NAME_LABEL,
    DEFAULT_RESOURCE_SCOPE,
    DEPLOYMENT_NAME_LABEL,
    DEPLOYMENT_WORKSPACE_LABEL,
    RESOURCE_SCOPE_LABEL,
    RESTART_POLICY_LABEL,
)
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.types import RestartPolicy


def _matching_labels(*, name: str, restart_policy: str = "Always", config_name: str = "cfg1") -> dict[str, str]:
    return {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "default",
        DEPLOYMENT_NAME_LABEL: name,
        RESTART_POLICY_LABEL: restart_policy,
        CONFIG_NAME_LABEL: config_name,
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }


@pytest.mark.asyncio
async def test_create_existing_matching_container_returns_read_status(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    existing = MagicMock()
    existing.labels = _matching_labels(name="srv")
    existing.id = "abc"
    existing.status = "running"
    existing.ports = {}
    existing.attrs = container_attrs()
    mock_docker_client.containers.get.return_value = existing
    mock_entities.get.return_value = sample_config()

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "READY"
    mock_docker_client.containers.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_policy", ["Never", "OnFailure"])
async def test_create_exited_one_shot_container_is_removed_and_recreated(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
    restart_policy: RestartPolicy,
) -> None:
    """Stale exited puller/job containers must not short-circuit create to SUCCEEDED."""
    existing = MagicMock()
    existing.labels = _matching_labels(name="srv-puller", restart_policy=restart_policy)
    existing.id = "stale123"
    existing.status = "exited"
    existing.ports = {}
    existing.attrs = container_attrs(status="exited", exit_code=0)
    mock_docker_client.containers.get.return_value = existing
    mock_entities.get.return_value = sample_config(restart_policy=restart_policy)
    fresh = MagicMock(id="fresh456")
    fresh.labels = _matching_labels(name="srv-puller", restart_policy=restart_policy)
    fresh.attrs = container_attrs(exit_code=0)
    if restart_policy == "Never":
        fresh.wait.return_value = {"StatusCode": 0}
    else:
        fresh.status = "running"
    mock_docker_client.containers.create.return_value = fresh

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv-puller",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    if restart_policy == "Never":
        assert update.status == "SUCCEEDED"
        assert update.exit_code == 0
        fresh.wait.assert_called_once()
    else:
        assert update.status == "STARTING"
        fresh.wait.assert_not_called()
    existing.remove.assert_called_once_with(force=True)
    mock_docker_client.containers.create.assert_called_once()


@pytest.mark.asyncio
async def test_create_running_one_shot_container_still_returns_status(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    existing = MagicMock()
    existing.labels = _matching_labels(name="srv-puller", restart_policy="OnFailure")
    existing.id = "running123"
    existing.status = "running"
    existing.ports = {}
    existing.attrs = container_attrs()
    mock_docker_client.containers.get.return_value = existing
    mock_entities.get.return_value = sample_config(restart_policy="OnFailure")

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv-puller",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    mock_docker_client.containers.create.assert_not_called()
    existing.remove.assert_not_called()


@pytest.mark.asyncio
async def test_create_exited_one_shot_returns_failed_when_removal_fails(
    docker_backend: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    existing = MagicMock()
    existing.labels = _matching_labels(name="srv-puller", restart_policy="Never")
    existing.status = "exited"
    existing.remove.side_effect = APIError("remove failed")
    mock_docker_client.containers.get.return_value = existing
    mock_entities.get.return_value = sample_config(restart_policy="Never")

    update = await docker_backend.create_deployment(
        workspace="default",
        name="srv-puller",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "FAILED"
    assert "remove exited container" in update.status_message
    existing.remove.assert_called_once_with(force=True)
    mock_docker_client.containers.create.assert_not_called()
