# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image pull auth tests for DockerDeploymentBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from backends.docker.docker_helpers import sample_config
from docker.errors import NotFound
from nemo_deployments_plugin.backends.docker.backend import DockerDeploymentBackend
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import Container, DeploymentConfig, EnvVar


def _nim_config(*, ngc_api_key: str | None = None) -> DeploymentConfig:
    return DeploymentConfig(
        name="cfg1",
        workspace="default",
        containers=[
            Container(
                name="main",
                image="nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:1.13.0",
                command=["nim"],
                args=["start"],
                env=[EnvVar(name="NGC_API_KEY", value=ngc_api_key)] if ngc_api_key else [],
            )
        ],
        restart_policy="Always",  # ty: ignore[unknown-argument]
    )


@pytest.fixture
def docker_backend_pull(mock_client: MagicMock, mock_entities: AsyncMock, mock_docker_client: MagicMock):
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.AsyncEntitiesClient"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(mock_client, {"docker_timeout": 60, "pull_images": True})
        backend._client = mock_docker_client
        yield backend


@pytest.mark.asyncio
async def test_create_pulls_nvcr_image_with_ngc_auth(
    docker_backend_pull: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = _nim_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    with patch(
        "nemo_deployments_plugin.backends.docker.backend.resolve_deployment_config_secrets",
        AsyncMock(return_value=_nim_config(ngc_api_key="ngc-test-key")),
    ):
        update = await docker_backend_pull.create_deployment(
            workspace="default",
            name="srv",
            config_name="cfg1",
            labels={"managed-by": MANAGED_BY_LABEL},
            backend_config={},
        )

    assert update.status == "STARTING"
    _, kwargs = mock_docker_client.images.pull.call_args
    assert kwargs.get("auth_config") == {"username": "$oauthtoken", "password": "ngc-test-key"}


@pytest.mark.asyncio
async def test_create_pulls_non_ngc_image_without_auth(
    docker_backend_pull: DockerDeploymentBackend,
    mock_entities: AsyncMock,
    mock_docker_client: MagicMock,
) -> None:
    mock_entities.get.return_value = sample_config()
    mock_docker_client.containers.get.side_effect = NotFound("missing")
    mock_docker_client.containers.create.return_value = MagicMock(id="abc123")

    update = await docker_backend_pull.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    args, kwargs = mock_docker_client.images.pull.call_args
    assert args[0] == "alpine:latest"
    assert not kwargs.get("auth_config")
