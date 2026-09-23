# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for docker backend unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_deployments_plugin.backends.docker.backend import DockerDeploymentBackend
from nemo_helix_plugin.capabilities import reset_capability_cache


@pytest.fixture(autouse=True)
def _reset_capability_cache() -> Iterator[None]:
    """Prevent probe_docker process cache from poisoning later fixtures."""
    reset_capability_cache()
    yield
    reset_capability_cache()


@pytest.fixture
def mock_sdk() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_entities() -> AsyncMock:
    client = AsyncMock()
    return client


@pytest.fixture
def mock_docker_client() -> MagicMock:
    client = MagicMock()
    client.containers = MagicMock()
    client.volumes = MagicMock()
    client.images = MagicMock()
    client.close = MagicMock()
    return client


@pytest.fixture
def docker_backend(
    mock_sdk: MagicMock, mock_entities: AsyncMock, mock_docker_client: MagicMock
) -> Iterator[DockerDeploymentBackend]:
    with (
        patch("nemo_deployments_plugin.backends.docker.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.docker.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("nemo_deployments_plugin.backends.docker.backend.get_shared_gpu_pool", return_value=None),
        patch("docker.from_env", return_value=mock_docker_client),
    ):
        backend = DockerDeploymentBackend(mock_sdk, {"docker_timeout": 60, "pull_images": False})
        backend._client = mock_docker_client
        yield backend
