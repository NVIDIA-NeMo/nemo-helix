# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Removing host images through the docker backend."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from nemo_deployments_plugin.backends.base import ImageInUseError, ImagesUnsupportedError
from nemo_deployments_plugin.backends.docker.backend import DockerDeploymentBackend
from nemo_deployments_plugin.backends.k8s.backend import K8sDeploymentBackend

from docker import errors as docker_errors


async def test_remove_image_never_forces(
    docker_backend: DockerDeploymentBackend, mock_docker_client: MagicMock
) -> None:
    assert await docker_backend.remove_image("nemo-agents/ws/a:1") is True
    mock_docker_client.images.remove.assert_called_once_with("nemo-agents/ws/a:1", force=False)


async def test_remove_image_reports_missing(
    docker_backend: DockerDeploymentBackend, mock_docker_client: MagicMock
) -> None:
    mock_docker_client.images.remove.side_effect = docker_errors.ImageNotFound("gone")

    assert await docker_backend.remove_image("nemo-agents/ws/a:1") is False


async def test_remove_image_in_use_raises(
    docker_backend: DockerDeploymentBackend, mock_docker_client: MagicMock
) -> None:
    response = MagicMock(status_code=409)
    mock_docker_client.images.remove.side_effect = docker_errors.APIError("conflict", response=response)

    with pytest.raises(ImageInUseError):
        await docker_backend.remove_image("nemo-agents/ws/a:1")


async def test_other_api_errors_propagate(
    docker_backend: DockerDeploymentBackend, mock_docker_client: MagicMock
) -> None:
    response = MagicMock(status_code=500)
    mock_docker_client.images.remove.side_effect = docker_errors.APIError("boom", response=response)

    with pytest.raises(docker_errors.APIError):
        await docker_backend.remove_image("nemo-agents/ws/a:1")


async def test_image_tags_lists_every_tag(
    docker_backend: DockerDeploymentBackend, mock_docker_client: MagicMock
) -> None:
    mock_docker_client.images.get.return_value = MagicMock(tags=["nemo-agents/ws/a:1", "reg.io/nemo-agents/ws/a:1"])

    assert await docker_backend.image_tags("nemo-agents/ws/a:1") == ["nemo-agents/ws/a:1", "reg.io/nemo-agents/ws/a:1"]


async def test_image_tags_reports_missing(
    docker_backend: DockerDeploymentBackend, mock_docker_client: MagicMock
) -> None:
    mock_docker_client.images.get.side_effect = docker_errors.ImageNotFound("gone")

    assert await docker_backend.image_tags("nemo-agents/ws/a:1") is None


async def test_backends_without_an_image_store_refuse() -> None:
    backend = K8sDeploymentBackend.__new__(K8sDeploymentBackend)

    with pytest.raises(ImagesUnsupportedError):
        await backend.remove_image("x")
    with pytest.raises(ImagesUnsupportedError):
        await backend.image_tags("x")
