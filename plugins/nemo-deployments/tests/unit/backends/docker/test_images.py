# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Listing and removing host images through the docker backend."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from nemo_deployments_plugin.backends.base import ImageInUseError, ImagesUnsupportedError
from nemo_deployments_plugin.backends.docker.backend import DockerDeploymentBackend
from nemo_deployments_plugin.backends.k8s.backend import K8sDeploymentBackend

from docker import errors as docker_errors


def _image(image_id: str, tags: list[str], labels: dict[str, str] | None = None) -> MagicMock:
    image = MagicMock(id=image_id, tags=tags, labels=labels or {})
    image.attrs = {"Size": 1024, "Created": "2026-11-01T00:00:00Z"}
    return image


def _container(image_id: str) -> MagicMock:
    return MagicMock(attrs={"ImageID": image_id})


async def test_list_images_matches_on_prefix_and_counts_containers(
    docker_backend: DockerDeploymentBackend, mock_docker_client: MagicMock
) -> None:
    mock_docker_client.images.list.return_value = [
        _image("sha256:a", ["nemo-agents/ws/a-1:0.1", "mine/a:latest"], {"com.nemo.agent.name": "a"}),
        _image("sha256:b", ["nemo-agents/other/b-1:0.1"]),
        _image("sha256:c", ["alpine:latest"]),
    ]
    mock_docker_client.containers.list.return_value = [_container("sha256:a"), _container("sha256:a")]

    images = await docker_backend.list_images(repository_prefix="nemo-agents/ws/")

    assert len(images) == 1
    assert images[0].id == "sha256:a"
    assert images[0].tags == ("mine/a:latest", "nemo-agents/ws/a-1:0.1")
    assert images[0].labels == {"com.nemo.agent.name": "a"}
    assert images[0].containers == 2
    assert images[0].size == 1024
    mock_docker_client.containers.list.assert_called_once_with(all=True, sparse=True)


async def test_list_images_skips_container_scan_without_matches(
    docker_backend: DockerDeploymentBackend, mock_docker_client: MagicMock
) -> None:
    mock_docker_client.images.list.return_value = [_image("sha256:c", ["alpine:latest"])]

    assert await docker_backend.list_images(repository_prefix="nemo-agents/ws/") == []
    mock_docker_client.containers.list.assert_not_called()


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
        await backend.list_images(repository_prefix="x/")
    with pytest.raises(ImagesUnsupportedError):
        await backend.remove_image("x")
    with pytest.raises(ImagesUnsupportedError):
        await backend.image_tags("x")
