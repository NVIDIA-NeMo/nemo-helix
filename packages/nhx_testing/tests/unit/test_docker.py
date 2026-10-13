# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Docker testing helpers."""

from unittest.mock import MagicMock

from nhx.testing.docker import MODELS_CONTROLLER_MANAGED_LABEL, cleanup_model_deployment_containers


def _container(name: str, labels: dict[str, str]) -> MagicMock:
    container = MagicMock()
    container.name = name
    container.labels = labels
    return container


def test_cleanup_model_deployment_containers_filters_by_owner_labels():
    """Cleanup removes only managed containers matching the supplied owner labels."""
    owner_labels = {
        "nhx.nvidia.com/test-run": "run-1",
        "nhx.nvidia.com/test-worker": "gw0",
    }
    matching_container = _container(
        "matching",
        {
            MODELS_CONTROLLER_MANAGED_LABEL: "models-controller",
            **owner_labels,
        },
    )
    other_worker_container = _container(
        "other-worker",
        {
            MODELS_CONTROLLER_MANAGED_LABEL: "models-controller",
            "nhx.nvidia.com/test-run": "run-1",
            "nhx.nvidia.com/test-worker": "gw1",
        },
    )

    docker_client = MagicMock()
    docker_client.containers.list.return_value = [matching_container, other_worker_container]

    removed = cleanup_model_deployment_containers(docker_client, labels=owner_labels)

    assert removed == 1
    docker_client.containers.list.assert_called_once_with(
        all=True,
        filters={"label": MODELS_CONTROLLER_MANAGED_LABEL},
        ignore_removed=True,
    )
    matching_container.stop.assert_called_once()
    matching_container.remove.assert_called_once_with(force=True)
    other_worker_container.stop.assert_not_called()
    other_worker_container.remove.assert_not_called()
