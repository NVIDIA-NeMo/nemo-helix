# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from nemo_deployments_plugin.backends.docker import gpu as gpu_module
from nemo_deployments_plugin.backends.docker.gpu import (
    DockerGPUPool,
    GPUAllocationError,
    discover_managed_gpu_allocations,
    get_shared_gpu_pool,
    parse_gpu_device_ids,
)
from nemo_deployments_plugin.backends.labels import DEPLOYMENT_NAME_LABEL, DEPLOYMENT_WORKSPACE_LABEL
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL


def test_parse_gpu_device_ids_extracts_nvidia_ids() -> None:
    requests = [
        {
            "Driver": "nvidia",
            "Count": 0,
            "DeviceIDs": ["0", "1"],
            "Capabilities": [["gpu"]],
        }
    ]
    assert parse_gpu_device_ids(requests) == [0, 1]


def test_parse_gpu_device_ids_ignores_non_nvidia() -> None:
    requests = [{"Driver": "amd", "DeviceIDs": ["0"]}]
    assert parse_gpu_device_ids(requests) == []


def test_restore_allocations_marks_gpus_in_use() -> None:
    pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1, 2])
    pool.restore_allocations({"smoke/srv": [0, 1]})
    assert pool.gpu_to_workload_id[0] == "smoke/srv"
    assert pool.gpu_to_workload_id[1] == "smoke/srv"
    assert pool.gpu_to_workload_id[2] is None
    with pytest.raises(GPUAllocationError):
        pool.allocate_gpu("other", num_requested=2)


def test_discover_managed_gpu_allocations_from_running_container() -> None:
    container = MagicMock()
    container.status = "running"
    container.name = "dep-smoke-srv"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "smoke",
        DEPLOYMENT_NAME_LABEL: "srv",
    }
    container.attrs = {
        "HostConfig": {
            "DeviceRequests": [
                {
                    "Driver": "nvidia",
                    "DeviceIDs": ["1"],
                    "Capabilities": [["gpu"]],
                }
            ]
        }
    }

    client = MagicMock()
    client.containers.list.return_value = [container]

    assert discover_managed_gpu_allocations(client) == {"smoke/srv": [1]}


def test_discover_managed_gpu_allocations_skips_exited_container() -> None:
    container = MagicMock()
    container.status = "exited"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "smoke",
        DEPLOYMENT_NAME_LABEL: "srv",
    }

    client = MagicMock()
    client.containers.list.return_value = [container]

    assert discover_managed_gpu_allocations(client) == {}


def test_get_shared_gpu_pool_recovers_running_allocations() -> None:
    gpu_module._pool = None
    container = MagicMock()
    container.status = "running"
    container.name = "dep-smoke-srv"
    container.labels = {
        "managed-by": MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: "smoke",
        DEPLOYMENT_NAME_LABEL: "srv",
    }
    container.attrs = {
        "HostConfig": {
            "DeviceRequests": [
                {
                    "Driver": "nvidia",
                    "DeviceIDs": ["0"],
                    "Capabilities": [["gpu"]],
                }
            ]
        }
    }
    mock_client = MagicMock()
    mock_client.containers.list.return_value = [container]

    with (
        patch.object(gpu_module, "detect_gpu_device_ids", return_value=[0, 1]),
        patch("docker.from_env", return_value=mock_client),
    ):
        pool = get_shared_gpu_pool()

    assert pool is not None
    assert pool.gpu_to_workload_id[0] == "smoke/srv"
    assert pool.gpu_to_workload_id[1] is None
    gpu_module._pool = None


def test_get_shared_gpu_pool_retries_after_recovery_failure() -> None:
    gpu_module._pool = None
    mock_client = MagicMock()
    mock_client.containers.list.side_effect = RuntimeError("docker unavailable")

    with (
        patch.object(gpu_module, "detect_gpu_device_ids", return_value=[0, 1]),
        patch("docker.from_env", return_value=mock_client),
    ):
        assert get_shared_gpu_pool() is None
        assert gpu_module._pool is None

        mock_client.containers.list.side_effect = None
        mock_client.containers.list.return_value = []

        pool = get_shared_gpu_pool()

    assert pool is not None
    assert pool.gpu_to_workload_id == {0: None, 1: None}
    gpu_module._pool = None


def test_get_shared_gpu_pool_propagates_invalid_reservation_config() -> None:
    gpu_module._pool = None
    docker_cfg = MagicMock()
    docker_cfg.get_reserved_gpu_ids.side_effect = ValueError("bad reserved_gpu_device_ids")
    platform_cfg = MagicMock()
    platform_cfg.docker = docker_cfg

    with (
        patch("nemo_helix_plugin.config.Configuration.get_service_config", return_value=platform_cfg),
        patch.object(gpu_module, "detect_gpu_device_ids") as detect,
    ):
        with pytest.raises(ValueError, match="bad reserved_gpu_device_ids"):
            get_shared_gpu_pool()
        detect.assert_not_called()
    gpu_module._pool = None
