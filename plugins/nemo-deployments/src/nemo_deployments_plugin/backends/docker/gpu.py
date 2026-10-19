# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe GPU pool for Docker deployments (plugin-local; not shared with models).

During the 759 cutover both pools may coexist briefly — consolidate into
nemo_helix_plugin when models docker backend is removed.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import docker

logger = logging.getLogger(__name__)


class GPUAllocationError(Exception):
    """Raised when GPU allocation fails due to insufficient resources."""


class DockerGPUPool:
    """Thread-safe pool of GPU device IDs for Docker device_requests."""

    def __init__(self, reserved_gpu_device_ids: list[int]) -> None:
        self.num_reserved_gpus = len(reserved_gpu_device_ids)
        self.gpu_to_workload_id: dict[int, str | None] = {gpu_id: None for gpu_id in reserved_gpu_device_ids}
        self._mutex = threading.Lock()

    def allocate_gpu(self, workload_id: str, num_requested: int = 1) -> list[int]:
        with self._mutex:
            if num_requested <= 0:
                raise GPUAllocationError(f"Invalid GPU request: {num_requested}. Must be a positive integer.")
            available_gpus = {gpu for gpu, workload in self.gpu_to_workload_id.items() if workload is None}
            if len(available_gpus) < num_requested:
                raise GPUAllocationError(
                    f"Not enough GPUs available. Requested {num_requested}, "
                    f"available {len(available_gpus)} out of {self.num_reserved_gpus} total."
                )
            gpu_ids: list[int] = []
            for _ in range(num_requested):
                gpu_id = available_gpus.pop()
                gpu_ids.append(gpu_id)
                self.gpu_to_workload_id[gpu_id] = workload_id
            logger.info("DockerGPUPool: allocated gpu_ids %s to workload %s", gpu_ids, workload_id)
            return gpu_ids

    def release_gpu(self, workload_id: str) -> list[int]:
        with self._mutex:
            gpu_ids = [gpu for gpu, workload in self.gpu_to_workload_id.items() if workload == workload_id]
            if gpu_ids:
                logger.info("DockerGPUPool: releasing gpu_ids %s from workload %s", gpu_ids, workload_id)
            for gpu_id in gpu_ids:
                self.gpu_to_workload_id[gpu_id] = None
            return gpu_ids

    def restore_allocations(self, allocations: dict[str, list[int]]) -> None:
        """Mark GPUs as allocated for workloads discovered from running containers."""
        with self._mutex:
            for workload_id, gpu_ids in allocations.items():
                for gpu_id in gpu_ids:
                    if gpu_id not in self.gpu_to_workload_id:
                        logger.warning(
                            "Skipping GPU %s for workload %s during pool recovery (not in reserved pool)",
                            gpu_id,
                            workload_id,
                        )
                        continue
                    existing = self.gpu_to_workload_id[gpu_id]
                    if existing is not None and existing != workload_id:
                        logger.warning(
                            "GPU %s already allocated to %s; skipping recovery claim for %s",
                            gpu_id,
                            existing,
                            workload_id,
                        )
                        continue
                    self.gpu_to_workload_id[gpu_id] = workload_id


_pool: DockerGPUPool | None = None
_pool_lock = threading.Lock()


def detect_gpu_device_ids() -> list[int]:
    """Return GPU indices from nvidia-smi when available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    ids: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.isdigit():
            ids.append(int(stripped))
    return ids


def parse_gpu_device_ids(device_requests: list[Any] | None) -> list[int]:
    """Extract NVIDIA GPU device IDs from Docker HostConfig DeviceRequests."""
    if not device_requests:
        return []
    gpu_ids: list[int] = []
    for request in device_requests:
        if isinstance(request, dict):
            driver = request.get("Driver")
            raw_ids = request.get("DeviceIDs")
        else:
            driver = getattr(request, "Driver", None)
            raw_ids = getattr(request, "DeviceIDs", None)
        if driver != "nvidia" or not raw_ids:
            continue
        for raw_id in raw_ids:
            try:
                gpu_ids.append(int(raw_id))
            except (TypeError, ValueError):
                continue
    return gpu_ids


def discover_managed_gpu_allocations(client: docker.DockerClient) -> dict[str, list[int]]:
    """Return workload_id -> GPU IDs for running deployment-managed containers."""
    from nemo_deployments_plugin.backends.labels import (
        DEPLOYMENT_NAME_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL,
        MANAGED_BY_KEY,
        deployment_key,
        managed_by_filter,
    )
    from nemo_deployments_plugin.constants import MANAGED_BY_LABEL

    try:
        containers = client.containers.list(all=True, filters=managed_by_filter())
    except Exception:
        logger.warning("Failed to list managed containers for GPU pool recovery", exc_info=True)
        raise

    allocations: dict[str, list[int]] = {}
    for container in containers:
        labels = container.labels or {}
        if labels.get(MANAGED_BY_KEY) != MANAGED_BY_LABEL:
            continue
        workspace = labels.get(DEPLOYMENT_WORKSPACE_LABEL)
        name = labels.get(DEPLOYMENT_NAME_LABEL)
        if not workspace or not name:
            continue
        if container.status != "running":
            continue
        try:
            container.reload()
        except Exception:
            logger.debug("Skipping container %s during GPU pool recovery", container.name, exc_info=True)
            continue
        host_config = container.attrs.get("HostConfig") or {}
        gpu_ids = parse_gpu_device_ids(host_config.get("DeviceRequests"))
        if not gpu_ids:
            continue
        workload_id = deployment_key(workspace, name)
        allocations.setdefault(workload_id, []).extend(gpu_ids)
    return allocations


def _recover_pool_allocations(pool: DockerGPUPool) -> bool:
    """Restore in-use GPUs from running containers. Returns False on transient failure."""
    try:
        import docker
    except ImportError:
        logger.debug("Docker SDK unavailable for GPU pool recovery")
        return True

    client = docker.from_env()
    try:
        allocations = discover_managed_gpu_allocations(client)
        pool.restore_allocations(allocations)
        if allocations:
            logger.info(
                "DockerGPUPool: recovered GPU allocations from %d managed container(s): %s",
                len(allocations),
                allocations,
            )
        return True
    except Exception:
        logger.warning("Failed to recover GPU allocations from managed containers", exc_info=True)
        return False
    finally:
        client.close()


def get_shared_gpu_pool() -> DockerGPUPool | None:
    """Lazy singleton GPU pool shared across docker executor instances in this process."""
    global _pool
    with _pool_lock:
        if _pool is None:
            reserved = None
            try:
                from nemo_helix_plugin.config import Configuration, NemoHelixConfig

                reserved = Configuration.get_service_config(NemoHelixConfig).docker.get_reserved_gpu_ids()
            except ImportError:
                logger.debug(
                    "NeMo Helix configuration is unavailable; using GPU detection",
                    exc_info=True,
                )
            except ValueError:
                logger.exception("Invalid platform.docker.reserved_gpu_device_ids; refusing to fall back to all GPUs")
                raise
            if reserved is not None:
                device_ids = reserved
            else:
                device_ids = detect_gpu_device_ids()
            if not device_ids:
                return None
            logger.info("DockerGPUPool: initializing with reserved GPU device IDs %s", device_ids)
            pool = DockerGPUPool(reserved_gpu_device_ids=device_ids)
            if not _recover_pool_allocations(pool):
                return None
            _pool = pool
        return _pool
