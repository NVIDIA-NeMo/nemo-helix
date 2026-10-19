# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared GPU pool for Docker-based services.

This module provides a thread-safe GPU allocation pool that can be shared
between the jobs and models services to prevent GPU resource conflicts.
"""

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class GPUAllocationError(Exception):
    """Raised when GPU allocation fails due to insufficient resources."""

    def __init__(
        self,
        message: str,
        *,
        requested: int | None = None,
        available: int | None = None,
        total: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.requested = requested
        self.available = available
        self.total = total

    @property
    def is_transient_capacity_exhaustion(self) -> bool:
        """Whether a retry could succeed after another workload releases GPUs."""
        return (
            self.requested is not None
            and self.available is not None
            and self.total is not None
            and self.requested > 0
            and self.available < self.requested <= self.total
        )


@dataclass
class GPUPoolStatus:
    """Status information for a GPU pool.

    Attributes:
        total: Total number of GPUs in the pool
        available: Number of currently available GPUs
        allocated: Number of currently allocated GPUs
        allocations: Dict mapping workload_id to list of GPU IDs allocated to it
        gpu_state: Dict mapping GPU ID to workload_id (or None if available)
    """

    total: int
    available: int
    allocated: int
    allocations: dict[str, list[int]] = field(default_factory=dict)
    gpu_state: dict[int, str | None] = field(default_factory=dict)


class DockerGPUPool:
    """A thread-safe pool of GPU resources for Docker containers.

    This class tracks which GPUs are allocated to which workloads and ensures
    that multiple containers don't try to use the same GPU simultaneously.

    Usage:
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1, 2, 3])

        # Allocate GPUs for a workload
        gpu_ids = pool.allocate_gpu("job-123", num_requested=2)
        # gpu_ids might be [0, 1]

        # Use gpu_ids when creating Docker container:
        # docker.types.DeviceRequest(device_ids=[str(id) for id in gpu_ids], ...)

        # Release GPUs when workload completes
        pool.release_gpu("job-123")
    """

    def __init__(self, reserved_gpu_device_ids: list[int]):
        """Initialize the GPU pool with the available GPU device IDs.

        Args:
            reserved_gpu_device_ids: List of GPU device IDs available for allocation.
                                    e.g., [0, 1, 2, 3] for a 4-GPU system.
        """
        self.num_reserved_gpus = len(reserved_gpu_device_ids)
        self.gpu_to_workload_id: dict[int, str | None] = {gpu_id: None for gpu_id in reserved_gpu_device_ids}
        self._mutex = threading.Lock()

    def allocate_gpu(self, workload_id: str, num_requested: int = 1) -> list[int]:
        """Allocate GPUs for a workload.

        Args:
            workload_id: Unique identifier for the workload (job ID, deployment key, etc.)
            num_requested: Number of GPUs to allocate. Must be a positive integer.

        Returns:
            List of allocated GPU device IDs

        Raises:
            GPUAllocationError: If num_requested is invalid (zero or negative), or if
                               not enough GPUs are available.
        """
        with self._mutex:
            # Validate num_requested: must be positive
            if num_requested <= 0:
                raise GPUAllocationError(f"Invalid GPU request: {num_requested}. Must be a positive integer.")

            available_gpus = {gpu for gpu, workload in self.gpu_to_workload_id.items() if workload is None}

            if len(available_gpus) < num_requested:
                available_count = len(available_gpus)
                raise GPUAllocationError(
                    f"Not enough GPUs available. Requested {num_requested}, "
                    f"available {available_count} out of {self.num_reserved_gpus} total.",
                    requested=num_requested,
                    available=available_count,
                    total=self.num_reserved_gpus,
                )
            gpu_ids = []
            for _ in range(num_requested):
                gpu_id = available_gpus.pop()
                gpu_ids.append(gpu_id)
                self.gpu_to_workload_id[gpu_id] = workload_id
            logger.info(f"DockerGPUPool: Allocated gpu_ids {gpu_ids} to workload {workload_id}")
            return gpu_ids

    def release_gpu(self, workload_id: str) -> list[int]:
        """Release GPUs allocated to a workload.

        Args:
            workload_id: The workload identifier used during allocation

        Returns:
            List of GPU device IDs that were released
        """
        with self._mutex:
            gpu_ids = [gpu for gpu, workload in self.gpu_to_workload_id.items() if workload == workload_id]
            if gpu_ids:
                logger.info(f"DockerGPUPool: Releasing gpu_ids {gpu_ids} from workload {workload_id}")
            for gpu_id in gpu_ids:
                self.gpu_to_workload_id[gpu_id] = None
            return gpu_ids

    def get_available_count(self) -> int:
        """Get the number of currently available GPUs."""
        with self._mutex:
            return sum(1 for workload in self.gpu_to_workload_id.values() if workload is None)

    def get_allocated_workloads(self) -> dict[int, str]:
        """Get a mapping of GPU IDs to their allocated workload IDs.

        Returns:
            Dict mapping GPU ID to workload ID (only for allocated GPUs)
        """
        with self._mutex:
            return {gpu: workload for gpu, workload in self.gpu_to_workload_id.items() if workload is not None}

    def _get_allocations_by_workload(self) -> dict[str, list[int]]:
        """Group allocated GPUs by workload ID. Caller must hold _mutex.

        Returns:
            Dict mapping workload_id to list of GPU IDs allocated to it
        """
        allocations: dict[str, list[int]] = {}
        for gpu_id, workload_id in self.gpu_to_workload_id.items():
            if workload_id is not None:
                allocations.setdefault(workload_id, []).append(gpu_id)
        return allocations

    def get_pool_status(self) -> GPUPoolStatus:
        """Get detailed status of the GPU pool for diagnostics.

        Returns:
            GPUPoolStatus dataclass with pool status information.
        """
        with self._mutex:
            available_count = sum(1 for workload in self.gpu_to_workload_id.values() if workload is None)

            return GPUPoolStatus(
                total=self.num_reserved_gpus,
                available=available_count,
                allocated=self.num_reserved_gpus - available_count,
                allocations=self._get_allocations_by_workload(),
                gpu_state=dict(self.gpu_to_workload_id),
            )

    def force_release_all(self) -> dict[str, list[int]]:
        """Force release all GPU allocations. USE WITH CAUTION - for debugging only.

        This method should only be used for debugging or emergency recovery.
        It will release all GPUs regardless of whether containers are still using them.

        Returns:
            Dict mapping workload_id to list of GPU IDs that were released
        """
        with self._mutex:
            # Group GPUs by workload before releasing
            released = self._get_allocations_by_workload()

            # Release all
            for gpu_id in self.gpu_to_workload_id:
                self.gpu_to_workload_id[gpu_id] = None

            if released:
                logger.warning(f"DockerGPUPool: Force released all GPUs: {released}")
            return released
