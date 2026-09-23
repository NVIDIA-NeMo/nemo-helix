# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Distributed training coordination utilities.

Provides role detection and file-based barrier synchronization for multi-node
training where multiple pods/containers run the same entry point.
"""

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Environment variables for distributed training injected by Volcano's pytorch plugin.
# Do not confuse these with the same env vars injected by torchrun.
# Here, WORLD_SIZE refers to number of nodes, while torchrun's WORLD_SIZE is the number of GPUs.
# RANK refers to the rank of the node, while torchrun's RANK is the global rank of the GPU.
RANK_ENVVAR = "RANK"
WORLD_SIZE_ENVVAR = "WORLD_SIZE"


class DistributedRole(Enum):
    """Role of this node in distributed training."""

    COORDINATOR = "coordinator"  # Rank 0 - runs all phases
    WORKER = "worker"  # Rank > 0 - only participates in training


@dataclass
class DistributedContext:
    """
    Distributed training context with file-based barrier synchronization.

    In multi-node training, all pods run the same entry point. This context
    provides:
    - Role detection (coordinator vs worker) based on RANK
    - File-based barriers for cross-pod synchronization

    File barriers work by:
    - Coordinator creates marker files to signal phase completion
    - Workers poll for marker files before proceeding
    - All ranks can sync via mutual signal-and-wait

    Attributes:
        role: Whether this node is coordinator (rank 0) or worker
        rank: This node's rank in the distributed job
        world_size: Total number of nodes participating
        barrier_dir: Directory for barrier marker files (on shared storage).
                     Must be provided by caller for multi-node; None for single-node.
    """

    role: DistributedRole
    rank: int
    world_size: int
    barrier_dir: Path
    _barrier_timeout: float = field(default=600.0, repr=False)
    _poll_interval: float = field(default=0.5, repr=False)

    @classmethod
    def from_env(cls, barrier_dir: Path) -> "DistributedContext":
        """
        Create distributed context from environment variables.

        The caller is responsible for constructing the barrier_dir path,
        including any task-specific namespacing for pause/resume support.

        Args:
            barrier_dir: Directory for barrier files (on shared storage).
                         Caller should namespace this by task ID for pause/resume support.

        Environment Variables:
            RANK: This node's rank (default: 0)
            WORLD_SIZE: Total number of nodes (default: 1)

        Returns:
            Configured DistributedContext
        """
        rank = int(os.environ.get(RANK_ENVVAR, "0"))
        world_size = int(os.environ.get(WORLD_SIZE_ENVVAR, "1"))

        role = DistributedRole.COORDINATOR if rank == 0 else DistributedRole.WORKER

        # Setup barrier directory if distributed
        if world_size > 1:
            # Coordinator cleans up stale barriers from previous task runs
            # (e.g., after pause/resume or retry). This must happen before
            # workers start waiting, so we do it here at initialization.
            if role == DistributedRole.COORDINATOR and barrier_dir.exists():
                logger.info(f"Cleaning up stale barriers from previous run: {barrier_dir}")
                shutil.rmtree(barrier_dir, ignore_errors=True)

            barrier_dir.mkdir(parents=True, exist_ok=True)

        ctx = cls(
            role=role,
            rank=rank,
            world_size=world_size,
            barrier_dir=barrier_dir,
        )

        logger.info(
            f"Distributed context: rank={rank}, world_size={world_size}, "
            f"role={role.value}, barriers={'enabled' if ctx.is_distributed else 'disabled'}"
        )

        return ctx

    @property
    def is_coordinator(self) -> bool:
        """True if this is the coordinator node (rank 0)."""
        return self.role == DistributedRole.COORDINATOR

    @property
    def is_distributed(self) -> bool:
        """True if running in multi-node mode."""
        return self.world_size > 1

    # --- Barrier Implementation ---

    def _marker_path(self, barrier_name: str, rank: int) -> Path:
        """Get path to barrier marker file for a specific rank."""
        return self.barrier_dir / f"{barrier_name}.rank{rank}.ready"

    def signal(self, barrier_name: str) -> None:
        """
        Signal that this rank has reached a synchronization point.

        Creates a marker file indicating this rank is ready.

        Args:
            barrier_name: Name of the barrier (should be unique per sync point)
        """
        if not self.is_distributed:
            return

        marker = self._marker_path(barrier_name, self.rank)
        marker.touch()
        logger.debug(f"Barrier signal: {barrier_name} (rank {self.rank})")

    def wait_for_coordinator(self, barrier_name: str, timeout: float | None = None) -> None:
        """
        Wait for the coordinator (rank 0) to signal.

        Used by workers to wait for coordinator to complete a phase.

        Args:
            barrier_name: Name of the barrier to wait for
            timeout: Override default timeout (seconds)

        Raises:
            TimeoutError: If coordinator doesn't signal within timeout
        """
        if not self.is_distributed:
            return

        if self.is_coordinator:
            # Coordinator doesn't wait for itself
            return

        timeout = timeout or self._barrier_timeout
        marker = self._marker_path(barrier_name, rank=0)
        start = time.time()

        logger.debug(f"Waiting for coordinator at barrier: {barrier_name}")

        while time.time() - start < timeout:
            if marker.exists():
                logger.debug(f"Coordinator signaled barrier: {barrier_name}")
                return
            time.sleep(self._poll_interval)

        raise TimeoutError(f"Timeout waiting for coordinator at barrier '{barrier_name}' after {timeout}s")

    def wait_all(self, barrier_name: str, timeout: float | None = None) -> None:
        """
        Wait for all ranks to reach this barrier.

        All ranks must call signal() before any rank proceeds.

        Args:
            barrier_name: Name of the barrier
            timeout: Override default timeout (seconds)

        Raises:
            TimeoutError: If not all ranks signal within timeout
        """
        if not self.is_distributed:
            return

        timeout = timeout or self._barrier_timeout
        start = time.time()

        logger.debug(f"Waiting for all ranks at barrier: {barrier_name}")

        while time.time() - start < timeout:
            ready_count = sum(1 for r in range(self.world_size) if self._marker_path(barrier_name, r).exists())
            if ready_count >= self.world_size:
                logger.debug(f"All ranks reached barrier: {barrier_name}")
                return
            time.sleep(self._poll_interval)

        # Report which ranks are missing for debugging
        missing = [r for r in range(self.world_size) if not self._marker_path(barrier_name, r).exists()]
        raise TimeoutError(f"Timeout at barrier '{barrier_name}' after {timeout}s. Missing ranks: {missing}")

    def sync_point(self, barrier_name: str, timeout: float | None = None) -> None:
        """
        Synchronization point where all ranks must arrive before any proceed.

        Combines signal() and wait_all() - this rank signals and then waits
        for all other ranks.

        Args:
            barrier_name: Name of the sync point
            timeout: Override default timeout (seconds)
        """
        self.signal(barrier_name)
        self.wait_all(barrier_name, timeout)

    def cleanup_barrier(self, barrier_name: str) -> None:
        """
        Clean up barrier marker files (coordinator only).

        Call after all ranks have passed the barrier.

        Args:
            barrier_name: Name of the barrier to clean up
        """
        if not self.is_distributed or not self.is_coordinator:
            return

        for r in range(self.world_size):
            marker = self._marker_path(barrier_name, r)
            try:
                if marker.exists():
                    marker.unlink()
            except OSError as e:
                logger.warning(f"Failed to clean up barrier marker {marker}: {e}")
