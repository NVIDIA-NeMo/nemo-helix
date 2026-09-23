# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Distributed training coordination utilities.

Provides role detection and file-based barrier synchronization for multi-node
training where multiple pods/containers run the same entry point.
"""

import logging
import os
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

# Marker file the coordinator writes when it fails before signaling a barrier.
# Workers poll for it so they can abort promptly instead of waiting out the timeout.
COORDINATOR_FAILURE_MARKER = "coordinator.failed"


class CoordinatorFailedError(RuntimeError):
    """Raised in a worker when the coordinator has published a failure marker."""


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
            barrier_dir.mkdir(parents=True, exist_ok=True)

            # Coordinator clears stale marker files from previous task runs
            # (e.g., after pause/resume or retry). We unlink individual markers
            # rather than rmtree the directory: rmtree can race with a worker that
            # has already created the directory or a live marker, deleting state
            # out from under it and deadlocking the barrier.
            if role == DistributedRole.COORDINATOR:
                logger.info(f"Cleaning up stale barrier markers from previous run: {barrier_dir}")
                for stale_marker in list(barrier_dir.glob("*.ready")) + list(
                    barrier_dir.glob(COORDINATOR_FAILURE_MARKER)
                ):
                    try:
                        stale_marker.unlink()
                    except OSError as e:
                        logger.warning(f"Failed to remove stale barrier marker {stale_marker}: {e}")

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

    def _failure_marker_path(self) -> Path:
        """Get path to the coordinator failure marker file."""
        return self.barrier_dir / COORDINATOR_FAILURE_MARKER

    def signal_failure(self) -> None:
        """
        Publish a coordinator failure marker (coordinator only).

        Workers blocked in :meth:`wait_for_coordinator` / :meth:`wait_all` poll
        for this marker and abort with :class:`CoordinatorFailedError` instead of
        stranding on the barrier until the timeout expires.
        """
        if not self.is_distributed or not self.is_coordinator:
            return

        try:
            self._failure_marker_path().touch()
            logger.info("Published coordinator failure marker to release waiting workers")
        except OSError as e:
            logger.warning(f"Failed to write coordinator failure marker: {e}")

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
            CoordinatorFailedError: If the coordinator published a failure marker
            TimeoutError: If coordinator doesn't signal within timeout
        """
        if not self.is_distributed:
            return

        if self.is_coordinator:
            # Coordinator doesn't wait for itself
            return

        timeout = self._barrier_timeout if timeout is None else timeout
        marker = self._marker_path(barrier_name, rank=0)
        failure_marker = self._failure_marker_path()
        start = time.time()

        logger.debug(f"Waiting for coordinator at barrier: {barrier_name}")

        while time.time() - start < timeout:
            if marker.exists():
                logger.debug(f"Coordinator signaled barrier: {barrier_name}")
                return
            if failure_marker.exists():
                raise CoordinatorFailedError(f"Coordinator reported failure while waiting at barrier '{barrier_name}'")
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
            CoordinatorFailedError: If the coordinator published a failure marker
            TimeoutError: If not all ranks signal within timeout
        """
        if not self.is_distributed:
            return

        timeout = self._barrier_timeout if timeout is None else timeout
        failure_marker = self._failure_marker_path()
        start = time.time()

        logger.debug(f"Waiting for all ranks at barrier: {barrier_name}")

        while time.time() - start < timeout:
            ready_count = sum(1 for r in range(self.world_size) if self._marker_path(barrier_name, r).exists())
            if ready_count >= self.world_size:
                logger.debug(f"All ranks reached barrier: {barrier_name}")
                return
            # A non-coordinator rank should bail out if the coordinator died; the
            # coordinator itself never waits on its own failure marker.
            if not self.is_coordinator and failure_marker.exists():
                raise CoordinatorFailedError(f"Coordinator reported failure while waiting at barrier '{barrier_name}'")
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
