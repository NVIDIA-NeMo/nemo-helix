# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level quickstart cluster management API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .config import QuickstartConfig
from .container import ContainerManager
from .platform_config import HelixConfig
from .preflight import CheckStatus, PreflightChecker, PreflightResult
from .storage import StorageManager


class PreflightError(Exception):
    """Raised when pre-flight checks fail."""

    def __init__(self, results: list[PreflightResult]):
        """Initialize preflight error.

        Args:
            results: List of preflight check results.
        """
        self.results = results
        failures = [r for r in results if r.status == CheckStatus.FAIL]
        message = "Pre-flight checks failed:\n" + "\n".join(f"  - {r.name}: {r.message}" for r in failures)
        super().__init__(message)


class QuickstartCluster:
    """High-level API for managing a quickstart cluster.

    This is the main SDK interface for programmatic quickstart management.

    Example:
        from nemo_helix_ext.quickstart import QuickstartCluster

        cluster = QuickstartCluster()
        cluster.start()
        print(cluster.status())
        cluster.stop()

    The cluster runs a single nhx-api container with:
    - Docker socket mounted for job execution (DOOD pattern)
    - Persistent data volume for storage
    - Configurable platform settings
    """

    def __init__(
        self,
        config: QuickstartConfig | None = None,
        platform_config: HelixConfig | None = None,
        platform_config_path: Path | None = None,
    ):
        """Initialize a quickstart cluster.

        Args:
            config: Quickstart configuration. Loaded from default path if not provided.
            platform_config: Platform configuration. Uses default if not provided.
            platform_config_path: Path to platform config YAML. Overrides platform_config.
        """
        self.config = config or QuickstartConfig.load()

        if platform_config_path:
            self.platform_config = HelixConfig.load(platform_config_path)
            self.config.platform_config_path = platform_config_path.resolve()
        elif platform_config:
            self.platform_config = platform_config
        else:
            self.platform_config = HelixConfig.get_default()

        self._container_manager = ContainerManager(self.config)
        self._preflight_checker = PreflightChecker(self.config)
        self._storage_manager = StorageManager(self.config.storage_path)

    def preflight(self) -> list[PreflightResult]:
        """Run pre-flight checks and return results.

        Returns:
            List of PreflightResult objects.
        """
        return self._preflight_checker.run_all()

    def start(
        self,
        skip_preflight: bool = False,
        pull: bool = True,
    ) -> None:
        """Start the quickstart cluster.

        Args:
            skip_preflight: Skip pre-flight checks (not recommended).
            pull: Pull the container image before starting.

        Raises:
            PreflightError: If pre-flight checks fail.
            docker.errors.APIError: If Docker operations fail.
        """
        if not skip_preflight:
            results = self.preflight()
            if self._preflight_checker.has_failures():
                raise PreflightError(results)

        # Initialize storage directories
        self._storage_manager.initialize()

        self._container_manager.start(
            platform_config=self.platform_config,
            pull=pull,
        )

    def stop(self) -> None:
        """Stop the quickstart cluster."""
        self._container_manager.stop()

    def destroy(self) -> None:
        """Stop the cluster and remove all data."""
        self._container_manager.destroy()

    def status(self) -> dict:
        """Get cluster status.

        Returns:
            Dictionary with status information including:
            - running: bool
            - status: str
            - health: str
            - url: str (if running)
        """
        status = self._container_manager.status()
        if status["running"]:
            effective_config = self._container_manager.get_effective_config()
            status["url"] = f"http://localhost:{effective_config.host_port}"
        return status

    def logs(self, follow: bool = False, tail: int | None = 100) -> Iterator[str]:
        """Stream cluster logs.

        Args:
            follow: Keep following log output.
            tail: Number of lines to show from the end, or None for all logs.

        Yields:
            Log lines as strings.
        """
        yield from self._container_manager.logs(follow=follow, tail=tail)

    def cluster_info(self) -> dict:
        """Get detailed cluster information.

        Returns:
            Dictionary with cluster configuration and state.
        """
        status = self.status()
        effective_config = self._container_manager.get_effective_config()
        return {
            **status,
            "config": {
                "image": effective_config.image,
                "port": effective_config.host_port,
                "storage_path": str(effective_config.storage_path),
                "docker_socket": str(effective_config.docker_socket),
            },
            "storage_size": self._storage_manager.get_size_human() if self._storage_manager.exists() else "0 B",
        }

    def is_running(self) -> bool:
        """Check if the cluster is running.

        Returns:
            True if the cluster container is running.
        """
        return self.status().get("running", False)

    def wait_for_healthy(self, timeout: int = 300, interval: int = 5) -> bool:
        """Wait for the cluster to become healthy.

        Args:
            timeout: Maximum time to wait in seconds.
            interval: Time between health checks in seconds.

        Returns:
            True if cluster became healthy within timeout, False otherwise.
        """
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.status()
            if status.get("health") == "healthy":
                return True
            if not status.get("running"):
                return False
            time.sleep(interval)
        return False
