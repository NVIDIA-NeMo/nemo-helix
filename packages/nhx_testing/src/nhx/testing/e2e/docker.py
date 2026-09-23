# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker-based E2E test backend using testcontainers.

This backend runs the NeMo Helix API in a Docker container with configuration
loaded from an external YAML file, requiring no external dependencies
like PostgreSQL or MinIO when using in-memory configs.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from nemo_helix_plugin.client.client import NemoClient
from nhx.common.docker.gpu_detection import detect_gpu_device_ids
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network

from .base import E2EBackend

if TYPE_CHECKING:
    from .config import E2EConfig

logger = logging.getLogger(__name__)

CONTAINER_PORT = 8080
HEALTH_ENDPOINT = "/health/ready"
STARTUP_TIMEOUT_SECONDS = 60
NHX_API_NETWORK_ALIAS = "nhx-quickstart"
NHX_API_CONTAINER_NAME_PREFIX = "nhx-api-test"
CLICKHOUSE_CONTAINER_PORT = 8123
CLICKHOUSE_HEALTH_ENDPOINT = "/ping"
CLICKHOUSE_STARTUP_TIMEOUT_SECONDS = 60
NHX_CLICKHOUSE_NETWORK_ALIAS = "nhx-intake-clickhouse"
NHX_CLICKHOUSE_CONTAINER_NAME_PREFIX = "nhx-clickhouse-test"
DEFAULT_E2E_CLICKHOUSE_IMAGE = "clickhouse/clickhouse-server:26.3"

# Docker client timeout in seconds. This needs to be higher than the default 60s
# to handle Docker-in-Docker (DinD) environments in CI where the Docker daemon
# may be slower to respond.
DOCKER_CLIENT_TIMEOUT_SECONDS = 180


def _api_container_name() -> str:
    """Return a readable, collision-resistant Docker container name for NeMo E2E."""
    return f"{NHX_API_CONTAINER_NAME_PREFIX}-{uuid.uuid4().hex[:8]}"


def _clickhouse_container_name() -> str:
    """Return a readable, collision-resistant Docker container name for NeMo E2E."""
    return f"{NHX_CLICKHOUSE_CONTAINER_NAME_PREFIX}-{uuid.uuid4().hex[:8]}"


def _clickhouse_image() -> str:
    """Return the ClickHouse image used by Docker E2E."""
    return os.environ.get("NHX_E2E_CLICKHOUSE_IMAGE", DEFAULT_E2E_CLICKHOUSE_IMAGE)


def _clickhouse_api_url() -> str:
    """Return the ClickHouse URL reachable from sibling containers in the E2E network."""
    return f"http://{NHX_CLICKHOUSE_NETWORK_ALIAS}:{CLICKHOUSE_CONTAINER_PORT}"


class Docker(E2EBackend):
    """Docker-based test backend using testcontainers.

    Runs NeMo Helix API in a Docker container with configuration loaded from
    an external YAML file. The config file determines the backend types
    (in-memory, PostgreSQL, etc.).

    Args:
        config_path: Path to the NeMo Helix configuration YAML file, or an E2EConfig object.
        **kwargs: Additional arguments passed to E2EBackend (registry, tag).
    """

    def __init__(
        self,
        config_path: str | Path | E2EConfig,
        *,
        gpu_requested: bool = False,
        **kwargs,
    ):
        super().__init__(
            config_path=config_path,
            registry=kwargs.get("registry"),
            tag=kwargs.get("tag"),
            gpu_requested=gpu_requested,
        )
        self.container: DockerContainer | None = None
        self.clickhouse_container: DockerContainer | None = None
        self.network: Network | None = None
        self._host_port: int | None = None
        self._data_dir: Path | None = None

    def _is_gpu_config(self) -> bool:
        """Return True if this run requested GPU (e.g. pytest --feature gpu)."""
        return self.gpu_requested

    def start(self) -> None:
        """Start the NeMo Helix API container with the specified configuration."""
        # Clean up any lingering state from a previous failed start attempt
        self._cleanup_existing_resources()

        logger.info(f"Starting NeMo Helix API container: {self.image}")
        logger.info(f"Using config: {self.config_path}")

        try:
            # Create a dedicated network for the test
            self.network = Network()
            self.network.create()

            # Create temporary directory for /data persistence (sqlite + files)
            self._data_dir = Path(tempfile.mkdtemp(prefix="nhx-e2e-data-"))
            logger.info(f"Created data directory: {self._data_dir}")

            self._start_clickhouse_container()

            # Create and configure the container
            # Use a longer timeout for DinD environments where Docker API calls can be slower
            self.container = DockerContainer(
                self.image,
                docker_client_kw={"timeout": DOCKER_CLIENT_TIMEOUT_SECONDS},
            )
            self.container.with_kwargs(init=True)
            self.container.with_name(_api_container_name())
            self.container.with_network(self.network)
            self.container.with_network_aliases(NHX_API_NETWORK_ALIAS)
            self.container.with_exposed_ports(CONTAINER_PORT)

            # Mount config file directly from the source path.
            # Note: We use the absolute path of the original config file instead of
            # copying to a temp directory. This is critical for Docker-in-Docker (DinD)
            # environments (like GitLab CI) where the project directory (/builds/...)
            # is shared between the CI runner and DinD container, but /tmp is not.
            config_abs_path = str(self.config_path.resolve())
            logger.info(f"Mounting config from: {config_abs_path}")
            self.container.with_volume_mapping(
                config_abs_path,
                "/etc/nhx/config.yaml",
                mode="ro",
            )

            # Mount Docker socket for Docker-on-Docker (DonD) mode
            # This allows the models controller to create NIM containers on the host
            self.container.with_volume_mapping(
                "/var/run/docker.sock",
                "/var/run/docker.sock",
                mode="rw",
            )

            # Mount /data directory for sqlite and file storage persistence
            # This ensures database and files persist correctly during tests
            self.container.with_volume_mapping(
                str(self._data_dir),
                "/data",
                mode="rw",
            )

            # Set environment variables
            self.container.with_env("MODE", "development")
            self.container.with_env("NHX_CONFIG_FILE_PATH", "/etc/nhx/config.yaml")
            self.container.with_env("NHX_CONFIG_WARNINGS_DISABLED", "1")
            # Run platform seed once after API server starts (single container, no separate Job)
            self.container.with_env("NHX_SEED_ON_STARTUP", "true")
            # Allow secrets service to create encryption keys during e2e tests
            self.container.with_env("NHX_SECRETS_ALLOW_KEY_CREATION", "1")
            # Enable mock provider mode for testing
            self.container.with_env("NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX", "igw-mock-")
            # Intake runs inside the API container, so localhost-published ClickHouse
            # ports are not reachable. Use the sidecar alias on the E2E network.
            self.container.with_env("NHX_INTAKE_CLICKHOUSE_URL", _clickhouse_api_url())
            # Ensure jobs use the same Docker network as the API for container execution
            self.container.with_env("NEMO_JOBS_DEFAULT_DOCKER_NETWORK", self.network.name)
            # Models Docker backend: use DonD mode so the API (in a container) can reach NIM
            # containers by name on the same network. Without this, health_url is localhost:port
            # and the API container cannot reach the NIM (localhost is the API itself).
            self.container.with_env("NEMO_DEPLOYMENTS_DOCKER_NETWORK", self.network.name)
            self.container.with_env("NEMO_DEPLOYMENTS_DOCKER_ENDPOINT_MODE", "network")
            self.container.with_env("MODELS_DOCKER_NETWORKING_MODE", "dond")
            self.container.with_env("MODELS_DOCKER_NETWORK", self.network.name)
            self.container.with_env("MODELS_DOCKER_CONTAINER_NAME", NHX_API_NETWORK_ALIAS)
            # Pass registry/tag overrides to the container so services use the correct images
            # for job execution (e.g., nhx-tasks). Without these, the YAML config defaults
            # would be used, which may not match the actual CI registry/tag.
            if self.registry:
                self.container.with_env("NHX_IMAGE_REGISTRY", self.registry)
            if self.tag:
                self.container.with_env("NHX_IMAGE_TAG", self.tag)

            # Forward NGC_API_KEY so the models controller can pass it to NIM containers
            # (required for pulling models like meta/llama-3.2-1b-instruct from NGC). Set when
            # running tests: NGC_API_KEY=xxx uv run pytest e2e/test_models.py --docker-gpu
            ngc_api_key = os.environ.get("NGC_API_KEY")
            if ngc_api_key:
                self.container.with_env("NGC_API_KEY", ngc_api_key)
                logger.info("Passing NGC_API_KEY into API container for NIM model downloads")

            # Forward NEMO_JOBS_IMAGE_REGISTRY_* so the jobs controller can authenticate
            # with a private registry to pull job images (e.g. nhx-tasks from nvcr.io).
            # Set when running tests against a private registry:
            #   NEMO_JOBS_IMAGE_REGISTRY=nvcr.io \
            #   NEMO_JOBS_IMAGE_REGISTRY_USER_NAME='$oauthtoken' \
            #   NEMO_JOBS_IMAGE_REGISTRY_PASSWORD=<ngc-token> \
            #   uv run pytest e2e/... --docker
            jobs_registry = os.environ.get("NEMO_JOBS_IMAGE_REGISTRY")
            if jobs_registry:
                self.container.with_env("NEMO_JOBS_IMAGE_REGISTRY", jobs_registry)
                jobs_registry_user = os.environ.get("NEMO_JOBS_IMAGE_REGISTRY_USER_NAME", "")
                jobs_registry_password = os.environ.get("NEMO_JOBS_IMAGE_REGISTRY_PASSWORD", "")
                self.container.with_env("NEMO_JOBS_IMAGE_REGISTRY_USER_NAME", jobs_registry_user)
                self.container.with_env("NEMO_JOBS_IMAGE_REGISTRY_PASSWORD", jobs_registry_password)
                logger.info("Passing NEMO_JOBS_IMAGE_REGISTRY credentials into API container for job image pulls")

            # When GPU is requested (e.g. pytest --feature gpu): pass GPU device IDs into
            # the API container via NHX_DOCKER_RESERVED_GPU_DEVICE_IDS so the models
            # controller can create GPU-backed deployments. Use env var if set (e.g. by CI),
            # otherwise detect on the host (where pytest runs) via NVML.
            if self._is_gpu_config():
                gpu_ids_str = os.environ.get("NHX_DOCKER_RESERVED_GPU_DEVICE_IDS")
                if not gpu_ids_str:
                    gpu_ids = detect_gpu_device_ids()
                    if gpu_ids:
                        gpu_ids_str = ",".join(str(i) for i in gpu_ids)
                if gpu_ids_str:
                    self.container.with_env(
                        "NHX_DOCKER_RESERVED_GPU_DEVICE_IDS",
                        gpu_ids_str,
                    )
                    logger.info("Passing GPU device IDs to container: %s", gpu_ids_str)
                else:
                    logger.warning(
                        "GPU config detected but no GPUs found (set NHX_DOCKER_RESERVED_GPU_DEVICE_IDS "
                        "e.g. to '0' in CI if the runner has one GPU); GPU workloads will fail."
                    )

            # Start container
            self.container.start()

            # Verify container is still running before getting port mapping
            self._verify_container_running()

            self._host_port = int(self.container.get_exposed_port(CONTAINER_PORT))

            logger.info(f"Container started, waiting for health check on port {self._host_port}")

            # Wait for container to be ready
            self._wait_for_healthy()

            logger.info(f"NeMo Helix API container ready at http://localhost:{self._host_port}")

        except Exception:
            # Clean up any partially created resources on failure
            logger.error("Failed to start container, cleaning up resources")
            self.stop()
            raise

    def _start_clickhouse_container(self) -> None:
        """Start a ClickHouse sidecar reachable by the API container."""
        if self.network is None:
            raise RuntimeError("Docker network must be created before ClickHouse")

        self.clickhouse_container = DockerContainer(
            _clickhouse_image(),
            docker_client_kw={"timeout": DOCKER_CLIENT_TIMEOUT_SECONDS},
        )
        self.clickhouse_container.with_kwargs(init=True)
        self.clickhouse_container.with_name(_clickhouse_container_name())
        self.clickhouse_container.with_network(self.network)
        self.clickhouse_container.with_network_aliases(NHX_CLICKHOUSE_NETWORK_ALIAS)
        self.clickhouse_container.with_exposed_ports(CLICKHOUSE_CONTAINER_PORT)
        self.clickhouse_container.with_env("CLICKHOUSE_USER", "default")
        self.clickhouse_container.with_env("CLICKHOUSE_PASSWORD", "")
        self.clickhouse_container.with_env("CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT", "1")
        self.clickhouse_container.with_env("CLICKHOUSE_SKIP_USER_SETUP", "1")

        logger.info("Starting ClickHouse sidecar container: %s", _clickhouse_image())
        self.clickhouse_container.start()
        self._wait_for_clickhouse_healthy()

    def _wait_for_clickhouse_healthy(self) -> None:
        """Wait for the ClickHouse sidecar to accept HTTP requests."""
        if self.clickhouse_container is None:
            raise RuntimeError("ClickHouse container was not created")

        container_host = self.clickhouse_container.get_container_host_ip()
        host_port = int(self.clickhouse_container.get_exposed_port(CLICKHOUSE_CONTAINER_PORT))
        health_url = f"http://{container_host}:{host_port}{CLICKHOUSE_HEALTH_ENDPOINT}"

        logger.info(f"Waiting for ClickHouse health check at: {health_url}")

        start_time = time.time()
        last_error: Exception | None = None

        while (time.time() - start_time) < CLICKHOUSE_STARTUP_TIMEOUT_SECONDS:
            try:
                response = httpx.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return
                last_error = Exception(f"ClickHouse health check returned status {response.status_code}")
            except httpx.RequestError as e:
                last_error = e

            time.sleep(1.0)

        msg = f"ClickHouse did not become healthy within {CLICKHOUSE_STARTUP_TIMEOUT_SECONDS}s"
        if last_error:
            msg += f": {last_error}"
        raise RuntimeError(msg)

    def _cleanup_existing_resources(self) -> None:
        """Clean up any lingering resources from a previous failed start.

        This handles the case where start() failed partway through and left
        the backend in a partially initialized state.
        """
        if self.container is not None:
            logger.warning("Found existing container from previous attempt, cleaning up")
            try:
                self.container.stop()
            except Exception as e:
                logger.warning(f"Error stopping existing container: {e}")
            self.container = None

        if self.clickhouse_container is not None:
            logger.warning("Found existing ClickHouse container from previous attempt, cleaning up")
            try:
                self.clickhouse_container.stop()
            except Exception as e:
                logger.warning(f"Error stopping existing ClickHouse container: {e}")
            self.clickhouse_container = None

        if self.network is not None:
            logger.warning("Found existing network from previous attempt, cleaning up")
            try:
                self.network.remove()
            except Exception as e:
                logger.warning(f"Error removing existing network: {e}")
            self.network = None

        if self._data_dir is not None:
            logger.warning("Found existing data directory from previous attempt, cleaning up")
            try:
                shutil.rmtree(self._data_dir)
            except Exception as e:
                logger.warning(f"Error removing data directory: {e}")
            self._data_dir = None

        self._host_port = None

    def _verify_container_running(self) -> None:
        """Verify the container is still running after start.

        Some containers may crash immediately after starting due to configuration
        errors or resource constraints. This check catches that early with a
        better error message.
        """
        if self.container is None:
            raise RuntimeError("Container was not created")

        try:
            wrapped = self.container.get_wrapped_container()
            wrapped.reload()  # Refresh container state from Docker daemon
            status = wrapped.status
            if status != "running":
                err_msg = f"Container exited immediately after starting (status: {status})"
                try:
                    stdout_bytes, stderr_bytes = self.container.get_logs()
                    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
                    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
                    log_section = []
                    if stdout:
                        log_section.append(f"--- container stdout ---\n{stdout}")
                    if stderr:
                        log_section.append(f"--- container stderr ---\n{stderr}")
                    if log_section:
                        err_msg += "\n\n" + "\n\n".join(log_section)
                except Exception as log_error:
                    err_msg += f"\n\n(Failed to retrieve container logs: {log_error})"
                raise RuntimeError(err_msg)
        except Exception as e:
            if "Container exited" in str(e):
                raise
            logger.warning(f"Failed to verify container status: {e}")
            # Continue anyway - the health check will catch issues

    def _wait_for_healthy(self) -> None:
        """Wait for the container to be healthy."""
        # Get the container host - this handles Docker Desktop on Mac/Windows correctly
        container_host = self.container.get_container_host_ip()
        base_url = f"http://{container_host}:{self._host_port}"
        health_url = f"{base_url}{HEALTH_ENDPOINT}"

        logger.info(f"Waiting for health check at: {health_url}")

        start_time = time.time()
        last_error: Exception | None = None

        while (time.time() - start_time) < STARTUP_TIMEOUT_SECONDS:
            try:
                response = httpx.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return
                last_error = Exception(f"Health check returned status {response.status_code}")
            except httpx.RequestError as e:
                last_error = e

            time.sleep(1.0)

        # If we get here, timeout occurred
        msg = f"Container did not become healthy within {STARTUP_TIMEOUT_SECONDS}s"
        if last_error:
            msg += f": {last_error}"

        # Embed container logs in the exception so they always appear in CI output
        # (logger output is swallowed by pytest unless --log-cli-level is set)
        try:
            if self.container:
                stdout_bytes, stderr_bytes = self.container.get_logs()
                stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
                stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
                log_section = []
                if stdout:
                    log_section.append(f"--- container stdout ---\n{stdout}")
                if stderr:
                    log_section.append(f"--- container stderr ---\n{stderr}")
                if log_section:
                    msg += "\n\n" + "\n\n".join(log_section)
        except Exception as log_error:
            msg += f"\n\n(Failed to retrieve container logs: {log_error})"

        raise RuntimeError(msg)

    def _collect_logs(self, log_dir: str = "docker/logs") -> None:
        """Collect logs from all Docker containers to a directory before teardown.

        Enumerates all containers (running and stopped) and writes their logs
        and inspect output to log_dir. Safe to call in a partially degraded state —
        individual failures are ignored.
        """
        source_container = self.container or self.clickhouse_container
        if source_container is None:
            return
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            docker_client = source_container.get_docker_client().client
            containers = docker_client.containers.list(all=True)
            for c in containers:
                name = c.name.lstrip("/")
                try:
                    (log_path / f"inspect-{name}.json").write_text(str(c.attrs))
                except Exception as e:
                    logger.debug(f"Could not write inspect for {name}: {e}")
                try:
                    logs = c.logs(stdout=True, stderr=True, timestamps=True)
                    (log_path / f"logs-{name}.txt").write_bytes(logs)
                except Exception as e:
                    logger.debug(f"Could not write logs for {name}: {e}")
            logger.info(f"Docker log collection complete — files written to: {log_dir}")
        except Exception as e:
            logger.warning(f"Docker log collection failed: {e}")

    def stop(self) -> None:
        """Stop and cleanup the container.

        This method is designed to be robust and clean up resources even when
        they are in a bad state. It will attempt to forcefully remove containers
        if graceful stop fails.
        """
        self._collect_logs()
        if self.container:
            # Try to get container ID for logging (optional - don't fail if unavailable)
            container_id = None
            try:
                wrapped = self.container.get_wrapped_container()
                container_id = wrapped.short_id
            except Exception as e:
                logger.debug(f"Could not get container ID for logging: {e}")

            try:
                self.container.stop()
                if container_id:
                    logger.info(f"Stopped container: {container_id}")
            except Exception as e:
                logger.warning(f"Error stopping container gracefully: {e}")
                # Try force remove if graceful stop failed
                try:
                    wrapped = self.container.get_wrapped_container()
                    wrapped.remove(force=True)
                    logger.info(f"Force removed container: {container_id}")
                except Exception as force_error:
                    logger.warning(f"Error force removing container: {force_error}")
            self.container = None

        if self.clickhouse_container:
            container_id = None
            try:
                wrapped = self.clickhouse_container.get_wrapped_container()
                container_id = wrapped.short_id
            except Exception as e:
                logger.debug(f"Could not get ClickHouse container ID for logging: {e}")

            try:
                self.clickhouse_container.stop()
                if container_id:
                    logger.info(f"Stopped ClickHouse container: {container_id}")
            except Exception as e:
                logger.warning(f"Error stopping ClickHouse container gracefully: {e}")
                try:
                    wrapped = self.clickhouse_container.get_wrapped_container()
                    wrapped.remove(force=True)
                    logger.info(f"Force removed ClickHouse container: {container_id}")
                except Exception as force_error:
                    logger.warning(f"Error force removing ClickHouse container: {force_error}")
            self.clickhouse_container = None

        if self.network:
            network_name = self.network.name
            try:
                self.network.remove()
                logger.info(f"Removed network: {network_name}")
            except Exception as e:
                logger.warning(f"Error removing network {network_name}: {e}")
            self.network = None

        if self._data_dir:
            try:
                shutil.rmtree(self._data_dir)
                logger.info(f"Removed data directory: {self._data_dir}")
            except Exception as e:
                logger.warning(f"Error removing data directory: {e}")
            self._data_dir = None

        self._host_port = None

    def get_client(self, principal_id: str | None = None) -> NemoClient:
        """Create a typed platform client for the containerized NeMo Helix API.

        Args:
            principal_id: Optional principal ID for authentication (X-NHX-Principal-Id header).

        Returns:
            Configured typed platform client.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self._host_port is None or self.container is None:
            raise RuntimeError("Container is not running. Call start() first.")

        container_host = self.container.get_container_host_ip()
        base_url = f"http://{container_host}:{self._host_port}"
        headers = {"X-NHX-Principal-Id": principal_id} if principal_id else None
        return NemoClient(base_url=base_url, default_headers=headers)

    @property
    def network_name(self) -> str | None:
        """Get the dedicated Docker network name for this backend."""
        return self.network.name if self.network is not None else None

    @property
    def network_alias(self) -> str:
        """Get the stable network alias used by sibling containers."""
        return NHX_API_NETWORK_ALIAS

    @property
    def container_port(self) -> int:
        """Get the internal API port exposed within the Docker network."""
        return CONTAINER_PORT

    @property
    def base_url(self) -> str:
        """Get the base URL for the running container.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self._host_port is None or self.container is None:
            raise RuntimeError("Container is not running. Call start() first.")
        container_host = self.container.get_container_host_ip()
        return f"http://{container_host}:{self._host_port}"

    def get_logs(self, tail: int | None = 200) -> tuple[str, str]:
        """Get container logs for debugging.

        Args:
            tail: Number of lines to return from the end. None for all logs.

        Returns:
            Tuple of (stdout, stderr) as strings.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.container is None:
            raise RuntimeError("Container is not running. Call start() first.")

        stdout_bytes, stderr_bytes = self.container.get_logs()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if tail is not None:
            stdout_lines = stdout.splitlines()
            stderr_lines = stderr.splitlines()
            stdout = "\n".join(stdout_lines[-tail:]) if len(stdout_lines) > tail else stdout
            stderr = "\n".join(stderr_lines[-tail:]) if len(stderr_lines) > tail else stderr

        return stdout, stderr
