# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker testing utilities for NeMo Helix integration tests.

This module provides utilities for Docker-based integration testing, including:
- DockerTestContext for tracking and cleaning up test containers/volumes
- Helper functions for creating Docker clients and mock NIM images
- Port range allocation for parallel test execution with pytest-xdist

Example usage in conftest.py:

    from nhx.testing.docker import (
        DockerTestContext,
        create_docker_client,
        build_mock_nim_image,
        get_worker_port_range,
        DEFAULT_MOCK_NIM_BASE_IMAGE,
        MOCK_NIM_NGINX_CONF,
    )

    @pytest.fixture(scope="module")
    def docker_client():
        client = create_docker_client()
        yield client

    @pytest.fixture(scope="module")
    def mock_nim_image(docker_client):
        image_name = "mock-nim:local"
        build_mock_nim_image(docker_client, image_name)
        yield image_name
"""

from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass, field

import pytest
from docker.errors import APIError, BuildError, DockerException, ImageNotFound, NotFound
from tenacity import retry, stop_after_attempt, wait_fixed

import docker

# =============================================================================
# Retry Configuration
# =============================================================================


@dataclass(frozen=True)
class DockerRetryConfig:
    """Configuration for Docker operation retries in DinD environments.

    These settings help handle transient failures when running tests against
    a remote Docker daemon (e.g., in CI with Docker-in-Docker).

    Example:
        # Use defaults
        ctx = DockerTestContext(docker_client=client)

        # Override for slower environments
        config = DockerRetryConfig(stop_timeout=30, stop_retries=5)
        ctx = DockerTestContext(docker_client=client, retry_config=config)
    """

    stop_timeout: int = 10  # seconds to wait for graceful stop
    stop_retries: int = 3  # number of stop attempts
    stop_retry_delay: int = 2  # seconds between stop retries
    remove_retries: int = 3  # number of remove attempts
    remove_retry_delay: int = 1  # seconds between remove retries


# Default retry configuration - can be used directly or as a reference
DEFAULT_RETRY_CONFIG = DockerRetryConfig()


# =============================================================================
# Mock NIM Configuration
# =============================================================================

MOCK_NIM_NGINX_CONF = """
events {}
http {
    server {
        listen 8000;
        default_type application/json;

        location = /v1/health/ready {
            return 200 '{"status": "ready"}';
        }
        location = /v1/health/live {
            return 200 '{"status": "live"}';
        }
        location = /v1/models {
            return 200 '{"object": "list", "data": [{"id": "mock-model", "object": "model", "created": 1234567890, "owned_by": "mock-nim"}]}';
        }
        location / {
            return 200 '{"message": "Mock NIM received request"}';
        }
    }
}
"""

# Default base image for mock NIM. Can be overridden via MOCK_NIM_BASE_IMAGE env var.
# In CI, set MOCK_NIM_BASE_IMAGE to use an internal Docker Hub cache to avoid rate limits.
DEFAULT_MOCK_NIM_BASE_IMAGE = "nginx:1.29.4-alpine-slim"

# Default base image for mock sidecar. Can be overridden via MOCK_SIDECAR_BASE_IMAGE env var.
DEFAULT_MOCK_SIDECAR_BASE_IMAGE = "alpine:3.23"

# Tag version for mock NIM and sidecar images. Bump this when you change mock behavior
# (e.g. nginx config, sidecar entrypoint) so CI and local builds rebuild instead of
# reusing a cached image with the same name.
MOCK_NIM_IMAGE_TAG = "0.0.1"
MOCK_SIDECAR_IMAGE_TAG = "0.0.1"


# =============================================================================
# Docker Client Helpers
# =============================================================================


def create_docker_client(fail_message: str | None = None) -> docker.DockerClient:
    """Create and validate a Docker client.

    Creates a Docker client from environment variables and verifies the daemon
    is accessible. Raises pytest.fail with a helpful message if Docker is not
    available.

    Args:
        fail_message: Optional custom failure message prefix.

    Returns:
        A validated Docker client.

    Raises:
        pytest.fail.Exception: If Docker client cannot be created or daemon is not responding.
    """
    try:
        client = docker.from_env()
    except DockerException as e:
        msg = fail_message or "Docker client initialization failed"
        raise pytest.fail.Exception(
            f"{msg}: {e}\n\n"
            "Please ensure Docker is installed and the Docker daemon is running:\n"
            "  - macOS/Windows: Start Docker Desktop\n"
            "  - Linux: Run 'sudo systemctl start docker' or 'sudo service docker start'\n"
            "  - Verify with: 'docker info'"
        ) from e

    # Verify the daemon is actually responding
    try:
        client.ping()
    except DockerException as e:
        raise pytest.fail.Exception(
            f"Docker daemon is not responding: {e}\n\n"
            "The Docker client was created but cannot communicate with the daemon.\n"
            "Please ensure the Docker daemon is running."
        ) from e

    return client


def build_mock_nim_image(
    docker_client: docker.DockerClient,
    image_name: str,
    nginx_conf: str = MOCK_NIM_NGINX_CONF,
    base_image: str | None = None,
) -> None:
    """Build an nginx-based mock NIM image for testing.

    The mock NIM provides standard NIM endpoints:
    - GET /v1/health/ready -> {"status": "ready"}
    - GET /v1/health/live -> {"status": "live"}
    - GET /v1/models -> mock model list
    - Catch-all handler for other endpoints

    Args:
        docker_client: Docker client to use for building.
        image_name: Name and tag for the built image (e.g., "mock-nim:local").
        nginx_conf: Nginx configuration content. Defaults to MOCK_NIM_NGINX_CONF.
        base_image: Base image to use. Defaults to DEFAULT_MOCK_NIM_BASE_IMAGE,
            but can be overridden via MOCK_NIM_BASE_IMAGE environment variable.
    """
    if base_image is None:
        base_image = os.environ.get("MOCK_NIM_BASE_IMAGE", DEFAULT_MOCK_NIM_BASE_IMAGE)
    dockerfile_content = f"FROM {base_image}\nCOPY nginx.conf /etc/nginx/nginx.conf\n"

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        dockerfile_bytes = dockerfile_content.encode("utf-8")
        dockerfile_info = tarfile.TarInfo(name="Dockerfile")
        dockerfile_info.size = len(dockerfile_bytes)
        tar.addfile(dockerfile_info, io.BytesIO(dockerfile_bytes))

        nginx_bytes = nginx_conf.encode("utf-8")
        nginx_info = tarfile.TarInfo(name="nginx.conf")
        nginx_info.size = len(nginx_bytes)
        tar.addfile(nginx_info, io.BytesIO(nginx_bytes))

    tar_buffer.seek(0)
    docker_client.images.build(fileobj=tar_buffer, custom_context=True, tag=image_name)


def ensure_mock_nim_image(docker_client: docker.DockerClient, image_name: str) -> str:
    """Ensure a mock NIM image exists, building it if necessary.

    This is idempotent - if the image already exists, it won't be rebuilt.
    Handles race conditions when multiple test workers try to build simultaneously.

    Args:
        docker_client: Docker client to use.
        image_name: Name and tag for the image.

    Returns:
        The image name.
    """
    try:
        docker_client.images.get(image_name)
    except ImageNotFound:
        try:
            build_mock_nim_image(docker_client, image_name)
        except BuildError:
            # Race condition - another worker may have built it
            try:
                docker_client.images.get(image_name)
            except ImageNotFound:
                raise

    return image_name


def build_mock_sidecar_image(
    docker_client: docker.DockerClient,
    image_name: str,
    base_image: str | None = None,
) -> None:
    """Build a minimal mock sidecar image for Docker backend integration tests.

    The models Docker backend creates a sidecar container (adapters controller) per
    deployment. Real nhx-core runs LoRA/PEFT sync; tests only need a container that
    starts and stays up. This builds an image that stays up until SIGTERM. The entrypoint traps
    SIGTERM and exits so container.stop(timeout=30) returns quickly in tests;
    otherwise the backend would wait the full 30s for the default stop timeout.

    Same convention as build_mock_nim_image: in-memory Dockerfile in a tar, no
    dedicated Dockerfile on disk.

    Args:
        docker_client: Docker client to use for building.
        image_name: Name and tag for the built image (e.g. "mock-sidecar-models:local").
        base_image: Base image. Defaults to DEFAULT_MOCK_SIDECAR_BASE_IMAGE, or
            MOCK_SIDECAR_BASE_IMAGE env var.
    """
    if base_image is None:
        base_image = os.environ.get("MOCK_SIDECAR_BASE_IMAGE", DEFAULT_MOCK_SIDECAR_BASE_IMAGE)
    # Trap SIGTERM and exit so container.stop(timeout=30) returns in ~1s during teardown
    dockerfile_content = (
        f'FROM {base_image}\nENTRYPOINT ["sh", "-c", "trap \\"exit 0\\" TERM; sleep infinity & wait"]\n'
    )

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        dockerfile_bytes = dockerfile_content.encode("utf-8")
        dockerfile_info = tarfile.TarInfo(name="Dockerfile")
        dockerfile_info.size = len(dockerfile_bytes)
        tar.addfile(dockerfile_info, io.BytesIO(dockerfile_bytes))

    tar_buffer.seek(0)
    docker_client.images.build(fileobj=tar_buffer, custom_context=True, tag=image_name)


def ensure_mock_sidecar_image(docker_client: docker.DockerClient, image_name: str) -> str:
    """Ensure a mock sidecar image exists, building it if necessary.

    Idempotent; same race-handling pattern as ensure_mock_nim_image.

    Args:
        docker_client: Docker client to use.
        image_name: Name and tag for the image (e.g. "mock-sidecar-models:local").

    Returns:
        The image name.
    """
    try:
        docker_client.images.get(image_name)
    except ImageNotFound:
        try:
            build_mock_sidecar_image(docker_client, image_name)
        except BuildError:
            try:
                docker_client.images.get(image_name)
            except ImageNotFound:
                raise

    return image_name


# Label used by the models Docker backend for NIM and sidecar containers.
# Used by cleanup_model_deployment_containers() so integration tests can tear down
# any leftover containers (e.g. after a failed test) without per-test try/finally.
MODELS_CONTROLLER_MANAGED_LABEL = "nhx.nvidia.com/managed-by=models-controller"


def _matches_labels(container_labels: dict[str, str], labels: dict[str, str]) -> bool:
    """Return True when all expected Docker labels are present."""
    return all(container_labels.get(key) == value for key, value in labels.items())


def cleanup_model_deployment_containers(
    docker_client: docker.DockerClient,
    labels: dict[str, str] | None = None,
) -> int:
    """Stop and remove models-controller containers matching optional owner labels.

    Finds containers with label nhx.nvidia.com/managed-by=models-controller
    (NIM, sidecar, and puller containers created by the Docker backend), stops
    and removes them. When labels are supplied, only containers matching all
    owner labels are removed. Intended for integration test teardown so failed
    tests don't leave stuck containers; use as a pytest fixture teardown.

    Uses the same retry logic as DockerTestContext for DinD compatibility.

    Args:
        docker_client: Docker client to use.
        labels: Optional labels used to scope cleanup to this test owner.

    Returns:
        Number of containers removed.
    """
    labels = labels or {}
    try:
        containers = docker_client.containers.list(
            all=True,
            filters={"label": MODELS_CONTROLLER_MANAGED_LABEL},
            ignore_removed=True,
        )
    except Exception:
        return 0

    cfg = DEFAULT_RETRY_CONFIG
    removed = 0

    for container in containers:
        try:
            if labels and not _matches_labels(container.labels or {}, labels):
                continue

            name = container.name

            @retry(
                stop=stop_after_attempt(cfg.stop_retries),
                wait=wait_fixed(cfg.stop_retry_delay),
                reraise=True,
            )
            def stop_container():
                container.stop(timeout=cfg.stop_timeout)

            try:
                stop_container()
            except Exception as e:
                print(f"Warning: Could not stop {name}: {e}")

            @retry(
                stop=stop_after_attempt(cfg.remove_retries),
                wait=wait_fixed(cfg.remove_retry_delay),
                reraise=True,
            )
            def remove_container():
                try:
                    container.remove(force=True)
                except APIError as e:
                    if "removal" in str(e).lower() or "already" in str(e).lower():
                        return
                    raise

            try:
                remove_container()
                removed += 1
            except Exception as e:
                print(f"Warning: Could not remove {name}: {e}")
        except NotFound:
            pass
        except Exception as e:
            print(f"Warning: Cleanup error for {container.name}: {e}")

    return removed


# =============================================================================
# Port Range Allocation
# =============================================================================


def get_worker_port_range(worker_id: str, ports_per_worker: int = 100) -> tuple[int, int]:
    """Calculate unique port range for a pytest-xdist worker.

    When running tests in parallel with pytest-xdist, each worker needs a unique
    port range to avoid conflicts. This function calculates non-overlapping
    ranges based on the worker ID.

    Args:
        worker_id: The xdist worker ID ("master", "gw0", "gw1", etc.)
        ports_per_worker: Number of ports to allocate per worker.

    Returns:
        Tuple of (start_port, end_port) inclusive.

    Environment Variables:
        MODELS_DOCKER_PORT_RANGE_START: Override the base port (default: 49152).
            Use this for DinD testing where you need to match exposed ports.

    Example:
        >>> get_worker_port_range("master")
        (49152, 49251)
        >>> get_worker_port_range("gw0")
        (49152, 49251)
        >>> get_worker_port_range("gw1")
        (49252, 49351)
    """
    # Use IANA ephemeral port range (49152-65535) to avoid conflicts with system services.
    # Can be overridden via environment variable for DinD testing.
    base_port = int(os.environ.get("MODELS_DOCKER_PORT_RANGE_START", "49152"))

    if worker_id == "master":
        worker_num = 0
    else:
        # Extract number from "gw0", "gw1", etc.
        worker_num = int(worker_id.replace("gw", ""))

    start_port = base_port + (worker_num * ports_per_worker)
    end_port = start_port + ports_per_worker - 1
    return start_port, end_port


# =============================================================================
# Docker Test Context
# =============================================================================


@dataclass
class DockerTestContext:
    """Context for Docker integration tests with cleanup support.

    Tracks containers and volumes created during tests and provides
    automatic cleanup with retry logic for DinD environments.

    Args:
        docker_client: Docker client instance.
        retry_config: Optional retry configuration. Defaults to DEFAULT_RETRY_CONFIG.
        containers: List of container names to track (usually empty at creation).
        volumes: List of volume names to track (usually empty at creation).

    Example usage:

        @pytest.fixture
        def docker_test_context(docker_client, request):
            ctx = DockerTestContext(docker_client=docker_client)
            yield ctx
            if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
                ctx.print_diagnostics()
            ctx.cleanup()

        # With custom retry config for slower environments:
        @pytest.fixture
        def docker_test_context(docker_client, request):
            config = DockerRetryConfig(stop_timeout=30, stop_retries=5)
            ctx = DockerTestContext(docker_client=docker_client, retry_config=config)
            yield ctx
            ctx.cleanup()

    In tests:

        def test_something(docker_test_context):
            # Register resources for cleanup
            docker_test_context.register_container("my-container")
            docker_test_context.register_volume("my-volume")
            # ... test code ...
            # Resources are automatically cleaned up after test
    """

    docker_client: docker.DockerClient
    retry_config: DockerRetryConfig = field(default_factory=lambda: DEFAULT_RETRY_CONFIG)
    containers: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)

    def register_container(self, name: str) -> None:
        """Register a container for cleanup."""
        self.containers.append(name)

    def register_volume(self, name: str) -> None:
        """Register a volume for cleanup."""
        self.volumes.append(name)

    def print_diagnostics(self) -> None:
        """Print diagnostic information for debugging test failures.

        Prints container logs and lists all containers managed by models-controller.
        Call this when a test fails to help debug the issue.
        """
        print("\n=== Docker Diagnostics ===")

        # Print container logs
        for container_name in self.containers:
            try:
                container = self.docker_client.containers.get(container_name)
                print(f"\n--- Logs for {container_name} (last 100 lines) ---")
                logs = container.logs(tail=100).decode("utf-8", errors="ignore")
                print(logs)
            except Exception as e:
                print(f"Could not get logs for {container_name}: {e}")

        # Print relevant containers
        print("\n--- Docker PS (test containers) ---")
        try:
            for c in self.docker_client.containers.list(all=True, ignore_removed=True):
                labels = c.labels
                if labels.get("nhx.nvidia.com/managed-by") == "models-controller":
                    owner = {
                        key: value
                        for key, value in labels.items()
                        if key in {"nhx.nvidia.com/test-run", "nhx.nvidia.com/test-worker"}
                    }
                    print(f"  {c.name}: {c.status} labels={owner}")
        except Exception as e:
            print(f"Could not list containers: {e}")

    def cleanup(self) -> None:
        """Clean up all registered resources with retry logic for DinD.

        Uses tenacity retry logic to handle transient failures that can occur in
        Docker-in-Docker environments under load.
        """
        cfg = self.retry_config

        # Stop and remove containers with retry logic
        for container_name in self.containers:
            try:
                container = self.docker_client.containers.get(container_name)

                # Stop container with tenacity retry
                @retry(
                    stop=stop_after_attempt(cfg.stop_retries),
                    wait=wait_fixed(cfg.stop_retry_delay),
                    reraise=True,
                )
                def stop_container():
                    container.stop(timeout=cfg.stop_timeout)

                try:
                    stop_container()
                except Exception as e:
                    print(f"Warning: Could not stop {container_name} after {cfg.stop_retries} attempts: {e}")

                # Remove container with tenacity retry
                @retry(
                    stop=stop_after_attempt(cfg.remove_retries),
                    wait=wait_fixed(cfg.remove_retry_delay),
                    reraise=True,
                )
                def remove_container():
                    try:
                        container.remove(force=True)
                    except APIError as e:
                        if "removal" in str(e).lower() or "already" in str(e).lower():
                            return  # Container being removed, success
                        raise

                try:
                    remove_container()
                except Exception as e:
                    print(f"Warning: Could not remove {container_name} after {cfg.remove_retries} attempts: {e}")

            except NotFound:
                pass  # Already removed
            except Exception as e:
                print(f"Warning: Cleanup error for {container_name}: {e}")

        # Remove volumes with retry logic
        for volume_name in self.volumes:

            @retry(
                stop=stop_after_attempt(cfg.remove_retries),
                wait=wait_fixed(cfg.remove_retry_delay),
                reraise=True,
            )
            def remove_volume():
                try:
                    volume = self.docker_client.volumes.get(volume_name)
                    volume.remove(force=True)
                except NotFound:
                    pass  # Already removed
                except APIError as e:
                    if "in use" in str(e).lower():
                        raise  # Retry
                    print(f"Warning: Could not remove volume {volume_name}: {e}")

            try:
                remove_volume()
            except Exception as e:
                print(f"Warning: Volume {volume_name} still in use after {cfg.remove_retries} attempts: {e}")
