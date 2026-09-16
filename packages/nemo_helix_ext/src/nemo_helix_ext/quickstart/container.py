# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container lifecycle management using Docker SDK."""

from __future__ import annotations

import logging
import os
import sys
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

from ._registry import image_registry_host
from .config import QuickstartConfig
from .platform_config import PlatformConfig

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from docker.models.containers import Container
    from docker.models.networks import Network
    from docker.types import Mount

    from docker import DockerClient


class PullProgress(TypedDict):
    """Progress update from image pull operation."""

    status: str
    progress: str | None
    layer_id: str | None
    current: int | None  # bytes downloaded for this layer
    total: int | None  # total bytes for this layer


class ContainerManager:
    """Manages the nhx-api container lifecycle using Docker SDK.

    This class handles:
    - Container creation and startup
    - Image pulling with authentication
    - Volume mounting (Docker socket for DOOD, data volume)
    - Environment variable configuration
    - Container lifecycle (start, stop, destroy)
    - Status and log retrieval
    """

    def __init__(self, config: QuickstartConfig):
        """Initialize container manager.

        Args:
            config: Quickstart configuration.
        """
        self.config = config
        self._client: DockerClient | None = None

    @property
    def client(self) -> DockerClient:
        """Get or create Docker client."""
        import docker

        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def start(
        self,
        platform_config: PlatformConfig | None = None,
        pull: bool = True,
        detach: bool = True,
    ) -> Container:
        """Start the nhx-api container.

        Args:
            platform_config: Platform configuration for the container.
                Uses default if not provided.
            pull: Pull the container image before starting.
            detach: Run container in detached mode.

        Returns:
            The started Container object.

        Raises:
            docker.errors.APIError: If Docker operations fail.
        """
        if platform_config is None:
            platform_config = PlatformConfig.get_default()

        # Pull image if requested
        if pull:
            self._pull_image()

        # Prepare mounts
        mounts = self._create_mounts()

        # Prepare environment variables
        environment = self._create_environment(platform_config)

        # Remove existing container, model deployments, and network if they exist.
        # Model deployments (NIMs) must be removed before the network, otherwise
        # network.remove() fails with "active endpoints".
        self._remove_existing_container()
        self._remove_model_deployments()
        self._remove_existing_network()

        # Create Docker network for the container
        network = self.client.networks.create(name=self.config.network_name)

        # Build container run arguments.
        # Rely on the image ENTRYPOINT (`nemo services run`) so quickstart uses
        # the container's default platform startup behavior.
        container_kwargs: dict = {
            "image": self.config.image,
            "name": self.config.container_name,
            "detach": detach,
            "mounts": mounts,
            "environment": environment,
            "ports": {f"{self.config.container_port}/tcp": self.config.host_port},
            "remove": False,  # Keep container for logs inspection
            "network": network.name,
        }

        # On Linux, run as current user and add Docker socket group.
        # macOS Docker Desktop uses a VM and doesn't support host UID/GID mapping.
        if sys.platform == "linux":
            # Run as current user so files in the mounted data volume (nhx-platform.db,
            # nhx-files) are owned by the host user, not root.
            container_kwargs["user"] = f"{os.getuid()}:{os.getgid()}"
            # Add Docker socket group so container can access /var/run/docker.sock
            docker_socket_gid = self._get_docker_socket_gid()
            if docker_socket_gid is not None:
                container_kwargs["group_add"] = [docker_socket_gid]

        # Store non-recoverable config fields as Docker labels so status/info
        # can be reconstructed from the running container without the config file.
        container_kwargs["labels"] = {
            "nhx.nvidia.com/inference-provider": self.config.inference_provider or "",
            "nhx.nvidia.com/use-gpu": "true" if self.config.use_gpu else "false",
            "nhx.nvidia.com/auth-enabled": "true" if self.config.auth_enabled else "false",
            "nhx.nvidia.com/host-port": str(self.config.host_port),
        }

        # Create and start container
        container = self.client.containers.run(**container_kwargs)

        # Store container ID in config
        self.config.container_id = container.id
        self.config.save()

        return container

    def _has_floating_tag(self) -> bool:
        """Check whether the configured image uses a floating (mutable) tag.

        Floating tags like ``latest`` can be updated in the registry while the
        local copy becomes stale, so callers should always pull when one is in
        use.
        """
        FLOATING_TAGS = {"latest"}
        # image might be "registry/repo:tag" or just "repo" (implicit latest)
        if ":" not in self.config.image.rsplit("/", 1)[-1]:
            return True  # no tag at all → Docker defaults to "latest"
        tag = self.config.image.rsplit(":", 1)[-1]
        return tag in FLOATING_TAGS

    def _pull_image(self) -> None:
        """Pull the container image with authentication if needed.

        Skips pulling if the image already exists locally and does not use a
        floating tag.  Floating tags (e.g. ``latest``) are always re-pulled so
        the local copy stays current.
        """
        if not self._has_floating_tag() and self.is_image_available():
            return

        # Parse registry from image name
        registry = image_registry_host(self.config.image) or None

        auth_config = None
        if registry and registry.split(":", 1)[0] == "nvcr.io":
            if self.config.ngc_api_key:
                auth_config = {
                    "username": "$oauthtoken",
                    "password": self.config.ngc_api_key.get_secret_value(),
                }

        self.client.images.pull(self.config.image, auth_config=auth_config)

    def pull_image_with_progress(self, auth_override: dict[str, str] | None = None) -> Iterator[PullProgress]:
        """Pull the container image with progress updates.

        Yields progress updates as the image is pulled. Skips if image
        already exists locally.

        Yields:
            PullProgress dictionaries with status, progress bar, and layer ID.
        """
        if not self._has_floating_tag() and self.is_image_available():
            yield {
                "status": "Image already available",
                "progress": None,
                "layer_id": None,
                "current": None,
                "total": None,
            }
            return

        # Build auth config (same logic as _pull_image)
        registry = image_registry_host(self.config.image) or None

        auth_config = None
        if auth_override and registry == auth_override["registry"]:
            auth_config = {
                "username": auth_override["username"],
                "password": auth_override["password"],
            }
        elif registry and registry.split(":", 1)[0] == "nvcr.io":
            if self.config.ngc_api_key:
                auth_config = {
                    "username": "$oauthtoken",
                    "password": self.config.ngc_api_key.get_secret_value(),
                }

        # Use low-level API for streaming progress
        api_client = self.client.api
        for line in api_client.pull(self.config.image, stream=True, decode=True, auth_config=auth_config):
            progress_detail = line.get("progressDetail") or {}
            yield {
                "status": line.get("status", ""),
                "progress": line.get("progress"),
                "layer_id": line.get("id"),
                "current": progress_detail.get("current"),
                "total": progress_detail.get("total"),
            }

    def _create_mounts(self) -> list[Mount]:
        """Create Docker mounts for the container.

        Returns:
            List of Mount objects for Docker socket and data volume.
        """
        from docker.types import Mount

        mounts = []

        # Mount Docker socket for DOOD (Docker-outside-of-Docker) pattern
        # This allows the container to spawn sibling containers for jobs
        mounts.append(
            Mount(
                target="/var/run/docker.sock",
                source=str(self.config.docker_socket),
                type="bind",
                read_only=False,
            )
        )

        # Mount shared data volume
        data_path = self.config.data_path
        data_path.mkdir(parents=True, exist_ok=True)
        mounts.append(
            Mount(
                target="/data",
                source=str(data_path),
                type="bind",
                read_only=False,
            )
        )

        # Mount platform config if specified
        if self.config.platform_config_path and self.config.platform_config_path.exists():
            mounts.append(
                Mount(
                    target="/etc/nhx/platform-config.yaml",
                    source=str(self.config.platform_config_path),
                    type="bind",
                    read_only=True,
                )
            )

        return mounts

    def _create_environment(self, platform_config: PlatformConfig) -> dict[str, str]:
        """Create environment variables for the container.

        Args:
            platform_config: Platform configuration to convert to env vars.

        Returns:
            Dictionary of environment variables.
        """
        env = platform_config.to_env_vars()

        # Add NGC API key for image pulls within container
        if self.config.ngc_api_key:
            env["NGC_API_KEY"] = self.config.ngc_api_key.get_secret_value()

        # Docker host for DOOD - container uses host's docker socket
        env["DOCKER_HOST"] = "unix:///var/run/docker.sock"

        # XDG base dirs under /data so libraries (e.g. garakapi) can create config/cache
        # when the container runs as non-root with no HOME set (PermissionError on /.config).
        env["XDG_CONFIG_HOME"] = "/data/.config"
        env["XDG_DATA_HOME"] = "/data/.local/share"
        env["XDG_CACHE_HOME"] = "/data/.cache"

        # Database in mounted /data so it persists across container restarts
        env["DATABASE_URL"] = "sqlite:////data/nhx-platform.db"

        # Secrets: allow key creation and persist key in /data so it survives restarts.
        # Keep explicit PlatformConfig env mappings if the caller supplied them.
        env.setdefault("NHX_SECRETS_ALLOW_KEY_CREATION", "1")
        env.setdefault("NHX_SECRETS_LOCAL_KEY_CREATION_PATH", "/data/nhx-encryption-key.txt")

        # Seed the platform on startup with entities
        env["NHX_SEED_ON_STARTUP"] = "true"

        # Set the default Docker network for jobs to the quickstart network.
        # This allows job containers to communicate with the quickstart container
        # on the same Docker network using the container name as the hostname.
        env["NEMO_JOBS_DEFAULT_DOCKER_NETWORK"] = self.config.network_name

        # Configure the models/NIM Docker backend for DonD (Docker-on-Docker) setup.
        # NIMs need to join the same network as the quickstart container so they can
        # communicate via container names (e.g., http://md-workspace-name:8000).
        env["NEMO_DEPLOYMENTS_DOCKER_NETWORK"] = self.config.network_name
        env["NEMO_DEPLOYMENTS_DOCKER_ENDPOINT_MODE"] = "network"
        env["MODELS_DOCKER_NETWORKING_MODE"] = "dond"
        env["MODELS_DOCKER_NETWORK"] = self.config.network_name
        # Pass the container name so the Models service can use it for localhost replacement
        env["MODELS_DOCKER_CONTAINER_NAME"] = self.config.container_name

        # Tell the service where to find the mounted config file
        if self.config.platform_config_path and self.config.platform_config_path.exists():
            env["NHX_CONFIG_FILE_PATH"] = "/etc/nhx/platform-config.yaml"

        # Auth configuration
        if self.config.auth_enabled:
            env["NHX_AUTH_ENABLED"] = "true"
            # Quickstart uses unsigned JWTs for local/testing bootstrap auth.
            env["NHX_AUTH_ALLOW_UNSIGNED_JWT"] = "true"
            env["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] = "http://localhost:8080"
            env["NHX_AUTH_POLICY_DECISION_POINT_PROVIDER"] = "embedded"
            # Faster auth propagation for quickstart (~2s max propagation)
            env["NHX_AUTH_BUNDLE_CACHE_SECONDS"] = "1"
            env["NHX_AUTH_POLICY_DATA_REFRESH_INTERVAL"] = "2"

            if self.config.admin_email:
                env["NHX_AUTH_ADMIN_EMAIL"] = self.config.admin_email

        # Pass parsed image components to configure PlatformConfig defaults inside container
        registry, tag = self.config.parse_image_components()
        if registry:
            env["NHX_IMAGE_REGISTRY"] = registry
        if tag:
            env["NHX_IMAGE_TAG"] = tag

        # Pass host GPU device IDs to the container (set during nemo quickstart configure).
        # The value is stored as a comma-separated string in config; no detection at up time.
        if self.config.use_gpu and self.config.reserved_gpu_device_ids is not None:
            env["NHX_DOCKER_RESERVED_GPU_DEVICE_IDS"] = self.config.reserved_gpu_device_ids.strip()

        # Pass registry credentials for the jobs backend to authenticate when pulling images
        # Only pass credentials if the registry requires authentication
        from .prompts import detect_registry_auth_type

        auth_type = detect_registry_auth_type(self.config.image)
        jobs_registry_host = image_registry_host(self.config.image) or None
        if auth_type == "ngc" and jobs_registry_host and self.config.ngc_api_key:
            # NGC uses $oauthtoken as username and NGC API key as password
            env["NEMO_JOBS_IMAGE_REGISTRY"] = jobs_registry_host
            env["NEMO_JOBS_IMAGE_REGISTRY_USER_NAME"] = "$oauthtoken"
            env["NEMO_JOBS_IMAGE_REGISTRY_PASSWORD"] = self.config.ngc_api_key.get_secret_value()
        elif auth_type == "user_pass" and jobs_registry_host and self.config.has_registry_credentials_for_image():
            registry_username = self.config.registry_username
            registry_password = self.config.registry_password
            if registry_username and registry_password:
                env["NEMO_JOBS_IMAGE_REGISTRY"] = jobs_registry_host
                env["NEMO_JOBS_IMAGE_REGISTRY_USER_NAME"] = registry_username
                env["NEMO_JOBS_IMAGE_REGISTRY_PASSWORD"] = registry_password.get_secret_value()
        return env

    def _get_docker_socket_gid(self) -> int | None:
        """Get the GID of the Docker socket file.

        When the container runs as a non-root user, it needs the Docker socket's
        group added to its supplementary groups to access /var/run/docker.sock.

        Returns:
            The GID of the Docker socket, or None if it cannot be determined.
        """
        try:
            stat_info = os.stat(self.config.docker_socket)
            return stat_info.st_gid
        except (FileNotFoundError, OSError):
            return None

    @staticmethod
    def _detect_host_gpu_device_ids() -> list[int] | None:
        """Detect GPU device IDs on the host.

        Runs on the host where GPU detection is available, so the container
        does not need GPU libraries itself.

        Returns:
            List of GPU device IDs (e.g., [0, 1, 2]), or None if detection fails.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                return list(range(gpu_count)) if gpu_count > 0 else None
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            return None

    @staticmethod
    def reconstruct_config_from_container(container: Container) -> QuickstartConfig:
        """Reconstruct QuickstartConfig from a running container's inspect data.

        Args:
            container: The running Docker container.

        Returns:
            QuickstartConfig populated from container inspect data.
        """
        attrs = container.attrs

        # Image
        if container.image.tags:
            image = container.image.tags[0]
        else:
            image = attrs.get("Config", {}).get("Image", "unknown")

        # Container name (strip leading slash that Docker adds)
        container_name = container.name.lstrip("/")

        # Network name
        networks = attrs.get("NetworkSettings", {}).get("Networks", {})
        network_name = next(iter(networks), "nhx-quickstart-network")

        # Port bindings: start with PortBindings, then label can override
        port_bindings = attrs.get("HostConfig", {}).get("PortBindings", {}) or {}
        host_port = 8080
        container_port = 8080
        for container_port_key, host_bindings in port_bindings.items():
            port_num = int(container_port_key.split("/")[0])
            container_port = port_num
            if host_bindings:
                host_port = int(host_bindings[0]["HostPort"])
            break

        # Labels
        labels = attrs.get("Config", {}).get("Labels", {}) or {}

        # Label override for host_port (handles NAT / port-remapping edge cases)
        label_host_port = labels.get("nhx.nvidia.com/host-port")
        if label_host_port:
            host_port = int(label_host_port)

        # Mounts
        mounts = attrs.get("Mounts", []) or []
        docker_socket: Path = Path("/var/run/docker.sock")
        storage_path: Path = Path.home() / ".config" / "nhx" / "quickstart"
        platform_config_path: Path | None = None
        for mount in mounts:
            dest = mount.get("Destination", "")
            source = mount.get("Source", "")
            if dest == "/var/run/docker.sock" and source:
                docker_socket = Path(source)
            elif dest == "/data" and source:
                # data_path = storage_path / "data", so storage_path is its parent
                storage_path = Path(source).parent
            elif dest == "/etc/nhx/platform-config.yaml" and source:
                platform_config_path = Path(source)

        config_kwargs: dict = {
            "image": image,
            "container_name": container_name,
            "network_name": network_name,
            "host_port": host_port,
            "container_port": container_port,
            "docker_socket": docker_socket,
            "storage_path": storage_path,
        }

        if platform_config_path is not None:
            config_kwargs["platform_config_path"] = platform_config_path

        return QuickstartConfig(**config_kwargs)

    def get_effective_config(self) -> QuickstartConfig:
        """Return QuickstartConfig from the running container, or self.config as fallback.

        Uses Docker inspect data when the container is running so that status
        and cluster_info reflect reality even if the config file has drifted.
        """
        from docker.errors import DockerException

        try:
            container = self._get_container()
        except DockerException:
            return self.config
        if container is None:
            return self.config
        try:
            container.reload()
            return self.reconstruct_config_from_container(container)
        except Exception:
            return self.config

    def _remove_existing_container(self) -> None:
        """Remove existing container if it exists."""
        from docker.errors import NotFound

        try:
            container = self.client.containers.get(self.config.container_name)
            container.stop(timeout=10)
            container.remove()
        except NotFound:
            pass

    def _remove_existing_network(self) -> None:
        """Remove existing Docker network, disconnecting any attached containers first.

        Handles the case where containers (e.g. NIMs, jobs) are still attached
        to the network after the quickstart container has been removed.  Without
        disconnecting them first, ``network.remove()`` fails with a 403
        "active endpoints" error from the Docker daemon.
        """
        from docker.errors import APIError, NotFound

        try:
            network = self.client.networks.get(self.config.network_name)
        except NotFound:
            return

        try:
            network.reload()
        except Exception as exc:
            logger.warning("Failed to reload network %s: %s", self.config.network_name, exc)

        try:
            containers = getattr(network, "containers", None) or []
            iter(containers)
        except TypeError:
            containers = []

        for container in containers:
            try:
                network.disconnect(container, force=True)
            except Exception as exc:
                logger.warning(
                    "Failed to disconnect container %s from network %s: %s",
                    getattr(container, "name", None) or getattr(container, "short_id", "unknown"),
                    self.config.network_name,
                    exc,
                )

        try:
            network.remove()
        except APIError as exc:
            if "active endpoints" in str(exc).lower():
                raise RuntimeError(
                    f"Cannot remove network '{self.config.network_name}' because containers "
                    f"are still connected to it. To recover:\n"
                    f"  1. docker network inspect {self.config.network_name}\n"
                    f"  2. docker stop <container-id> && docker rm <container-id>  "
                    f"(for each listed container)\n"
                    f"  3. docker network rm {self.config.network_name}\n"
                    f"Then retry your command."
                ) from exc
            logger.error("Failed to remove network %s: %s", self.config.network_name, exc)
            raise

    def _remove_model_deployments(self, timeout: int = 30) -> None:
        """Stop and remove all model deployment containers (managed-by models-controller).

        Call this after stopping the nhx-api container so the models controller
        does not see deployments in 'stopping' state and mark them as ERROR
        in the API.
        """
        try:
            containers = self.client.containers.list(
                all=True, filters={"label": ["nhx.nvidia.com/managed-by=models-controller"]}
            )
        except Exception:
            containers = []
        for container in containers:
            container.stop(timeout=timeout)
            container.remove()

    def stop(self, remove: bool = False, timeout: int = 30) -> None:
        """Stop the nhx-api container, then model deployments, then optionally remove the network.

        Stops the main quickstart container first so the models controller does
        not see model deployment containers in 'stopping' state and mark them
        as ERROR. Then stops and removes all remaining containers with the
        managed-by=models-controller (model deployments). When remove=True, also removes the main
        container and the quickstart network and clears the stored container ID.

        Args:
            remove: If True, remove the main container (if present), remove the
                quickstart network, and clear the stored container ID from config.
            timeout: Seconds to wait for each container to stop gracefully.
        """
        container = self._get_container()
        if container:
            container.stop(timeout=timeout)
            if remove:
                container.remove()
                self.config.container_id = None
                self.config.save()
        self._remove_model_deployments(timeout)
        if remove:
            self._remove_existing_network()

    def destroy(self) -> None:
        """Stop container and remove all data.

        This performs a complete cleanup including:
        - Stopping and removing the container
        - Removing the storage directory
        """
        self.stop(remove=True)

        # Remove data directory
        import shutil

        if self.config.storage_path.exists():
            shutil.rmtree(self.config.storage_path)

    def status(self) -> dict:
        """Get container status information.

        Returns:
            Dictionary with status information including:
            - running: bool
            - status: str (container state)
            - id: str (short container ID)
            - name: str
            - image: str
            - ports: dict
            - health: str (health check status)
        """
        from docker.errors import DockerException

        try:
            container = self._get_container()
        except DockerException:
            return {"running": False, "status": "docker-unavailable"}
        if not container:
            return {"running": False, "status": "not found"}
        network = self._get_network()
        if not network:
            return {"running": False, "status": "network not found"}

        container.reload()
        return {
            "running": container.status == "running",
            "status": container.status,
            "id": container.short_id,
            "name": container.name,
            "network": network.name,
            "image": container.image.tags[0] if container.image.tags else "unknown",
            "ports": container.ports,
            "health": self._get_health_status(container),
        }

    def _get_health_status(self, container: Container) -> str:
        """Get container health check status.

        Args:
            container: Container to check.

        Returns:
            Health status string.
        """
        health = container.attrs.get("State", {}).get("Health", {})
        return health.get("Status", "unknown")

    def logs(self, follow: bool = False, tail: int | None = 100) -> Iterator[str]:
        """Stream container logs.

        Args:
            follow: Keep following log output.
            tail: Number of lines to show from the end, or None for all logs.

        Yields:
            Log lines as strings.
        """
        container = self._get_container()
        if not container:
            return

        # Docker API accepts "all" string or an integer for tail
        docker_tail: str | int = "all" if tail is None else tail

        if follow:
            for line in container.logs(stream=True, follow=True, tail=docker_tail):
                yield line.decode("utf-8")
        else:
            yield container.logs(tail=docker_tail).decode("utf-8")

    def _get_container(self) -> Container | None:
        """Get the container by ID or name.

        Returns:
            Container object or None if not found.
        """
        from docker.errors import NotFound

        try:
            if self.config.container_id:
                return self.client.containers.get(self.config.container_id)
            return self.client.containers.get(self.config.container_name)
        except NotFound:
            return None

    def _get_network(self) -> Network | None:
        """Get the Docker network name for the quickstart container.

        Returns:
            The Docker network name.
        """
        from docker.errors import NotFound

        try:
            return self.client.networks.get(self.config.network_name)
        except NotFound:
            return None

    def is_image_available(self) -> bool:
        """Check if the container image is available locally.

        Returns:
            True if the image exists locally, False otherwise.
        """
        from docker.errors import ImageNotFound

        try:
            self.client.images.get(self.config.image)
            return True
        except ImageNotFound:
            return False
