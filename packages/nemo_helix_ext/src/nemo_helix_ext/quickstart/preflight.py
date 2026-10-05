# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight checks for quickstart cluster."""

from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass
from enum import Enum

from nemo_helix_plugin.capabilities import probe_docker

from .config import QuickstartConfig


def detect_gpus() -> tuple[bool, str]:
    """Check if NVIDIA GPUs are available on the system.

    Returns:
        Tuple of (gpus_detected, message).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_list = result.stdout.strip()
        if gpu_list and "No devices found" not in gpu_list:
            # Count GPUs
            gpu_count = len(gpu_list.splitlines())
            return True, f"Detected {gpu_count} GPU(s)"
        return False, "No NVIDIA GPUs detected"
    except FileNotFoundError:
        return False, "nvidia-smi not found (NVIDIA drivers not installed)"
    except subprocess.CalledProcessError:
        return False, "nvidia-smi failed (GPU may not be available)"


class CheckStatus(Enum):
    """Status of a pre-flight check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class PreflightResult:
    """Result of a single pre-flight check."""

    name: str
    status: CheckStatus
    message: str
    details: str | None = None


class PreflightChecker:
    """Runs pre-flight checks before starting the quickstart cluster.

    Pre-flight checks verify that the system is ready to run the quickstart
    container, including Docker availability, socket permissions, storage
    directories, port availability, and credentials.
    """

    def __init__(self, config: QuickstartConfig):
        """Initialize pre-flight checker.

        Args:
            config: Quickstart configuration to validate.
        """
        self.config = config
        self.results: list[PreflightResult] = []
        self._nhx_already_running = False
        self._image_pull_required: bool | None = None

    def run_all(self) -> list[PreflightResult]:
        """Run all pre-flight checks and return results.

        Returns:
            List of PreflightResult objects for each check.
        """
        self.results = []
        self._nhx_already_running = False
        self._image_pull_required = None
        self._check_docker_available()
        self._check_docker_socket_permissions()
        self._check_storage_directory()
        self._check_port_available()
        self._check_image_pullable()
        self._check_credentials()
        self._check_gpu_available()
        return self.results

    def _check_docker_available(self) -> None:
        """Verify Docker daemon is running and accessible."""
        # Preflight can be re-run after the user starts Docker.
        result = probe_docker(use_cache=False)
        if result.available:
            self.results.append(
                PreflightResult(
                    name="Docker Available",
                    status=CheckStatus.PASS,
                    message="Docker daemon is running",
                )
            )
        else:
            self.results.append(
                PreflightResult(
                    name="Docker Available",
                    status=CheckStatus.FAIL,
                    message="Docker daemon is not accessible",
                    details=result.detail or "Docker daemon unreachable",
                )
            )

    def _check_docker_socket_permissions(self) -> None:
        """Verify host docker socket exists and has correct permissions."""
        socket_path = self.config.docker_socket

        if not socket_path.exists():
            self.results.append(
                PreflightResult(
                    name="Docker Socket",
                    status=CheckStatus.FAIL,
                    message=f"Docker socket not found at {socket_path}",
                )
            )
            return

        # Check if socket is readable/writable
        if not os.access(socket_path, os.R_OK | os.W_OK):
            self.results.append(
                PreflightResult(
                    name="Docker Socket",
                    status=CheckStatus.WARN,
                    message="Docker socket may not have correct permissions",
                    details="Container may need to run with elevated privileges",
                )
            )
            return

        self.results.append(
            PreflightResult(
                name="Docker Socket",
                status=CheckStatus.PASS,
                message="Docker socket is accessible",
            )
        )

    def _check_storage_directory(self) -> None:
        """Verify storage directory exists or can be created."""
        storage_path = self.config.storage_path
        try:
            storage_path.mkdir(parents=True, exist_ok=True)
            (storage_path / "data").mkdir(exist_ok=True)
            (storage_path / "state").mkdir(exist_ok=True)

            # Verify we can write to the directory
            test_file = storage_path / ".write_test"
            test_file.touch()
            test_file.unlink()

            self.results.append(
                PreflightResult(
                    name="Storage Directory",
                    status=CheckStatus.PASS,
                    message=f"Storage directory ready at {storage_path}",
                )
            )
        except PermissionError:
            self.results.append(
                PreflightResult(
                    name="Storage Directory",
                    status=CheckStatus.FAIL,
                    message=f"Cannot create storage directory at {storage_path}",
                    details="Check directory permissions",
                )
            )
        except Exception as e:
            self.results.append(
                PreflightResult(
                    name="Storage Directory",
                    status=CheckStatus.FAIL,
                    message=f"Storage directory error: {e}",
                )
            )

    def _check_port_available(self) -> None:
        """Check if the configured host port is available or NeMo Helix is already running."""
        from .validators import is_quickstart_running

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", self.config.host_port))  # noqa: S104  # nosec B104
            self.results.append(
                PreflightResult(
                    name="Port Available",
                    status=CheckStatus.PASS,
                    message=f"Port {self.config.host_port} is available",
                )
            )
        except OSError:
            # Port is in use - check if NeMo Helix quickstart is already running
            if is_quickstart_running(self.config):
                self.results.append(
                    PreflightResult(
                        name="Port Available",
                        status=CheckStatus.PASS,
                        message=f"NeMo Helix Quickstart already running on port {self.config.host_port}",
                        details="No restart needed",
                    )
                )
                self._nhx_already_running = True
            else:
                self.results.append(
                    PreflightResult(
                        name="Port Available",
                        status=CheckStatus.FAIL,
                        message=f"Port {self.config.host_port} is already in use by another service",
                        details="Stop the service using this port or configure a different port",
                    )
                )

    def _check_image_pullable(self) -> None:
        """Verify the container image can be pulled (or exists locally)."""
        if not self.config.image:
            self.results.append(
                PreflightResult(
                    name="Container Image",
                    status=CheckStatus.FAIL,
                    message="No container image configured",
                    details="Set NHX_IMAGE_TAG, provide --image, or ensure your NGC key has access to a nightly build",
                )
            )
            return

        def add_pull_required_result(*, details: str | None = None) -> None:
            self._image_pull_required = True
            self.results.append(
                PreflightResult(
                    name="Container Image",
                    status=CheckStatus.PASS,
                    message=f"Image {self.config.image} will be pulled on start",
                    details=details,
                )
            )

        try:
            from docker.errors import DockerException, ImageNotFound

            import docker

            from .container import ContainerManager

            client = docker.from_env()
            has_floating_tag = ContainerManager(self.config)._has_floating_tag()
            # First check if image exists locally
            try:
                client.images.get(self.config.image)
                self._image_pull_required = has_floating_tag
                message = (
                    f"Image {self.config.image} will be refreshed on start"
                    if has_floating_tag
                    else f"Image {self.config.image} available locally"
                )
                self.results.append(
                    PreflightResult(
                        name="Container Image",
                        status=CheckStatus.PASS,
                        message=message,
                    )
                )
            except ImageNotFound:
                # Will need to pull - this is normal, not a warning
                add_pull_required_result()
        except DockerException as e:
            from .prompts import detect_registry_auth_type

            if detect_registry_auth_type(self.config.image) == "user_pass":
                add_pull_required_result(details=f"Could not inspect local image: {e}")
                return

            self.results.append(
                PreflightResult(
                    name="Container Image",
                    status=CheckStatus.FAIL,
                    message="Cannot verify container image",
                    details=str(e),
                )
            )

    def _check_credentials(self) -> None:
        """Verify registry credentials if pulling from private registry."""
        if not self.config.image:
            return

        if self.config.is_ngc_registry():
            from .validators import validate_ngc_credentials

            if not self.config.ngc_api_key:
                self.results.append(
                    PreflightResult(
                        name="Registry Credentials",
                        status=CheckStatus.FAIL,
                        message="NGC API key required for nvcr.io images",
                        details="Set NGC_API_KEY environment variable or run 'nemo quickstart configure'",
                    )
                )
                return

            validation = validate_ngc_credentials(self.config.ngc_api_key.get_secret_value())
            if not validation.valid:
                self.results.append(
                    PreflightResult(
                        name="Registry Credentials",
                        status=CheckStatus.FAIL,
                        message="NGC credentials invalid",
                        details=validation.message,
                    )
                )
                return

            self.results.append(
                PreflightResult(
                    name="Registry Credentials",
                    status=CheckStatus.PASS,
                    message="NGC credentials configured",
                )
            )
            return

        from .prompts import detect_registry_auth_type
        from .validators import validate_image_registry_access, validate_registry_credentials

        if detect_registry_auth_type(self.config.image) == "user_pass":
            registry = self.config.get_registry_host()
            login_hint = (
                f"Run 'nemo quickstart auth --registry {registry}' and try again."
                if registry
                else "Run 'nemo quickstart auth' and try again."
            )
            if not self.config.has_registry_credentials_for_image():
                self.results.append(
                    PreflightResult(
                        name="Registry Credentials",
                        status=CheckStatus.FAIL,
                        message="Registry credentials are required",
                        details=login_hint,
                    )
                )
                return

            assert self.config.registry_username is not None
            assert self.config.registry_password is not None
            credential_validation = validate_registry_credentials(
                registry,
                self.config.registry_username,
                self.config.registry_password.get_secret_value(),
            )
            if not credential_validation.valid:
                self.results.append(
                    PreflightResult(
                        name="Registry Credentials",
                        status=CheckStatus.FAIL,
                        message="Registry credentials invalid",
                        details=f"{credential_validation.message}. {login_hint}",
                    )
                )
                return

            validation = validate_image_registry_access(self.config.image)
            if not validation.valid:
                self.results.append(
                    PreflightResult(
                        name="Registry Credentials",
                        status=CheckStatus.FAIL,
                        message="Registry credentials invalid or image is inaccessible",
                        details=f"{validation.message}. {login_hint}",
                    )
                )
                return

            self.results.append(
                PreflightResult(
                    name="Registry Credentials",
                    status=CheckStatus.PASS,
                    message="Registry credentials validated",
                )
            )
            return

        self.results.append(
            PreflightResult(
                name="Registry Credentials",
                status=CheckStatus.PASS,
                message="No registry credentials required",
            )
        )

    def _check_gpu_available(self) -> None:
        """Verify GPU availability if host-gpu inference is configured."""
        # Only check GPU if user selected host-gpu inference
        if not self.config.use_gpu:
            # User selected nvidia-build, no GPU needed
            return

        gpus_available, message = detect_gpus()

        if gpus_available:
            self.results.append(
                PreflightResult(
                    name="GPU Availability",
                    status=CheckStatus.PASS,
                    message=message,
                )
            )
        else:
            self.results.append(
                PreflightResult(
                    name="GPU Availability",
                    status=CheckStatus.FAIL,
                    message=message,
                    details="Host GPU inference requires an NVIDIA GPU. "
                    "Install NVIDIA drivers or use 'nvidia-build' inference instead.",
                )
            )

    def has_failures(self) -> bool:
        """Return True if any check failed."""
        return any(r.status == CheckStatus.FAIL for r in self.results)

    def has_warnings(self) -> bool:
        """Return True if any check has warnings."""
        return any(r.status == CheckStatus.WARN for r in self.results)

    def get_failures(self) -> list[PreflightResult]:
        """Get list of failed checks."""
        return [r for r in self.results if r.status == CheckStatus.FAIL]

    def get_warnings(self) -> list[PreflightResult]:
        """Get list of checks with warnings."""
        return [r for r in self.results if r.status == CheckStatus.WARN]

    def is_already_running(self) -> bool:
        """Return True if NeMo Helix quickstart is already running."""
        return self._nhx_already_running
