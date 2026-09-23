# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential and configuration validators."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from nemo_helix_plugin.capabilities import probe_docker

from .config import QuickstartConfig


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    message: str

    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean contexts."""
        return self.valid


def validate_docker_available() -> ValidationResult:
    """Check if Docker daemon is available via an uncached reachability probe (≤5s).

    Returns:
        ValidationResult indicating success or failure.
    """
    # CLI may re-run after the user starts Docker; do not pin a prior miss.
    result = probe_docker(use_cache=False)
    if result.available:
        return ValidationResult(True, "Docker is available")
    return ValidationResult(False, result.detail or "Docker is not available")


def validate_ngc_credentials(api_key: str) -> ValidationResult:
    """Validate NGC API key by attempting a docker login.

    Args:
        api_key: The NGC API key to validate.

    Returns:
        ValidationResult indicating success or failure.
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "login",
                "nvcr.io",
                "-u",
                "$oauthtoken",
                "--password-stdin",
            ],
            capture_output=True,
            input=api_key,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return ValidationResult(True, "NGC credentials are valid")
        return ValidationResult(False, f"NGC login failed: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        return ValidationResult(False, "NGC login timed out")
    except FileNotFoundError:
        return ValidationResult(False, "Docker CLI not found")
    except Exception as e:
        return ValidationResult(False, f"NGC validation error: {e}")


def validate_registry_credentials(
    registry: str,
    username: str,
    password: str,
) -> ValidationResult:
    """Validate custom registry credentials.

    Args:
        registry: Registry URL.
        username: Registry username.
        password: Registry password.

    Returns:
        ValidationResult indicating success or failure.
    """
    try:
        result = subprocess.run(
            ["docker", "login", registry, "-u", username, "--password-stdin"],
            capture_output=True,
            input=password,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return ValidationResult(True, "Registry credentials are valid")
        return ValidationResult(False, f"Registry login failed: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        return ValidationResult(False, "Registry login timed out")
    except FileNotFoundError:
        return ValidationResult(False, "Docker CLI not found")
    except Exception as e:
        return ValidationResult(False, f"Registry validation error: {e}")


def validate_image_registry_access(image: str) -> ValidationResult:
    """Validate that Docker can access an image manifest without pulling layers.

    Args:
        image: Fully qualified image reference to inspect.

    Returns:
        ValidationResult indicating success or failure.
    """
    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", image],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return ValidationResult(True, "Image manifest is accessible")
        detail = result.stderr.strip() or result.stdout.strip()
        if detail:
            return ValidationResult(False, f"Image manifest check failed: {detail}")
        return ValidationResult(False, "Image manifest check failed")
    except subprocess.TimeoutExpired:
        return ValidationResult(False, "Image manifest check timed out")
    except FileNotFoundError:
        return ValidationResult(False, "Docker CLI not found")
    except Exception as e:
        return ValidationResult(False, f"Image manifest validation error: {e}")


def validate_docker_socket(socket_path: Path) -> ValidationResult:
    """Validate docker socket exists and is accessible.

    Args:
        socket_path: Path to the docker socket.

    Returns:
        ValidationResult indicating success or failure.
    """
    import os

    if not socket_path.exists():
        return ValidationResult(False, f"Docker socket not found at {socket_path}")

    if not os.access(socket_path, os.R_OK | os.W_OK):
        return ValidationResult(
            False,
            f"Docker socket at {socket_path} is not readable/writable",
        )

    return ValidationResult(True, "Docker socket is accessible")


def validate_storage_path(storage_path: Path) -> ValidationResult:
    """Validate storage path can be created and written to.

    Args:
        storage_path: Path to the storage directory.

    Returns:
        ValidationResult indicating success or failure.
    """
    try:
        storage_path.mkdir(parents=True, exist_ok=True)
        test_file = storage_path / ".write_test"
        test_file.touch()
        test_file.unlink()
        return ValidationResult(True, f"Storage path {storage_path} is writable")
    except PermissionError:
        return ValidationResult(False, f"Cannot write to storage path {storage_path}")
    except Exception as e:
        return ValidationResult(False, f"Storage path error: {e}")


def is_quickstart_running(config: QuickstartConfig) -> bool:
    """Check if NeMo Helix quickstart container is already running via Docker.

    Args:
        config: Quickstart configuration with container name.

    Returns:
        True if the quickstart container is running, False otherwise.
    """
    try:
        import docker

        client = docker.from_env()
        container = client.containers.get(config.container_name)
        return container.status == "running"
    except Exception:
        return False


def validate_port_available(port: int, config: QuickstartConfig | None = None) -> ValidationResult:
    """Validate that a port is available for binding.

    Args:
        port: Port number to check.
        config: Optional quickstart config to check if NeMo Helix is using the port.

    Returns:
        ValidationResult indicating success or failure.
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))  # noqa: S104  # nosec B104
        return ValidationResult(True, f"Port {port} is available")
    except OSError:
        # Port is in use - check if it's the quickstart container
        if config is not None and is_quickstart_running(config):
            return ValidationResult(True, f"Port {port} is in use by quickstart")
        return ValidationResult(False, f"Port {port} is already in use")


def validate_config(config: QuickstartConfig) -> list[ValidationResult]:
    """Run all validations on a config.

    Args:
        config: The QuickstartConfig to validate.

    Returns:
        List of ValidationResult objects.
    """
    results = []

    # Always check Docker
    results.append(validate_docker_available())

    # Check docker socket
    results.append(validate_docker_socket(config.docker_socket))

    # Check storage path
    results.append(validate_storage_path(config.storage_path))

    # Check port availability (with config to detect if quickstart is using the port)
    results.append(validate_port_available(config.host_port, config))

    # When host-gpu is enabled, reserved_gpu_device_ids must be set (no backwards compatibility)
    if config.use_gpu:
        if not config.reserved_gpu_device_ids or not config.reserved_gpu_device_ids.strip():
            results.append(
                ValidationResult(
                    False,
                    "GPU mode is enabled but GPU device IDs are not set. Re-run configure and select host-gpu, "
                    "or set GPU device IDs in advanced options.",
                )
            )
        else:
            results.append(ValidationResult(True, "GPU device IDs are set"))

    # Check credentials based on image registry
    if config.is_ngc_registry() and config.ngc_api_key:
        results.append(validate_ngc_credentials(config.ngc_api_key.get_secret_value()))
    return results
