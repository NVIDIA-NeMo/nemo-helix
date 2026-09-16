# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quickstart configuration management."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_serializer, model_validator
from pydantic.functional_serializers import SerializationInfo
from typing_extensions import Self

from ._registry import image_registry_host
from .gpu_config import parse_comma_separated_non_negative_integers


def _try_secure_chmod_dir(path: Path) -> None:
    """Best-effort owner-only permissions on a directory (e.g. ``/tmp`` may reject chmod)."""
    try:
        os.chmod(path, stat.S_IRWXU)  # 700
    except PermissionError:
        pass


def _write_secure_yaml(path: Path, config_data: dict) -> None:
    """Write YAML atomically with owner read/write permissions (600).

    File permission failures remain fatal so credentials are never left world-readable.
    """
    mode = stat.S_IRUSR | stat.S_IWUSR  # 600
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w") as f:
            fd = -1
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
    finally:
        if fd >= 0:
            os.close(fd)
    # Existing files keep prior mode under O_TRUNC; enforce 600 after write.
    os.chmod(path, mode)


InferenceProviderType = Literal["nvidia-build", "host-gpu"]

# Registry and repo placeholder for SDK-stamped nightly/milestone tags.
_NIGHTLY_IMAGE_REPO = "nvcr.io/nvidia/platform-api"
# Registry and repo for public GA releases.  The exact path is not yet confirmed —
# it will be either nvcr.io/nvidia/nemo-microservices/nhx-api or
# nvcr.io/nvidia/nemo-helix/nhx-api.  Update this constant once the GA repo is
# finalised.
_PUBLIC_IMAGE_REPO = "nvcr.io/nvidia/nemo-microservices/nhx-api"


def _is_internal_tag(tag: str) -> bool:
    """Return True for nightly and milestone tags that live in the internal registry.

    Internal tag formats:
    - ``nightly-20260223``  (nightly build)
    - ``26.02-k10``         (milestone / RC build)

    Everything else (e.g. ``26.03``) is treated as a public GA release.
    """
    import re

    return bool(re.fullmatch(r"nightly-\d{8}|\d{2}\.\d{2}-k\d+", tag))


class QuickstartConfig(BaseModel):
    """Configuration for quickstart cluster.

    Stored in ~/.config/nhx/quickstart.yaml with environment variable overrides.

    Environment variables use the NHX_QUICKSTART_ prefix:
        - NHX_QUICKSTART_IMAGE: Container image
        - NHX_QUICKSTART_HOST_PORT: Host port
        - NHX_QUICKSTART_STORAGE_PATH: Storage directory
        - NHX_QUICKSTART_DOCKER_SOCKET: Docker socket path
        - NGC_API_KEY: NGC API key for image pulls
    """

    model_config = {"extra": "ignore"}  # Ignore unknown fields from old config files

    # Container image configuration
    image: str = Field(
        default="",
        description="Container image to run",
    )
    container_name: str = Field(
        default="nhx-quickstart",
        description="Name for the container",
    )
    network_name: str = Field(
        default="nhx-quickstart-network",
        description="Docker network name for the quickstart container",
    )

    # Credentials - use SecretStr for sensitive data
    ngc_api_key: SecretStr | None = Field(
        default_factory=lambda: SecretStr(key) if (key := os.getenv("NGC_API_KEY")) else None,
        description="NGC API key for authentication",
    )
    registry_host: str | None = Field(
        default=None,
        description="Registry host for authenticated quickstart image pulls",
    )
    registry_username: str | None = Field(
        default=None,
        description="Registry username for authenticated quickstart image pulls",
    )
    registry_password: SecretStr | None = Field(
        default=None,
        description="Registry password/token for authenticated quickstart image pulls",
    )
    # Runtime state (not saved to config file)
    container_id: str | None = Field(
        default=None,
        exclude=True,
        description="Running container ID",
    )

    # Inference provider configuration
    inference_provider: InferenceProviderType | None = Field(
        default=None,
        description="Deployment mode: 'nvidia-build' for cloud-only inference or 'host-gpu' for local GPU (inference, safe synthesizer)",
    )
    use_gpu: bool = Field(
        default=False,
        description="Whether to pass GPU through to the container (for host-gpu mode: inference, safe synthesizer)",
    )

    # Authentication configuration
    auth_enabled: bool = Field(
        default=False,
        description="Whether to enable authentication/authorization",
    )
    admin_email: str | None = Field(
        default=None,
        description="Bootstrap admin email for PlatformAdmin role",
    )

    # Storage configuration
    storage_path: Path = Field(
        default_factory=lambda: Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "nhx" / "quickstart",
        description="Storage directory for persistent data",
    )
    docker_socket: Path = Field(
        default=Path("/var/run/docker.sock"),
        description="Docker socket path",
    )

    # Network configuration
    host_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="Host port to expose the API",
    )
    container_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="Container port for the API",
    )

    # Platform config path (optional override)
    platform_config_path: Path | None = Field(
        default=None,
        description="Path to platform configuration YAML file",
    )

    # GPU device IDs for host-gpu mode (comma-separated string, e.g. "0,1,2").
    # Set during configure; required when use_gpu is True. None or empty = invalid for host-gpu.
    reserved_gpu_device_ids: str | None = Field(
        default=None,
        description="Comma-separated GPU device IDs (e.g. '0,1,2'). Set via nemo quickstart configure when using host-gpu.",
    )

    @model_validator(mode="before")
    @classmethod
    def ignore_legacy_registry_credentials(cls, data: Any) -> Any:
        """Ignore stale registry passwords from legacy registry_user configs."""
        if isinstance(data, dict) and "registry_user" in data and "registry_username" not in data:
            cleaned = dict(data)
            cleaned.pop("registry_password", None)
            return cleaned
        return data

    @field_serializer("ngc_api_key", "registry_password", when_used="json")
    def serialize_secrets(self, value: SecretStr | None, info: SerializationInfo) -> str | None:
        """Serialize secrets, revealing only when context requests it."""
        if value is None:
            return None
        if info.context and info.context.get("include_secrets"):
            return value.get_secret_value()
        return "***REDACTED***"

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default configuration file path.

        Checks XDG_CONFIG_HOME first, then falls back to ~/.config/nhx/quickstart.yaml.
        """
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config_home:
            return Path(xdg_config_home) / "nhx" / "quickstart.yaml"
        return Path.home() / ".config" / "nhx" / "quickstart.yaml"

    @classmethod
    def _load_from_env(cls) -> dict[str, str]:
        """Load configuration from environment variables with NHX_QUICKSTART_ prefix."""
        env_values: dict[str, str] = {}
        for field_name in cls.model_fields:
            env_key = f"NHX_QUICKSTART_{field_name.upper()}"
            if val := os.environ.get(env_key):
                env_values[field_name] = val
        # NGC_API_KEY is handled by the field's default_factory
        return env_values

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        """Load configuration from YAML file with environment variable overrides.

        Args:
            path: Path to config file. Uses default path if not provided.

        Returns:
            QuickstartConfig instance with merged file and environment settings.
        """
        config_path = path or cls.get_default_config_path()

        # Load from file first (lowest priority)
        config_data: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}

        # Merge env vars (higher priority than file)
        env_values = cls._load_from_env()
        config_data.update(env_values)

        return cls(**config_data)

    def save(self, path: Path | None = None) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save to. Uses default path if not provided.
        """
        config_path = path or self.get_default_config_path()

        # Ensure parent directory exists with secure permissions (owner-only access)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        _try_secure_chmod_dir(config_path.parent)

        # Serialize with secrets revealed
        config_data = self.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"container_id"},  # Don't persist runtime state
            context={"include_secrets": True},
        )

        # Convert Path objects to strings for YAML
        if "storage_path" in config_data:
            config_data["storage_path"] = str(config_data["storage_path"])
        if "docker_socket" in config_data:
            config_data["docker_socket"] = str(config_data["docker_socket"])
        if "platform_config_path" in config_data:
            config_data["platform_config_path"] = str(config_data["platform_config_path"])

        _write_secure_yaml(config_path, config_data)

    @classmethod
    def remove(cls) -> None:
        """Remove the configuration file."""
        config_path = cls.get_default_config_path()
        if config_path.exists():
            config_path.unlink()

    @property
    def data_path(self) -> Path:
        """Get the data directory path."""
        return self.storage_path / "data"

    def is_ngc_registry(self) -> bool:
        """Check if the configured image uses NGC registry."""
        # Strip explicit port (e.g. "nvcr.io:443") before matching.
        return self.get_registry_host().lower().split(":", 1)[0] == "nvcr.io"

    def get_registry_host(self) -> str:
        """Extract registry host from image name."""
        return image_registry_host(self.image)

    def has_registry_credentials_for_image(self) -> bool:
        """Return True when stored registry credentials match the configured image host."""
        registry_host = self.get_registry_host()
        return bool(
            registry_host and self.registry_host == registry_host and self.registry_username and self.registry_password
        )

    def resolve_best_image(self) -> str:
        """Return the best available image for the installed SDK build.

        If no image is explicitly configured, this method looks up the image
        tag for the installed SDK build and selects the right registry based
        on whether the tag is an internal or public release:

        - **Internal tags** (``nightly-YYYYMMDD``, ``YY.MM-kN``) live in the
          private NGC registry.  An NGC API key is required and a registry
          manifest check is performed to confirm access before returning the
          image.  Falls back to ``self.image`` if the key lacks access.
        - **Public GA tags** (e.g. ``26.03``) live in the public NGC registry.
          No key is required; the image is returned directly if it exists.

        The image tag is sourced from (in priority order):

        1. ``NHX_IMAGE_TAG`` environment variable — overrides the SDK-baked
           tag so you can test against a specific release without reinstalling
           (e.g. ``NHX_IMAGE_TAG=26.03``).
        2. ``__image_tag__`` stamped into the installed ``nemo-helix`` SDK
           (``nemo_helix._version``) at release time.

        The access check is performed via the Docker daemon's distribution
        endpoint (a lightweight manifest HEAD — no image data is transferred).

        Returns:
            The resolved image string if available, otherwise ``self.image``.
        """
        if self.image is not None and self.image != "":
            return self.image

        # NHX_IMAGE_TAG overrides the SDK-baked tag for pre-release testing.
        image_tag: str | None = os.environ.get("NHX_IMAGE_TAG") or None
        if image_tag is None:
            try:
                from nemo_helix._version import __image_tag__
            except ImportError:
                return self.image
            image_tag = __image_tag__

        if not image_tag:
            return self.image

        if _is_internal_tag(image_tag):
            api_image = f"{_NIGHTLY_IMAGE_REPO}:{image_tag}"
            return api_image
        else:
            # Public GA release: NGC key required (nvcr.io requires authentication),
            # but any valid key grants access — skip the Docker round-trip.
            if not self.ngc_api_key:
                return self.image
            return f"{_PUBLIC_IMAGE_REPO}:{image_tag}"

    def parse_image_components(self) -> tuple[str, str]:
        """Parse image into (registry, tag) components.

        Image format: [registry/][name][:tag]
        Examples:
            - "nvcr.io/nvidia/nemo-microservices/nhx-api:25.10"
              → ("nvcr.io/nvidia/nemo-microservices", "25.10")
            - "my-registry/nhx-api:local"
              → ("my-registry", "local")
            - "registry:5000/image:v1"
              → ("registry:5000", "v1")
            - "nhx-api:latest"
              → ("", "latest")
            - "nhx-api"
              → ("", "latest")

        Returns:
            Tuple of (image_registry, image_tag)
        """
        image = self.image

        # Extract tag (after last colon, if not part of a port)
        if ":" in image:
            # Handle case where colon might be a port (e.g., registry:5000/image)
            last_colon = image.rfind(":")
            last_slash = image.rfind("/")
            if last_colon > last_slash:
                tag = image[last_colon + 1 :]
                image = image[:last_colon]
            else:
                tag = "latest"
        else:
            tag = "latest"

        # Extract registry (everything before the image name)
        parts = image.rsplit("/", 1)
        if len(parts) == 2:
            registry = parts[0]
        else:
            registry = ""

        return registry, tag

    def parse_reserved_gpu_device_ids(self) -> list[int] | None:
        """Parse reserved_gpu_device_ids string into a list of integers.

        Uses the shared parser in gpu_config for consistency with
        CUDA_VISIBLE_DEVICES parsing.

        Returns:
            List of GPU device IDs; None if field is None or contains invalid values;
            empty list if field is empty or whitespace-only string.
        """
        if self.reserved_gpu_device_ids is None:
            return None
        s = self.reserved_gpu_device_ids.strip()
        if not s:
            return []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            return []
        return parse_comma_separated_non_negative_integers(s)
