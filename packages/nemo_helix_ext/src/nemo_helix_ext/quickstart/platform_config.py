# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform configuration for the nhx-api container."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from typing_extensions import Self


class SecretsConfig(BaseModel):
    """Secrets service configuration for quickstart development/demo runs."""

    model_config = {"extra": "allow"}

    allow_key_creation: bool = Field(
        default=True,
        description=(
            "Allow quickstart to create a local encryption key when no explicit "
            "secrets encryption provider is configured."
        ),
    )
    local_key_creation_path: str = Field(
        default="/data/nhx-encryption-key.txt",
        description="Container path where quickstart persists the local secrets encryption key.",
    )


class PlatformConfig(BaseModel):
    """Platform configuration passed to the nhx-api container.

    This configuration can be passed to the container either via:
    1. Environment variables (via to_env_vars())
    2. Mounted YAML file at /etc/nhx/platform-config.yaml
    """

    model_config = {"extra": "allow"}

    nvidia_api_key: str | None = Field(
        default=None,
        description="NVIDIA API key for inference services",
    )
    secrets: SecretsConfig = Field(
        default_factory=SecretsConfig,
        description="Secrets service configuration for quickstart.",
    )

    @classmethod
    def get_default(cls) -> Self:
        """Return default platform configuration."""
        return cls()

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load platform config from YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            PlatformConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the YAML is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Platform config file not found: {path}")

        with open(path, "r") as f:
            try:
                config_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing platform config at {path}: {e}") from e

        return cls.model_validate(config_data)

    def save(self, path: Path) -> None:
        """Save platform config to YAML file.

        Args:
            path: Path to save the configuration.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        config_data = self.model_dump(mode="json", exclude_none=True)

        with open(path, "w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

    def to_env_vars(self) -> dict[str, str]:
        """Convert config to environment variables for the container.

        Returns:
            Dictionary of environment variable name to value.
        """
        env: dict[str, str] = {}

        if self.nvidia_api_key:
            env["NVIDIA_API_KEY"] = self.nvidia_api_key

        env["NHX_SECRETS_ALLOW_KEY_CREATION"] = "1" if self.secrets.allow_key_creation else "0"
        if self.secrets.local_key_creation_path:
            env["NHX_SECRETS_LOCAL_KEY_CREATION_PATH"] = self.secrets.local_key_creation_path

        return env
