# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config discovery and loading for E2E tests.

This module provides functionality to:
- Discover config files in e2e/configs/
- Load configs as-is (no inheritance or merging; platform does merge overwrite at runtime)
- Infer backend type from config name or explicit field

Usage:
    from nhx.testing.e2e.config import discover_configs, load_config

    # Discover all configs in e2e/configs/
    configs = discover_configs(Path("e2e/configs"))
    # Returns: {"docker": Path("e2e/configs/docker.yaml"), ...}

    # Load a config (file used as-is; platform applies merge overwrite at runtime)
    e2e_config = load_config(Path("e2e/configs/docker.yaml"), repo_root)
    # Returns E2EConfig with path and loaded config dict
"""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

# Default config file for docker platform (relative to repo root).
DEFAULT_QUICKSTART_CONFIG = "e2e/quickstart/default.yaml"

# Auth-enabled config for docker platform (used when --feature auth is set).
AUTH_QUICKSTART_CONFIG = "e2e/quickstart/auth.yaml"


@dataclass
class E2EConfig:
    """Configuration for an E2E test run.

    Config file is used as-is; backends mount the file path and the platform
    applies merge overwrite (defaults + file) at runtime.

    Attributes:
        name: Config name derived from filename (e.g., "docker", "docker_auth_enabled")
        path: Absolute path to the config file
        backend: The backend type ("docker" or "kubernetes")
        config: Loaded config dict from the file (for inspection; backends use path)
    """

    name: str
    path: Path
    backend: Literal["docker", "kubernetes", "external"]
    config: dict

    def __repr__(self) -> str:
        return f"E2EConfig(name={self.name!r}, backend={self.backend!r})"


def discover_configs(configs_dir: Path) -> dict[str, Path]:
    """Discover all E2E config files in a directory.

    Finds all *.yaml files in the given directory and returns a mapping
    from config name (filename without extension) to absolute path.

    Args:
        configs_dir: Directory to search for config files.

    Returns:
        Dict mapping config names to their absolute paths.
        Example: {"docker": Path("/abs/path/e2e/configs/docker.yaml")}
    """
    configs: dict[str, Path] = {}

    if not configs_dir.exists():
        return configs

    for config_file in configs_dir.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = config_file.resolve()

    return configs


def load_config(config_path: Path, repo_root: Path) -> E2EConfig:
    """Load an E2E config file as-is (no inheritance or merging).

    The file is used directly; the platform applies merge overwrite
    (defaults + file) at runtime when loading the config.

    Args:
        config_path: Path to the config file to load.
        repo_root: Repository root (kept for API compatibility; not used).

    Returns:
        E2EConfig with path and loaded config dict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    _ = repo_root  # Kept for API compatibility
    config_path = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_data = yaml.safe_load(f) or {}

    config_name = config_path.stem
    backend = infer_backend(config_name, config_data)

    return E2EConfig(
        name=config_name,
        path=config_path,
        backend=backend,
        config=config_data,
    )


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.

    Recursively merges nested dicts. For scalars and lists, override
    values replace base values entirely.

    Args:
        base: Base dictionary.
        override: Dictionary with values to override.

    Returns:
        New merged dictionary.
    """
    result = copy.deepcopy(base)

    for key, override_value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(override_value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], override_value)
        else:
            # Scalars and lists: override replaces base
            result[key] = copy.deepcopy(override_value)

    return result


def infer_backend(config_name: str, config_data: dict) -> Literal["docker", "kubernetes", "external"]:
    """Infer the backend type from config.

    First checks for explicit `e2e.backend` field, then falls back to
    inferring from the config filename prefix.

    Args:
        config_name: Name of the config (filename without extension).
        config_data: Loaded config data.

    Returns:
        Backend type: "docker", "kubernetes", or "external".
    """
    # Check for explicit backend setting
    e2e_settings = config_data.get("e2e", {})
    explicit_backend = e2e_settings.get("backend")

    if explicit_backend:
        if explicit_backend not in ("docker", "kubernetes", "external"):
            raise ValueError(f"Invalid backend '{explicit_backend}'. Must be 'docker', 'kubernetes', or 'external'.")
        return explicit_backend

    # Infer from filename prefix
    if config_name.startswith("external"):
        return "external"
    elif config_name.startswith("docker"):
        return "docker"
    elif config_name.startswith("kubernetes") or config_name.startswith("k8s"):
        return "kubernetes"

    # Default to docker
    return "docker"
