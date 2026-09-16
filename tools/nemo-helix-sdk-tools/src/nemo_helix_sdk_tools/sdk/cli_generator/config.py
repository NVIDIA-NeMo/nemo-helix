# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from nemo_helix_sdk_tools.sdk.core.common import get_project_dir

VALID_WAIT_CONFIG_TYPES = {"inference_deployment", "platform_job"}
VALID_WATCH_CONFIG_TYPES = {"inference_deployment", "platform_job"}


def get_cli_generator_root() -> Path:
    """Get the root directory for CLI generation."""
    return Path(__file__).parent


def get_templates_dir() -> Path:
    """Get the templates directory."""
    return get_cli_generator_root() / "templates"


def get_overrides_dir() -> Path:
    """Get the overrides directory."""
    return get_cli_generator_root() / "overrides"


def get_target_commands_dir() -> Path:
    """Get the target commands/api directory."""
    repo_root = get_project_dir()
    return repo_root / "packages" / "nemo_helix_ext" / "src" / "nemo_helix_ext" / "cli" / "commands" / "api"


@dataclass
class ColumnConfig:
    """Column configuration for code generation.

    This is separate from the runtime Column class in formatters.py
    to keep the generator independent of the CLI runtime.
    """

    field: str
    header: str | None = None


class CLIConfig:
    """Load and query CLI configuration from YAML file."""

    def __init__(self, config_path: Path):
        """Initialize with path to cli_config.yaml.

        Args:
            config_path: Path to the CLI configuration YAML file
        """
        self._config_path = config_path
        self._config: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load the YAML configuration file."""
        with open(self._config_path) as f:
            self._config = yaml.safe_load(f)

    def get_columns(
        self,
        resource_path: list[str],
        method_name: str,
    ) -> list[ColumnConfig] | None:
        """Get table columns for a resource method.

        Args:
            resource_path: Path to the resource (e.g., ["customization", "jobs"])
            method_name: Name of the method (e.g., "list", "list_namespace")

        Returns:
            List of (header, field) tuples, or None if not configured
        """
        # Find matching resource in flat config
        if config := self.get_method_config(resource_path, method_name):
            columns_config = config.get("columns")
            if columns_config is not None:
                return self._parse_columns(columns_config)

        # No match found, return defaults
        return self._get_default_columns(method_name)

    def _get_default_columns(self, method_name: str) -> list[ColumnConfig] | None:
        """Get default columns from config.

        Args:
            method_name: Name of the method (e.g., "list")

        Returns:
            List of (header, field) tuples, or None if no defaults
        """
        defaults = self._config.get("defaults", {})
        method_defaults = defaults.get(method_name)

        if method_defaults is None:
            # Fallback to "list" defaults for list_* methods
            if method_name.startswith("list"):
                method_defaults = defaults.get("list")

        if method_defaults is None:
            return None

        columns_config = method_defaults.get("columns")
        if columns_config is None:
            return None

        return self._parse_columns(columns_config)

    def get_resource_config(self, resource_path: list[str]) -> dict[str, object] | None:
        """
        Get configuration for a resource.
        Args:
            resource_path: Path to the resource (e.g., ["customization", "jobs"])
        Returns:
            Configuration dictionary, or None if not found
        """
        config_list = self._config.get("config", [])
        for entry in config_list:
            if entry.get("resource") == resource_path:
                return entry

        return None

    def get_top_level_command_config(self, command_name: str) -> dict[str, object]:
        """Get root command metadata for a generated top-level API command."""
        raw_top_level_config = self._config.get("top_level")
        top_level_config = raw_top_level_config if isinstance(raw_top_level_config, dict) else {}
        command_config = top_level_config.get(command_name, {})
        if isinstance(command_config, dict):
            return command_config
        return {}

    def get_method_config(self, resource_path: list[str], method_name: str) -> dict[str, object] | None:
        """
        Get configuration for a method.
        Args:
            resource_path: Path to the resource (e.g., ["customization", "jobs"])
            method_name: Name of the method (e.g., "list", "list_namespace")
        Returns:
            Configuration dictionary, or None if not found
        """
        if rc := self.get_resource_config(resource_path):
            return rc.get("methods", {}).get(method_name)

        return None

    def _parse_columns(self, columns_config: list) -> list[ColumnConfig]:
        """Parse columns config into ColumnConfig objects.

        Supports two formats:
        - Simple: just field name string (e.g., "id")
        - Full: dict with "field" and optional "header" keys

        Args:
            columns_config: List of column definitions

        Returns:
            List of ColumnConfig objects
        """
        result = []
        for col in columns_config:
            if isinstance(col, str):
                # Simple format: field name only
                result.append(ColumnConfig(field=col))
            else:
                # Full format: dict with field and optional header
                field = col["field"]
                header = col.get("header")
                result.append(ColumnConfig(field=field, header=header))

        return result

    def should_skip(self, resource_path: list[str]) -> bool:
        """
        Check if a resource should be skipped from generation.

        A resource is skipped if it or any of its parent resources are marked with skip: true.
        For example, if [evaluation] has skip: true, then [evaluation, benchmarks] is also skipped.
        """
        # Check all prefixes from shortest to longest (e.g., [evaluation], [evaluation, benchmarks])
        for i in range(1, len(resource_path) + 1):
            prefix = resource_path[:i]
            if rc := self.get_resource_config(prefix):
                if rc.get("skip", False):
                    return True

        return False

    def get_method_override(self, resource_path: list[str], method_name: str) -> Path | None:
        """Get the full path to an override file if one exists."""
        if method_config := self.get_method_config(resource_path, method_name):
            if override_rel_path := method_config.get("override"):  # type: ignore[union-attr]
                return get_overrides_dir() / str(override_rel_path)
        return None

    def should_skip_method(self, resource_path: list[str], method_name: str) -> bool:
        """Check if a specific method should be skipped from generation."""
        if method_config := self.get_method_config(resource_path, method_name):
            return method_config.get("skip", False)  # type: ignore[union-attr]
        return False

    def get_wait_config(self, resource_path: list[str], method_name: str) -> dict[str, object] | None:
        """Get inline wait configuration for a generated command."""
        return self._get_lifecycle_config(
            resource_path,
            method_name,
            config_key="wait",
            valid_types=VALID_WAIT_CONFIG_TYPES,
        )

    def get_watch_config(self, resource_path: list[str], method_name: str) -> dict[str, object] | None:
        """Get inline watch configuration for a generated command."""
        return self._get_lifecycle_config(
            resource_path,
            method_name,
            config_key="watch",
            valid_types=VALID_WATCH_CONFIG_TYPES,
        )

    def _get_lifecycle_config(
        self,
        resource_path: list[str],
        method_name: str,
        *,
        config_key: str,
        valid_types: set[str],
    ) -> dict[str, object] | None:
        if method_config := self.get_method_config(resource_path, method_name):
            lifecycle_config = method_config.get(config_key)
            if lifecycle_config is None:
                return None
            resource = ".".join(resource_path)
            if not isinstance(lifecycle_config, dict):
                raise ValueError(
                    f"Invalid {config_key} config for {resource}.{method_name}. Expected a mapping, got "
                    f"{type(lifecycle_config).__name__}."
                )

            lifecycle_type = lifecycle_config.get("type")
            valid_types_list = ", ".join(sorted(valid_types))
            try:
                lifecycle_type_is_valid = lifecycle_type in valid_types
            except TypeError as exc:
                raise ValueError(
                    f"Invalid {config_key} config type={lifecycle_type!r} for resource_path={resource_path!r}, "
                    f"method_name={method_name!r}. Expected one of: {valid_types_list}"
                ) from exc

            if not lifecycle_type_is_valid:
                raise ValueError(
                    f"Invalid {config_key} config type {lifecycle_type!r} for {resource}.{method_name}. "
                    f"Expected one of: {valid_types_list}"
                )
            resource_label = lifecycle_config.get("resource_label")
            if not isinstance(resource_label, str) or not resource_label.strip():
                raise ValueError(
                    f"Invalid {config_key} config resource_label {resource_label!r} for {resource}.{method_name}. "
                    "Expected a non-empty string."
                )
            return lifecycle_config
        return None

    def get_examples(self, resource_path: list[str], method_name: str) -> list[str] | None:
        """Return endpoint-specific CLI examples when configured."""

        method_config = self.get_method_config(resource_path, method_name)
        examples = method_config.get("examples") if method_config else None
        if examples is None:
            return None
        resource = ".".join(resource_path)
        if (
            not isinstance(examples, list)
            or not examples
            or not all(isinstance(example, str) and example.strip() for example in examples)
        ):
            raise ValueError(f"Invalid CLI examples for {resource}.{method_name}; expected non-empty strings")
        return [str(example) for example in examples]

    def get_additional_methods(self, resource_path: list[str]) -> dict[str, Any]:
        """Get additional methods for a resource.

        Returns:
            Dict mapping method name to config (e.g., {"upload": {"override": "filesets/upload.py"}})
        """
        if rc := self.get_resource_config(resource_path):
            return rc.get("additional_methods", {})  # type: ignore[union-attr]
        return {}

    def get_all_resources_with_additional_methods(self) -> list[tuple[str, ...]]:
        """Get all resource paths that have additional_methods configured.

        Returns:
            List of resource paths as tuples (e.g., [("files",), ("files", "filesets")])
        """
        resources = []
        config_list = self._config.get("config", [])
        for entry in config_list:
            if isinstance(entry, dict) and entry.get("additional_methods"):
                resource_path = entry.get("resource", [])
                if resource_path:
                    resources.append(tuple(resource_path))
        return resources

    def get_param_help(self, resource_path: list[str], method_name: str) -> dict[str, str]:
        """Get help text overrides for specific parameters of a method.

        Supports the ``{{ENTITY_TYPES}}`` placeholder, which is replaced with a
        comma-separated list of all entity types discovered in the repo.

        Args:
            resource_path: Path to the resource (e.g., ["entities"])
            method_name: Name of the method (e.g., "list")

        Returns:
            Dict mapping parameter name to help text string (empty dict if none configured).
        """
        if method_config := self.get_method_config(resource_path, method_name):
            raw: dict[str, str] = method_config.get("param_help", {})  # type: ignore[union-attr]
            return {name: _resolve_placeholders(text, get_project_dir()) for name, text in raw.items()}
        return {}


def _resolve_placeholders(text: str, project_root: Path) -> str:
    """Replace ``{{ENTITY_TYPES}}`` placeholder with discovered entity types."""
    if "{{ENTITY_TYPES}}" not in text:
        return text
    entity_types = discover_entity_types(project_root)
    return text.replace("{{ENTITY_TYPES}}", ", ".join(entity_types))


@lru_cache
def discover_entity_types(project_root: Path) -> list[str]:
    """Scan the repo for ``__entity_type__`` assignments and return a sorted list of values.

    Searches services/, packages/, and plugins/ for ``__entity_type__ = "..."`` assignments.

    Exclusions:
    - Test files (any path component named ``tests`` or matching ``test_*.py``).
    - Reference/demo plugins (``plugins/example-plugin/``).
    """
    pattern = re.compile(r'__entity_type__(?::[^=]+)?\s*=\s*["\'](\w+)["\']')
    entity_types: set[str] = set()

    # Plugins that are reference implementations, not production plugins.
    # Their entity types should not appear in the CLI help text.
    _EXCLUDED_PLUGINS = {"example-plugin"}

    for scan_dir in ["services", "packages", "plugins"]:
        scan_path = project_root / scan_dir
        for py_file in scan_path.rglob("*.py"):
            # Skip test directories and test files.
            rel = py_file.relative_to(scan_path)
            parts = rel.parts
            if any(p == "tests" or p.startswith("test_") for p in parts):
                continue
            # Skip excluded plugins.
            if scan_dir == "plugins" and parts[0] in _EXCLUDED_PLUGINS:
                continue
            try:
                entity_types.update(pattern.findall(py_file.read_text(encoding="utf-8")))
            except OSError:
                continue
    return sorted(entity_types)
