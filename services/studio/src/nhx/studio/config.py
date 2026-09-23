# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Studio service configuration."""

import logging
from fnmatch import fnmatchcase
from functools import cached_property
from pathlib import Path
from typing import Any

from nhx.common.config import Configuration, create_service_config_class
from nhx.studio.env_mappings import ENV_MAPPINGS
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StudioOtelConfig(BaseModel):
    """OpenTelemetry configuration for Studio UI browser telemetry."""

    collector_url: str = Field(
        default="",
        description="Internal OTLP/HTTP collector base URL used by the Studio telemetry proxy.",
    )
    service_name: str = Field(
        default="nemo-studio-ui",
        description="OpenTelemetry service.name resource attribute for Studio UI telemetry.",
    )
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost",
            "http://localhost:*",
        ],
        description=(
            "Browser origins allowed to post telemetry through the Studio telemetry proxy. "
            "Entries use shell-style '*' wildcards."
        ),
    )

    def is_origin_allowed(self, origin: str, same_origin: str | None = None) -> bool:
        """Return whether the given browser Origin is allowed."""
        if not origin:
            return False
        if same_origin and origin == same_origin:
            return True
        return any(fnmatchcase(origin, allowed_origin) for allowed_origin in self.allowed_origins)


class StudioConfig(create_service_config_class("studio")):  # type: ignore[misc]  # ty: ignore[unsupported-base]
    """Configuration for the Studio service.

    This configuration is loaded from the 'studio' section of the
    global config file or from environment variables with the prefix
    NHX_STUDIO_.

    StudioConfig also provides access to global settings needed for
    runtime environment variable injection into the UI bundle.
    """

    static_files_path: Path | None = Field(
        default=None,
        description=(
            "Path to the directory containing the built static UI assets. "
            "When unset, Studio looks for the `static/` directory bundled alongside the "
            "`nhx.studio` package (populated by the wheel build), then `/static/studio` "
            "(where NeMo Helix container images place the bundle), then "
            "`web/packages/studio/dist` in a source checkout."
        ),
    )
    platform_base_url: str = Field(
        default="",
        description="Base URL of the platform. This is used by the Studio UI to make API calls.",
    )
    telemetry_enabled: bool = Field(
        default=False,
        description="Enable Studio UI browser telemetry export.",
    )
    otel: StudioOtelConfig = Field(
        default_factory=StudioOtelConfig,
        description="Studio UI OpenTelemetry settings.",
    )

    @cached_property
    def global_settings(self) -> dict[str, Any]:
        """Get the full global settings dict (all YAML sections).

        Returns:
            Dict containing all global configuration sections (platform, entities, etc.)
        """
        try:
            return Configuration.get_global_settings_from_env()
        except Exception as e:
            logger.warning(f"Failed to get global settings: {e}")
            return {}

    def _resolve_config_path(self, path: str) -> str | None:
        """Resolve a dot-notation path on the global settings.

        Args:
            path: Dot-notation path (e.g., 'platform.base_url' or 'entities.backend')

        Returns:
            The resolved value as a string, or None if not found
        """
        try:
            value: Any = self.global_settings
            for part in path.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)
                if value is None:
                    return None
            return str(value) if value is not None else None
        except (AttributeError, KeyError, TypeError):
            return None

    def _resolve_field_path(self, path: str) -> str | None:
        """Resolve a dot-notation path on this StudioConfig instance."""
        try:
            value: Any = self
            for part in path.split("."):
                value = getattr(value, part, None)
                if value is None:
                    return None
            if value == "":
                return None
            if isinstance(value, bool):
                return str(value).lower()
            return str(value)
        except (AttributeError, TypeError):
            return None

    @cached_property
    def env_replacements(self) -> dict[str, str]:
        """Environment variable replacements for the UI bundle (cached).

        Uses ENV_MAPPINGS to dynamically resolve markers to their config values.
        See env_mappings.py for the mapping definitions.

        The config_path in ENV_MAPPINGS uses dot notation to access any section:
        - "platform.base_url" -> config['platform']['base_url']
        - "entities.backend" -> config['entities']['backend']

        Returns:
            Dict mapping STUDIO_UI_* markers to their actual values
        """
        replacements: dict[str, str] = {}

        for mapping in ENV_MAPPINGS:
            value = self._resolve_config_path(mapping.config_path)
            if value == "":
                value = None
            # Fall back to own pydantic-settings fields for paths under "studio.*"
            # (env vars like NHX_STUDIO_PLATFORM_BASE_URL set pydantic fields but
            # aren't reflected in the YAML-backed global_settings dict).
            if value is None and mapping.config_path.startswith("studio."):
                value = self._resolve_field_path(mapping.config_path.removeprefix("studio."))
            if value == "":
                value = None
            if value is None and mapping.config_path == "studio.platform_base_url":
                value = self._resolve_config_path("platform.base_url")
            if value == "":
                value = None
            if value is not None:
                replacements[mapping.marker] = value
                logger.debug(f"Resolved {mapping.marker} -> {value}")
            elif mapping.default is not None:
                replacements[mapping.marker] = mapping.default
                logger.debug(f"Using default for {mapping.marker} -> {mapping.default}")
            else:
                logger.debug(f"Could not resolve config path '{mapping.config_path}' for {mapping.marker}")

        logger.info(f"Studio env replacements configured: {len(replacements)} mappings")
        return replacements
