# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for platform seeding."""

import os
import re
from pathlib import Path

from pydantic import Field, TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict

_BOOL_ENV_PARSER = TypeAdapter(bool)


class PlatformSeedConfig(BaseSettings):
    """
    Configuration for platform seed runs.

    Environment variables use the NHX_PLATFORM_SEED_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="NHX_PLATFORM_SEED_",
        env_nested_delimiter="_",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Master switch: run platform seeding when true")
    auth_enabled: bool = Field(
        default=True,
        description="Seed auth role bindings",
    )
    guardrails_enabled: bool = Field(default=True, description="Seed guardrail configs from config store")
    model_provider_enabled: bool = Field(default=True, description="Seed the default nvidia-build model provider")

    guardrails_config_store_path: Path = Field(
        default_factory=lambda: Path(os.getenv("CONFIG_STORE_PATH", "/dev/null")),
        description="Path to guardrails config store directory",
    )
    wait_for_ready_enabled: bool = Field(
        default=True,
        description="Wait for dependency services (entities, auth, files) via /status before seeding",
    )
    wait_for_ready_retries: int = Field(
        default=60,
        description="Max number of readiness checks (each spaced by wait_for_ready_interval_seconds)",
    )
    wait_for_ready_interval_seconds: float = Field(
        default=5.0,
        description="Seconds between readiness checks",
    )

    @staticmethod
    def plugin_seed_enabled_env_var(plugin_name: str) -> str:
        """Return the per-plugin seed toggle env var name for a discovered seed job."""
        normalized_name = re.sub(r"[^A-Za-z0-9]+", "_", plugin_name).strip("_")
        return f"NHX_PLATFORM_SEED_{normalized_name.upper()}_ENABLED"

    def is_plugin_seed_enabled(self, plugin_name: str) -> bool:
        """Return whether a discovered plugin seed job should run."""
        env_var = self.plugin_seed_enabled_env_var(plugin_name)
        raw_value = os.getenv(env_var)
        if raw_value is None:
            return True
        return _BOOL_ENV_PARSER.validate_python(raw_value)
