# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Intake service."""

import os
from pathlib import Path
from typing import Any, cast

from nhx.common.config import EnvironmentFirstSettings, create_service_config_class
from pydantic import Field
from pydantic_settings import SettingsConfigDict

DEFAULT_ATIF_MAX_SUBAGENT_DEPTH = 64
MAX_ATIF_MAX_SUBAGENT_DEPTH = 256
DEFAULT_SPAN_RETENTION_DAYS = 90
DEFAULT_CLICKHOUSE_URL = "http://localhost:8123"
DEFAULT_CLICKHOUSE_VERSION = "26.3"
DEFAULT_CLICKHOUSE_IMAGE = f"clickhouse/clickhouse-server:{DEFAULT_CLICKHOUSE_VERSION}"
CLICKHOUSE_URL_ENV_VAR = "NHX_INTAKE_CLICKHOUSE_URL"


class ClickHouseConfig(EnvironmentFirstSettings):
    """Configuration for Intake's ClickHouse-backed spans storage."""

    model_config = SettingsConfigDict(env_prefix="NHX_INTAKE_CLICKHOUSE_")

    url: str = Field(
        default=DEFAULT_CLICKHOUSE_URL,
        description="HTTP URL for the ClickHouse server used by Intake spans storage",
    )
    user: str = Field(
        default="default",
        description="ClickHouse username for Intake spans storage",
    )
    password: str = Field(
        default="",
        description="ClickHouse password for Intake spans storage",
    )
    database: str = Field(
        default="intake",
        description="ClickHouse database for Intake spans",
    )
    image: str = Field(
        default=DEFAULT_CLICKHOUSE_IMAGE,
        description="Container image used only when Intake provisions local ClickHouse",
    )
    data_dir: Path | None = Field(
        default=None,
        description="Host data directory used only when Intake provisions local ClickHouse",
    )


def should_provision_local_clickhouse(config: ClickHouseConfig) -> bool:
    """Return whether Intake owns the default local ClickHouse lifecycle.

    An explicitly exported URL is always operator-owned, including the URL Helm
    injects for its embedded ClickHouse. A non-default URL loaded from any other
    configuration source is operator-owned as well.
    """

    return CLICKHOUSE_URL_ENV_VAR not in os.environ and config.url.rstrip("/") == DEFAULT_CLICKHOUSE_URL


_BaseIntakeConfig = cast(Any, create_service_config_class("intake"))


class IntakeConfig(_BaseIntakeConfig):
    """
    Configuration for the Intake service.

    Environment variables use the NHX_INTAKE_ prefix.
    """

    clickhouse_config: ClickHouseConfig = Field(
        default_factory=ClickHouseConfig,
        description="ClickHouse connection settings for Intake spans storage.",
    )
    otlp_max_body_bytes: int = Field(
        default=5 * 1024 * 1024,
        ge=1024,
        description="Maximum accepted body size for OTLP ingest requests, in bytes.",
    )
    atif_max_subagent_depth: int = Field(
        default=DEFAULT_ATIF_MAX_SUBAGENT_DEPTH,
        ge=1,
        le=MAX_ATIF_MAX_SUBAGENT_DEPTH,
        description="Maximum number of trajectory levels accepted for recursive ATIF subagents.",
    )
    denormalization_interval_seconds: float = Field(
        default=60.0,
        gt=0,
        description=(
            "How often the background worker drains the dirty set and denormalizes the distinct "
            "agent/model name fields from ClickHouse onto Evaluation entities. Bounds how stale those "
            "fields can be after ingest; ingest never blocks on it."
        ),
    )
