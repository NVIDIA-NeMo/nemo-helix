# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Entity Store service."""

from nhx.common.config import create_service_config_class, internal_field
from nhx.common.config.base import DatabaseConfig
from pydantic import Field


class EntitiesConfig(create_service_config_class("entities")):
    """Configuration for the Entities Service."""

    run_migrations: bool = Field(
        default=True,
        description="Run Alembic migrations (upgrade head) at startup.",
    )

    database_config: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="",
    )

    workspace_cleanup_interval: int = Field(
        default=10,
        description="Interval in seconds to run the workspace cleanup routine.",
    )

    db_health_check_interval_seconds: float = Field(
        default=10.0,
        description="Interval in seconds for the background database health ping (used for readiness).",
    )

    principal_bindings_cache_enabled: bool = Field(
        default=True,
        description=(
            "Enable in-process caching of per-principal role-binding list queries. "
            "Set via NHX_ENTITIES_PRINCIPAL_BINDINGS_CACHE_ENABLED."
        ),
    )
    principal_bindings_cache_ttl_sec: float = Field(
        default=60.0,
        description=(
            "TTL in seconds for principal role-binding cache entries (NHX_ENTITIES_PRINCIPAL_BINDINGS_CACHE_TTL_SEC)."
        ),
    )
    principal_bindings_cache_max_size: int = Field(
        default=256,
        ge=1,
        description=(
            "Max distinct principal keys in the role-binding cache before LRU eviction "
            "(NHX_ENTITIES_PRINCIPAL_BINDINGS_CACHE_MAX_SIZE)."
        ),
    )

    startup_retry_timeout_seconds: int = internal_field(
        default=120,
        description="Maximum seconds to retry database init at startup (e.g. Postgres still booting).",
    )
    startup_retry_interval_seconds: int = internal_field(
        default=2,
        description="Seconds between database init retry attempts during startup.",
    )
    startup_retry_log_interval_seconds: int = internal_field(
        default=10,
        description="Minimum seconds between warning logs while retrying database init at startup.",
    )

    def get_async_sqlalchemy_url(self) -> str:
        """Get the database URL with async driver suffix.

        Transforms:
        - postgresql:// -> postgresql+asyncpg://
        - sqlite:// -> sqlite+aiosqlite://

        Idempotent - won't double-add async suffix if already present.

        Args:
            config: Entities service configuration

        Returns:
            Async-compatible database URL string
        """
        url = self.database_config.sqlalchemy_database_url()

        # Only add async suffix if not already present
        if "postgresql://" in url and "postgresql+asyncpg://" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://")
        if "sqlite://" in url and "sqlite+aiosqlite://" not in url:
            url = url.replace("sqlite://", "sqlite+aiosqlite://")

        return url
