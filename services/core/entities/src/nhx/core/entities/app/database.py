# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database engine and session management for Entities service."""

from nhx.core.entities.config import EntitiesConfig
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_async_engine_for_entities(
    config: EntitiesConfig,
    pool_pre_ping: bool = True,
    **kwargs,
) -> AsyncEngine:
    """Create an async SQLAlchemy engine for the entities database.

    For SQLite, this configures:
    - timeout from database_config.connect_timeout_seconds: Wait for locks (default 30s)
    - WAL mode: Enables concurrent reads during writes

    For PostgreSQL (asyncpg), this configures:
    - timeout from database_config.connect_timeout_seconds: Connection timeout (default 30s).

    Args:
        config: Entities service configuration
        database_url: Database URL. If None, uses get_async_database_url(config)
        pool_pre_ping: Enable connection health checks (default True)
        **kwargs: Additional arguments passed to create_async_engine

    Returns:
        AsyncEngine configured for the database type
    """
    database_url = config.get_async_sqlalchemy_url()
    echo = config.database_config.echo
    connect_timeout = config.database_config.connect_timeout_seconds

    engine_kwargs = {"pool_pre_ping": pool_pre_ping, "echo": echo, **kwargs}

    if "sqlite" in database_url:
        # SQLite: timeout for lock acquisition (seconds)
        engine_kwargs["connect_args"] = {"timeout": connect_timeout}
    elif "postgresql" in database_url:
        # PostgreSQL (asyncpg): connection timeout in seconds
        engine_kwargs["connect_args"] = {"timeout": connect_timeout}

    engine = create_async_engine(database_url, **engine_kwargs)

    if "sqlite" in database_url:
        # Enable WAL mode for better concurrent access
        # This allows readers and writers to work simultaneously
        timeout_ms = connect_timeout * 1000

        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={timeout_ms}")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine
