# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database engine and session management for Entities service."""

import asyncio

import aiosqlite
from nhx.core.entities.config import EntitiesConfig
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_async_engine_for_entities(
    config: EntitiesConfig,
    pool_pre_ping: bool | None = None,
    **kwargs,
) -> AsyncEngine:
    """Create an async SQLAlchemy engine for the entities database.

    For SQLite, this configures:
    - timeout from database_config.connect_timeout_seconds: Wait for locks (default 30s)
    - WAL mode: Enables concurrent reads during writes

    For PostgreSQL (asyncpg), this configures:
    - timeout from database_config.connect_timeout_seconds: Connection timeout (default 30s).
    - pool_size from database_config.connections_limit: connections kept open and reused. SQLAlchemy's
      default of 5 forces a burst of concurrent requests to open (and then close) overflow connections.
    - pool_recycle from database_config.pool_recycle_seconds: replace connections older than this, so one
      that idled past a proxy or server timeout is not reused.

    Args:
        config: Entities service configuration
        pool_pre_ping: Health-check each pooled connection on checkout. If None, uses
            database_config.pool_pre_ping (default off: a ping costs about three database round trips)
        **kwargs: Additional arguments passed to create_async_engine

    Returns:
        AsyncEngine configured for the database type
    """
    database_url = config.get_async_sqlalchemy_url()
    echo = config.database_config.echo
    connect_timeout = config.database_config.connect_timeout_seconds

    if pool_pre_ping is None:
        pool_pre_ping = config.database_config.pool_pre_ping

    engine_kwargs = {"pool_pre_ping": pool_pre_ping, "echo": echo, **kwargs}

    if "sqlite" in database_url:
        # SQLite: timeout for lock acquisition (seconds)
        engine_kwargs["connect_args"] = {"timeout": connect_timeout}
    elif "postgresql" in database_url:
        # PostgreSQL (asyncpg): connection timeout in seconds
        engine_kwargs["connect_args"] = {"timeout": connect_timeout}
        engine_kwargs.setdefault("pool_size", config.database_config.connections_limit)
        if config.database_config.pool_recycle_seconds > 0:
            engine_kwargs.setdefault("pool_recycle", config.database_config.pool_recycle_seconds)

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
            _stop_aiosqlite_worker_once(dbapi_conn.driver_connection)

    return engine


def _stop_aiosqlite_worker_once(connection: aiosqlite.Connection) -> None:
    """Make repeat ``stop()`` calls on an aiosqlite connection no-ops.

    aiosqlite's worker thread exits after the first stop request it handles, so a second
    ``stop()`` returns a future that never resolves. SQLAlchemy can trigger exactly that when a
    connection is invalidated during shutdown: its shielded graceful ``close()`` and its
    force-close fallback each call ``stop()``, and if the force-close lands first the close task
    waits forever. ``close()`` treats a ``None`` future as already stopped.
    """
    stop = connection.stop
    stopped = False

    def stop_once() -> asyncio.Future | None:
        nonlocal stopped
        if stopped:
            return None
        stopped = True
        return stop()

    connection.stop = stop_once  # ty: ignore[invalid-assignment]
