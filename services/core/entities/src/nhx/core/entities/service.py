# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entities service implementation."""

import asyncio
import logging
from typing import List

from nhx.common.service import RouterConfig, Service
from nhx.core.entities.app.repository import dispose_async_engine, initialize_async_engine, ping_database
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.initialize import initialize_database

logger = logging.getLogger(__name__)


class EntitiesService(Service[EntitiesConfig]):
    """Entities service for NeMo Helix."""

    dependencies: list[str] = []

    def __init__(self):
        """Initialize the entities service."""
        super().__init__(name="entities", module_name="nhx.core.entities")
        self._db_conn_healthy: bool = False
        self._db_ping_task: asyncio.Task[None] | None = None

    @property
    def title(self) -> str:
        return "NeMo Helix Entities Microservice"

    @property
    def description(self) -> str:
        return "Generic entity storage service with schema-agnostic design."

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the entities service."""
        from nhx.core.entities.api.v2.entities import router as entities_router
        from nhx.core.entities.api.v2.projects import router as projects_router
        from nhx.core.entities.api.v2.workspaces import router as workspaces_router

        return [
            RouterConfig(
                entities_router,
                tag="Entity Store",
                description="Operations related to entities",
            ),
            RouterConfig(
                projects_router,
                tag="Entity Store",
                description="Operations related to projects",
            ),
            RouterConfig(
                workspaces_router,
                tag="Entity Store",
                description="Operations related to workspaces",
            ),
        ]

    async def _db_health_loop(self) -> None:
        """Background loop that pings the database on an interval and updates _db_conn_healthy."""
        if not self.service_config:
            logger.warning("Entities config not set, skipping database health loop")
            return
        logger.debug("Starting database health check loop")
        interval = self.service_config.db_health_check_interval_seconds
        while True:
            try:
                self._db_conn_healthy = await ping_database()
                if self._db_conn_healthy:
                    logger.debug("Database health check passed")
                else:
                    logger.warning("Database health check failed")
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logger.debug("Database health check loop cancelled")
                break
            except Exception as e:
                logger.error("Database health ping error: %s", e)
                self._db_conn_healthy = False

    async def on_startup(self) -> None:
        """Initialize database and dependencies on startup.

        Retries for up to ``startup_retry_timeout_seconds`` if the database is not yet
        ready (e.g. Postgres pod still booting). While retrying, logs a warning at
        most every ``startup_retry_log_interval_seconds`` (not on every attempt). On success
        sets _db_conn_healthy and starts the background health ping loop. On failure
        after retries, raises so is_ready() remains False.
        """
        if not self.service_config:
            logger.debug("Entities config not set, skipping database initialization")
            return

        cfg = self.service_config
        loop = asyncio.get_running_loop()
        deadline = loop.time() + cfg.startup_retry_timeout_seconds
        last_exc: Exception | None = None
        next_warning_at = loop.time() + cfg.startup_retry_log_interval_seconds

        # Retry database setup and initialization until startup_retry_timeout_seconds.
        # If successful, set _db_conn_healthy to True and start the background health ping loop.
        # If unsuccessful, set _db_conn_healthy to False and raise an exception.
        self._db_ping_task = asyncio.create_task(self._db_health_loop())
        while loop.time() < deadline:
            try:
                await initialize_async_engine(cfg)
                logger.debug("Async engine initialized")
                await initialize_database(run_migrations=cfg.run_migrations)
                self._db_conn_healthy = True
                return
            except Exception as e:
                last_exc = e
                now = loop.time()
                logger.debug(
                    "Database initialization failed (will retry): %s",
                    e,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                if now >= next_warning_at:
                    logger.warning(
                        "Database not ready yet (will retry every %ss): %s",
                        cfg.startup_retry_interval_seconds,
                        e,
                    )
                    next_warning_at = now + cfg.startup_retry_log_interval_seconds
            await asyncio.sleep(cfg.startup_retry_interval_seconds)

        logger.exception(
            "Database initialization did not succeed after %s seconds",
            cfg.startup_retry_timeout_seconds,
        )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Database initialization timed out after {cfg.startup_retry_timeout_seconds} seconds")

    async def startup(self) -> None:
        """Do nothing on startup. Database connectivity is checked in on_startup(), and health checks are performed in the background."""
        pass

    async def is_ready(self) -> bool:
        """Return True if the database connection is healthy, False otherwise."""
        return self._db_conn_healthy

    async def on_shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Entities service shutting down")
        if self._db_ping_task is not None and not self._db_ping_task.done():
            self._db_ping_task.cancel()
            try:
                await self._db_ping_task
            except asyncio.CancelledError:
                pass
            self._db_ping_task = None

        await dispose_async_engine()
        await super().on_shutdown()
