# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Platform HTTP surface for the vendored scaled-evals control plane."""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.service import ExceptionHandler, NemoService, RouterSpec
from nemo_scaled_evals_plugin.authz import scope, stamp_router_authz
from nemo_scaled_evals_plugin.migrations import apply_sql
from scaled_evals.api import s3
from scaled_evals.api.db import close_pool, open_pool
from scaled_evals.api.routers import (
    admin,
    agent_bundles,
    benchmark_imports,
    benchmark_runs,
    benchmarks,
    config_profiles,
    credentials,
    evaluations,
    ops,
    tasks,
    teams,
    users,
)
from scaled_evals.api.settings import settings

logger = logging.getLogger(__name__)

# Switchyard lease/publish router intentionally omitted from the plugin mount.
_V1_ROUTERS = (
    ops.router,
    tasks.router,
    benchmarks.router,
    benchmark_imports.router,
    benchmark_runs.router,
    credentials.router,
    config_profiles.router,
    evaluations.router,
    users.router,
    admin.router,
    agent_bundles.router,
    teams.router,
)


async def _redacted_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    """Return 422s without echoing the submitted request body.

    FastAPI's default handler includes the offending value in each error's
    ``input`` (and sometimes ``ctx``). On ``POST /v1/credentials`` the offending
    value is the whole submitted model, so a rejected write echoes the plaintext
    ``key``/``yaml`` back to the caller. Strip those two fields globally and keep
    the structural ``type``/``loc``/``msg`` that callers actually need.
    """
    errors = [{k: v for k, v in e.items() if k not in ("input", "ctx")} for e in exc.errors()]  # type: ignore[attr-defined]
    return JSONResponse(status_code=422, content={"detail": errors})


class ScaledEvalsService(NemoService):
    """Mount scaled-evals ``/v1`` routers under ``/apis/scaled-evals``."""

    name: ClassVar[str] = "scaled-evals"
    dependencies: ClassVar[list[str]] = ["jobs"]

    def get_routers(self) -> list[RouterSpec]:
        health = APIRouter()

        @health.get("/healthz")
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def healthz() -> dict[str, object]:
            return {
                "plugin": self.name,
                "status": "ok",
                "mode": "scaled-evals-cp",
            }

        specs: list[RouterSpec] = [
            RouterSpec(
                router=health,
                tag="Scaled Evals",
                description="Plugin health probe.",
            )
        ]
        for router in _V1_ROUTERS:
            stamp_router_authz(router)
            specs.append(
                RouterSpec(
                    router=router,
                    tag="Scaled Evals",
                    description="Scaled-evals control-plane routes.",
                    prefix="/v1",
                )
            )
        return specs

    def get_exception_handlers(self) -> dict[type[Exception], ExceptionHandler]:
        return {RequestValidationError: _redacted_validation_error}

    async def on_startup(self) -> None:
        """Bring the plugin up without ever aborting platform startup.

        Nothing here may raise. Anything that escapes propagates through the
        lifespan and uvicorn exits with "Application startup failed", taking the
        whole platform API down over one plugin's misconfiguration.
        """
        await asyncio.to_thread(self._apply_migrations)
        await asyncio.to_thread(self._ensure_object_store)
        try:
            # Resolves settings, so an unusable config raises here and not only
            # in _apply_migrations. `get_conn` reopens lazily per request, so a
            # failure now degrades the plugin rather than wedging it.
            open_pool(wait=False)
        except Exception:
            logger.exception("scaled-evals: connection pool unavailable; /v1/readyz will report failure")

    def _apply_migrations(self) -> None:
        """Bring the plugin's database up to date before the pool serves traffic.

        Failure is logged, not raised: a plugin must not take the whole platform
        API down because its own Postgres is unreachable. That matches
        ``open_pool(wait=False)``, and ``/v1/readyz`` already reports ``schema`` as
        a required check, so a database that never migrated fails readiness
        instead of serving 500s.
        """
        try:
            # Reading settings has to be inside the guard too: it resolves lazily,
            # so a bad CREDENTIALS_ENCRYPTION_KEY raises here rather than at import.
            if not settings.run_migrations:
                logger.info("scaled-evals: SCALED_EVALS_RUN_MIGRATIONS is off, skipping migrations")
                return
            schema_count, migration_count = apply_sql(
                settings.resolved_database_url(),
                schema=settings.database_schema,
                wait_seconds=settings.migration_wait_seconds,
            )
        except Exception:
            logger.exception("scaled-evals: database migration failed; /v1/readyz will report schema failure")
            return
        logger.info(
            "scaled-evals: database ready in schema %r (%s schema files, %s migrations)",
            settings.database_schema,
            schema_count,
            migration_count,
        )

    def _ensure_object_store(self) -> None:
        """Create the blob bucket before the API hands out upload URLs.

        Same contract as `_apply_migrations`: logged, never raised, with
        `/v1/readyz` reporting `object_store` as the authoritative check.
        """
        try:
            bucket = s3.ensure_bucket()
        except Exception:
            logger.exception("scaled-evals: object store unavailable; /v1/readyz will report object_store failure")
            return
        logger.info("scaled-evals: object store bucket %r ready", bucket)

    async def on_shutdown(self) -> None:
        close_pool()
