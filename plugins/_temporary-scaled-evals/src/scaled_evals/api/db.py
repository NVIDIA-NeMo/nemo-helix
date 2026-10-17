# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from threading import Lock
from typing import Annotated

import psycopg
from fastapi import Depends, HTTPException
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from scaled_evals.api.agent_bundle_store import AgentBundleStore
from scaled_evals.api.benchmark_import_store import BenchmarkImportStore
from scaled_evals.api.benchmark_run_store import BenchmarkRunStore
from scaled_evals.api.benchmark_store import BenchmarkStore
from scaled_evals.api.config_profile_store import ConfigProfileStore
from scaled_evals.api.evaluation_store import EvaluationStore
from scaled_evals.api.repositories.benchmark_archive_repository import BenchmarkArchiveRepository
from scaled_evals.api.repositories.ops_repository import OperationsRepository
from scaled_evals.api.settings import settings
from scaled_evals.api.task_store import TaskStore

_pool: ConnectionPool | None = None
_pool_lock = Lock()


def open_pool(*, wait: bool = False) -> ConnectionPool:
    """Create and start the process-local API pool once.

    Startup is non-blocking by default so the process can serve liveness while
    Postgres is unavailable. Connection checkout and readiness probes retain
    their own bounded timeouts.
    """
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                conninfo=settings.resolved_database_url(),
                kwargs={"row_factory": dict_row},
                min_size=settings.database_pool_min_size,
                max_size=settings.database_pool_max_size,
                timeout=settings.database_pool_timeout_seconds,
                open=False,
                name="scaled-evals-api",
            )
            _pool.open(wait=wait)
        return _pool


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def pooled_connection(*, timeout: float | None = None) -> Iterator[psycopg.Connection]:
    pool = open_pool()
    with pool.connection(timeout=timeout) as conn:
        yield conn


class Database:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    @property
    def agent_bundles(self) -> AgentBundleStore:
        return AgentBundleStore()

    @property
    def tasks(self) -> TaskStore:
        return TaskStore()

    @property
    def benchmarks(self) -> BenchmarkStore:
        return BenchmarkStore()

    @property
    def benchmark_imports(self) -> BenchmarkImportStore:
        return BenchmarkImportStore()

    @property
    def benchmark_archives(self) -> BenchmarkArchiveRepository:
        return BenchmarkArchiveRepository(self.conn)

    @property
    def benchmark_runs(self) -> BenchmarkRunStore:
        return BenchmarkRunStore()

    @property
    def config_profiles(self) -> ConfigProfileStore:
        return ConfigProfileStore()

    @property
    def evaluations(self) -> EvaluationStore:
        return EvaluationStore()

    @property
    def ops(self) -> OperationsRepository:
        return OperationsRepository(self.conn)

    def commit(self) -> None:
        self.conn.commit()


def get_conn() -> Iterator[psycopg.Connection]:
    """Yield a pooled connection, or fail the request as 503 rather than 500.

    Every `/v1` route reaches the database through this dependency, so an
    unreachable Postgres surfaces once here instead of as an unhandled
    PoolTimeout per route. That distinction matters diagnostically: a 500 with a
    pool traceback reads like a bug in whichever endpoint happened to be called,
    which is exactly how a Postgres outage was first misread as a broken archive
    route.
    """
    try:
        with pooled_connection() as conn:
            yield conn
    except PoolTimeout as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "database_unavailable", "message": "database is unavailable"},
        ) from exc


def get_db(conn: Annotated[psycopg.Connection, Depends(get_conn)]) -> Database:
    return Database(conn)


@contextmanager
def stream_database() -> Iterator[Database]:
    """Check out a database connection for one short SSE poll only."""
    with pooled_connection(timeout=settings.database_pool_timeout_seconds) as conn:
        yield Database(conn)


def get_stream_database_factory() -> Callable[[], AbstractContextManager[Database]]:
    """Return a factory without checking out a connection for the request lifetime."""
    return stream_database
