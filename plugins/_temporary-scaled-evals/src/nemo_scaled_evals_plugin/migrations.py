# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the vendored scaled-evals SQL to the plugin's own Postgres.

This replaces the three shell+psql call sites the standalone repo used (compose
entrypoint, `schema-migrate` compose service, Helm `schema-init` Job), none of
which came across with the vendored code. Semantics are copied from that Job:
load `db/schema` **only** on a fresh database, then apply every `db/migrations`
file in filename order, every time.

Re-applying the whole migration set on each boot looks wasteful but is the
upstream contract: the set has no version ledger, and each file is written to be
re-appliable (`ADD COLUMN IF NOT EXISTS`, `to_regclass` early-exit, and so on).
Standalone CI proves it by applying the set twice; `tests/test_migrations.py`
keeps that guarantee here.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import psycopg
import scaled_evals
from psycopg.sql import SQL, Identifier

logger = logging.getLogger(__name__)

_CONNECT_RETRY_INTERVAL_SECONDS = 2.0

# Serializes DDL across concurrent API replicas and any operator running the
# console script.
_MIGRATION_LOCK_ID = 1936024439

# `evaluations` exists in every released schema, so it survives the
# benchmark->task rename in migration 008 and is a rename-stable "already
# initialized" sentinel. Unqualified, so it resolves through the search_path
# that `resolved_database_url()` sets rather than assuming `public`.
_SCHEMA_SENTINEL = "evaluations"


def sql_root() -> Path:
    """Locate the vendored `db/` tree in either an installed or a source layout."""
    package_root = Path(scaled_evals.__file__).resolve().parent
    candidates = (
        # Wheel: pyproject force-includes `db` as `scaled_evals/db`.
        package_root / "db",
        # Source checkout: src/scaled_evals/ -> src/ -> plugin root.
        package_root.parent.parent / "db",
    )
    for root in candidates:
        if (root / "schema").is_dir() and (root / "migrations").is_dir():
            return root
    raise FileNotFoundError(f"scaled-evals SQL not found; looked in {[str(c) for c in candidates]}")


def _sql_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.sql"))


def _run_file(conn: psycopg.Connection, path: Path) -> None:
    logger.debug("scaled-evals: applying %s", path.name)
    # No parameters, so psycopg uses the simple query protocol and accepts the
    # multi-statement file the way `psql -f` would. Bytes, because the str overload
    # is typed LiteralString to discourage exactly the dynamic SQL we need here.
    conn.execute(path.read_bytes())


def _ensure_schema(conn: psycopg.Connection, schema: str) -> None:
    """Create the target schema and prove the connection actually lands in it.

    Postgres silently drops unknown entries from `search_path`, so a missing
    schema would not error — it would quietly resolve every unqualified table in
    the vendored SQL to `public` and scatter 19 tables through the platform's
    own schema. Verify instead of assuming.
    """
    # `as_bytes` for the same reason as _run_file: the str overload is typed
    # LiteralString. Identifier() still does the quoting.
    statement = SQL("CREATE SCHEMA IF NOT EXISTS {}").format(Identifier(schema))
    conn.execute(statement.as_bytes(conn))
    row = conn.execute("SELECT current_schema()").fetchone()
    current = row[0] if row else None
    if current != schema:
        raise RuntimeError(
            f"expected to write into schema {schema!r} but the connection resolves to "
            f"{current!r}; the DSN is missing its search_path option"
        )


def _connect(dsn: str, wait_seconds: float) -> psycopg.Connection:
    """Connect, tolerating a Postgres that has not finished starting.

    Nothing orders startup for us. Under Kubernetes the API and both workers are
    scheduled in parallel with the database, and a fresh volume adds `initdb` to
    its critical path, so losing this race is the normal case rather than the
    exceptional one. Migrations run once per boot and are never retried, so a
    connection refused here leaves the schema absent for the process lifetime:
    the API serves 503 from `/v1/readyz` and the workers fault on missing tables.

    The wait is bounded because a real misconfiguration cannot be told apart from
    a slow start. psycopg reports no SQLSTATE at connect time for a refused
    connection, a rejected password or a missing database alike, so waiting on
    the wrong host would otherwise never end.
    """
    deadline = time.monotonic() + wait_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            return psycopg.connect(dsn, autocommit=True)
        except psycopg.OperationalError:
            if time.monotonic() >= deadline:
                raise
            logger.info(
                "scaled-evals: Postgres unreachable (attempt %s); retrying for up to %.0fs",
                attempt,
                wait_seconds,
            )
            time.sleep(_CONNECT_RETRY_INTERVAL_SECONDS)


def apply_sql(dsn: str, *, schema: str | None = None, wait_seconds: float = 0.0) -> tuple[int, int]:
    """Bring the scaled-evals database up to date.

    Args:
        dsn: libpq connection string for the scaled-evals database. Must already
            carry the `search_path` option when `schema` is given — use
            `settings.resolved_database_url()`.
        schema: schema to create and assert the connection resolves to. `None`
            keeps whatever the DSN selects (i.e. `public`).
        wait_seconds: how long to keep retrying a database that is not accepting
            connections yet. `0.0` attempts once.

    Returns:
        tuple[int, int]: schema files loaded (0 on an existing database) and
        migration files applied.

    """
    root = sql_root()
    migration_files = _sql_files(root / "migrations")

    # Own connection rather than the API pool: this needs autocommit and the
    # default tuple row factory, and it runs before the pool is opened.
    with _connect(dsn, wait_seconds) as conn:
        # Session-scoped lock, so it is released even if this process is killed.
        conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
        if schema:
            _ensure_schema(conn, schema)
        sentinel = conn.execute("SELECT to_regclass(%s)", (_SCHEMA_SENTINEL,)).fetchone()
        is_fresh = sentinel is None or sentinel[0] is None
        schema_files = _sql_files(root / "schema") if is_fresh else []

        if schema_files:
            # All or nothing. The sentinel table is created by 04_evaluations.sql, in the
            # middle of the sequence, so a fresh install that commits part way leaves the
            # sentinel present and `is_fresh` false on the next boot: the remaining files
            # are skipped forever, and they use bare CREATE TABLE, so nothing repairs it.
            conn.autocommit = False
            try:
                for path in schema_files:
                    _run_file(conn, path)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.autocommit = True
        for path in migration_files:
            _run_file(conn, path)

    return len(schema_files), len(migration_files)


def main() -> None:
    """Run the applier as a one-shot command (compose service, k8s Job, runbook)."""
    from scaled_evals.api.settings import settings

    parser = argparse.ArgumentParser(description="Apply scaled-evals schema and migrations.")
    parser.add_argument("--dsn", default=None, help="Override SCALED_EVALS_DATABASE_URL.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    schema_count, migration_count = apply_sql(
        args.dsn or settings.resolved_database_url(),
        schema=settings.database_schema,
        wait_seconds=settings.migration_wait_seconds,
    )
    logger.info(
        "scaled-evals database ready (%s schema files, %s migrations)",
        schema_count or "0 - already initialized",
        migration_count,
    )
