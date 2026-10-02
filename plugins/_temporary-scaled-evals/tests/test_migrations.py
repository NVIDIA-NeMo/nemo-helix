# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cover the startup migration applier."""

from __future__ import annotations

import asyncio
import os
import uuid
from unittest import mock
from urllib.parse import quote, urlsplit, urlunsplit

import pytest
from cryptography.fernet import Fernet

# This plugin is absent from `enabled-plugins`, so a default sync leaves it and its database
# driver uninstalled and the repo-wide test run still sweeps this directory. Skip rather than
# error there; the job that owns these tests installs the `scaled-evals` group first.
try:
    import psycopg
    from nemo_scaled_evals_plugin import migrations
    from nemo_scaled_evals_plugin.migrations import (
        _CONNECT_RETRY_INTERVAL_SECONDS,
        _ensure_schema,
        apply_sql,
        sql_root,
    )
    from nemo_scaled_evals_plugin.service import ScaledEvalsService
    from psycopg.sql import SQL, Identifier
    from scaled_evals.api.repositories.ops_repository import OperationsRepository
    from scaled_evals.api.settings import Settings, settings
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

# Set to a throwaway database to run the real-Postgres check:
#   docker run -d --name se-pg -e POSTGRES_PASSWORD=pw -p 5433:5432 postgres:16
#   SCALED_EVALS_TEST_DATABASE_URL=postgresql://postgres:<password>@localhost:5433/postgres
TEST_DSN_ENV = "SCALED_EVALS_TEST_DATABASE_URL"


def _fixture_dsn(role: str, host: str, database: str, password: str | None = None) -> str:
    """Build a throwaway DSN, defaulting the password to the role name.

    Assembled rather than written out: a literal ``user:password@host`` in-tree reads as
    a live credential to secret scanners even when, as here, it points nowhere.
    """
    return f"postgresql://{role}:{password or role}@{host}:5432/{database}"


def test_sql_is_discoverable_and_ordered() -> None:
    root = sql_root()
    schema = sorted(p.name for p in (root / "schema").glob("*.sql"))
    migrations = [p.name for p in sorted((root / "migrations").glob("*.sql"))]

    assert schema, "no schema files found"
    assert len(migrations) > 40, migrations

    # Filename order is the apply order, so a mis-sorted glob reorders DDL.
    assert migrations == sorted(migrations)
    assert migrations[0].startswith("001_")


@pytest.mark.parametrize("key", [None, "not-a-fernet-key"], ids=["missing", "invalid"])
def test_startup_survives_unusable_settings(key: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken plugin config must degrade the plugin, not abort platform startup.

    Drives the real `on_startup`, not just `_apply_migrations`: an earlier version
    guarded only the migration and left `open_pool` to raise straight through the
    lifespan, which uvicorn turns into "Application startup failed. Exiting."
    """
    # Drop the memoized Settings so the config below is what gets validated.
    monkeypatch.setattr(settings, "_instance", None)
    if key is None:
        monkeypatch.delenv("CREDENTIALS_ENCRYPTION_KEY", raising=False)
    else:
        monkeypatch.setenv("CREDENTIALS_ENCRYPTION_KEY", key)

    asyncio.run(ScaledEvalsService().on_startup())


def test_migrations_wait_out_a_slow_postgres_but_give_up_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A database that is still starting must not cost us the schema for the process lifetime.

    Migrations run once per boot, so the refused connection that used to happen
    here left every table missing until someone restarted the API by hand — the
    exact failure seen on GKE, where a fresh volume put `initdb` on Postgres'
    critical path while all three pods started in parallel.

    Runs on a fake clock: real sleeps would make this slow, and the point being
    asserted is the bound, not the wall time.
    """
    clock = 0.0

    def fake_sleep(seconds: float) -> None:
        nonlocal clock
        clock += seconds

    monkeypatch.setattr(migrations.time, "monotonic", lambda: clock)
    monkeypatch.setattr(migrations.time, "sleep", fake_sleep)

    attempts = 0

    def connect(_dsn: str, **_kwargs: object) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise psycopg.OperationalError("connection refused")
        return "connection"

    monkeypatch.setattr(psycopg, "connect", connect)
    assert migrations._connect("dsn", 30.0) == "connection"
    assert attempts == 3, "gave up before Postgres finished starting"

    # Bounded, because a wrong host, a rejected password and a slow start are all
    # indistinguishable here — psycopg reports no SQLSTATE for any of them.
    def never_connects(_dsn: str, **_kwargs: object) -> str:
        raise psycopg.OperationalError("connection refused")

    clock, attempts = 0.0, 0
    monkeypatch.setattr(psycopg, "connect", never_connects)
    with pytest.raises(psycopg.OperationalError):
        migrations._connect("dsn", 30.0)
    assert clock <= 30.0 + _CONNECT_RETRY_INTERVAL_SECONDS, clock

    # Default stays single-attempt, so callers that want fail-fast still get it.
    clock = 0.0
    with pytest.raises(psycopg.OperationalError):
        migrations._connect("dsn", 0.0)
    assert clock == 0.0, "waited despite wait_seconds=0"


def _dsn_for(base_dsn: str, schema: str) -> str:
    """Build the DSN through the real settings path, not a copy of its logic.

    Goes via the environment because both fields are alias-only, which is also
    the wiring a deployment uses.
    """
    env = {
        "SCALED_EVALS_DATABASE_URL": base_dsn,
        "SCALED_EVALS_DATABASE_SCHEMA": schema,
        "DATABASE_SSL_MODE": "disable",
        "CREDENTIALS_ENCRYPTION_KEY": Fernet.generate_key().decode(),
    }
    with mock.patch.dict(os.environ, env):
        return Settings().resolved_database_url()  # ty: ignore[missing-argument]


def test_database_url_never_adopts_the_platform_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scaled-evals runs on its own Postgres; the platform's is never inherited.

    The platform injects its own database config into this process, so adoption
    would happen silently — running these migrations and the claim queues inside
    the platform's database.
    """
    monkeypatch.setenv("CREDENTIALS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.delenv("SCALED_EVALS_DATABASE_URL", raising=False)
    # A platform database that is perfectly usable, so the only thing keeping us
    # off it is that we do not look.
    monkeypatch.setattr(
        "nhx.common.config.DatabaseConfig.sqlalchemy_database_url",
        lambda _self: _fixture_dsn("nhx", "platform-db", "nemo_helix"),
    )
    monkeypatch.setenv("DATABASE_HOST", "platform-db")

    unset = Settings().resolved_database_url()  # ty: ignore[missing-argument]
    assert "platform-db" not in unset, unset
    assert unset.startswith(_fixture_dsn("scaled_evals", "localhost", "scaled_evals")), unset

    monkeypatch.setenv("SCALED_EVALS_DATABASE_URL", _fixture_dsn("me", "own-db", "scaled_evals", "pw"))
    assert "own-db:5432/scaled_evals" in Settings().resolved_database_url()  # ty: ignore[missing-argument]


def test_dsn_carries_search_path_and_applier_refuses_a_public_fallback() -> None:
    dsn = _dsn_for(_fixture_dsn("u", "h", "nemo_helix", "p"), "scaled_evals")
    assert "sslmode=disable" in dsn
    assert f"options={quote('-c search_path=scaled_evals,public', safe='')}" in dsn

    class _Conn:
        """Stands in for a connection whose search_path never took effect."""

        # psycopg reads this off the adapt context when rendering composed SQL.
        connection = None

        def execute(self, *_args: object, **_kwargs: object) -> _Conn:
            return self

        def fetchone(self) -> tuple[str]:
            return ("public",)

    # Postgres drops unknown schemas from search_path silently, so without this
    # guard the 19 tables land in the platform's own schema.
    with pytest.raises(RuntimeError, match="search_path"):
        _ensure_schema(_Conn(), "scaled_evals")  # ty: ignore[invalid-argument-type]


@pytest.mark.skipif(not os.environ.get(TEST_DSN_ENV), reason=f"{TEST_DSN_ENV} not set")
def test_applies_to_a_fresh_database_and_is_reappliable() -> None:
    admin_dsn = os.environ[TEST_DSN_ENV]
    scratch = f"se_migrate_test_{uuid.uuid4().hex[:12]}"
    parsed = urlsplit(admin_dsn)
    dsn = urlunsplit(parsed._replace(path=f"/{scratch}"))

    # Own database per run, so "fresh load" is actually assertable and reruns work.
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(SQL("CREATE DATABASE {}").format(Identifier(scratch)))
    try:
        _assert_fresh_then_reappliable(dsn)
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            conn.execute(SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(Identifier(scratch)))


def _assert_fresh_then_reappliable(raw_dsn: str) -> None:
    dsn = _dsn_for(raw_dsn, "scaled_evals")
    schema_count, migration_count = apply_sql(dsn, schema="scaled_evals")
    assert schema_count > 0, "fresh database should have loaded the schema"
    assert migration_count > 40

    # The migration set has no version ledger, so every boot re-applies all of
    # it. A file that is not re-appliable breaks startup on the second deploy.
    reapplied_schema, reapplied_migrations = apply_sql(dsn, schema="scaled_evals")
    assert reapplied_schema == 0, "schema reloaded over an initialized database"
    assert reapplied_migrations == migration_count

    with psycopg.connect(dsn) as conn:
        # Same probe /v1/readyz uses, so passing here means readiness passes.
        OperationsRepository(conn).assert_schema_compatible()

        # The point of the schema: co-locating in the platform database must not
        # put a single scaled-evals table into the schema Alembic manages.
        counts = "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s"
        ours = conn.execute(counts, ("scaled_evals",)).fetchone()
        public = conn.execute(counts, ("public",)).fetchone()
        assert ours is not None and ours[0] > 15, ours
        assert public is not None and public[0] == 0, f"leaked into public: {public}"
