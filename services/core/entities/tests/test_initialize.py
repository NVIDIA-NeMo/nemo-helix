# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for entities initialization helpers."""

import pytest
from nhx.common.config.base import DatabaseConfig
from nhx.core.entities.app.database import create_async_engine_for_entities
from nhx.core.entities.app.repository import dispose_async_engine, initialize_async_engine
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.initialize import _engine_url_for_alembic


def test_engine_url_rendering_for_alembic_uses_unmasked_password(monkeypatch):
    """Given DB config, Alembic should receive non-masked URL string."""
    monkeypatch.setenv("DATABASE_DIALECT", "postgresql")
    monkeypatch.setenv("DATABASE_USER", "nhx")
    monkeypatch.setenv("DATABASE_PASSWORD", "supersecret")
    monkeypatch.setenv("DATABASE_NAME", "nhx")
    monkeypatch.setenv("DATABASE_HOST", "db.example")
    monkeypatch.setenv("DATABASE_PORT", "5432")

    config = EntitiesConfig()
    engine = create_async_engine_for_entities(config)

    # SQLAlchemy masks password in URL.__str__()
    assert "***" in str(engine.url)
    assert "supersecret" not in str(engine.url)
    # Alembic path must use real password so DB auth succeeds
    assert "supersecret" in _engine_url_for_alembic(engine)


async def test_a_second_engine_for_a_different_database_is_refused_until_the_first_is_disposed(tmp_path):
    """A silently shared store is the failure mode: a nested test context would read the outer one's rows."""
    first = EntitiesConfig(database_config=DatabaseConfig(url=f"sqlite:///{tmp_path}/first.db"))
    second = EntitiesConfig(database_config=DatabaseConfig(url=f"sqlite:///{tmp_path}/second.db"))
    await dispose_async_engine()
    try:
        await initialize_async_engine(first)
        await initialize_async_engine(first)  # same database: idempotent
        with pytest.raises(RuntimeError, match="already initialized for a different database"):
            await initialize_async_engine(second)
        await dispose_async_engine()
        await initialize_async_engine(second)
    finally:
        await dispose_async_engine()
