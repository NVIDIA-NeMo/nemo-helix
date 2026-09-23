# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL-parity safety net for in-memory filter evaluation.

Each parametrized FilterOperation tree is run against the same seeded rows on
every available SQL backend (SQLite always; PostgreSQL via testcontainers when
Docker is available) and through ``InMemoryFilterRepository``, all via the same
``op.apply(repo)`` front door:
  1. ``SQLAlchemyFilterRepository`` (the SQL source of truth).
  2. ``InMemoryFilterRepository`` over the ORM instances (the in-memory backend).

All three must select exactly the same set of row ids. Production runs
PostgreSQL while dev/CI default to SQLite, so the PostgreSQL leg is what catches
cross-backend divergences a SQLite-only run would miss (AIRCORE-749): the
non-numeric/null/absent numeric-cast (a hard error on PostgreSQL) and the
absent-JSON-key semantics for ``$eq None`` are both covered here.

``InMemoryFilterRepository`` is a native-Python evaluator, NOT a byte-for-byte
SQL mirror (see its class docstring). The remaining documented divergences not
yet reconciled in the SQL layer — int-vs-string ``$eq`` and boolean text
rendering under ``$like``/``$in``/``$nin`` — are excluded here and pinned in the
plugin's test_filter_matches.py.
"""

from importlib.util import find_spec

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.common.api.in_memory_filter import InMemoryFilterRepository
from nhx.core.entities.app.repository.sqlalchemy.filter import SQLAlchemyFilterRepository
from sqlalchemy import JSON, Column, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class FakeEntity(Base):
    __tablename__ = "fake_entity"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    data = Column(JSON)


# The compared JSON fields (score, tier, flag) are present on every row so the
# suite exercises agreeing semantics, not SQL's literal-"null" handling of
# absent keys (a documented native divergence pinned in the unit tests). A
# plain-column NULL (name on row 5) and an explicit/absent ``k`` for $eq-None
# coverage are the only nullable bits, and $eq agrees with SQL on both.
#
# Rows 6-9 carry SQL LIKE metacharacters (``_``/``%``) in ``name``/``data.tier``,
# each paired with a near-identical row that a wildcard interpretation would
# wrongly match. They pin the AIRCORE-749 contract that ``$like`` is a literal
# substring (``_``/``%`` are ordinary characters), agreeing with the in-memory
# backend. All keep score/tier/flag/tags present so no absent-key divergence is
# introduced into the existing cases.
#
# ``data.n2`` is numeric on one row, non-numeric text on another, explicit null on a
# third, and absent elsewhere — to pin the AIRCORE-749 numeric-comparison contract:
# only an actual JSON number participates in $gt/$lt; non-numeric/null/absent are
# no-match on both backends (and PostgreSQL must not error casting the text row).
SEED = [
    dict(
        id=1, name="llama", data={"score": 5, "tier": "free", "flag": True, "k": None, "n2": 5, "tags": ["red", "blue"]}
    ),
    dict(id=2, name="Llama-2", data={"score": 9, "tier": "pro", "flag": False, "n2": "notnum", "tags": ["red"]}),
    # "redish" is a deliberate prefix near-miss for "red" — quote-delimited matching must exclude it.
    dict(
        id=3,
        name="zephyr",
        data={"score": 10, "tier": "pro", "flag": True, "k": "v", "n2": None, "tags": ["green", "redish"]},
    ),
    dict(id=4, name="mistral", data={"score": 100, "tier": "enterprise", "flag": False, "tags": []}),
    dict(id=5, name=None, data={"score": 1, "tier": "free", "flag": False, "tags": ["blue"]}),
    # `_` is a single-char wildcard under LIKE; "prod_db" must not match "prodXdb".
    dict(id=6, name="prod_db", data={"score": 7, "tier": "free", "flag": True, "tags": []}),
    dict(id=7, name="prodXdb", data={"score": 8, "tier": "pro", "flag": False, "tags": []}),
    # `%` is a multi-char wildcard under LIKE; "50%off" must not match "50pctoff".
    # data.tier "a_c" must not match "axc" (exercises the JSON cast-to-text path).
    dict(id=8, name="50%off", data={"score": 11, "tier": "a_c", "flag": True, "tags": []}),
    dict(id=9, name="50pctoff", data={"score": 12, "tier": "axc", "flag": False, "tags": []}),
]


def _docker_available() -> bool:
    """Whether a usable Docker daemon is reachable (for the testcontainers PG leg)."""
    if find_spec("docker") is None:
        return False
    from docker.errors import DockerException

    import docker

    try:
        client = docker.from_env()
        try:
            client.ping()
        finally:
            client.close()
        return True
    except (DockerException, OSError):
        return False


@pytest.fixture(scope="session")
def _pg_engine():
    """Session-scoped PostgreSQL engine via testcontainers; skipped without Docker.

    Production runs PostgreSQL while dev/CI default to SQLite, so the SQLite-only
    leg cannot catch cross-backend divergences (AIRCORE-749). This runs the same
    cases against a real Postgres (matching the prod ``json`` column type).
    """
    if not _docker_available():
        pytest.skip("Docker unavailable; skipping PostgreSQL parity leg")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16") as pg:
        engine = create_engine(pg.get_connection_url())
        try:
            yield engine
        finally:
            engine.dispose()


@pytest.fixture(params=["sqlite", "postgres"])
def db(request):
    """Seed FakeEntity in SQLite and (when Docker is available) PostgreSQL.

    The same cases run against both backends so SQL↔SQL and SQL↔in-memory
    divergences are caught. The table is recreated per test so the session-scoped
    Postgres engine stays clean between cases.
    """
    if request.param == "sqlite":
        engine = create_engine("sqlite:///:memory:")
    else:
        engine = request.getfixturevalue("_pg_engine")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            session.add_all([FakeEntity(**row) for row in SEED])
            session.commit()
            yield session
    finally:
        Base.metadata.drop_all(engine)


def C(operator, field, value):
    return ComparisonOperation(operator=operator, field=field, value=value)


def AND(*ops):
    return LogicalOperation(operator=FilterOperator.AND, operations=list(ops))


def OR(*ops):
    return LogicalOperation(operator=FilterOperator.OR, operations=list(ops))


def NOT(op):
    return LogicalOperation(operator=FilterOperator.NOT, operations=[op])


# (label, FilterOperation tree)
CASES = [
    ("eq_name_hit", C(FilterOperator.EQ, "name", "llama")),
    ("eq_name_none", C(FilterOperator.EQ, "name", None)),
    ("eq_data_tier", C(FilterOperator.EQ, "data.tier", "pro")),
    ("eq_data_score_int", C(FilterOperator.EQ, "data.score", 5)),
    ("eq_data_flag_true", C(FilterOperator.EQ, "data.flag", True)),
    ("eq_data_flag_false", C(FilterOperator.EQ, "data.flag", False)),
    ("eq_data_k_none", C(FilterOperator.EQ, "data.k", None)),
    ("like_name", C(FilterOperator.LIKE, "name", "llama")),
    ("like_name_lower", C(FilterOperator.LIKE, "name", "LAMA")),
    ("like_data_tier", C(FilterOperator.LIKE, "data.tier", "pr")),
    ("like_data_miss", C(FilterOperator.LIKE, "data.tier", "zzz")),
    # AIRCORE-749: `_`/`%` are literal substrings, not SQL wildcards.
    ("like_name_underscore_literal", C(FilterOperator.LIKE, "name", "prod_db")),
    ("like_name_percent_literal", C(FilterOperator.LIKE, "name", "50%off")),
    ("like_data_tier_underscore_literal", C(FilterOperator.LIKE, "data.tier", "a_c")),
    ("in_name", C(FilterOperator.IN, "name", ["llama", "mistral"])),
    ("in_data_tier", C(FilterOperator.IN, "data.tier", ["pro", "free"])),
    ("in_data_score", C(FilterOperator.IN, "data.score", [5, 10])),
    ("nin_name", C(FilterOperator.NIN, "name", ["llama"])),
    ("nin_data_tier", C(FilterOperator.NIN, "data.tier", ["pro"])),
    ("nin_data_score", C(FilterOperator.NIN, "data.score", [5, 9])),
    ("contains_tags_red", C(FilterOperator.CONTAINS, "data.tags", "red")),
    ("contains_tags_blue", C(FilterOperator.CONTAINS, "data.tags", "blue")),
    ("contains_tags_absent", C(FilterOperator.CONTAINS, "data.tags", "nope")),
    ("not_contains_tags_red", NOT(C(FilterOperator.CONTAINS, "data.tags", "red"))),
    (
        "and_contains_tags",
        AND(C(FilterOperator.CONTAINS, "data.tags", "blue"), C(FilterOperator.EQ, "data.tier", "free")),
    ),
    ("gt_data_score", C(FilterOperator.GT, "data.score", 9)),
    ("gte_data_score", C(FilterOperator.GTE, "data.score", 10)),
    ("lt_data_score", C(FilterOperator.LT, "data.score", 10)),
    ("lte_data_score", C(FilterOperator.LTE, "data.score", 9)),
    ("gt_data_tier_text", C(FilterOperator.GT, "data.tier", "a")),
    ("lt_data_tier_text", C(FilterOperator.LT, "data.tier", "g")),
    # AIRCORE-749: ordered comparison only matches actual JSON numbers; non-numeric
    # text / null / absent are no-match on both backends (PostgreSQL must not error
    # casting the "notnum" row).
    ("gt_data_n2_numeric_only", C(FilterOperator.GT, "data.n2", 4)),
    ("lt_data_n2_numeric_only", C(FilterOperator.LT, "data.n2", 10)),
    ("and_tree", AND(C(FilterOperator.EQ, "data.tier", "pro"), C(FilterOperator.GT, "data.score", 9))),
    ("or_tree", OR(C(FilterOperator.EQ, "name", "llama"), C(FilterOperator.EQ, "name", "zephyr"))),
    ("not_tree", NOT(C(FilterOperator.EQ, "data.tier", "pro"))),
    (
        "nested_and_or_not",
        AND(
            OR(C(FilterOperator.EQ, "data.tier", "pro"), C(FilterOperator.EQ, "data.tier", "free")),
            NOT(C(FilterOperator.LT, "data.score", 9)),
        ),
    ),
]


@pytest.mark.parametrize("label,op", CASES, ids=[c[0] for c in CASES])
def test_matches_matches_sql(db, label, op):
    condition = op.apply(SQLAlchemyFilterRepository(FakeEntity))
    sql_ids = {r.id for r in db.execute(select(FakeEntity).where(condition)).scalars().all()}

    all_rows = db.execute(select(FakeEntity)).scalars().all()
    py_ids = {r.id for r in all_rows if op.apply(InMemoryFilterRepository(r))}

    assert py_ids == sql_ids, f"{label}: in-memory={sorted(py_ids)} != SQL={sorted(sql_ids)}"
