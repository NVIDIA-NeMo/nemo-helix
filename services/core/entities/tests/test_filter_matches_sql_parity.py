# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL-parity safety net for in-memory filter evaluation.

Each parametrized FilterOperation tree is run two ways against the same seeded
SQLite rows, both through the same ``op.apply(repo)`` front door:
  1. ``SQLAlchemyFilterRepository`` (the SQL source of truth).
  2. ``InMemoryFilterRepository`` over the ORM instances (the in-memory backend).

The two must select exactly the same set of row ids. ``InMemoryFilterRepository``
is a native-Python evaluator, NOT a byte-for-byte SQL mirror (see its class
docstring), so this suite only covers the cases where native
and SQL semantics agree — strings, real numbers, native booleans (via ``$eq``),
and logical trees. It deliberately excludes the cases where the SQL backends
disagree with each other or rely on JSON-to-text coercion (e.g. int-vs-string
``$eq``, boolean text rendering, non-numeric-text numeric casts); those are
pinned as documented divergences in the plugin's test_filter_matches.py.
"""

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
SEED = [
    dict(id=1, name="llama", data={"score": 5, "tier": "free", "flag": True, "k": None, "tags": ["red", "blue"]}),
    dict(id=2, name="Llama-2", data={"score": 9, "tier": "pro", "flag": False, "tags": ["red"]}),
    # "redish" is a deliberate prefix near-miss for "red" — quote-delimited matching must exclude it.
    dict(id=3, name="zephyr", data={"score": 10, "tier": "pro", "flag": True, "k": "v", "tags": ["green", "redish"]}),
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


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([FakeEntity(**row) for row in SEED])
        session.commit()
        yield session


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
