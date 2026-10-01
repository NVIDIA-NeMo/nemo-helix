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
SEED = [
    dict(
        id=1,
        name="llama",
        data={
            "score": 5,
            "tier": "free",
            "flag": True,
            "k": None,
            "tags": ["red", "blue"],
            "meta": [{"key": "owner", "value": "alice"}, {"key": "team", "value": "eval"}],
            "members": ["ws/task_a#d1", "ws/task-b#d2"],
            "tag_map": {"latest": 2, "v1.2": 1},
        },
    ),
    dict(
        id=2,
        name="Llama-2",
        data={
            "score": 9,
            "tier": "pro",
            "flag": False,
            "tags": ["red"],
            "meta": [{"key": "owner", "value": "bob"}, {"key": "level", "value": 3}],
            # Near misses for "ws/task_a#": a longer workspace, "ws/taskXa" if "_" were a LIKE wildcard,
            # and the prefix after an escaped quote inside an element.
            "members": ["other-ws/task_a#d1", "ws/taskXa#d1", 'x"ws/task_a#d1'],
            "tag_map": {"latest": 1, "v1": 1},
        },
    ),
    # "redish" is a deliberate prefix near-miss for "red" — quote-delimited matching must exclude it.
    dict(
        id=3,
        name="zephyr",
        data={
            "score": 10,
            "tier": "pro",
            "flag": True,
            "k": "v",
            "tags": ["green", "redish"],
            "meta": [{"key": "owner", "value": None}, {"key": "team", "value": "alice"}],
            "members": ["ws/task_a#d9"],
            "tag_map": {},
        },
    ),
    # "meta" as a bare object, not an array: its members must not be treated as elements.
    dict(
        id=4,
        name="mistral",
        data={
            "score": 100,
            "tier": "enterprise",
            "flag": False,
            "tags": [],
            "meta": {"key": "owner", "value": "alice"},
        },
    ),
    dict(id=5, name=None, data={"score": 1, "tier": "free", "flag": False, "tags": ["blue"]}),
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
    ("elem_match_pair", C(FilterOperator.ELEM_MATCH, "data.meta", {"key": "owner", "value": "alice"})),
    ("elem_match_single_field", C(FilterOperator.ELEM_MATCH, "data.meta", {"key": "team"})),
    ("elem_match_int_value", C(FilterOperator.ELEM_MATCH, "data.meta", {"key": "level", "value": 3})),
    ("elem_match_null_value", C(FilterOperator.ELEM_MATCH, "data.meta", {"key": "owner", "value": None})),
    # Row 1 has owner=alice and team=eval on different elements; neither element satisfies both.
    ("elem_match_across_elements", C(FilterOperator.ELEM_MATCH, "data.meta", {"key": "team", "value": "alice"})),
    ("elem_match_absent_field", C(FilterOperator.ELEM_MATCH, "data.nope", {"key": "owner"})),
    ("not_elem_match", NOT(C(FilterOperator.ELEM_MATCH, "data.meta", {"key": "owner", "value": "alice"}))),
    ("contains_prefix_any_revision", C(FilterOperator.CONTAINS_PREFIX, "data.members", "ws/task_a#")),
    ("contains_prefix_exact_member", C(FilterOperator.CONTAINS_PREFIX, "data.members", "ws/task-b#d2")),
    ("contains_prefix_absent_field", C(FilterOperator.CONTAINS_PREFIX, "data.nope", "ws/")),
    ("not_contains_prefix", NOT(C(FilterOperator.CONTAINS_PREFIX, "data.members", "ws/task_a#"))),
    ("has_key_dotted", C(FilterOperator.HAS_KEY, "data.tag_map", "v1.2")),
    ("has_key_prefix_of_dotted", C(FilterOperator.HAS_KEY, "data.tag_map", "v1")),
    ("has_key_common", C(FilterOperator.HAS_KEY, "data.tag_map", "latest")),
    ("has_key_absent_field", C(FilterOperator.HAS_KEY, "data.nope", "latest")),
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
    condition = op.apply(SQLAlchemyFilterRepository(FakeEntity, dialect_name="sqlite"))
    sql_ids = {r.id for r in db.execute(select(FakeEntity).where(condition)).scalars().all()}

    all_rows = db.execute(select(FakeEntity)).scalars().all()
    py_ids = {r.id for r in all_rows if op.apply(InMemoryFilterRepository(r))}

    assert py_ids == sql_ids, f"{label}: in-memory={sorted(py_ids)} != SQL={sorted(sql_ids)}"


@pytest.mark.parametrize(
    "criteria,expected_ids",
    [
        ({"key": "owner", "value": "alice"}, {1}),
        ({"key": "team"}, {1, 3}),
        ({"key": "level", "value": 3}, {2}),
        ({"key": "owner", "value": None}, {3}),
        ({"key": "team", "value": "alice"}, {3}),
    ],
)
def test_elem_match_requires_one_element_to_satisfy_every_criterion(db, criteria, expected_ids):
    """Each criterion must hold on the same element, and a non-array field (row 4) never matches."""
    condition = C(FilterOperator.ELEM_MATCH, "data.meta", criteria).apply(
        SQLAlchemyFilterRepository(FakeEntity, dialect_name="sqlite")
    )
    assert {r.id for r in db.execute(select(FakeEntity).where(condition)).scalars().all()} == expected_ids


@pytest.mark.parametrize(
    "op",
    [
        C(FilterOperator.ELEM_MATCH, "data.meta", {"key": "owner"}),
        C(FilterOperator.CONTAINS_PREFIX, "data.members", "ws/"),
    ],
)
def test_element_operators_reject_unknown_dialect(op):
    with pytest.raises(ValueError, match="dialect"):
        op.apply(SQLAlchemyFilterRepository(FakeEntity))


@pytest.mark.parametrize(
    "op,expected_ids",
    [
        (C(FilterOperator.CONTAINS_PREFIX, "data.members", "ws/task_a#"), {1, 3}),
        (C(FilterOperator.CONTAINS_PREFIX, "data.members", "ws/task-b#d2"), {1}),
        (C(FilterOperator.HAS_KEY, "data.tag_map", "v1.2"), {1}),
        (C(FilterOperator.HAS_KEY, "data.tag_map", "v1"), {2}),
        (C(FilterOperator.HAS_KEY, "data.tag_map", "latest"), {1, 2}),
    ],
)
def test_contains_prefix_and_has_key_select_expected_rows(db, op, expected_ids):
    """Prefixes stay quote-anchored with ``_`` literal, and a dotted key is one key, not a path."""
    condition = op.apply(SQLAlchemyFilterRepository(FakeEntity, dialect_name="sqlite"))
    assert {r.id for r in db.execute(select(FakeEntity).where(condition)).scalars().all()} == expected_ids
