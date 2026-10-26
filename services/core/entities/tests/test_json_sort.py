# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar JSON sorting through the real SQLite repository and PostgreSQL compiler."""

from collections.abc import AsyncIterator
from datetime import datetime

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.core.entities.app.repository.sqlalchemy.base import Base
from nhx.core.entities.app.repository.sqlalchemy.entity import SQLAlchemyEntityRepository
from nhx.core.entities.app.repository.sqlalchemy.models import DBEntity, DBWorkspace
from sqlalchemy import event, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
async def json_sort_repository(tmp_path) -> AsyncIterator[SQLAlchemyEntityRepository]:
    """Yield a real repository over an isolated SQLite database in tmp_path."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'json-sort.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        session.add(DBWorkspace(id="workspace-default", name="default"))
        await session.commit()
    try:
        yield SQLAlchemyEntityRepository(sessions)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("path", ["data.task_index", "data.nested.task_index"])
@pytest.mark.parametrize("descending,expected", [(False, [2, 10, 100]), (True, [100, 10, 2])])
async def test_json_numeric_order_precedes_pagination(json_sort_repository, path, descending, expected):
    """Numeric JSON values must determine page boundaries, including nested paths."""
    for index in [100, 2, 10]:
        data = {"task_index": index}
        if "nested" in path:
            data = {"nested": data}
        await json_sort_repository.create_entity(
            workspace="default", entity_type="member", name=f"task-{index}", data=data
        )
    actual = []
    for page in [1, 2, 3]:
        rows, total = await json_sort_repository.list_entities(
            workspace="default",
            entity_type="member",
            sort=("-" if descending else "") + path,
            page=page,
            page_size=1,
        )
        assert total == 3 and len(rows) == 1
        data = rows[0].data
        actual.append(data["nested"]["task_index"] if "nested" in path else data["task_index"])
    assert actual == expected


async def test_json_numeric_order_preserves_floats_and_negative_values(json_sort_repository):
    """Scalar extraction must preserve numeric types rather than cast them to text or integers."""
    for name, value in [("ten", 10.5), ("two", 2), ("hundred", 100), ("negative", -1.25)]:
        await json_sort_repository.create_entity(
            workspace="default", entity_type="member", name=name, data={"value": value}
        )
    rows, total = await json_sort_repository.list_entities(workspace="default", entity_type="member", sort="data.value")
    assert total == 4
    assert [row.data["value"] for row in rows] == [-1.25, 2, 10.5, 100]


async def test_sqlite_json_missing_null_and_mixed_scalar_order(json_sort_repository):
    """SQLite sorts SQL NULL first, then numbers numerically, then strings lexically."""
    for name, data in [
        ("missing", {}),
        ("null", {"value": None}),
        ("ten", {"value": 10}),
        ("two", {"value": 2}),
        ("text-nine", {"value": "9"}),
        ("text-one", {"value": "1"}),
        ("text-eleven", {"value": "11"}),
    ]:
        await json_sort_repository.create_entity(workspace="default", entity_type="member", name=name, data=data)
    rows, total = await json_sort_repository.list_entities(workspace="default", entity_type="member", sort="data.value")
    assert total == 7
    assert {row.name for row in rows[:2]} == {"missing", "null"}
    assert [row.name for row in rows[2:]] == ["two", "ten", "text-one", "text-eleven", "text-nine"]
    rows, _ = await json_sort_repository.list_entities(workspace="default", entity_type="member", sort="-data.value")
    assert [row.name for row in rows[:5]] == ["text-nine", "text-eleven", "text-one", "ten", "two"]
    assert {row.name for row in rows[5:]} == {"missing", "null"}


@pytest.mark.parametrize(
    "sort,expected",
    [
        ("data.label", ["apple", "mango", "zebra"]),
        ("-data.label", ["zebra", "mango", "apple"]),
        ("name", ["apple", "mango", "zebra"]),
        ("-name", ["zebra", "mango", "apple"]),
    ],
)
async def test_json_string_and_base_column_sort_regressions(json_sort_repository, sort, expected):
    """String data and direct entity columns retain ascending and descending ordering."""
    for name in ["zebra", "apple", "mango"]:
        await json_sort_repository.create_entity(
            workspace="default", entity_type="member", name=name, data={"label": name}
        )
    rows, total = await json_sort_repository.list_entities(workspace="default", entity_type="member", sort=sort)
    assert total == 3
    assert [row.name for row in rows] == expected


async def test_json_sort_preserves_parent_phase_filters_and_totals(json_sort_repository):
    """Filtering happens before numeric sorting and pagination, with correct parent-scoped totals."""
    parent = await json_sort_repository.create_entity(workspace="default", entity_type="run", name="selected", data={})
    other = await json_sort_repository.create_entity(workspace="default", entity_type="run", name="other", data={})
    for owner, index, phase in [
        (parent, 100, "failed"),
        (parent, 2, "failed"),
        (parent, 10, "running"),
        (other, 1, "failed"),
    ]:
        await json_sort_repository.create_entity(
            workspace="default",
            entity_type="member",
            name=f"task-{index}",
            parent=owner.id,
            data={"task_index": index, "phase": phase},
        )
    filters = LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=parent.id),
            ComparisonOperation(field="data.phase", operator=FilterOperator.EQ, value="failed"),
        ],
    )
    rows, total = await json_sort_repository.list_entities(
        workspace="default",
        entity_type="member",
        filter_op=filters,
        sort="data.task_index",
        page=2,
        page_size=1,
    )
    assert total == 2
    assert [row.name for row in rows] == ["task-100"]


@pytest.mark.parametrize("sort", ["data.task_index", "-data.task_index", "data.nested.task_index"])
def test_postgresql_json_sort_uses_jsonb_numeric_comparison(json_sort_repository, sort):
    """PostgreSQL must compare JSONB scalar values, preserving backend-native NULL ordering."""
    expression = json_sort_repository._get_sort_key(sort, dialect_name="postgresql")
    sql = str(
        select(DBEntity)
        .order_by(expression)
        .compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    ).split("ORDER BY", 1)[1]
    assert "AS JSONB" in sql
    assert "->>" not in sql and "AS VARCHAR" not in sql
    assert "NULLS FIRST" not in sql and "NULLS LAST" not in sql
    assert sql.rstrip().endswith("DESC" if sort.startswith("-") else "ASC")


@pytest.mark.parametrize("sort", ["created_at", "-created_at", "name", "-name"])
def test_postgresql_base_sort_uses_direct_column(json_sort_repository, sort):
    """Base-field sorts bypass JSON casting and keep the requested direction."""
    expression = json_sort_repository._get_sort_key(sort, dialect_name="postgresql")
    sql = str(expression.compile(dialect=postgresql.dialect()))
    assert sql == f"entities.{sort.lstrip('-')} {'DESC' if sort.startswith('-') else 'ASC'}"


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
@pytest.mark.parametrize(
    "field", ["bogus", "metadata", "registry", "__class__", "to_pydantic", "data", "workspace_rel"]
)
@pytest.mark.parametrize("prefix", ["", "-"])
def test_non_sortable_attributes_are_rejected(json_sort_repository, dialect_name, field, prefix):
    """Invalid client input must fail validation before building a database query."""
    with pytest.raises(ValueError, match=f"Invalid sort field {field}"):
        json_sort_repository._get_sort_key(prefix + field, dialect_name=dialect_name)


@pytest.mark.parametrize("sort", ["", "-", "--name", "---created_at", "data.", "data..k", "data.k."])
@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_malformed_sort_syntax_is_rejected(json_sort_repository, sort, dialect_name):
    with pytest.raises(ValueError, match="Invalid sort field"):
        json_sort_repository._get_sort_key(sort, dialect_name=dialect_name)


@pytest.mark.parametrize("path", ["data.display-name", "data.display name", "data.nested.123"])
def test_json_sort_preserves_non_identifier_keys(json_sort_repository, path):
    """JSON object keys are not restricted to Python identifiers."""
    expression = json_sort_repository._get_sort_key(path, dialect_name="postgresql")
    assert "AS JSONB" in str(expression.compile(dialect=postgresql.dialect()))


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "workspace",
        "entity_type",
        "name",
        "parent",
        "project",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
        "db_version",
    ],
)
@pytest.mark.parametrize("prefix", ["", "-"])
def test_existing_scalar_column_sorts_are_preserved(json_sort_repository, field, prefix):
    expression = json_sort_repository._get_sort_key(prefix + field, dialect_name="postgresql")
    assert str(expression.compile(dialect=postgresql.dialect())) == (f"entities.{field} {'DESC' if prefix else 'ASC'}")


@pytest.mark.parametrize("sort", [None, "created_at", "-created_at", "data.k", "-data.k", "data.missing"])
@pytest.mark.parametrize("page_size", [1, 2, 4])
async def test_tied_sort_pages_use_unique_id_order(json_sort_repository, sort, page_size):
    """Ties have a defined order, including missing JSON paths and default timestamps."""
    timestamp = datetime(2026, 1, 1)
    async with json_sort_repository.session_maker() as session:
        # Insertion order intentionally differs from the required tie-break order.
        for index in [5, 2, 4, 1, 3, 0]:
            session.add(
                DBEntity(
                    id=f"member-{index}",
                    workspace="default",
                    entity_type="member",
                    name=f"task-{index}",
                    data={"k": "equal"},
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
        await session.commit()

    statements = []
    async with json_sort_repository.session_maker() as session:
        engine = session.get_bind()

        def capture_statement(conn, clauseelement, multiparams, params, execution_options):
            statements.append(clauseelement)

        event.listen(engine, "before_execute", capture_statement)
        try:
            ids = []
            for page in range(1, (6 + page_size - 1) // page_size + 1):
                rows, total = await json_sort_repository.list_entities(
                    workspace="default",
                    entity_type="member",
                    sort=sort,
                    page=page,
                    page_size=page_size,
                    session=session,
                )
                assert total == 6
                ids.extend(row.id for row in rows)
        finally:
            event.remove(engine, "before_execute", capture_statement)

    assert ids == [f"member-{index}" for index in range(6)]
    # SQLite can appear stable even without a total order. Check the actual
    # paginated query against the PostgreSQL compiler as well.
    for statement in statements:
        if statement._limit_clause is not None:
            sql = str(statement.compile(dialect=postgresql.dialect()))
            order_by = sql.split("ORDER BY", 1)[1].split("LIMIT", 1)[0].strip()
            assert order_by.endswith(", entities.id ASC")
