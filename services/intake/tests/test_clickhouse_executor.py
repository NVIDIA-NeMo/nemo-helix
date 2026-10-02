# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed ClickHouse repository executor."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from clickhouse_connect.driver.exceptions import ClickHouseError
from clickhouse_connect.driver.external import ExternalData
from nhx.intake.repository.clickhouse.executor import (
    ClickHouseExecutor,
    ClickHouseExternalData,
    ClickHouseInsert,
    ClickHouseInsertError,
    ClickHouseQuery,
    ClickHouseQueryError,
)
from nhx.intake.repository.clickhouse.tables import ClickHouseTable, qualified_table
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient


class _Client:
    database = "intake"

    def __init__(self, *, error: ClickHouseError | None = None) -> None:
        self.error = error
        self.statements: list[str] = []
        self.parameters: list[dict[str, Any]] = []
        self.external_data: list[object | None] = []
        self.inserts: list[tuple[str, list[tuple[object, ...]], list[str]]] = []

    async def query(
        self,
        statement: str,
        *,
        parameters: dict[str, Any],
        external_data: object | None = None,
    ) -> SimpleNamespace:
        self.statements.append(statement)
        self.parameters.append(parameters)
        self.external_data.append(external_data)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            column_names=["session_id", "span_count"],
            result_rows=[("session-a", 3)],
        )

    async def insert(
        self,
        table: str,
        rows: list[tuple[object, ...]],
        *,
        column_names: list[str],
    ) -> None:
        if self.error is not None:
            raise self.error
        self.inserts.append((table, rows, column_names))


def _executor(client: _Client) -> ClickHouseExecutor:
    return ClickHouseExecutor(cast(ClickHouseSpanClient, client))


def test_query_bind_preserves_base_parameters() -> None:
    query = ClickHouseQuery(
        name="sessions.get",
        statement="SELECT %(session_id)s",
        parameters={"metric_key": "cost.total"},
    )

    bound = query.bind(session_id="session-a")

    assert query.parameters == {"metric_key": "cost.total"}
    assert bound.parameters == {
        "metric_key": "cost.total",
        "session_id": "session-a",
    }


@pytest.mark.asyncio
async def test_executor_returns_mapped_rows_and_bound_parameters() -> None:
    client = _Client()
    query = ClickHouseQuery(
        name="sessions.get",
        statement="SELECT %(session_id)s",
        parameters={"session_id": "session-a"},
    )

    rows = await _executor(client).fetch_all(query)

    assert rows == [{"session_id": "session-a", "span_count": 3}]
    assert client.statements == [query.statement]
    assert client.parameters == [{"session_id": "session-a"}]


@pytest.mark.asyncio
async def test_executor_returns_first_scalar() -> None:
    scalar = await _executor(_Client()).fetch_scalar(
        ClickHouseQuery(
            name="sessions.first_id",
            statement="SELECT session_id",
        )
    )

    assert scalar == "session-a"


@pytest.mark.asyncio
async def test_executor_builds_driver_external_data_inside_boundary() -> None:
    client = _Client()
    query = ClickHouseQuery(
        name="traces.latest_started_at_by_group",
        statement="SELECT * FROM trace_refs",
        external_data=ClickHouseExternalData(
            file_name="trace_refs.jsonl",
            data=b'{"group_id":"group-a","trace_id":"trace-a"}',
            fmt="JSONEachRow",
            structure="group_id String, trace_id String",
        ),
    )

    await _executor(client).fetch_all(query)

    external_data = cast(ExternalData, client.external_data[0])
    assert external_data.query_params == {
        "trace_refs_format": "JSONEachRow",
        "trace_refs_structure": "group_id String, trace_id String",
    }


@pytest.mark.asyncio
async def test_executor_translates_clickhouse_errors_without_sql_details() -> None:
    query = ClickHouseQuery(
        name="sessions.get",
        statement="SELECT secret FROM hidden",
    )

    with pytest.raises(ClickHouseQueryError) as exc_info:
        await _executor(_Client(error=ClickHouseError("driver details"))).fetch_all(query)

    assert exc_info.value.query_name == "sessions.get"
    assert str(exc_info.value) == "ClickHouse query failed: sessions.get"
    assert query.statement not in str(exc_info.value)


@pytest.mark.asyncio
async def test_executor_inserts_rows_into_registered_table() -> None:
    client = _Client()
    insert = ClickHouseInsert(
        name="annotations.save",
        table=ClickHouseTable.ANNOTATIONS,
        rows=[("annotation-a", "default")],
        column_names=["annotation_id", "workspace"],
    )

    await _executor(client).insert(insert)

    assert client.inserts == [
        (
            "annotations",
            [("annotation-a", "default")],
            ["annotation_id", "workspace"],
        )
    ]


@pytest.mark.asyncio
async def test_executor_translates_insert_errors_without_row_details() -> None:
    insert = ClickHouseInsert(
        name="annotations.save",
        table=ClickHouseTable.ANNOTATIONS,
        rows=[("secret-value",)],
        column_names=["value"],
    )

    with pytest.raises(ClickHouseInsertError) as exc_info:
        await _executor(_Client(error=ClickHouseError("driver details"))).insert(insert)

    assert exc_info.value.insert_name == "annotations.save"
    assert str(exc_info.value) == "ClickHouse insert failed: annotations.save"
    assert "secret-value" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_executor_rejects_unregistered_insert_table() -> None:
    insert = ClickHouseInsert(
        name="invalid.save",
        table=cast(ClickHouseTable, "system.tables"),
        rows=[("value",)],
        column_names=["value"],
    )

    with pytest.raises(TypeError, match="Expected ClickHouseTable"):
        await _executor(_Client()).insert(insert)


def test_table_registry_quotes_known_tables() -> None:
    assert qualified_table("intake", ClickHouseTable.SPANS) == "`intake`.`spans`"


def test_table_registry_rejects_unregistered_names() -> None:
    with pytest.raises(TypeError, match="Expected ClickHouseTable"):
        qualified_table("intake", cast(ClickHouseTable, "system.tables"))
