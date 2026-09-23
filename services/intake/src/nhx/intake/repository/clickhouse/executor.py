# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed runtime query boundary for Intake ClickHouse repositories."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from clickhouse_connect.driver.exceptions import ClickHouseError
from clickhouse_connect.driver.external import ExternalData
from nhx.intake.repository.clickhouse.tables import ClickHouseTable, qualified_table
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClickHouseExternalData:
    """One typed external-data payload used by a repository query."""

    file_name: str
    data: bytes
    fmt: str
    structure: str


@dataclass(frozen=True)
class ClickHouseQuery:
    """One named, parameterized ClickHouse read statement."""

    name: str
    statement: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    external_data: ClickHouseExternalData | None = None

    def bind(self, **parameters: object) -> ClickHouseQuery:
        """Return a copy with additional bound parameters."""

        return ClickHouseQuery(
            name=self.name,
            statement=self.statement,
            parameters={**self.parameters, **parameters},
            external_data=self.external_data,
        )


@dataclass(frozen=True)
class ClickHouseInsert:
    """One named insert into a registered runtime table."""

    name: str
    table: ClickHouseTable
    rows: Sequence[Sequence[Any]]
    column_names: Sequence[str]


class ClickHouseQueryError(RuntimeError):
    """Raised when a named repository query fails."""

    def __init__(self, query_name: str) -> None:
        self.query_name = query_name
        super().__init__(f"ClickHouse query failed: {query_name}")


class ClickHouseInsertError(RuntimeError):
    """Raised when a named repository insert fails."""

    def __init__(self, insert_name: str) -> None:
        self.insert_name = insert_name
        super().__init__(f"ClickHouse insert failed: {insert_name}")


class ClickHouseExecutor:
    """Execute named repository operations without exposing the raw driver."""

    def __init__(self, client: ClickHouseSpanClient) -> None:
        self._client = client

    def table(self, table: ClickHouseTable) -> str:
        return qualified_table(self._client.database, table)

    async def fetch_all(self, query: ClickHouseQuery) -> list[dict[str, Any]]:
        started_at = perf_counter()
        try:
            if query.external_data is None:
                result = await self._client.query(
                    query.statement,
                    parameters=dict(query.parameters),
                )
            else:
                external_data = query.external_data
                result = await self._client.query(
                    query.statement,
                    parameters=dict(query.parameters),
                    external_data=ExternalData(
                        file_name=external_data.file_name,
                        data=external_data.data,
                        fmt=external_data.fmt,
                        structure=external_data.structure,
                    ),
                )
        except ClickHouseError as exc:
            logger.exception("ClickHouse repository query failed", extra={"query_name": query.name})
            raise ClickHouseQueryError(query.name) from exc
        finally:
            logger.debug(
                "ClickHouse repository query finished",
                extra={
                    "query_name": query.name,
                    "duration_ms": (perf_counter() - started_at) * 1000,
                },
            )

        columns: Sequence[str] = result.column_names
        rows: Sequence[Sequence[Any]] = result.result_rows
        return [dict(zip(columns, row, strict=True)) for row in rows]

    async def fetch_scalar(self, query: ClickHouseQuery) -> Any | None:
        """Return the first column of the first row, or ``None`` when no row exists."""

        rows = await self.fetch_all(query)
        return next(iter(rows[0].values())) if rows else None

    async def insert(self, insert: ClickHouseInsert) -> None:
        if not insert.rows:
            return
        if not isinstance(insert.table, ClickHouseTable):
            raise TypeError(f"Expected ClickHouseTable, got {type(insert.table).__name__}")

        started_at = perf_counter()
        try:
            await self._client.insert(
                insert.table.value,
                insert.rows,
                column_names=insert.column_names,
            )
        except ClickHouseError as exc:
            logger.exception("ClickHouse repository insert failed", extra={"insert_name": insert.name})
            raise ClickHouseInsertError(insert.name) from exc
        finally:
            logger.debug(
                "ClickHouse repository insert finished",
                extra={
                    "insert_name": insert.name,
                    "row_count": len(insert.rows),
                    "duration_ms": (perf_counter() - started_at) * 1000,
                },
            )
