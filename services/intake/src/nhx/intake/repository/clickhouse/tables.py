# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed registry of runtime-owned Intake ClickHouse tables."""

from enum import StrEnum

from nhx.intake.spans.clickhouse_migrations import quote_clickhouse_identifier


class ClickHouseTable(StrEnum):
    ANNOTATIONS = "annotations"
    EVALUATOR_RESULTS = "evaluator_results"
    SPANS = "spans"
    TRACE_INDEX = "trace_index"


def qualified_table(database: str, table: ClickHouseTable) -> str:
    """Return a safely quoted table name from the closed runtime registry."""

    if not isinstance(table, ClickHouseTable):
        raise TypeError(f"Expected ClickHouseTable, got {type(table).__name__}")
    return f"{quote_clickhouse_identifier(database)}.{quote_clickhouse_identifier(table.value)}"
