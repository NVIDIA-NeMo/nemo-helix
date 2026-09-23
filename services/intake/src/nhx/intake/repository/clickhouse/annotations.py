# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse implementation of Intake annotation persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from nhx.common.api.common import PaginatedResult
from nhx.intake.repository.annotations import AnnotationsRepository
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseInsert, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.spans.domain import Annotation, AnnotationKind, AnnotationListFilter
from nhx.intake.spans.storage import dict_to_row, make_pagination

ANNOTATION_COLUMNS = [
    "annotation_id",
    "workspace",
    "span_id",
    "session_id",
    "kind",
    "name",
    "value_text",
    "value_numeric",
    "text",
    "metadata",
    "created_by",
    "created_at",
    "ingested_at",
    "is_deleted",
]

ANNOTATION_SORT_COLUMNS = {
    "created_at": "created_at",
}


class ClickHouseAnnotationsRepository(AnnotationsRepository):
    def __init__(self, executor: ClickHouseExecutor) -> None:
        self._executor = executor

    async def save_annotations(self, annotations: list[Annotation]) -> None:
        if not annotations:
            return
        rows = [dict_to_row(_annotation_to_row(item), ANNOTATION_COLUMNS) for item in annotations]
        await self._executor.insert(
            ClickHouseInsert(
                name="annotations.save",
                table=ClickHouseTable.ANNOTATIONS,
                rows=rows,
                column_names=ANNOTATION_COLUMNS,
            )
        )

    async def get_annotation(self, *, workspace: str, annotation_id: str) -> Annotation | None:
        rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name="annotations.get",
                statement=f"""
                SELECT *
                FROM {self._executor.table(ClickHouseTable.ANNOTATIONS)} FINAL
                WHERE workspace = %(workspace)s
                  AND annotation_id = %(annotation_id)s
                  AND is_deleted = 0
                LIMIT 1
                """,
                parameters={"workspace": workspace, "annotation_id": annotation_id},
            )
        )
        if not rows:
            return None
        return _row_to_annotation(rows[0])

    async def list_annotations(
        self,
        *,
        filters: AnnotationListFilter,
        page: int,
        page_size: int,
        sort: str,
    ) -> PaginatedResult[Annotation]:
        where_sql, parameters = _annotation_where(filters)
        table = self._executor.table(ClickHouseTable.ANNOTATIONS)
        total_results = int(
            await self._executor.fetch_scalar(
                ClickHouseQuery(
                    name="annotations.list.count",
                    statement=f"SELECT count() FROM {table} FINAL WHERE {where_sql}",
                    parameters=parameters,
                )
            )
            or 0
        )
        offset = (page - 1) * page_size
        rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name="annotations.list.rows",
                statement=f"""
                SELECT *
                FROM {table} FINAL
                WHERE {where_sql}
                ORDER BY {_annotation_order_by(sort)}
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                parameters={**parameters, "limit": page_size, "offset": offset},
            )
        )
        annotations = [_row_to_annotation(row) for row in rows]
        return PaginatedResult(
            data=annotations,
            pagination=make_pagination(
                page=page,
                page_size=page_size,
                current_page_size=len(annotations),
                total_results=total_results,
            ),
        )

    async def soft_delete_annotation(self, *, annotation: Annotation) -> None:
        """Tombstone the annotation_id by writing a new row with is_deleted=1.

        The tombstone uses a fresh `ingested_at` so the ReplacingMergeTree
        version column strictly exceeds the live row and the deletion wins
        on next merge. `FINAL` reads filter out tombstoned rows.
        """

        row = _annotation_to_row(annotation, is_deleted=True)
        row["ingested_at"] = datetime.now(timezone.utc)
        rows = [dict_to_row(row, ANNOTATION_COLUMNS)]
        await self._executor.insert(
            ClickHouseInsert(
                name="annotations.soft_delete",
                table=ClickHouseTable.ANNOTATIONS,
                rows=rows,
                column_names=ANNOTATION_COLUMNS,
            )
        )


def _annotation_where(filters: AnnotationListFilter) -> tuple[str, dict[str, Any]]:
    clauses = ["workspace = %(workspace)s", "is_deleted = 0"]
    parameters: dict[str, Any] = {"workspace": filters.workspace}
    if filters.span_id is not None:
        clauses.append("span_id = %(span_id)s")
        parameters["span_id"] = filters.span_id
    if filters.session_id is not None:
        clauses.append("session_id = %(session_id)s")
        parameters["session_id"] = filters.session_id
    if filters.kind is not None:
        clauses.append("kind = %(kind)s")
        parameters["kind"] = filters.kind.value
    if filters.name is not None:
        clauses.append("name = %(name)s")
        parameters["name"] = filters.name
    if filters.value_text is not None:
        clauses.append("value_text = %(value_text)s")
        parameters["value_text"] = filters.value_text
    if filters.value_numeric_gte is not None:
        clauses.append("value_numeric >= %(value_numeric_gte)s")
        parameters["value_numeric_gte"] = filters.value_numeric_gte
    if filters.value_numeric_lte is not None:
        clauses.append("value_numeric <= %(value_numeric_lte)s")
        parameters["value_numeric_lte"] = filters.value_numeric_lte
    if filters.created_by is not None:
        clauses.append("created_by = %(created_by)s")
        parameters["created_by"] = filters.created_by
    if filters.created_at_gte is not None:
        clauses.append("created_at >= %(created_at_gte)s")
        parameters["created_at_gte"] = filters.created_at_gte
    if filters.created_at_lte is not None:
        clauses.append("created_at <= %(created_at_lte)s")
        parameters["created_at_lte"] = filters.created_at_lte
    return " AND ".join(clauses), parameters


def _annotation_order_by(sort: str) -> str:
    direction = "DESC" if sort.startswith("-") else "ASC"
    field = sort.removeprefix("-")
    column = ANNOTATION_SORT_COLUMNS.get(field)
    if column is None:
        raise ValueError(f"Unsupported annotation sort field: {field}")
    return f"{column} {direction}, annotation_id ASC"


def _annotation_to_row(annotation: Annotation, *, is_deleted: bool = False) -> dict[str, Any]:
    return {
        "annotation_id": annotation.annotation_id,
        "workspace": annotation.workspace,
        # span_id stored as empty string when absent (matches `external_parent_span_id` convention).
        "span_id": annotation.span_id or "",
        "session_id": annotation.session_id,
        "kind": annotation.kind.value,
        "name": annotation.name,
        "value_text": annotation.value_text,
        "value_numeric": annotation.value_numeric,
        "text": annotation.text,
        "metadata": json.dumps(annotation.metadata) if annotation.metadata is not None else None,
        "created_by": annotation.created_by,
        "created_at": annotation.created_at,
        "ingested_at": annotation.ingested_at,
        "is_deleted": 1 if is_deleted else 0,
    }


def _row_to_annotation(row: dict[str, Any]) -> Annotation:
    metadata_raw = row.get("metadata")
    metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) and metadata_raw else None
    span_id_raw = row.get("span_id")
    span_id = span_id_raw if isinstance(span_id_raw, str) and span_id_raw else None
    return Annotation(
        annotation_id=row["annotation_id"],
        workspace=row["workspace"],
        span_id=span_id,
        session_id=row["session_id"],
        kind=AnnotationKind(row["kind"]),
        name=row.get("name"),
        value_text=row.get("value_text"),
        value_numeric=row.get("value_numeric"),
        text=row.get("text"),
        metadata=metadata,
        created_by=row.get("created_by"),
        created_at=row["created_at"],
        ingested_at=row["ingested_at"],
    )
