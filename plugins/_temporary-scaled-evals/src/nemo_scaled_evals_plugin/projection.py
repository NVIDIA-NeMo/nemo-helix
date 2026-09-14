# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project evaluation rows into Entity Store and read them back.

The writer is driven by the controller and is the only producer, so a row never
needs a cross-entity transaction to stay consistent. The reader reproduces the
SQL predicates in `EvaluationRepository.list`/`get` closely enough that the
existing response schemas are rebuilt byte-for-byte from the projection.
"""

from __future__ import annotations

import builtins
import datetime as dt
import json
import logging
from threading import Lock
from typing import Any

from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.entities.base import EntityNotFoundError, SyncEntityClient
from nemo_platform_plugin.entities.client import EntitiesClient
from nemo_platform_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_platform_plugin.sdk_provider import get_platform_sdk
from nemo_scaled_evals_plugin.entities import PROJECTED_COLUMNS, ScaledEvaluation, searchable_blob
from scaled_evals.api.repositories.base_repository import normalize_order, substring_search_pattern
from scaled_evals.api.schemas.common import decode_cursor
from scaled_evals.api.settings import settings

LOG = logging.getLogger(__name__)

# Entity-store filters address promoted fields through the `data` JSON column.
# Only `filter_obj` gets this prefix added for us; a structured FilterOperation
# is serialized verbatim, so it has to be spelled out here.
_DATA = "data"


def _field(name: str) -> str:
    return f"{_DATA}.{name}"


def _eq(name: str, value: Any) -> ComparisonOperation:
    return ComparisonOperation(operator=FilterOperator.EQ, field=_field(name), value=value)


def _all(operations: list[FilterOperation]) -> FilterOperation:
    if len(operations) == 1:
        return operations[0]
    return LogicalOperation(operator=FilterOperator.AND, operations=operations)


def jsonable(value: Any) -> Any:
    """Return `value` with timestamps and other SQL scalars made JSON-safe."""
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


def projected_detail(row: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-safe subset of `row` the responses are rebuilt from."""
    return jsonable({key: value for key, value in row.items() if key in PROJECTED_COLUMNS})


def row_to_entity(row: dict[str, Any], *, workspace: str) -> ScaledEvaluation:
    """Build the projection entity for one evaluation row."""
    return ScaledEvaluation(
        name=str(row["id"]),
        workspace=workspace,
        evaluation_id=str(row["id"]),
        owner_id=row.get("owner_id"),
        task_id=str(row["task_id"]),
        benchmark_run_id=row.get("benchmark_run_id"),
        standalone=row.get("benchmark_run_id") is None,
        status=str(row["status"]),
        visibility=str(row["visibility"]),
        framework=str(row["framework"]),
        runtime=str(row["runtime"]),
        deleted=row.get("deleted_at") is not None,
        row_created_at=row["created_at"],
        row_updated_at=row["updated_at"],
        search_blob=searchable_blob(row),
        detail=projected_detail(row),
    )


def entity_to_row(entity: ScaledEvaluation) -> dict[str, Any]:
    """Return the projected row, shaped as the SQL read paths returned it."""
    return dict(entity.detail)


class EvaluationProjectionWriter:
    """Upsert evaluation rows into Entity Store, newest change first."""

    def __init__(self, client: SyncEntityClient, *, workspace: str) -> None:
        self._client = client
        self._workspace = workspace

    def watermark(self) -> dt.datetime | None:
        """Return the newest `row_updated_at` already projected, if any.

        Recovering the watermark from the projection keeps the writer stateless:
        no new SQL column to add now and delete later, and a restarted
        controller resumes instead of replaying every row.
        """
        page = self._client.list(
            ScaledEvaluation,
            workspace=self._workspace,
            sort="-row_updated_at",
            page_size=1,
        )
        if not page.data:
            return None
        return page.data[0].row_updated_at

    def project(self, row: dict[str, Any]) -> None:
        """Write one row's projection, overwriting any earlier version."""
        entity = row_to_entity(row, workspace=self._workspace)
        try:
            existing = self._client.get(ScaledEvaluation, entity.name, workspace=self._workspace)
        except EntityNotFoundError:
            self._client.create(entity)
            return
        # Copy the fresh projection onto the stored object rather than updating
        # the new one: `id` and `db_version` are read-only views over private
        # attrs, and carrying them across is what makes this a compare-and-swap.
        # A racing write loses the swap and is retried on the next pass.
        for field in ScaledEvaluation.model_fields:
            setattr(existing, field, getattr(entity, field))
        self._client.update(existing)


class EvaluationProjectionReader:
    """Serve the evaluation list/get reads from the projection.

    Method signatures mirror `EvaluationRepository` so the routers only choose a
    source; they do not change shape.
    """

    def __init__(self, client: SyncEntityClient, *, workspace: str) -> None:
        self._client = client
        self._workspace = workspace

    def list(
        self,
        *,
        limit: int,
        cursor: str | None,
        order: str,
        status: str | None,
        task_id: str | None,
        shared: bool,
        benchmark_run_id: str | None = None,
        owner_id: str | None = None,
        q: str | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """Return up to `limit + 1` projected rows, matching the SQL ordering."""
        direction = normalize_order(order)
        conditions: list[FilterOperation] = [_eq("deleted", False)]
        if owner_id is not None:
            conditions.append(_eq("owner_id", owner_id))
        if benchmark_run_id is not None:
            conditions.append(_eq("benchmark_run_id", benchmark_run_id))
        else:
            conditions.append(_eq("standalone", True))
        if status is not None:
            conditions.append(_eq("status", status))
        if task_id is not None:
            conditions.append(_eq("task_id", task_id))
        if shared:
            conditions.append(
                ComparisonOperation(
                    operator=FilterOperator.NIN,
                    field=_field("visibility"),
                    value=["private"],
                )
            )
        # ponytail: the pattern is backslash-escaped for SQL's `ESCAPE '\'`,
        # which `$like` does not honour, so a query containing % or _ matches
        # less here than in Postgres. Under-matching, never over-matching, so
        # no row leaks; revisit if the store documents an escape.
        if pattern := substring_search_pattern(q):
            conditions.append(
                ComparisonOperation(
                    operator=FilterOperator.LIKE,
                    field=_field("search_blob"),
                    value=pattern.lower(),
                )
            )
        if keyset := self._keyset(cursor, direction):
            conditions.append(keyset)

        prefix = "" if direction == "asc" else "-"
        page = self._client.list(
            ScaledEvaluation,
            workspace=self._workspace,
            filter_operation=_all(conditions),
            sort=f"{prefix}row_created_at",
            page_size=limit + 1,
        )
        # ponytail: the store sorts one field, so the (created_at, id)
        # tiebreaker is reapplied here. That orders the page correctly but does
        # not decide which rows the store picked, so evaluations sharing a
        # created_at across a page boundary can still repeat or be skipped.
        # Ceiling accepted while Postgres is authoritative; the fix is a
        # composite sort key in the store, not more local sorting.
        entities = sorted(
            page.data,
            key=lambda item: (item.row_created_at, item.evaluation_id),
            reverse=direction == "desc",
        )
        return [entity_to_row(entity) for entity in entities]

    def get(self, evaluation_id: str) -> dict[str, Any] | None:
        """Return one projected row, or None when absent or soft-deleted."""
        try:
            entity = self._client.get(ScaledEvaluation, evaluation_id, workspace=self._workspace)
        except EntityNotFoundError:
            return None
        if entity.deleted:
            return None
        return entity_to_row(entity)

    def _keyset(self, cursor: str | None, direction: str) -> FilterOperation | None:
        """Return the `(created_at, id)` row comparison the SQL cursor encodes."""
        position = decode_cursor(cursor)
        if position is None:
            return None
        operator = FilterOperator.GT if direction == "asc" else FilterOperator.LT
        created_at = jsonable(position.created_at)
        return LogicalOperation(
            operator=FilterOperator.OR,
            operations=[
                ComparisonOperation(operator=operator, field=_field("row_created_at"), value=created_at),
                _all(
                    [
                        _eq("row_created_at", created_at),
                        ComparisonOperation(
                            operator=operator,
                            field=_field("evaluation_id"),
                            value=position.id,
                        ),
                    ]
                ),
            ],
        )


_reader: EvaluationProjectionReader | None = None
_reader_lock = Lock()


def evaluation_reader() -> EvaluationProjectionReader:
    """Return the process-wide projection reader, building it on first use.

    Sync routes run in a threadpool, so the client is shared rather than rebuilt
    per request; one HTTP client per process, not per read.
    """
    global _reader
    with _reader_lock:
        if _reader is None:
            entities = client_from_platform(
                get_platform_sdk(as_service="scaled-evals", internal=True),
                EntitiesClient,
            )
            _reader = EvaluationProjectionReader(
                SyncEntityClient(entities),
                workspace=settings.entity_store_workspace,
            )
        return _reader


def parity_report(row: dict[str, Any], projected: dict[str, Any]) -> list[str]:
    """Return the fields where a Postgres row and its read-back projection differ.

    `projected` must come from a real read (`EvaluationProjectionReader.get`),
    not from re-running the mapping: the drift worth catching is whatever the
    store does to the payload in transit, and comparing the mapping against
    itself would always agree.
    """
    expected = projected_detail(row)
    # Compare through JSON so tuple/list and int/float encodings agree the way
    # they would after a real round trip through the store.
    differences = []
    for key in sorted(set(expected) | set(projected)):
        if json.dumps(expected.get(key), sort_keys=True) != json.dumps(projected.get(key), sort_keys=True):
            differences.append(key)
    return differences
