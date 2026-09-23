# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span repository tests."""

from datetime import datetime, timezone

import pytest
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.span import (
    SPAN_COLUMNS,
    SPAN_GROUP_COLUMN_FIELDS,
    SPAN_GROUP_SORT_FIELDS,
    ClickHouseSpanRepository,
    _group_order_by,
    _order_by,
)
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.spans.api.spans_schemas import SpanGroupBy, SpanGroupSortField
from nhx.intake.spans.domain import SpanListFilter, SpanStatus
from nhx.intake.spans.storage import make_pagination


class _QueryResult:
    def __init__(self, rows: list[tuple[object, ...]], columns: list[str] | None = None) -> None:
        self.result_rows = rows
        self.column_names = columns or (["count()"] if rows and len(rows[0]) == 1 else [])


class _Client(ClickHouseExecutor):
    def __init__(self, query_results: list[_QueryResult] | None = None) -> None:
        self.queries: list[str] = []
        self.parameters: list[dict[str, object]] = []
        self.query_results = query_results or []

    def table(self, table: ClickHouseTable) -> str:
        return table.value

    async def fetch_all(self, query: ClickHouseQuery) -> list[dict[str, object]]:
        self.queries.append(query.statement)
        self.parameters.append(dict(query.parameters))
        if self.query_results:
            result = self.query_results.pop(0)
        elif query.statement.lstrip().startswith("SELECT count()"):
            result = _QueryResult([(0,)], ["count()"])
        else:
            result = _QueryResult([])
        return [dict(zip(result.column_names, row, strict=True)) for row in result.result_rows]


_GROUP_STARTED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _repository(client: _Client) -> ClickHouseSpanRepository:
    return ClickHouseSpanRepository(client)


def test_order_by_whitelists_supported_span_sort_keys():
    assert (
        _order_by("started_at") == "start_time ASC, nullIf(attributes_number['nemo.step_id'], 0) ASC NULLS LAST, id ASC"
    )
    assert (
        _order_by("-started_at")
        == "start_time DESC, nullIf(attributes_number['nemo.step_id'], 0) DESC NULLS LAST, id ASC"
    )


def test_order_by_rejects_unsupported_span_sort_keys():
    with pytest.raises(ValueError, match="Unsupported span sort field"):
        _order_by("started_at DESC; DROP TABLE spans")


def test_make_pagination_computes_total_pages():
    pagination = make_pagination(page=2, page_size=25, current_page_size=10, total_results=60)

    assert pagination.page == 2
    assert pagination.page_size == 25
    assert pagination.current_page_size == 10
    assert pagination.total_results == 60
    assert pagination.total_pages == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"page": 0, "page_size": 10, "current_page_size": 1, "total_results": 1}, "page must be >= 1"),
        ({"page": 1, "page_size": 0, "current_page_size": 0, "total_results": 0}, "page_size must be >= 1"),
        (
            {"page": 1, "page_size": 10, "current_page_size": 11, "total_results": 11},
            "current_page_size must be between 0 and page_size",
        ),
        (
            {"page": 1, "page_size": 10, "current_page_size": 0, "total_results": -1},
            "total_results must be >= 0",
        ),
    ],
)
def test_make_pagination_rejects_invalid_inputs(kwargs: dict[str, int], match: str):
    with pytest.raises(ValueError, match=match):
        make_pagination(**kwargs)


@pytest.mark.asyncio
async def test_list_spans_counts_final_rows():
    client = _Client()
    repository = _repository(client)

    await repository.list_spans(
        filters=SpanListFilter(workspace="workspace-a"),
        page=1,
        page_size=10,
        sort="started_at",
        mode="summary",
    )

    assert client.queries[0] == "SELECT count() FROM spans FINAL WHERE workspace = %(workspace)s AND is_deleted = 0"


@pytest.mark.asyncio
async def test_list_spans_reads_final_rows():
    client = _Client()
    repository = _repository(client)

    await repository.list_spans(
        filters=SpanListFilter(workspace="workspace-a"),
        page=1,
        page_size=10,
        sort="started_at",
        mode="summary",
    )

    assert "FROM spans FINAL" in client.queries[1]
    assert "'' AS input" in client.queries[1]
    assert "'' AS output" in client.queries[1]
    assert "payload_char_limit" not in client.parameters[1]


@pytest.mark.asyncio
async def test_list_spans_preview_truncates_payloads_in_clickhouse():
    client = _Client()
    repository = _repository(client)

    await repository.list_spans(
        filters=SpanListFilter(workspace="workspace-a"),
        page=1,
        page_size=10,
        sort="started_at",
        mode="preview",
    )

    assert "substringUTF8(input, 1, %(payload_char_limit)s) AS input" in client.queries[1]
    assert "substringUTF8(output, 1, %(payload_char_limit)s) AS output" in client.queries[1]
    assert client.parameters[1]["payload_char_limit"] == 300


@pytest.mark.asyncio
async def test_list_spans_detailed_reads_full_payloads():
    client = _Client()
    repository = _repository(client)

    await repository.list_spans(
        filters=SpanListFilter(workspace="workspace-a"),
        page=1,
        page_size=10,
        sort="started_at",
        mode="detailed",
    )

    assert "input AS input" in client.queries[1]
    assert "output AS output" in client.queries[1]
    assert "substringUTF8(input" not in client.queries[1]
    assert "payload_char_limit" not in client.parameters[1]


def test_span_group_by_enum_matches_repository_columns():
    assert {field.value for field in SpanGroupBy} == set(SPAN_GROUP_COLUMN_FIELDS)


@pytest.mark.asyncio
async def test_list_span_groups_groups_by_columns():
    client = _Client(
        query_results=[
            _QueryResult([(2,)]),
            _QueryResult(
                [
                    ("session-a", "trace-a", 3, _GROUP_STARTED_AT),
                    ("session-b", "trace-b", 1, _GROUP_STARTED_AT),
                ],
                ["session_id", "trace_id", "span_count", "started_at"],
            ),
        ]
    )
    repository = _repository(client)

    result = await repository.list_span_groups(
        filters=SpanListFilter(workspace="workspace-a"),
        group_by=["session_id", "trace_id"],
        page=1,
        page_size=10,
        sort="-span_count",
    )

    assert result.pagination.total_results == 2
    assert result.data[0].group == {"session_id": "session-a", "trace_id": "trace-a"}
    assert result.data[0].span_count == 3
    assert result.data[0].started_at == _GROUP_STARTED_AT
    assert "FROM spans FINAL" in client.queries[0]
    assert "GROUP BY session_id, trace_id" in client.queries[0]
    assert "ORDER BY span_count DESC, session_id ASC, trace_id ASC" in client.queries[1]


@pytest.mark.asyncio
async def test_list_span_groups_sorts_by_time_so_recent_work_is_one_call():
    """Newest first over groups is what makes 'what did this agent run lately' cheap.

    Without it a caller has to page spans newest first and collect distinct ids, which
    only ever gives the traces that fit inside the scanned window.
    """
    client = _Client(
        query_results=[
            _QueryResult([(2,)]),
            _QueryResult(
                [
                    ("trace-new", 1, _GROUP_STARTED_AT),
                    ("trace-old", 9, datetime(2026, 1, 1, tzinfo=timezone.utc)),
                ],
                ["trace_id", "span_count", "started_at"],
            ),
        ]
    )
    repository = _repository(client)

    result = await repository.list_span_groups(
        filters=SpanListFilter(workspace="workspace-a"),
        group_by=["trace_id"],
        page=1,
        page_size=10,
        sort="-started_at",
    )

    assert "min(start_time) AS started_at" in client.queries[0]
    assert "ORDER BY started_at DESC, trace_id ASC" in client.queries[1]
    # The newest group leads even though it is the smallest.
    assert [group.group["trace_id"] for group in result.data] == ["trace-new", "trace-old"]


def test_group_order_by_whitelists_supported_group_sort_keys():
    assert _group_order_by("started_at", ["trace_id"]) == "started_at ASC, trace_id ASC"
    assert _group_order_by("-started_at", ["trace_id"]) == "started_at DESC, trace_id ASC"
    assert _group_order_by("-span_count", ["trace_id"]) == "span_count DESC, trace_id ASC"


def test_group_order_by_rejects_unsupported_group_sort_keys():
    with pytest.raises(ValueError, match="Unsupported span group sort field"):
        _group_order_by("started_at DESC; DROP TABLE spans", ["trace_id"])


def test_group_sort_enum_matches_repository_sort_fields():
    assert {field.value.removeprefix("-") for field in SpanGroupSortField} == SPAN_GROUP_SORT_FIELDS


@pytest.mark.asyncio
async def test_list_span_groups_reuses_span_filters():
    client = _Client()
    repository = _repository(client)

    await repository.list_span_groups(
        filters=SpanListFilter(workspace="workspace-a", status=SpanStatus.ERROR),
        group_by=["trace_id"],
        page=1,
        page_size=10,
        sort="-span_count",
    )

    assert "status = %(status)s" in client.queries[0]
    assert "GROUP BY trace_id" in client.queries[0]
    assert client.parameters[0]["status"] == "error"


@pytest.mark.asyncio
async def test_list_span_groups_rejects_unsupported_group_field():
    client = _Client()
    repository = _repository(client)

    with pytest.raises(ValueError, match="Unsupported span group-by field"):
        await repository.list_span_groups(
            filters=SpanListFilter(workspace="workspace-a"),
            group_by=["trace_id; DROP TABLE spans"],
            page=1,
            page_size=10,
            sort="-span_count",
        )


@pytest.mark.asyncio
async def test_get_span_prefers_external_span_id_over_numeric_internal_id():
    started_at = datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
    started_at_us = 1767225600123456
    row = _span_row(internal_id=7, external_span_id="123", started_at=started_at)
    client = _Client(
        query_results=[
            _QueryResult(
                [(7, "session-a", started_at_us, 0)],
                ["id", "session_id", "start_time_us", "current_is_deleted"],
            ),
            _QueryResult([row], SPAN_COLUMNS),
        ]
    )
    repository = _repository(client)

    span = await repository.get_span(workspace="workspace-a", span_id="123")

    assert span is not None
    assert span.id == 7
    assert span.external_span_id == "123"
    assert len(client.queries) == 2
    assert "FINAL" not in client.queries[0]
    assert "external_span_id = %(span_id)s" in client.queries[0]
    assert client.parameters[0] == {"workspace": "workspace-a", "span_id": "123"}
    assert "FROM spans FINAL" in client.queries[1]
    assert "session_id = %(session_id)s" in client.queries[1]
    assert "start_time = fromUnixTimestamp64Micro(%(start_time_us)s)" in client.queries[1]
    assert "id = %(id)s" in client.queries[1]
    assert client.parameters[1] == {
        "workspace": "workspace-a",
        "session_id": "session-a",
        "start_time_us": started_at_us,
        "id": 7,
    }


@pytest.mark.asyncio
async def test_get_span_does_not_fall_back_to_internal_id_after_external_miss():
    client = _Client(query_results=[_QueryResult([], ["id", "session_id", "start_time_us", "current_is_deleted"])])
    repository = _repository(client)

    span = await repository.get_span(workspace="workspace-a", span_id="123")

    assert span is None
    assert len(client.queries) == 1
    assert "external_span_id = %(span_id)s" in client.queries[0]
    assert client.parameters[0] == {"workspace": "workspace-a", "span_id": "123"}


def _span_row(*, internal_id: int, external_span_id: str, started_at: datetime) -> tuple[object, ...]:
    zero_time = datetime.fromtimestamp(0, tz=timezone.utc)
    values: dict[str, object] = {
        "workspace": "workspace-a",
        "session_id": "session-a",
        "trace_id": "trace-a",
        "id": internal_id,
        "source_format": "test",
        "external_span_id": external_span_id,
        "external_parent_span_id": "",
        "kind": "LLM",
        "name": "span-a",
        "status": "success",
        "start_time": started_at,
        "end_time": zero_time,
        "attributes_string": {},
        "attributes_number": {},
        "attributes_bool": {},
        "input": "",
        "output": "",
        "event_ts": started_at,
        "is_deleted": 0,
    }
    return tuple(values[column] for column in SPAN_COLUMNS)
