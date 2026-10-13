# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace repository tests."""

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseExternalData, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.clickhouse.trace import TRACE_COLUMNS, ClickHouseTraceRepository, _order_by
from nhx.intake.spans.domain import TraceListFilter

_TRACE_PAGE_COLUMNS = [*TRACE_COLUMNS, "started_at_us"]


class _QueryResult:
    def __init__(self, rows: list[tuple[object, ...]], columns: list[str] | None = None) -> None:
        self.result_rows = rows
        self.column_names = columns or (["count()"] if rows and len(rows[0]) == 1 else [])


class _Client(ClickHouseExecutor):
    def __init__(self, query_results: list[_QueryResult] | None = None) -> None:
        self.queries: list[str] = []
        self.parameters: list[dict[str, object]] = []
        self.external_data: list[ClickHouseExternalData | None] = []
        self.query_results = query_results or []

    def table(self, table: ClickHouseTable) -> str:
        return table.value

    async def fetch_all(self, query: ClickHouseQuery) -> list[dict[str, object]]:
        self.queries.append(query.statement)
        self.parameters.append(dict(query.parameters))
        self.external_data.append(query.external_data)
        if self.query_results:
            result = self.query_results.pop(0)
        elif query.statement.lstrip().startswith("SELECT count()"):
            result = _QueryResult([(0,)], ["count()"])
        else:
            result = _QueryResult([])
        return [dict(zip(result.column_names, row, strict=True)) for row in result.result_rows]


def _repository(client: _Client) -> ClickHouseTraceRepository:
    return ClickHouseTraceRepository(client)


def _with_started_at_us(row: tuple[object, ...], started_at_us: int) -> tuple[object, ...]:
    return (*row, started_at_us)


def test_order_by_whitelists_supported_trace_sort_keys():
    assert _order_by("started_at") == "started_at ASC, id ASC"
    assert _order_by("-started_at") == "started_at DESC, id ASC"


def test_order_by_rejects_unsupported_trace_sort_keys():
    with pytest.raises(ValueError, match="Unsupported trace sort field"):
        _order_by("started_at DESC; DROP TABLE spans")


@pytest.mark.asyncio
async def test_summary_mode_reads_root_spans_without_metric_aggregates():
    client = _Client()
    repository = _repository(client)

    await repository.list_traces(
        filters=TraceListFilter(workspace="workspace-a", trace_ids=["trace-a", "trace-b"]),
        page=1,
        page_size=10,
        sort="started_at",
        mode="summary",
    )

    assert client.queries[0].lstrip().startswith("SELECT count()")
    assert "FROM trace_index AS trace_roots FINAL" in client.queries[0]
    assert "trace_roots.root_status AS status" in client.queries[0]
    assert "trace_roots.root_input" not in client.queries[0]
    assert "trace_roots.root_output" not in client.queries[0]
    assert "LIMIT 1 BY trace_roots.workspace, trace_roots.source_format, trace_roots.trace_id" in client.queries[0]
    assert "span_versions" not in client.queries[0]
    assert "trace_roots.trace_id IN %(trace_ids)s" in client.queries[0]
    assert client.parameters[0]["trace_ids"] == ["trace-a", "trace-b"]
    assert "sumIf" not in client.queries[1]
    assert "groupUniqArrayIf" not in client.queries[1]
    assert "span_versions" not in client.queries[1]
    assert "trace_roots.root_input" not in client.queries[1]
    assert "trace_roots.root_output" not in client.queries[1]
    assert "'' AS input" in client.queries[1]
    assert "'' AS output" in client.queries[1]
    assert "payload_char_limit" not in client.parameters[1]


@pytest.mark.asyncio
async def test_latest_trace_started_at_by_group_aggregates_all_references_in_one_query():
    latest = datetime(2026, 1, 2, tzinfo=timezone.utc)
    client = _Client(query_results=[_QueryResult([("insight-a", latest)], ["group_id", "started_at"])])
    repository = _repository(client)

    result = await repository.latest_trace_started_at_by_group(
        workspace="workspace-a",
        trace_refs_by_group={
            "insight-a": ["trace-old", "trace-new"],
            "insight-empty": [],
            "insight-missing": ["trace-missing"],
        },
    )

    assert result == {"insight-a": latest}
    assert len(client.queries) == 1
    assert "FROM trace_refs" in client.queries[0]
    assert "max(traces.started_at) AS started_at" in client.queries[0]
    assert "GROUP BY refs.group_id" in client.queries[0]
    assert client.parameters[0] == {"workspace": "workspace-a"}
    external_data = client.external_data[0]
    assert external_data is not None
    assert external_data.fmt == "JSONEachRow"
    assert external_data.structure == "group_id String, trace_id String"
    assert [json.loads(line) for line in external_data.data.splitlines()] == [
        {"group_id": "insight-a", "trace_id": "trace-old"},
        {"group_id": "insight-a", "trace_id": "trace-new"},
        {"group_id": "insight-missing", "trace_id": "trace-missing"},
    ]


@pytest.mark.asyncio
async def test_preview_mode_bounds_payloads_and_adds_trace_aggregate_block():
    started_at = datetime(2026, 1, 1, microsecond=123456, tzinfo=timezone.utc)
    page_row = _with_started_at_us(
        _trace_row(started_at=started_at, ended_at=None, ingested_at=started_at, detailed=False),
        1767225600123456,
    )
    client = _Client(
        query_results=[
            _QueryResult([(1,)]),
            _QueryResult([page_row], _TRACE_PAGE_COLUMNS),
        ]
    )
    repository = _repository(client)

    await repository.list_traces(
        filters=TraceListFilter(workspace="workspace-a"),
        page=1,
        page_size=10,
        sort="started_at",
        mode="preview",
    )

    assert len(client.queries) == 3
    assert "trace_roots.root_input" not in client.queries[1]
    assert "trace_roots.root_output" not in client.queries[1]
    assert "toUnixTimestamp64Micro(traces.started_at) AS started_at_us" in client.queries[1]
    assert "LIMIT %(limit)s OFFSET %(offset)s" in client.queries[1]
    assert "substringUTF8(trace_roots.root_input, 1, %(payload_char_limit)s)" in client.queries[2]
    assert "substringUTF8(trace_roots.root_output, 1, %(payload_char_limit)s)" in client.queries[2]
    assert client.parameters[2]["payload_char_limit"] == 300
    assert "sumIf" in client.queries[2]
    assert "count() AS span_count" in client.queries[2]
    assert "page_traces" not in client.queries[2]
    assert "trace_page_refs" not in client.queries[2]
    assert "IN %(page_trace_keys)s" in client.queries[2]
    assert "trace_id IN %(page_trace_ids)s" in client.queries[2]
    assert "LIMIT 1 BY trace_roots.workspace, trace_roots.source_format, trace_roots.trace_id" in client.queries[2]
    assert "indexOf(%(page_trace_keys)s" not in client.queries[2]
    assert client.external_data[2] is None
    assert client.parameters[2]["page_trace_ids"] == ["trace-a"]
    assert client.parameters[2]["page_trace_keys"] == [("otel", "trace-a")]
    assert "root_started_at >= fromUnixTimestamp64Micro(%(page_started_at_min_us)s)" in client.queries[2]
    assert "root_started_at <= fromUnixTimestamp64Micro(%(page_started_at_max_us)s)" in client.queries[2]
    assert client.parameters[2]["page_started_at_min_us"] == 1767225600123456
    assert client.parameters[2]["page_started_at_max_us"] == 1767225600123456
    assert "page_root_keys" not in client.parameters[2]


@pytest.mark.asyncio
async def test_detailed_mode_adds_trace_aggregate_block():
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    page_row = _with_started_at_us(
        _trace_row(started_at=started_at, ended_at=None, ingested_at=started_at, detailed=False),
        1767225600000000,
    )
    client = _Client(
        query_results=[
            _QueryResult([(1,)]),
            _QueryResult([page_row], _TRACE_PAGE_COLUMNS),
        ]
    )
    repository = _repository(client)

    await repository.list_traces(
        filters=TraceListFilter(workspace="workspace-a"),
        page=1,
        page_size=10,
        sort="started_at",
        mode="detailed",
    )

    assert "FROM trace_index AS trace_roots FINAL" in client.queries[0]
    assert "span_versions" not in client.queries[0]
    assert "trace_roots.root_input" not in client.queries[1]
    assert "trace_roots.root_output" not in client.queries[1]
    assert "AS trace_spans" in client.queries[2]
    assert "span_versions.trace_id IN %(page_trace_ids)s" in client.queries[2]
    assert "(span_versions.source_format, span_versions.trace_id) IN %(page_trace_keys)s" in client.queries[2]
    assert "FROM spans AS span_versions FINAL" in client.queries[2]
    assert "argMax(span_versions." not in client.queries[2]
    assert "span_versions.attributes_string AS attributes_string" in client.queries[2]
    assert "span_versions.attributes_number AS attributes_number" in client.queries[2]
    assert "sumIf" in client.queries[2]
    assert "groupUniqArrayIf" in client.queries[2]
    assert "count() AS span_count" in client.queries[2]
    assert "trace_roots.root_input AS input" in client.queries[2]
    assert "trace_roots.root_output AS output" in client.queries[2]
    assert "substringUTF8(trace_roots.root_input" not in client.queries[2]
    assert "payload_char_limit" not in client.parameters[2]


@pytest.mark.asyncio
async def test_list_traces_maps_detailed_row():
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ended_at = started_at + timedelta(milliseconds=2500)
    ingested_at = started_at + timedelta(seconds=3)
    row = _trace_row(started_at=started_at, ended_at=ended_at, ingested_at=ingested_at)
    page_row = _with_started_at_us(
        _trace_row(started_at=started_at, ended_at=ended_at, ingested_at=ingested_at, detailed=False),
        1767225600000000,
    )
    client = _Client(
        query_results=[
            _QueryResult([(1,)]),
            _QueryResult([page_row], _TRACE_PAGE_COLUMNS),
            _QueryResult([row], TRACE_COLUMNS),
        ]
    )
    repository = _repository(client)

    result = await repository.list_traces(
        filters=TraceListFilter(workspace="workspace-a"),
        page=1,
        page_size=10,
        sort="-started_at",
        mode="detailed",
    )

    trace = result.data[0]
    assert trace.id == "trace-a"
    assert trace.session_id == "session-a"
    assert trace.root_span_id == "span-root"
    assert trace.name == "root"
    assert trace.input == "root input"
    assert trace.output == "root output"
    assert trace.duration_ms == 2500
    assert trace.project == "project-a"
    assert trace.evaluation_name == "experiment-a"
    assert trace.test_case_name == "case-a"
    assert trace.input_tokens == 420
    assert trace.output_tokens == 310
    assert trace.cached_tokens == 128
    assert trace.total_tokens == 858
    assert trace.cost_usd == 0.0061
    assert trace.cost_input_usd == 0.0024
    assert trace.cost_output_usd == 0.0037
    assert trace.models == ["model-a", "model-b"]
    assert trace.providers == ["openai"]
    assert trace.span_count == 3
    assert trace.error_count == 1


@pytest.mark.asyncio
async def test_non_summary_empty_page_skips_hydration_query():
    client = _Client(query_results=[_QueryResult([(0,)])])
    repository = _repository(client)

    result = await repository.list_traces(
        filters=TraceListFilter(workspace="workspace-a"),
        page=1,
        page_size=10,
        sort="-started_at",
        mode="preview",
    )

    assert result.data == []
    assert len(client.queries) == 2
    assert all(external_data is None for external_data in client.external_data)


@pytest.mark.asyncio
async def test_non_summary_reconciles_hydration_misses_and_restores_page_order(caplog: pytest.LogCaptureFixture):
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    page_rows = [
        _with_started_at_us(
            _trace_row(
                trace_id=trace_id,
                started_at=started_at,
                ended_at=None,
                ingested_at=started_at,
                detailed=False,
            ),
            1767225600000000,
        )
        for trace_id in ("trace-a", "trace-b", "trace-c")
    ]
    hydrated_rows = [
        _trace_row(trace_id=trace_id, started_at=started_at, ended_at=None, ingested_at=started_at)
        for trace_id in ("trace-c", "trace-a")
    ]
    client = _Client(
        query_results=[
            _QueryResult([(3,)]),
            _QueryResult(page_rows, _TRACE_PAGE_COLUMNS),
            _QueryResult(hydrated_rows, TRACE_COLUMNS),
        ]
    )
    repository = _repository(client)

    with caplog.at_level(logging.WARNING):
        result = await repository.list_traces(
            filters=TraceListFilter(workspace="workspace-a"),
            page=1,
            page_size=10,
            sort="-started_at",
            mode="preview",
        )

    assert [trace.id for trace in result.data] == ["trace-a", "trace-c"]
    assert result.pagination.current_page_size == 2
    assert result.pagination.total_results == 2
    assert result.pagination.total_pages == 1
    record = caplog.records[-1]
    assert record.message == "Trace page hydration omitted refs returned by the page query"
    assert getattr(record, "dropped_trace_refs") == [("otel", "trace-b")]


@pytest.mark.asyncio
async def test_summary_mode_maps_no_aggregate_fields():
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = _trace_row(started_at=started_at, ended_at=None, ingested_at=started_at, detailed=False)
    client = _Client(query_results=[_QueryResult([(1,)]), _QueryResult([row], TRACE_COLUMNS)])
    repository = _repository(client)

    result = await repository.list_traces(
        filters=TraceListFilter(workspace="workspace-a"),
        page=1,
        page_size=10,
        sort="-started_at",
        mode="summary",
    )

    trace = result.data[0]
    assert trace.status.value == "error"
    assert trace.input is None
    assert trace.output is None
    assert trace.input_tokens is None
    assert trace.total_tokens is None
    assert trace.cost_usd is None
    assert trace.models is None
    assert trace.providers is None
    assert trace.span_count is None
    assert trace.error_count is None


@pytest.mark.asyncio
async def test_trace_started_at_filter_is_applied_to_trace_index():
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    client = _Client()
    repository = _repository(client)

    await repository.list_traces(
        filters=TraceListFilter(workspace="workspace-a", started_at_gte=started_at),
        page=1,
        page_size=10,
        sort="started_at",
        mode="summary",
    )

    assert "trace_roots.root_started_at >= %(started_at_gte)s" in client.queries[0]
    assert client.parameters[0]["started_at_gte"] == started_at


@pytest.mark.asyncio
async def test_root_filters_use_trace_index_columns():
    client = _Client()
    repository = _repository(client)

    await repository.list_traces(
        filters=TraceListFilter(
            workspace="workspace-a",
            evaluation_name="experiment-a",
        ),
        page=1,
        page_size=10,
        sort="started_at",
        mode="detailed",
    )

    assert "trace_roots.evaluation_name = %(filter_evaluation_name)s" in client.queries[0]
    assert "candidate_spans" not in client.queries[0]
    assert client.parameters[0]["filter_evaluation_name"] == "experiment-a"


@pytest.mark.asyncio
async def test_agent_filter_uses_the_trace_index_column():
    client = _Client()
    repository = _repository(client)

    await repository.list_traces(
        filters=TraceListFilter(workspace="workspace-a", agent_name="support-bot"),
        page=1,
        page_size=10,
        sort="started_at",
        mode="detailed",
    )

    assert "trace_roots.agent_name = %(filter_agent_name)s" in client.queries[0]
    # Agent scoping must not fall back to probing the spans attribute map.
    assert "candidate_spans" not in client.queries[0]
    assert client.parameters[0]["filter_agent_name"] == "support-bot"


def _trace_row(
    *,
    trace_id: str = "trace-a",
    started_at: datetime,
    ended_at: datetime | None,
    ingested_at: datetime,
    detailed: bool = True,
) -> tuple[object, ...]:
    values: dict[str, object | None] = {
        "id": trace_id,
        "workspace": "workspace-a",
        "session_id": "session-a",
        "source_format": "otel",
        "root_span_id": "span-root",
        "name": "root",
        "input": "root input" if detailed else "",
        "output": "root output" if detailed else "",
        "project": "project-a",
        "evaluation_name": "experiment-a",
        "test_case_name": "case-a",
        "started_at": started_at,
        "ended_at": ended_at,
        "status": "error",
        "input_tokens": 420 if detailed else None,
        "output_tokens": 310 if detailed else None,
        "cached_tokens": 128 if detailed else None,
        "total_tokens": 858 if detailed else None,
        "cost_usd": 0.0061 if detailed else None,
        "cost_input_usd": 0.0024 if detailed else None,
        "cost_output_usd": 0.0037 if detailed else None,
        "models": ["model-a", "model-b"] if detailed else None,
        "providers": ["openai"] if detailed else None,
        "span_count": 3 if detailed else None,
        "error_count": 1 if detailed else None,
        "ingested_at": ingested_at,
    }
    return tuple(values[column] for column in TRACE_COLUMNS)


_METRIC_COLUMNS = [
    "bucket_start",
    "run_count",
    "failed_run_count",
    *[
        f"{field}_{suffix}"
        for field in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens", "cost_usd")
        for suffix in ("sum", "mean", "quantiles")
    ],
    "latency_ms_mean",
    "latency_ms_quantiles",
]


def _metric_row(*, bucket_start, run_count, failed_run_count, **rollups):
    """Build one _METRIC_COLUMNS-shaped row; unnamed rollups come back empty."""

    row = [bucket_start, run_count, failed_run_count]
    for field in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens", "cost_usd"):
        row.extend(rollups.get(field, (None, None, [])))
    row.extend(rollups.get("latency_ms", (None, [])))
    return tuple(row)


@pytest.mark.asyncio
async def test_trace_metrics_buckets_by_day_in_the_requested_timezone():
    bucket_start = datetime(2026, 8, 14, tzinfo=timezone.utc)
    client = _Client(
        [
            _QueryResult(
                [
                    _metric_row(
                        bucket_start=bucket_start,
                        run_count=3,
                        failed_run_count=1,
                        input_tokens=(300, 100.0, [180.0, 190.0]),
                        output_tokens=(150, 50.0, [90.0, 95.0]),
                        cost_usd=(0.5, 0.16, [0.3, 0.4]),
                        latency_ms=(2000.0, [1200.0, 3900.0, 4300.0, 4800.0]),
                    )
                ],
                _METRIC_COLUMNS,
            )
        ]
    )
    repository = _repository(client)

    points = await repository.trace_metrics(
        filters=TraceListFilter(workspace="workspace-a", agent_name="support-bot"),
        bucket="day",
        timezone_name="America/Los_Angeles",
    )

    query = client.queries[0]
    assert "toStartOfDay(trace_roots.root_started_at, %(metrics_timezone)s)" in query
    assert "trace_roots.agent_name = %(filter_agent_name)s" in query
    assert client.parameters[0]["metrics_timezone"] == "America/Los_Angeles"
    assert client.parameters[0]["filter_agent_name"] == "support-bot"

    assert len(points) == 1
    point = points[0]
    assert point.bucket_start == bucket_start
    assert point.run_count == 3
    assert point.failed_run_count == 1
    assert point.input_tokens.sum == 300
    assert point.input_tokens.mean == 100.0
    assert point.input_tokens.p90 == 180.0
    assert point.input_tokens.p99 == 190.0
    assert point.cost_usd.sum == 0.5
    assert point.latency_ms.mean == 2000.0
    assert point.latency_ms.p50 == 1200.0
    assert point.latency_ms.p95 == 4300.0
    assert point.latency_ms.p99 == 4800.0
    # A mean is not derivable from percentiles, so it must come from the query.
    assert "avg(roots.latency_ms) AS latency_ms_mean" in query
    # One combined aggregate per metric rather than a quantile state per percentile.
    assert "quantiles(0.9, 0.99)(rollups.input_tokens) AS input_tokens_quantiles" in query
    assert "quantiles(0.5, 0.9, 0.95, 0.99)(roots.latency_ms) AS latency_ms_quantiles" in query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bucket", "expected_sql"),
    [
        ("hour", "toStartOfHour(trace_roots.root_started_at, %(metrics_timezone)s)"),
        ("week", "toDateTime(toStartOfWeek(trace_roots.root_started_at, 1, %(metrics_timezone)s)"),
        ("month", "toDateTime(toStartOfMonth(trace_roots.root_started_at, %(metrics_timezone)s)"),
    ],
)
async def test_trace_metrics_supports_coarser_buckets(bucket, expected_sql):
    client = _Client([_QueryResult([], _METRIC_COLUMNS)])
    repository = _repository(client)

    await repository.trace_metrics(
        filters=TraceListFilter(workspace="workspace-a"),
        bucket=bucket,
        timezone_name="UTC",
    )

    assert expected_sql in client.queries[0]


@pytest.mark.asyncio
async def test_trace_metrics_total_bucket_collapses_to_one_row_without_a_start():
    client = _Client(
        [
            _QueryResult(
                [
                    _metric_row(
                        bucket_start=datetime(1970, 1, 1, tzinfo=timezone.utc),
                        run_count=5,
                        failed_run_count=0,
                    )
                ],
                _METRIC_COLUMNS,
            )
        ]
    )
    repository = _repository(client)

    points = await repository.trace_metrics(
        filters=TraceListFilter(workspace="workspace-a"),
        bucket="total",
        timezone_name="UTC",
    )

    assert "toStartOfDay" not in client.queries[0]
    # The constant bucket value is an implementation detail and must not surface.
    assert points[0].bucket_start is None
    assert points[0].run_count == 5
    # Traces with no token attributes report null rather than a misleading zero.
    assert points[0].input_tokens.sum is None
    assert points[0].input_tokens.p90 is None
    assert points[0].latency_ms.p50 is None


@pytest.mark.asyncio
async def test_trace_metrics_reads_tokens_from_final_spans():
    client = _Client([_QueryResult([], _METRIC_COLUMNS)])
    repository = _repository(client)

    await repository.trace_metrics(
        filters=TraceListFilter(workspace="workspace-a"),
        bucket="day",
        timezone_name="UTC",
    )

    query = client.queries[0]
    # spans is a ReplacingMergeTree, so the rollup must read through FINAL rather
    # than summing raw versions.
    assert "FROM spans AS span_versions FINAL" in query
    assert "argMax(span_versions." not in query
    assert "LEFT JOIN rollups" in query
    # session_id leads the spans sort key after workspace, so it must be carried from
    # the selected roots for the primary key to prune before the trace-id bloom filter.
    assert "span_versions.session_id IN (SELECT session_id FROM roots)" in query
    # The rollup CTE must be scoped by source_format too, not just trace_id, so a
    # trace_id shared across ingest formats does not drag in unrelated spans.
    assert (
        "(span_versions.source_format, span_versions.trace_id) IN (SELECT source_format, trace_id FROM roots)" in query
    )
    # Joined on the full trace identity, matching the trace hydration query.
    assert "roots.workspace = rollups.workspace" in query
    assert "roots.source_format = rollups.source_format" in query
