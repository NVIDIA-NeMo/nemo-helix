# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse migration helpers for Intake spans."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from nhx.intake.config import DEFAULT_SPAN_RETENTION_DAYS
from nhx.intake.spans.span_attribute_catalog import SpanAttributeField, bag_keys, spec_for_field

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ClickHouseMigrationSettings(Protocol):
    url: str
    user: str
    password: str
    database: str


@dataclass(frozen=True)
class ClickHouseUrl:
    host: str
    port: int
    secure: bool


def run_clickhouse_migrations(settings: ClickHouseMigrationSettings) -> None:
    """Run ClickHouse migrations for the configured database in order.

    Each migration's CREATE TABLE uses IF NOT EXISTS, so re-running an already
    applied migration is a no-op. The version table records what's been
    applied so future migrations only run new steps.
    """

    ensure_clickhouse_database(settings)
    client = _get_sync_client(settings, database=settings.database)
    try:
        _ensure_version_table(client, settings)
        applied_versions = _applied_versions(client, settings)
        for version, migrate in _MIGRATIONS:
            if version in applied_versions:
                continue
            migrate(client, settings)
            client.insert(
                "clickhouse_alembic_version",
                [[version]],
                column_names=["version_num"],
                database=settings.database,
            )
    finally:
        _close_client(client)


def ensure_clickhouse_database(settings: ClickHouseMigrationSettings) -> None:
    """Create the configured ClickHouse database."""

    client = _get_sync_client(settings, database="default")
    try:
        client.command(f"CREATE DATABASE IF NOT EXISTS {quote_clickhouse_identifier(settings.database)}")
    finally:
        _close_client(client)


def _ensure_version_table(client, settings: ClickHouseMigrationSettings) -> None:
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {_table(settings, "clickhouse_alembic_version")}
        (
            version_num String,
            dt DateTime DEFAULT now()
        )
        ENGINE = ReplacingMergeTree(dt)
        ORDER BY version_num
        """
    )


def _applied_versions(client, settings: ClickHouseMigrationSettings) -> set[str]:
    result = client.query(f"SELECT version_num FROM {_table(settings, 'clickhouse_alembic_version')} FINAL")
    return {str(row[0]) for row in result.result_rows}


def _create_spans_schema(client, settings: ClickHouseMigrationSettings) -> None:
    # WARNING: Lossy! This drops the spans table and recreates it. Breaking, but currently in internal development only.
    client.command(f"DROP TABLE IF EXISTS {_table(settings, 'spans')}")
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {_table(settings, "spans")}
        (
            workspace LowCardinality(String),
            session_id String,
            trace_id String,
            id UInt64 MATERIALIZED cityHash64(workspace, source_format, trace_id, external_span_id),
            source_format LowCardinality(String),
            external_span_id String,
            external_parent_span_id String DEFAULT '',

            kind LowCardinality(String) DEFAULT 'UNKNOWN',
            name String DEFAULT '',
            status LowCardinality(String) DEFAULT 'unknown',

            start_time DateTime64(6) CODEC(Delta(8), ZSTD(1)),
            end_time DateTime64(6) DEFAULT toDateTime64(0, 6) CODEC(Delta(8), ZSTD(1)),

            attributes_string Map(LowCardinality(String), String) CODEC(ZSTD(1)),
            attributes_number Map(LowCardinality(String), Float64) CODEC(ZSTD(1)),
            attributes_bool Map(LowCardinality(String), Bool) CODEC(ZSTD(1)),

            input String CODEC(ZSTD(3)),
            output String CODEC(ZSTD(3)),

            event_ts DateTime64(6),
            is_deleted UInt8 DEFAULT 0,

            INDEX idx_trace_id trace_id TYPE bloom_filter(0.001) GRANULARITY 1,
            INDEX idx_id id TYPE bloom_filter(0.001) GRANULARITY 1,
            INDEX idx_external_span_id external_span_id TYPE bloom_filter(0.01) GRANULARITY 1,
            INDEX idx_external_parent_id external_parent_span_id TYPE bloom_filter(0.01) GRANULARITY 1,

            INDEX idx_start_time start_time TYPE minmax GRANULARITY 1,
            INDEX idx_kind kind TYPE set(16) GRANULARITY 4,
            INDEX idx_status status TYPE set(4) GRANULARITY 4,
            INDEX idx_source_format source_format TYPE set(8) GRANULARITY 4,

            INDEX idx_attrs_str_key mapKeys(attributes_string) TYPE tokenbf_v1(1024, 2, 0) GRANULARITY 1,
            INDEX idx_attrs_str_val mapValues(attributes_string) TYPE ngrambf_v1(4, 5000, 2, 0) GRANULARITY 1,
            INDEX idx_attrs_num_key mapKeys(attributes_number) TYPE tokenbf_v1(1024, 2, 0) GRANULARITY 1,
            INDEX idx_attrs_num_val mapValues(attributes_number) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_attrs_bool_key mapKeys(attributes_bool) TYPE tokenbf_v1(1024, 2, 0) GRANULARITY 1,

            INDEX idx_input_fts lower(input) TYPE ngrambf_v1(4, 32768, 3, 0) GRANULARITY 1,
            INDEX idx_output_fts lower(output) TYPE ngrambf_v1(4, 32768, 3, 0) GRANULARITY 1
        )
        ENGINE = ReplacingMergeTree(event_ts, is_deleted)
        PARTITION BY toYYYYMM(start_time)
        PRIMARY KEY (workspace, session_id, start_time)
        ORDER BY (workspace, session_id, start_time, id)
        TTL toDate(start_time) + INTERVAL {DEFAULT_SPAN_RETENTION_DAYS} DAY
        SETTINGS
            index_granularity = 8192,
            ttl_only_drop_parts = 1
        """
    )


def _create_evaluator_results_schema(client, settings: ClickHouseMigrationSettings) -> None:
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {_table(settings, "evaluator_results")}
        (
            evaluator_result_id String,
            span_id String,
            session_id String,
            workspace LowCardinality(String),
            name LowCardinality(String),
            value Nullable(Float64),
            string_value Nullable(String),
            data_type Enum8(
                'NUMERIC' = 1,
                'CATEGORICAL' = 2,
                'BOOLEAN' = 3,
                'TEXT' = 4
            ),
            comment Nullable(String),
            created_by Nullable(String),
            created_at DateTime64(3),
            ingested_at DateTime64(3)
        )
        ENGINE = ReplacingMergeTree(ingested_at)
        ORDER BY (workspace, session_id, span_id, name, evaluator_result_id)
        """
    )


def _create_annotations_schema(client, settings: ClickHouseMigrationSettings) -> None:
    # `span_id` uses an empty-string default (matches `external_parent_span_id`
    # on `spans`) so it can participate cleanly in ORDER BY. Empty means
    # "session-level annotation".
    #
    # Skip indexes let direct-id lookups (GET /annotations/{annotation_id}) and
    # per-span listings (GET /spans/{span_id}/annotations) avoid scanning the
    # full workspace when leading ORDER BY keys aren't supplied.
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {_table(settings, "annotations")}
        (
            annotation_id String,
            workspace LowCardinality(String),
            span_id String DEFAULT '',
            session_id String,

            kind Enum8(
                'feedback' = 1,
                'label' = 2,
                'note' = 3,
                'metadata' = 4
            ),
            name LowCardinality(Nullable(String)),
            value_text Nullable(String),
            value_numeric Nullable(Float64),
            text Nullable(String) CODEC(ZSTD(3)),
            metadata Nullable(String) CODEC(ZSTD(3)),

            created_by Nullable(String),
            created_at DateTime64(3),
            ingested_at DateTime64(3),
            is_deleted UInt8 DEFAULT 0,

            INDEX idx_annotation_id annotation_id TYPE bloom_filter(0.001) GRANULARITY 1,
            INDEX idx_span_id span_id TYPE bloom_filter(0.01) GRANULARITY 1,
            INDEX idx_value_numeric value_numeric TYPE minmax GRANULARITY 1,
            INDEX idx_created_at created_at TYPE minmax GRANULARITY 1
        )
        ENGINE = ReplacingMergeTree(ingested_at, is_deleted)
        ORDER BY (workspace, session_id, span_id, kind, annotation_id)
        """
    )


def _add_evaluator_results_skip_indexes(client, settings: ClickHouseMigrationSettings) -> None:
    """Add skip indexes for direct-id lookups, per-span listings, and value/time range queries."""

    table = _table(settings, "evaluator_results")
    client.command(
        f"ALTER TABLE {table} ADD INDEX IF NOT EXISTS"
        " idx_evaluator_result_id evaluator_result_id TYPE bloom_filter(0.001) GRANULARITY 1"
    )
    client.command(
        f"ALTER TABLE {table} ADD INDEX IF NOT EXISTS idx_span_id span_id TYPE bloom_filter(0.01) GRANULARITY 1"
    )
    client.command(f"ALTER TABLE {table} ADD INDEX IF NOT EXISTS idx_value value TYPE minmax GRANULARITY 1")
    client.command(f"ALTER TABLE {table} ADD INDEX IF NOT EXISTS idx_created_at created_at TYPE minmax GRANULARITY 1")
    # Build the indexes for already-stored rows; new inserts populate them automatically.
    client.command(f"ALTER TABLE {table} MATERIALIZE INDEX idx_evaluator_result_id")
    client.command(f"ALTER TABLE {table} MATERIALIZE INDEX idx_span_id")
    client.command(f"ALTER TABLE {table} MATERIALIZE INDEX idx_value")
    client.command(f"ALTER TABLE {table} MATERIALIZE INDEX idx_created_at")


def _create_trace_index_schema(client, settings: ClickHouseMigrationSettings) -> None:
    """Create the root-span trace index and insert-time projection."""

    table = _table(settings, "trace_index")
    view = _table(settings, "trace_index_mv")
    client.command(f"DROP TABLE IF EXISTS {_table(settings, 'evaluation_sessions_mv')}")
    client.command(f"DROP TABLE IF EXISTS {_table(settings, 'evaluation_sessions')}")
    client.command(f"DROP TABLE IF EXISTS {view}")
    client.command(f"DROP TABLE IF EXISTS {table}")

    project_key = spec_for_field(SpanAttributeField.PROJECT).bag_key
    # Ingest writes canonical keys, while the backfill must preserve associations on historical rows.
    evaluation_name_expr = _coalesced_string_attribute(SpanAttributeField.EVALUATION_NAME)
    test_case_name_expr = _coalesced_string_attribute(SpanAttributeField.TEST_CASE_NAME)
    agent_name_key = spec_for_field(SpanAttributeField.AGENT_NAME).bag_key
    agent_version_key = spec_for_field(SpanAttributeField.AGENT_VERSION).bag_key

    # Note this is logically a single table. CH requires creating an underlying table and then a view that writes to that table.
    client.command(
        f"""
        CREATE TABLE {table}
        (
            workspace LowCardinality(String),
            source_format LowCardinality(String),
            trace_id String,
            session_id String,
            root_span_id String,
            root_name String DEFAULT '',
            root_status LowCardinality(String) DEFAULT 'unknown',
            root_input String CODEC(ZSTD(3)),
            root_output String CODEC(ZSTD(3)),

            project String DEFAULT '',
            evaluation_name String DEFAULT '',
            test_case_name String DEFAULT '',

            agent_name String DEFAULT '',
            agent_version String DEFAULT '',

            root_started_at DateTime64(6) CODEC(Delta(8), ZSTD(1)),
            root_ended_at Nullable(DateTime64(6)) CODEC(Delta(8), ZSTD(1)),
            latency_ms Nullable(Float64),

            event_ts DateTime64(6),
            is_deleted UInt8 DEFAULT 0,

            INDEX idx_trace_id trace_id TYPE bloom_filter(0.001) GRANULARITY 1,
            INDEX idx_session_id session_id TYPE bloom_filter(0.01) GRANULARITY 1,
            INDEX idx_evaluation_name evaluation_name TYPE bloom_filter(0.01) GRANULARITY 1,
            INDEX idx_test_case_name test_case_name TYPE bloom_filter(0.01) GRANULARITY 1,
            INDEX idx_agent_name agent_name TYPE bloom_filter(0.01) GRANULARITY 1,
            INDEX idx_root_status root_status TYPE set(4) GRANULARITY 4,
            INDEX idx_source_format source_format TYPE set(8) GRANULARITY 4
        )
        ENGINE = ReplacingMergeTree(event_ts, is_deleted)
        PARTITION BY toYYYYMM(root_started_at)
        PRIMARY KEY (workspace, root_started_at)
        ORDER BY (workspace, root_started_at, trace_id, root_span_id)
        TTL toDate(root_started_at) + INTERVAL {DEFAULT_SPAN_RETENTION_DAYS} DAY
        SETTINGS
            index_granularity = 256,
            ttl_only_drop_parts = 1
        """
    )
    trace_index_select_sql = f"""
        SELECT
            workspace,
            source_format,
            trace_id,
            session_id,
            external_span_id AS root_span_id,
            name AS root_name,
            status AS root_status,
            input AS root_input,
            output AS root_output,
            attributes_string['{project_key}'] AS project,
            {evaluation_name_expr} AS evaluation_name,
            {test_case_name_expr} AS test_case_name,
            attributes_string['{agent_name_key}'] AS agent_name,
            attributes_string['{agent_version_key}'] AS agent_version,
            start_time AS root_started_at,
            nullIf(end_time, toDateTime64(0, 6)) AS root_ended_at,
            if(end_time = toDateTime64(0, 6), NULL, dateDiff('millisecond', start_time, end_time)) AS latency_ms,
            event_ts,
            is_deleted
        FROM {_table(settings, "spans")}
        WHERE external_parent_span_id = ''
        """
    client.command(
        f"""
        CREATE MATERIALIZED VIEW {view}
        TO {table}
        AS
        {trace_index_select_sql}
        """
    )
    client.command(
        f"""
        INSERT INTO {table}
        {trace_index_select_sql}
        """
    )


_MIGRATIONS: list[tuple[str, Callable[..., None]]] = [
    ("ch_spans_0002", _create_spans_schema),
    ("ch_evaluator_results_0001", _create_evaluator_results_schema),
    ("ch_annotations_0001", _create_annotations_schema),
    ("ch_evaluator_results_0002", _add_evaluator_results_skip_indexes),
    ("ch_trace_index_0003", _create_trace_index_schema),
    # NeMo-owned span attribute keys moved under the ``nemo.*`` namespace
    # (e.g., ``test_case.id`` → ``nemo.test_case.id``). The trace_index MV's SELECT clause
    # resolves bag keys from the catalog at creation time, so any environment that already
    # applied 0003 has the old keys baked in. Re-running the schema function drops and
    # recreates the MV with the current catalog keys.
    ("ch_trace_index_0004_nemo_keys", _create_trace_index_schema),
    # The trace_index column ``experiment_id`` was renamed to ``evaluation_id`` (column,
    # ``idx_evaluation_id``, and the MV SELECT alias). Environments that already applied 0004 still
    # have the old ``experiment_id`` column, while the read path now queries ``evaluation_id`` — so
    # re-run the rebuild under a new key. The function drops and recreates trace_index with the new
    # column and backfills losslessly from ``spans`` (the durable source of truth).
    ("ch_trace_index_0005_evaluation_id", _create_trace_index_schema),
    # The evaluation span-attribute bag key was renamed ``nemo.experiment.id`` -> ``nemo.evaluation.name``
    # (finishing the Experiment->Evaluation rename). The MV bakes the SELECT at creation time, so
    # environments that applied 0005 still resolve evaluation_id from the old key. Re-run the rebuild:
    # the MV now coalesces both keys, so spans already ingested under ``nemo.experiment.id`` keep their
    # evaluation association while new spans use the canonical key.
    ("ch_trace_index_0006_nemo_evaluation_name", _create_trace_index_schema),
    # The test-case span-attribute bag key was renamed ``nemo.test_case.id`` -> ``nemo.test_case.name``,
    # and the trace_index columns now describe the name values they store. Rebuild the derived table
    # and MV with canonical columns while the backfill coalesces canonical and historical bag keys.
    ("ch_trace_index_0007_nemo_test_case_name", _create_trace_index_schema),
    # trace_index gained ``agent_name`` / ``agent_version`` so agent-scoped trace
    # listing and metric rollups filter on a real column instead of scanning the spans attribute
    # map. Rebuild backfills them from ``spans``, which shares trace_index's retention window, so
    # no retained history is lost. A separate revision because 0007 already shipped.
    ("ch_trace_index_0008_agent", _create_trace_index_schema),
]
CURRENT_SCHEMA_VERSION = _MIGRATIONS[-1][0]


def _coalesced_string_attribute(field: SpanAttributeField) -> str:
    keys = bag_keys(spec_for_field(field))
    return "coalesce(" + ", ".join(f"nullIf(attributes_string['{key}'], '')" for key in keys) + ", '')"


def _table(settings: ClickHouseMigrationSettings, name: str) -> str:
    return f"{quote_clickhouse_identifier(settings.database)}.{quote_clickhouse_identifier(name)}"


def quote_clickhouse_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Invalid ClickHouse identifier: {identifier!r}")
    return f"`{identifier}`"


def _import_clickhouse_connect():
    import clickhouse_connect

    return clickhouse_connect


def _get_sync_client(settings: ClickHouseMigrationSettings, *, database: str):
    clickhouse_connect = _import_clickhouse_connect()
    parsed = parse_clickhouse_url(settings.url)
    return clickhouse_connect.get_client(
        host=parsed.host,
        port=parsed.port,
        secure=parsed.secure,
        username=settings.user,
        password=settings.password,
        database=database,
    )


def _close_client(client) -> None:
    close = getattr(client, "close", None)
    if close is not None:
        close()


def parse_clickhouse_url(url: str) -> ClickHouseUrl:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("NHX_INTAKE_CLICKHOUSE_URL must use http or https")
    if parsed.hostname is None:
        raise ValueError("NHX_INTAKE_CLICKHOUSE_URL must include a host")
    host = parsed.hostname
    secure = parsed.scheme == "https"
    port = parsed.port or (8443 if secure else 8123)
    return ClickHouseUrl(host=host, port=port, secure=secure)
