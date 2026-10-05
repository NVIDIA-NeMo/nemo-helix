# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse migration helper tests."""

import re
from pathlib import Path

import nhx.intake.spans.clickhouse_migrations as clickhouse_migrations
import pytest
from nhx.intake.spans.clickhouse_migrations import parse_clickhouse_url
from nhx.intake.spans.span_attribute_catalog import SpanAttributeField, spec_for_field


def test_parse_clickhouse_url_rejects_hostless_url():
    with pytest.raises(ValueError, match="include a host"):
        parse_clickhouse_url("http://:8123")


def test_parse_clickhouse_url_keeps_default_ports():
    assert parse_clickhouse_url("clickhouse.local").port == 8123
    assert parse_clickhouse_url("https://clickhouse.local").port == 8443


def test_spans_schema_keeps_cityhash_identity_expression():
    source = Path(clickhouse_migrations.__file__).read_text(encoding="utf-8")
    match = re.search(r"\bid\s+UInt64\s+MATERIALIZED\s+cityHash64\(([^)]*)\)", source)

    assert match is not None
    assert [part.strip() for part in match.group(1).split(",")] == [
        "workspace",
        "source_format",
        "trace_id",
        "external_span_id",
    ]


def test_trace_index_schema_is_root_span_projection():
    source = Path(clickhouse_migrations.__file__).read_text(encoding="utf-8")
    function_match = re.search(
        r"def _create_trace_index_schema\(.*?^_MIGRATIONS",
        source,
        re.DOTALL | re.MULTILINE,
    )
    assert function_match is not None
    source = function_match.group(0)

    table_match = re.search(
        r"CREATE TABLE \{table\}.*?ttl_only_drop_parts = 1",
        source,
        re.DOTALL,
    )

    assert table_match is not None
    ddl = source

    assert '"trace_index"' in source
    assert '"trace_index_mv"' in source
    assert "CREATE TABLE {table}" in ddl
    assert "CREATE MATERIALIZED VIEW {view}" in ddl
    assert "TO {table}" in ddl
    assert "INSERT INTO {table}" in source
    assert "WHERE external_parent_span_id = ''" in ddl
    # Both identifiers resolve via canonical keys plus historical aliases, so the backfill keeps older
    # spans associated.
    assert "{evaluation_name_expr} AS evaluation_name" in ddl
    assert "{test_case_name_expr} AS test_case_name" in ddl
    assert "_coalesced_string_attribute(SpanAttributeField.EVALUATION_NAME)" in ddl
    assert "_coalesced_string_attribute(SpanAttributeField.TEST_CASE_NAME)" in ddl
    assert "root_status LowCardinality(String)" in ddl
    assert "root_input String" in ddl
    assert "PRIMARY KEY (workspace, root_started_at)" in ddl
    assert "ORDER BY (workspace, root_started_at, trace_id, root_span_id)" in ddl
    assert "INDEX idx_evaluation_name evaluation_name" in ddl
    assert "INDEX idx_test_case_name test_case_name" in ddl
    assert "index_granularity = 256" in ddl
    # Agent identity is denormalized so agent-scoped listing and metric rollups filter on a
    # column instead of probing the spans attribute map.
    assert "attributes_string['{agent_name_key}'] AS agent_name" in ddl
    assert "attributes_string['{agent_version_key}'] AS agent_version" in ddl
    assert "INDEX idx_agent_name agent_name" in ddl


def test_trace_index_mv_keys_match_attribute_catalog():
    evaluation_spec = spec_for_field(SpanAttributeField.EVALUATION_NAME)
    assert evaluation_spec.bag_key == "nemo.evaluation.name"
    # The legacy key is still accepted on ingest so pre-rename producers keep associating.
    assert "nemo.experiment.id" in evaluation_spec.source_keys
    test_case_spec = spec_for_field(SpanAttributeField.TEST_CASE_NAME)
    assert test_case_spec.bag_key == "nemo.test_case.name"
    assert test_case_spec.bag_aliases == ("nemo.test_case.id",)
    # The MV bakes these keys in at creation time, so a catalog rename needs a new migration.
    assert spec_for_field(SpanAttributeField.AGENT_NAME).bag_key == "gen_ai.agent.name"
    assert spec_for_field(SpanAttributeField.AGENT_VERSION).bag_key == "agent.version"
