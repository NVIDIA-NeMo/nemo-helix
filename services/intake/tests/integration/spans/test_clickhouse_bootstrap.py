# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse schema bootstrap tests."""

from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.intake.repository.clickhouse.tables import ClickHouseTable, qualified_table
from nhx.intake.service import IntakeService
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient, bootstrap_schema
from nhx.intake.spans.clickhouse_migrations import quote_clickhouse_identifier


def test_clickhouse_server_matches_supported_lts(
    clickhouse_client: ClickHouseSpanClient,
    clickhouse_version: str,
    run_async,
) -> None:
    result = run_async(clickhouse_client.query("SELECT version()"))

    assert len(result.result_rows) == 1
    assert str(result.result_rows[0][0]).startswith(f"{clickhouse_version}.")


def test_clickhouse_bootstrap_is_idempotent(clickhouse_client: ClickHouseSpanClient, run_async):
    run_async(bootstrap_schema(clickhouse_client))
    run_async(bootstrap_schema(clickhouse_client))

    version_table = (
        f"{quote_clickhouse_identifier(clickhouse_client.database)}."
        f"{quote_clickhouse_identifier('clickhouse_alembic_version')}"
    )
    result = run_async(clickhouse_client.query(f"SELECT version_num FROM {version_table} FINAL ORDER BY version_num"))
    assert result.result_rows == [
        ("ch_annotations_0001",),
        ("ch_evaluator_results_0001",),
        ("ch_evaluator_results_0002",),
        ("ch_spans_0002",),
        ("ch_trace_index_0003",),
        ("ch_trace_index_0004_nemo_keys",),
        ("ch_trace_index_0005_evaluation_id",),
        ("ch_trace_index_0006_nemo_evaluation_name",),
        ("ch_trace_index_0007_nemo_test_case_name",),
        ("ch_trace_index_0008_agent",),
    ]
    expected_ttl = {
        ClickHouseTable.SPANS: "TTL toDate(start_time) + toIntervalDay(90)",
        ClickHouseTable.TRACE_INDEX: "TTL toDate(root_started_at) + toIntervalDay(90)",
    }
    for table, ttl in expected_ttl.items():
        create = run_async(
            clickhouse_client.query(f"SHOW CREATE TABLE {qualified_table(clickhouse_client.database, table)}")
        )
        assert ttl in str(create.result_rows[0][0])


def test_intake_service_readiness_bootstraps_and_probes_service_owned_clickhouse(client: TestClient, run_async):
    app = cast(FastAPI, client.app)
    service = cast(IntakeService, app.state.intake_service)

    assert service.clickhouse_client is not None
    assert run_async(service.is_ready()) is True
    assert service.clickhouse_client._bootstrapped is True

    response = client.get("/apis/intake/v2/workspaces/default/spans")

    assert response.status_code == 200, response.text
    assert service.clickhouse_client._bootstrapped is True
