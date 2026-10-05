# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session detail read API tests."""

from datetime import datetime, timezone
from decimal import Decimal

from fastapi.testclient import TestClient
from nhx.intake.repository.clickhouse.session import _session_detail_query
from nhx.intake.repository.clickhouse.tables import ClickHouseTable, qualified_table
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient


def test_session_detail_rolls_up_all_current_spans(
    client: TestClient,
    clickhouse_client: ClickHouseSpanClient,
    make_otlp_request,
    run_async,
) -> None:
    base_ns = int(datetime.now(timezone.utc).replace(microsecond=0).timestamp() * 1_000_000_000)
    session_id = "session-detail-rollup"
    first_trace = make_otlp_request(
        [
            {
                "name": "first-root",
                "span_id": "0000000000001001",
                "start_time_unix_nano": base_ns,
                "end_time_unix_nano": base_ns + 1_000_000_000,
                "attributes": {"gen_ai.conversation.id": session_id},
            },
            {
                "name": "first-llm",
                "span_id": "0000000000001002",
                "parent_span_id": "0000000000001001",
                "start_time_unix_nano": base_ns + 100_000_000,
                "end_time_unix_nano": base_ns + 900_000_000,
                "attributes": {
                    "gen_ai.conversation.id": session_id,
                    "gen_ai.usage.input_tokens": 100,
                    "gen_ai.usage.output_tokens": 40,
                    "gen_ai.usage.total_tokens": 140,
                    "llm.cost.total": 0.0014,
                },
            },
        ],
        trace_id="00000000000000000000000000001001",
    )
    second_trace = make_otlp_request(
        [
            {
                "name": "second-root",
                "span_id": "0000000000002001",
                "start_time_unix_nano": base_ns + 2_000_000_000,
                "end_time_unix_nano": base_ns + 5_000_000_000,
                "error": True,
                "attributes": {
                    "gen_ai.conversation.id": session_id,
                    "gen_ai.usage.input_tokens": 60,
                    "gen_ai.usage.output_tokens": 20,
                    "gen_ai.usage.total_tokens": 80,
                    "llm.cost.total": 0.0008,
                },
            }
        ],
        trace_id="00000000000000000000000000002001",
    )

    for body in (first_trace, second_trace):
        response = client.post(
            "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
            content=body,
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert response.status_code == 200, response.text

    response = client.get(f"/apis/intake/v2/workspaces/default/sessions/{session_id}")

    assert response.status_code == 200, response.text
    session = response.json()
    assert session["id"] == session_id
    assert session["workspace"] == "default"
    assert session["status"] == "error"
    assert session["duration_ms"] == 5000
    assert session["trace_count"] == 2
    assert session["span_count"] == 3
    assert "error_count" not in session
    assert session["input_tokens"] == 160
    assert session["output_tokens"] == 60
    assert session["total_tokens"] == 220
    assert Decimal(str(session["cost_usd"])) == Decimal("0.0022")
    assert "traces" not in session
    assert "spans" not in session
    assert "input" not in session
    assert "output" not in session

    query = _session_detail_query(qualified_table(clickhouse_client.database, ClickHouseTable.SPANS))
    plan = run_async(
        clickhouse_client.query(
            f"EXPLAIN indexes = 1 {query.statement}",
            parameters={**query.parameters, "workspace": "default", "session_id": session_id},
        )
    )
    plan_text = "\n".join(str(row[0]) for row in plan.result_rows)
    assert "PrimaryKey" in plan_text
    assert "workspace" in plan_text
    assert "session_id" in plan_text

    missing = client.get("/apis/intake/v2/workspaces/default/sessions/missing")
    assert missing.status_code == 404
