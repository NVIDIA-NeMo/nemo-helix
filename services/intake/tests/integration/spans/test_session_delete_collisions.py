# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deletion must not emit tombstones for trace-index keys shared by different owners."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from nhx.intake.repository.clickhouse.tables import ClickHouseTable, qualified_table
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient

BASE = "/apis/intake/v2/workspaces/default"


def _ingest(
    client: TestClient,
    started_at: datetime,
    *,
    session: str = "target",
    source: str = "source-a",
    trace: str = "shared-trace",
    span: str = "shared-root",
    parent: str | None = None,
    workspace: str = "default",
) -> None:
    response = client.post(
        f"/apis/intake/v2/workspaces/{workspace}/ingest/spans",
        json={
            "source": source,
            "spans": [
                {
                    "session_id": session,
                    "trace_id": trace,
                    "span_id": span,
                    "parent_span_id": parent,
                    "name": span,
                    "started_at": started_at.isoformat(),
                    "ended_at": (started_at + timedelta(seconds=1)).isoformat(),
                }
            ],
        },
    )
    assert response.status_code == 201, response.text


@pytest.mark.parametrize("owner_difference", ["session", "source", "session-and-source"])
@pytest.mark.parametrize("merge_index", [False, True])
def test_session_delete_rejects_conflicting_owners_even_after_index_merge(
    client: TestClient, clickhouse_client: ClickHouseSpanClient, run_async, owner_difference: str, merge_index: bool
) -> None:
    started_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    _ingest(client, started_at)
    _ingest(client, started_at, span="child", parent="shared-root")
    _ingest(client, started_at, trace="unrelated-trace", span="unrelated-root")
    other_session = "other" if "session" in owner_difference else "target"
    other_source = "source-b" if "source" in owner_difference else "source-a"
    _ingest(client, started_at, session=other_session, source=other_source)
    index = qualified_table(clickhouse_client.database, ClickHouseTable.TRACE_INDEX)
    spans = qualified_table(clickhouse_client.database, ClickHouseTable.SPANS)
    if merge_index:
        run_async(clickhouse_client.command(f"OPTIMIZE TABLE {index} FINAL"))
        rows = run_async(clickhouse_client.query(f"SELECT count() FROM {index} WHERE trace_id = 'shared-trace'"))
        # The index has lost an owner; the spans table still retains the evidence.
        assert rows.result_rows[0][0] == 1

    response = client.delete(f"{BASE}/sessions/target")

    assert response.status_code == 409, response.text
    assert "conflicting trace-index ownership" in response.json()["detail"]
    expected = {"target": 4} if other_session == "target" else {"target": 3, "other": 1}
    for session_id, count in expected.items():
        stored = client.get(f"{BASE}/spans", params={"filter[session_id]": session_id})
        assert stored.status_code == 200, stored.text
        assert len(stored.json()["data"]) == count
    for table in (spans, index):
        deleted = run_async(clickhouse_client.query(f"SELECT count() FROM {table} WHERE is_deleted = 1"))
        assert deleted.result_rows[0][0] == 0


@pytest.mark.parametrize("deleted_index_owner", [False, True])
def test_session_delete_rejects_owner_retained_only_in_index_history(
    client: TestClient, clickhouse_client: ClickHouseSpanClient, run_async, deleted_index_owner: bool
) -> None:
    started_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    _ingest(client, started_at)
    run_async(
        clickhouse_client.insert(
            "trace_index",
            [
                [
                    "default",
                    "other",
                    "source-a",
                    started_at,
                    "shared-trace",
                    "shared-root",
                    datetime.now(timezone.utc),
                    int(deleted_index_owner),
                ]
            ],
            column_names=[
                "workspace",
                "session_id",
                "source_format",
                "root_started_at",
                "trace_id",
                "root_span_id",
                "event_ts",
                "is_deleted",
            ],
        )
    )

    response = client.delete(f"{BASE}/sessions/target")

    assert response.status_code == 409, response.text
    session = client.get(f"{BASE}/sessions/target")
    assert session.status_code == 200, session.text
    assert session.json()["span_count"] == 1


@pytest.mark.parametrize("difference", ["time", "trace", "root", "workspace", "same-owner-replay", "rootless"])
def test_session_delete_allows_distinct_keys_and_same_owner_versions(client: TestClient, difference: str) -> None:
    started_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    _ingest(client, started_at, parent="missing-root" if difference == "rootless" else None)
    other_workspace = "other" if difference == "workspace" else "default"
    if other_workspace == "other":
        created = client.post("/apis/entities/v2/workspaces", json={"name": "other"})
        assert created.status_code == 201, created.text
    _ingest(
        client,
        started_at + timedelta(microseconds=1) if difference == "time" else started_at,
        session="target" if difference == "same-owner-replay" else "other",
        trace="other-trace" if difference == "trace" else "shared-trace",
        span="other-root" if difference == "root" else "shared-root",
        workspace=other_workspace,
    )

    response = client.delete(f"{BASE}/sessions/target")

    assert response.status_code == 204, response.text
    assert client.get(f"{BASE}/sessions/target").status_code == 404
    if difference != "same-owner-replay":
        kept = client.get(f"/apis/intake/v2/workspaces/{other_workspace}/sessions/other")
        assert kept.status_code == 200, kept.text
        assert kept.json()["span_count"] == 1
