# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stale root cleanup and recovery after span/index writes partially succeed."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery, ClickHouseQueryError
from nhx.intake.repository.clickhouse.tables import ClickHouseTable, qualified_table
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient

BASE = "/apis/intake/v2/workspaces/default"


def _ingest(
    client: TestClient,
    *,
    evaluation: str | None = None,
    parent: str | None = None,
    started: datetime,
    session: str = "target",
    source: str = "recovery-test",
) -> None:
    response = client.post(
        f"{BASE}/ingest/spans",
        json={
            "source": source,
            "spans": [
                {
                    "session_id": session,
                    "trace_id": session,
                    "span_id": "root",
                    "parent_span_id": parent,
                    "name": "root",
                    "started_at": started.isoformat(),
                    "ended_at": (started + timedelta(seconds=1)).isoformat(),
                    "attributes": {"nemo.evaluation.name": evaluation} if evaluation else {},
                }
            ],
        },
    )
    assert response.status_code == 201, response.text


def _assert_no_telemetry(client: TestClient) -> None:
    for resource in ("spans", "traces"):
        response = client.get(f"{BASE}/{resource}", params={"filter[session_id]": "target"})
        assert response.status_code == 200, response.text
        assert response.json()["data"] == []


@pytest.mark.parametrize("changed_start", [False, True])
def test_delete_cleans_historical_root_after_it_becomes_child(client: TestClient, changed_start: bool) -> None:
    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    _ingest(client, started=started)
    _ingest(client, started=started + timedelta(seconds=int(changed_start)), parent="new-parent")
    _ingest(client, started=started, session="other")

    response = client.delete(f"{BASE}/sessions/target")
    assert response.status_code == 204, response.text
    _assert_no_telemetry(client)
    assert client.get(f"{BASE}/sessions/other").status_code == 200
    assert client.delete(f"{BASE}/sessions/target").status_code == 404


@pytest.mark.parametrize("through_evaluation", [False, True])
@pytest.mark.parametrize("failure_stage", ["sessions.delete", "sessions.delete_index"])
def test_delete_retry_repairs_index_after_all_spans_are_tombstoned(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, through_evaluation: bool, failure_stage: str
) -> None:
    evaluation = None
    if through_evaluation:
        group = client.post(f"{BASE}/experiments", json={"name": "group"}).json()
        response = client.post(
            f"{BASE}/evaluations",
            json={
                "name": "evaluation",
                "experiment_group_id": group["id"],
                "dataset_name": "test",
            },
        )
        assert response.status_code == 201, response.text
        evaluation = "evaluation"
    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    _ingest(client, started=started, evaluation=evaluation)
    # Leave a historical index root that the span MV cannot repair.
    _ingest(client, started=started, evaluation=evaluation, parent="new-parent")

    execute = ClickHouseExecutor.execute
    failed = False

    async def fail_index_once(self: ClickHouseExecutor, query: ClickHouseQuery) -> None:
        nonlocal failed
        if query.name == failure_stage and not failed:
            failed = True
            if failure_stage == "sessions.delete":
                # Simulate a source insert that committed before its downstream
                # processing reported an error to the caller.
                await execute(self, query)
            raise ClickHouseQueryError(query.name)
        await execute(self, query)

    monkeypatch.setattr(ClickHouseExecutor, "execute", fail_index_once)
    path = f"{BASE}/evaluations/evaluation" if through_evaluation else f"{BASE}/sessions/target"
    response = client.delete(path)
    assert failed
    assert response.status_code == 503, response.text
    assert client.get(f"{BASE}/sessions/target").status_code == 404

    response = client.delete(path)
    assert response.status_code == 204, response.text
    _assert_no_telemetry(client)


@pytest.mark.parametrize("merge_index", [False, True])
def test_historical_root_cleanup_rejects_conflicting_child_owners(
    client: TestClient, clickhouse_client: ClickHouseSpanClient, run_async, merge_index: bool
) -> None:
    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    for source in ("source-a", "source-b"):
        _ingest(client, started=started, source=source)
        _ingest(client, started=started, source=source, parent="parent")
    index = qualified_table(clickhouse_client.database, ClickHouseTable.TRACE_INDEX)
    if merge_index:
        run_async(clickhouse_client.command(f"OPTIMIZE TABLE {index} FINAL"))
    response = client.delete(f"{BASE}/sessions/target")
    assert response.status_code == 409, response.text
    response = client.get(f"{BASE}/spans", params={"filter[session_id]": "target"})
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 2
