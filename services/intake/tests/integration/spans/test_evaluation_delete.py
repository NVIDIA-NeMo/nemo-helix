# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation deletion cleans complete sessions while protecting shared evidence."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from nhx.intake.repository.clickhouse.tables import ClickHouseTable, qualified_table
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient

BASE = "/apis/intake/v2/workspaces/default"


@pytest.mark.parametrize("through_group", [False, True])
@pytest.mark.parametrize("evaluation_key", ["nemo.evaluation.name", "nemo.experiment.id"])
def test_evaluation_delete_preserves_shared_sessions_until_last_evaluation(
    client: TestClient, through_group: bool, evaluation_key: str
) -> None:
    for name in ("first", "second"):
        group = client.post(f"{BASE}/experiments", json={"name": name})
        assert group.status_code == 201, group.text
        evaluation = client.post(
            f"{BASE}/evaluations",
            json={"name": name, "experiment_group_id": group.json()["id"], "dataset_name": "test"},
        )
        assert evaluation.status_code == 201, evaluation.text

    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    # Only child spans carry evaluation context. The whole session (including
    # untagged roots and sibling traces) must be deleted when eligible.
    for session, owners in (("exclusive", ["first"]), ("shared", ["first", "second"]), ("unrelated", [])):
        spans = []
        for index in range(2):
            for span_id, owner in [("root", None), *[(f"child-{i}", name) for i, name in enumerate(owners)]]:
                spans.append(
                    {
                        "session_id": session,
                        "trace_id": f"{session}-{index}",
                        "span_id": span_id,
                        "parent_span_id": "root" if owner else None,
                        "name": span_id,
                        "started_at": started.isoformat(),
                        "ended_at": (started + timedelta(seconds=1)).isoformat(),
                        "attributes": {evaluation_key: owner} if owner else {},
                    }
                )
        response = client.post(f"{BASE}/ingest/spans", json={"source": "cascade-test", "spans": spans})
        assert response.status_code == 201, response.text

    annotation = client.post(f"{BASE}/annotations", json={"session_id": "exclusive", "kind": "note", "text": "retain"})
    assert annotation.status_code == 201, annotation.text
    resource = "experiments" if through_group else "evaluations"
    deleted = client.delete(f"{BASE}/{resource}/first")
    assert deleted.status_code == 204, deleted.text
    assert client.get(f"{BASE}/sessions/exclusive").status_code == 404
    assert client.get(f"{BASE}/sessions/shared").status_code == 200
    assert client.get(f"{BASE}/sessions/unrelated").status_code == 200
    assert client.get(f"{BASE}/annotations/{annotation.json()['annotation_id']}").status_code == 200

    # Reuse the old name while the shared session remains. Its links must resolve
    # the deleted owner, not this new live evaluation with the same original name.
    group_response = client.get(f"{BASE}/experiments/first")
    if through_group:
        group_response = client.post(f"{BASE}/experiments", json={"name": "first"})
    assert group_response.status_code in (200, 201), group_response.text
    replacement = client.post(
        f"{BASE}/evaluations",
        json={"name": "first", "experiment_group_id": group_response.json()["id"], "dataset_name": "new"},
    )
    assert replacement.status_code == 201, replacement.text
    deleted = client.delete(f"{BASE}/{resource}/second")
    assert deleted.status_code == 204, deleted.text
    assert client.get(f"{BASE}/sessions/shared").status_code == 404
    for resource in ("traces", "spans"):
        for session in ("exclusive", "shared"):
            response = client.get(f"{BASE}/{resource}", params={"filter[session_id]": session})
            assert response.status_code == 200, response.text
            assert response.json()["data"] == []
    assert client.get(f"{BASE}/sessions/unrelated").status_code == 200


@pytest.mark.parametrize("merge_before", [False, True])
def test_evaluation_rename_relinks_roots_children_history_and_trace_index_without_changing_other_owners(
    client: TestClient, clickhouse_client: ClickHouseSpanClient, run_async, merge_before: bool
) -> None:
    group = client.post(f"{BASE}/experiments", json={"name": "group"}).json()
    for name in ("first", "second"):
        response = client.post(
            f"{BASE}/evaluations",
            json={"name": name, "experiment_group_id": group["id"], "dataset_name": "test"},
        )
        assert response.status_code == 201, response.text
    started = datetime.now(timezone.utc) - timedelta(minutes=1)
    rows = [
        [
            "default",
            "shared",
            "trace",
            "source",
            "root",
            "",
            started,
            started,
            started,
            {"nemo.evaluation.name": "first", "nemo.experiment.id": "second", "custom.link": "first"},
            0,
        ],
        [
            "default",
            "shared",
            "trace",
            "source",
            "child",
            "root",
            started,
            started,
            started,
            {"nemo.evaluation.name": "second", "nemo.experiment.id": "first"},
            0,
        ],
        [
            "default",
            "shared",
            "legacy-trace",
            "source",
            "legacy-root",
            "",
            started,
            started,
            started,
            {"nemo.experiment.id": "first"},
            0,
        ],
        [
            "default",
            "historical",
            "deleted-trace",
            "source",
            "deleted-root",
            "",
            started,
            started,
            started,
            {"nemo.evaluation.name": "first"},
            1,
        ],
        [
            "other",
            "shared",
            "trace",
            "source",
            "root",
            "",
            started,
            started,
            started,
            {"nemo.evaluation.name": "first"},
            0,
        ],
    ]
    columns = [
        "workspace",
        "session_id",
        "trace_id",
        "source_format",
        "external_span_id",
        "external_parent_span_id",
        "start_time",
        "end_time",
        "event_ts",
        "attributes_string",
        "is_deleted",
    ]
    run_async(clickhouse_client.insert("spans", rows, column_names=columns))
    if merge_before:
        for table in (ClickHouseTable.SPANS, ClickHouseTable.TRACE_INDEX):
            run_async(
                clickhouse_client.command(f"OPTIMIZE TABLE {qualified_table(clickhouse_client.database, table)} FINAL")
            )

    response = client.delete(f"{BASE}/evaluations/first")
    assert response.status_code == 204, response.text
    trash = client.get(f"{BASE}/evaluations", params={"filter[is_deleted]": "true"})
    assert trash.status_code == 200, trash.text
    archived = next(row["name"] for row in trash.json()["data"] if row["name"].startswith("first-deleted-"))
    assert client.get(f"{BASE}/sessions/shared").status_code == 200
    assert client.get(f"{BASE}/sessions/historical").status_code == 404

    stored = run_async(
        clickhouse_client.query(
            "SELECT workspace, external_span_id, attributes_string, is_deleted FROM spans ORDER BY workspace, external_span_id"
        )
    )
    for workspace, span_id, attributes, deleted in stored.result_rows:
        if workspace == "other":
            assert attributes["nemo.evaluation.name"] == "first"
            continue
        for key in ("nemo.evaluation.name", "nemo.experiment.id"):
            assert attributes.get(key) != "first"
        if span_id == "root":
            assert attributes["nemo.evaluation.name"] == archived
            assert attributes["nemo.experiment.id"] == "second"
            assert attributes["custom.link"] == "first"
        elif span_id == "child":
            assert attributes["nemo.evaluation.name"] == "second"
            assert attributes["nemo.experiment.id"] == archived
        else:
            assert archived in attributes.values()
        assert deleted == (1 if span_id == "deleted-root" else 0)
    index = run_async(
        clickhouse_client.query("SELECT workspace, root_span_id, evaluation_name, is_deleted FROM trace_index")
    )
    for workspace, root_id, evaluation_name, deleted in index.result_rows:
        assert evaluation_name == ("first" if workspace == "other" else archived)
        assert deleted == (1 if root_id == "deleted-root" else 0)
    traces = client.get(f"{BASE}/traces", params={"filter[session_id]": "shared"})
    assert traces.status_code == 200, traces.text
    assert {trace["evaluation_context"]["evaluation_name"] for trace in traces.json()["data"]} == {archived}

    replacement = client.post(
        f"{BASE}/evaluations", json={"name": "first", "experiment_group_id": group["id"], "dataset_name": "new"}
    )
    assert replacement.status_code == 201, replacement.text
    assert client.delete(f"{BASE}/evaluations/second").status_code == 204
    assert client.get(f"{BASE}/sessions/shared").status_code == 404
