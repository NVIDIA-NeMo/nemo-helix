# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP ingest validates the evaluation its spans name, as ATIF ingest does."""

from fastapi.testclient import TestClient

OTLP_INGEST = "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces"
EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"


def _create_evaluation(client: TestClient, name: str) -> None:
    group = client.post(EXPERIMENTS, json={"name": "otlp-context-group"})
    if group.status_code == 409:
        group = client.get(f"{EXPERIMENTS}/otlp-context-group")
    group.raise_for_status()
    created = client.post(
        EVALUATIONS,
        json={
            "name": name,
            "experiment_group_id": group.json()["id"],
            "dataset_name": "otlp-context-dataset",
            "dataset_version": "v1",
        },
    )
    assert created.status_code == 201, created.text


def _post(client: TestClient, body: bytes):
    return client.post(OTLP_INGEST, content=body, headers={"Content-Type": "application/x-protobuf"})


def _span(evaluation_name: str, *, name: str = "agent-run", span_id: str | None = None) -> dict[str, object]:
    span: dict[str, object] = {"name": name, "attributes": {"nemo.evaluation.name": evaluation_name}}
    if span_id is not None:
        span["span_id"] = span_id
    return span


def test_otlp_ingest_reports_an_unknown_evaluation_in_the_error_list(client: TestClient, make_otlp_request) -> None:
    response = _post(client, make_otlp_request([_span("otlp-missing-evaluation")]))

    assert response.status_code == 200, response.text
    assert response.json()["errors"] == [
        "evaluation 'otlp-missing-evaluation': Evaluation 'otlp-missing-evaluation' must be created "
        "before it can be logged."
    ]


def test_otlp_ingest_stores_nothing_when_the_evaluation_is_unknown(client: TestClient, make_otlp_request) -> None:
    # The check has to precede the write, or the rejected spans are already stored.
    session_id = "otlp-rejected-session"
    body = make_otlp_request(
        [
            {
                "name": "agent-run",
                "attributes": {
                    "nemo.evaluation.name": "otlp-missing-evaluation-2",
                    "gen_ai.conversation.id": session_id,
                },
            }
        ],
        trace_id="a" * 31 + "2",
    )

    assert _post(client, body).status_code == 200

    stored = client.get(
        "/apis/intake/v2/workspaces/default/spans",
        params={"filter[session_id]": session_id, "filter[source]": "otel"},
    )
    assert stored.status_code == 200, stored.text
    assert stored.json()["data"] == []


def test_otlp_ingest_accepts_spans_naming_an_existing_evaluation(client: TestClient, make_otlp_request) -> None:
    _create_evaluation(client, "otlp-known-evaluation")

    response = _post(client, make_otlp_request([_span("otlp-known-evaluation")], trace_id="b" * 31 + "3"))

    assert response.status_code == 200, response.text
    assert response.json() == {"errors": []}


def test_otlp_ingest_without_an_evaluation_name_is_unaffected(client: TestClient, make_otlp_request) -> None:
    body = make_otlp_request([{"name": "plain-span"}], trace_id="c" * 31 + "4")

    response = _post(client, body)

    assert response.status_code == 200, response.text
    assert response.json() == {"errors": []}


def test_spans_for_a_known_evaluation_survive_a_sibling_naming_a_missing_one(
    client: TestClient, make_otlp_request
) -> None:
    # This endpoint reports problems and stores the rest rather than failing the request.
    _create_evaluation(client, "otlp-batch-known")
    body = make_otlp_request(
        [
            {
                "name": "known",
                "span_id": f"{1:016x}",
                "attributes": {"nemo.evaluation.name": "otlp-batch-known", "gen_ai.conversation.id": "batch-known"},
            },
            {
                "name": "missing",
                "span_id": f"{2:016x}",
                "attributes": {"nemo.evaluation.name": "otlp-batch-missing", "gen_ai.conversation.id": "batch-missing"},
            },
        ],
        trace_id="d" * 31 + "5",
    )

    response = _post(client, body)

    assert response.status_code == 200, response.text
    assert response.json()["errors"] == [
        "evaluation 'otlp-batch-missing': Evaluation 'otlp-batch-missing' must be created before it can be logged."
    ]
    kept = client.get(
        "/apis/intake/v2/workspaces/default/spans",
        params={"filter[session_id]": "batch-known", "filter[source]": "otel"},
    )
    assert [span["name"] for span in kept.json()["data"]] == ["known"]
    dropped = client.get(
        "/apis/intake/v2/workspaces/default/spans",
        params={"filter[session_id]": "batch-missing", "filter[source]": "otel"},
    )
    assert dropped.json()["data"] == []


def test_otlp_ingest_rejects_a_deleted_evaluation(client: TestClient, make_otlp_request) -> None:
    _create_evaluation(client, "otlp-deleted-evaluation")
    deleted = client.delete(f"{EVALUATIONS}/otlp-deleted-evaluation")
    assert deleted.status_code == 204, deleted.text

    response = _post(client, make_otlp_request([_span("otlp-deleted-evaluation")], trace_id="e" * 31 + "6"))

    assert response.status_code == 200, response.text
    assert "deleted" in response.json()["errors"][0].lower()
