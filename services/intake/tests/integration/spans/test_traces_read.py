# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace summary read API tests."""

from datetime import datetime, timezone
from decimal import Decimal

from fastapi.testclient import TestClient


def _create_evaluation(client: TestClient, name: str) -> None:
    """Create the Evaluation the spans below name; ingest rejects one that resolves to nothing."""
    group = client.post("/apis/intake/v2/workspaces/default/experiments", json={"name": "trace-read-group"})
    if group.status_code == 409:
        group = client.get("/apis/intake/v2/workspaces/default/experiments/trace-read-group")
    group.raise_for_status()
    created = client.post(
        "/apis/intake/v2/workspaces/default/evaluations",
        json={
            "name": name,
            "experiment_group_id": group.json()["id"],
            "dataset_name": "trace-read-dataset",
            "dataset_version": "v1",
        },
    )
    assert created.status_code == 201, created.text


def test_traces_read_returns_core_trace_summary(client: TestClient, make_otlp_request):
    _create_evaluation(client, "experiment-a")
    base_ns = int(datetime.now(timezone.utc).replace(microsecond=0).timestamp() * 1_000_000_000)
    input_text = "i" * 350
    output_text = "o" * 350
    body = make_otlp_request(
        [
            {
                "name": "root-agent",
                "span_id": "0000000000000001",
                "start_time_unix_nano": base_ns,
                "end_time_unix_nano": base_ns + 100_000_000,
                "attributes": {
                    "openinference.span.kind": "AGENT",
                    "gen_ai.conversation.id": "trace-session",
                    "project": "project-a",
                    # Emit the legacy evaluation and test-case keys on purpose: ingest normalizes them
                    # to the canonical name keys, so this asserts the dual-read path end to end (a
                    # pre-rename producer still associates its traces to the evaluation and test case).
                    "nemo.experiment.id": "experiment-a",
                    "nemo.test_case.id": "case-a",
                    "deployment.environment.name": "prod",
                    "tag.tags": ["trace-read"],
                    "metadata": {"owner": "trace-test"},
                    "input.value": input_text,
                    "output.value": output_text,
                },
            },
            {
                "name": "llm-call",
                "span_id": "0000000000000002",
                "parent_span_id": "0000000000000001",
                "start_time_unix_nano": base_ns + 1_000_000_000,
                "end_time_unix_nano": base_ns + 1_200_000_000,
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "gen_ai.conversation.id": "trace-session",
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-4o-mini",
                    "gen_ai.usage.input_tokens": 420,
                    "gen_ai.usage.output_tokens": 310,
                    "gen_ai.usage.cached_tokens": 128,
                    "gen_ai.usage.total_tokens": 858,
                    "llm.cost.prompt": 0.0024,
                    "llm.cost.completion": 0.0037,
                    "llm.cost.total": 0.0061,
                },
            },
        ]
    )

    ingest_response = client.post(
        "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
        content=body,
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert ingest_response.status_code == 200, ingest_response.text

    traces_response = client.get(
        "/apis/intake/v2/workspaces/default/traces",
        params={
            "filter[session_id]": "trace-session",
            "filter[evaluation_name]": "experiment-a",
            "page_size": 20,
        },
    )
    assert traces_response.status_code == 200, traces_response.text
    payload = traces_response.json()
    assert payload["pagination"]["total_results"] == 1
    trace = payload["data"][0]
    assert trace["id"] == "00000000000000000000000000000001"
    assert trace["session_id"] == "trace-session"
    assert trace["workspace"] == "default"
    assert trace["root_span_id"] == "0000000000000001"
    assert trace["name"] == "root-agent"
    assert trace["input"] == input_text[:300]
    assert trace["output"] == output_text[:300]
    assert trace["status"] == "success"
    assert trace["input_tokens"] == 420
    assert trace["output_tokens"] == 310
    assert trace["cached_tokens"] == 128
    assert trace["total_tokens"] == 858
    assert Decimal(str(trace["cost_usd"])) == Decimal("0.0061")
    assert Decimal(str(trace["cost_input_usd"])) == Decimal("0.0024")
    assert Decimal(str(trace["cost_output_usd"])) == Decimal("0.0037")
    assert trace["span_count"] == 2
    assert trace["error_count"] == 0
    assert trace["evaluation_context"] == {
        "evaluation_name": "experiment-a",
        "test_case_name": "case-a",
        "evaluation_id": "experiment-a",
        "test_case_id": "case-a",
    }
    assert "evaluation_id" not in trace
    assert "experiment_id" not in trace
    assert "test_case_id" not in trace
    assert "source_format" not in trace
    assert "project" not in trace
    # models and providers were withheld when this API was designed (#9); ASTD-391's
    # details card needs them, so they are now part of the rollup response.
    assert trace["models"] == ["gpt-4o-mini"]
    assert trace["providers"] == ["openai"]

    get_response = client.get(f"/apis/intake/v2/workspaces/default/traces/{trace['id']}")
    assert get_response.status_code == 200, get_response.text
    assert get_response.json()["id"] == trace["id"]
    assert get_response.json()["input"] == input_text
    assert get_response.json()["output"] == output_text

    summary_response = client.get(
        "/apis/intake/v2/workspaces/default/traces",
        params={"filter[session_id]": "trace-session", "mode": "summary", "page_size": 20},
    )
    assert summary_response.status_code == 200, summary_response.text
    summary_trace = summary_response.json()["data"][0]
    assert summary_trace["id"] == trace["id"]
    assert summary_trace["status"] == "success"
    assert summary_trace["evaluation_context"] == trace["evaluation_context"]
    assert "evaluation_id" not in summary_trace
    assert "experiment_id" not in summary_trace
    assert "test_case_id" not in summary_trace
    assert "input_tokens" not in summary_trace
    assert "cost_usd" not in summary_trace
    assert "span_count" not in summary_trace
    # summary skips the span aggregate join, so these have no value to report
    assert "models" not in summary_trace
    assert "providers" not in summary_trace
    assert "input" not in summary_trace
    assert "output" not in summary_trace

    preview_response = client.get(
        "/apis/intake/v2/workspaces/default/traces",
        params={"filter[session_id]": "trace-session", "mode": "preview", "page_size": 20},
    )
    assert preview_response.status_code == 200, preview_response.text
    preview_trace = preview_response.json()["data"][0]
    assert preview_trace["input"] == input_text[:300]
    assert preview_trace["output"] == output_text[:300]
    assert preview_trace["span_count"] == 2


def test_traces_read_picks_earliest_root_when_trace_has_multiple_roots(client: TestClient, make_otlp_request):
    base_ns = int(datetime.now(timezone.utc).replace(microsecond=0).timestamp() * 1_000_000_000)
    body = make_otlp_request(
        [
            {
                "name": "earliest-root",
                "span_id": "0000000000000101",
                "start_time_unix_nano": base_ns,
                "end_time_unix_nano": base_ns + 100_000_000,
                "attributes": {"gen_ai.conversation.id": "multi-root-session"},
            },
            {
                "name": "later-root",
                "span_id": "0000000000000102",
                "start_time_unix_nano": base_ns + 1_000_000_000,
                "end_time_unix_nano": base_ns + 1_100_000_000,
                "attributes": {"gen_ai.conversation.id": "multi-root-session"},
            },
        ],
        trace_id="00000000000000000000000000000101",
    )

    ingest_response = client.post(
        "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
        content=body,
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert ingest_response.status_code == 200, ingest_response.text

    traces_response = client.get(
        "/apis/intake/v2/workspaces/default/traces",
        params={"filter[session_id]": "multi-root-session", "page_size": 20},
    )
    assert traces_response.status_code == 200, traces_response.text
    payload = traces_response.json()
    assert payload["pagination"]["total_results"] == 1
    trace = payload["data"][0]
    assert trace["id"] == "00000000000000000000000000000101"
    assert trace["name"] == "earliest-root"
    assert trace["root_span_id"] == "0000000000000101"


def test_reused_trace_and_span_ids_remain_scoped_to_their_sessions(client: TestClient, make_otlp_request):
    """Distinct start times preserve both physical index rows, even when OTLP IDs collide."""
    trace_id = "1234567890abcdef1234567890abcdef"
    base_ns = int(datetime.now(timezone.utc).replace(microsecond=0).timestamp() * 1_000_000_000)
    evaluations = ["collision-evaluation-a", "collision-evaluation-b"]
    for evaluation in evaluations:
        _create_evaluation(client, evaluation)

    def ingest(index: int, tokens: int) -> None:
        session = f"collision-session-{index}"
        attributes = {
            "gen_ai.conversation.id": session,
            "nemo.evaluation.name": evaluations[index],
            "nemo.test_case.name": "shared-case",
            "gen_ai.agent.name": f"agent-{index}",
            "gen_ai.agent.version": f"version-{index}",
        }
        started_at = base_ns + index * 10_000_000_000
        body = make_otlp_request(
            [
                {
                    "name": f"root-{index}",
                    "span_id": "0000000000000001",
                    "start_time_unix_nano": started_at,
                    "end_time_unix_nano": started_at + 2_000_000_000,
                    "attributes": {**attributes, "openinference.span.kind": "AGENT"},
                },
                {
                    "name": f"llm-{index}",
                    "span_id": "0000000000000002",
                    "parent_span_id": "0000000000000001",
                    "start_time_unix_nano": started_at + 1_000_000_000,
                    "end_time_unix_nano": started_at + 2_000_000_000,
                    "attributes": {
                        **attributes,
                        "openinference.span.kind": "LLM",
                        "gen_ai.usage.input_tokens": tokens,
                    },
                },
            ],
            trace_id=trace_id,
        )
        response = client.post(
            "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
            content=body,
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert response.status_code == 200, response.text

    def assert_reads(expected_tokens: list[int]) -> None:
        response = client.get(f"/apis/intake/v2/workspaces/default/traces/{trace_id}")
        assert response.status_code == 409, response.text
        ambiguous_span = client.get("/apis/intake/v2/workspaces/default/spans/0000000000000002")
        assert ambiguous_span.status_code == 409, ambiguous_span.text
        listed = client.get("/apis/intake/v2/workspaces/default/traces", params={"filter[id]": trace_id})
        assert listed.status_code == 200, listed.text
        assert listed.json()["pagination"]["total_results"] == 2
        batch_traces = {trace["session_id"]: trace for trace in listed.json()["data"]}
        batched = client.get("/apis/intake/v2/workspaces/default/evaluations")
        assert batched.status_code == 200, batched.text
        batch_evaluations = {evaluation["name"]: evaluation for evaluation in batched.json()["data"]}
        for index, tokens in enumerate(expected_tokens):
            session = f"collision-session-{index}"
            response = client.get(
                f"/apis/intake/v2/workspaces/default/traces/{trace_id}",
                params={"session_id": session},
            )
            assert response.status_code == 200, response.text
            trace = response.json()
            assert trace["session_id"] == session
            assert trace["name"] == f"root-{index}"
            assert trace["span_count"] == 2
            assert trace["input_tokens"] == tokens
            span_response = client.get(
                "/apis/intake/v2/workspaces/default/spans/0000000000000002", params={"session_id": session}
            )
            assert span_response.status_code == 200, span_response.text
            assert span_response.json()["session_id"] == session
            assert span_response.json()["name"] == f"llm-{index}"
            assert batch_traces[session]["input_tokens"] == tokens
            evaluation_response = client.get(f"/apis/intake/v2/workspaces/default/evaluations/{evaluations[index]}")
            assert evaluation_response.status_code == 200, evaluation_response.text
            evaluation = evaluation_response.json()
            assert evaluation["run_count"] == 1
            assert evaluation["tokens"]["sum"] == tokens
            assert evaluation["agent_names"] == [f"agent-{index}"]
            assert evaluation["agent_versions"] == [f"version-{index}"]
            for field in ("run_count", "tokens", "agent_names", "agent_versions"):
                assert batch_evaluations[evaluations[index]][field] == evaluation[field]

    ingest(0, 11)
    ingest(1, 29)
    assert_reads([11, 29])
    ingest(0, 47)
    assert_reads([47, 29])
    missing = client.get(
        f"/apis/intake/v2/workspaces/default/traces/{trace_id}", params={"session_id": "another-session"}
    )
    assert missing.status_code == 404
