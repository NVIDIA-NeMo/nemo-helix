# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for ``nemo intake`` against a scripted typed client."""

from __future__ import annotations

import json

import httpx
import pytest
from nmp.intake.cli import IntakeCLI

WS = "/apis/intake/v2/workspaces/default"

TRACE = {
    "id": "trace-1",
    "root_span_id": "span-1",
    "session_id": "sess-1",
    "workspace": "default",
    "name": "run",
    "started_at": "2026-01-01T00:00:00Z",
    "status": "success",
}
SPAN = {
    "span_id": "span-1",
    "session_id": "sess-1",
    "workspace": "default",
    "kind": "LLM",
    "name": "chat",
    "source": "otel",
    "trace_id": "trace-1",
    "started_at": "2026-01-01T00:00:00Z",
    "status": "success",
    "ingested_at": "2026-01-01T00:00:01Z",
}
SPAN_GROUP = {"group": {"trace_id": "trace-1"}, "span_count": 3, "started_at": "2026-01-01T00:00:00Z"}
SESSION = {
    "id": "sess-1",
    "workspace": "default",
    "started_at": "2026-01-01T00:00:00Z",
    "status": "success",
    "trace_count": 1,
    "span_count": 3,
}
ANNOTATION = {
    "kind": "note",
    "annotation_id": "ann-1",
    "workspace": "default",
    "session_id": "sess-1",
    "text": "looks good",
    "created_at": "2026-01-01T00:00:00Z",
    "ingested_at": "2026-01-01T00:00:00Z",
}
EVALUATOR_RESULT = {
    "evaluator_result_id": "eval-1",
    "span_id": "span-1",
    "session_id": "sess-1",
    "workspace": "default",
    "name": "faithfulness/v1",
    "value": 0.9,
    "data_type": "NUMERIC",
    "created_at": "2026-01-01T00:00:00Z",
    "ingested_at": "2026-01-01T00:00:00Z",
}
TRACE_METRICS = {
    "bucket": "day",
    "timezone": "UTC",
    "data": [
        {
            "bucket_start": "2026-01-01T00:00:00Z",
            "run_count": 2,
            "failed_run_count": 0,
            "input_tokens": {"sum": 10},
            "output_tokens": {"sum": 5},
            "cached_tokens": {},
            "total_tokens": {"sum": 15},
            "cost_usd": {"sum": 0.01},
            "latency_ms": {"mean": 120.0},
        }
    ],
}


def test_plugin_metadata() -> None:
    assert IntakeCLI.name == "intake"
    assert IntakeCLI.description == "Intake operations."
    groups = {group.name for group in IntakeCLI().get_cli().registered_groups}
    assert groups == {"annotations", "evaluator-results", "ingest", "sessions", "spans", "traces"}


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------


def test_traces_get(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["traces", "get", "trace-1", "--mode", "summary"], [httpx.Response(200, json=TRACE)]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{WS}/traces/trace-1"
    assert dict(recorder.last.url.params) == {"mode": "summary"}
    assert json.loads(result.stdout)["id"] == "trace-1"


def test_traces_get_without_mode_sends_no_query(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["traces", "get", "trace-1", "--workspace", "other"], [httpx.Response(200, json=TRACE)]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/intake/v2/workspaces/other/traces/trace-1"
    assert dict(recorder.last.url.params) == {}


def test_traces_get_not_found(intake_cli) -> None:
    result, _ = intake_cli.run(
        ["traces", "get", "nope"], [httpx.Response(404, json={"detail": "Trace default/nope not found"})]
    )

    assert result.exit_code == 3
    assert "Not found: (404) Trace default/nope not found" in result.stderr


def test_traces_get_without_workspace_is_usage_error(intake_cli) -> None:
    result, recorder = intake_cli.run(["traces", "get", "trace-1"], workspace=None)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_traces_list_filters_mode_sort_and_pages(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "traces",
            "list",
            "--filter.agent-name",
            "bot",
            "--filter.status",
            "success",
            "--mode",
            "detailed",
            "--sort",
            "-started_at",
            "--page",
            "1",
            "--page-size",
            "1",
        ],
        [intake_cli.page([TRACE], 1, 3)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{WS}/traces"
    params = dict(recorder.last.url.params)
    assert json.loads(params.pop("filter")) == {"agent_name": "bot", "status": "success"}
    assert params == {"mode": "detailed", "sort": "-started_at", "page": "1", "page_size": "1"}
    body = json.loads(result.stdout)
    assert [item["id"] for item in body["data"]] == ["trace-1"]
    assert body["pagination"]["total_pages"] == 3
    assert "More pages" in result.stderr


def test_traces_list_json_filter_with_started_at_range(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["traces", "list", "--filter", '{"started_at": {"gte": "2026-01-01T00:00:00Z"}}', "--filter.id", "trace-1"],
        [intake_cli.page([TRACE], 1, 1)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(dict(recorder.last.url.params)["filter"]) == {
        "started_at": {"gte": "2026-01-01T00:00:00Z"},
        "id": "trace-1",
    }


def test_traces_list_all_pages(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["traces", "list", "--page-size", "1", "--all-pages"],
        [intake_cli.page([TRACE], 1, 2), intake_cli.page([{**TRACE, "id": "trace-2"}], 2, 2)],
    )

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["id"] for item in body["data"]] == ["trace-1", "trace-2"]
    assert "More pages" not in result.stderr


def test_traces_list_table_default_columns(intake_cli) -> None:
    result, _ = intake_cli.run(["traces", "list", "-f", "table"], [intake_cli.page([TRACE], 1, 1)])

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "workspace" in output
    assert "session_id" not in output and "status" not in output


def test_traces_list_stream_requires_json(intake_cli) -> None:
    result, recorder = intake_cli.run(["traces", "list", "--stream", "-f", "table"])

    assert result.exit_code == 2
    assert recorder.requests == []


def test_traces_get_metrics(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "traces",
            "get-metrics",
            "--bucket",
            "day",
            "--timezone",
            "America/Los_Angeles",
            "--filter.agent-name",
            "bot",
        ],
        [httpx.Response(200, json=TRACE_METRICS)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{WS}/traces/metrics"
    params = dict(recorder.last.url.params)
    assert json.loads(params.pop("filter")) == {"agent_name": "bot"}
    assert params == {"bucket": "day", "timezone": "America/Los_Angeles"}
    body = json.loads(result.stdout)
    assert body["bucket"] == "day"
    assert body["data"][0]["run_count"] == 2


def test_traces_get_metrics_bad_timezone_maps_to_remote_error(intake_cli) -> None:
    result, _ = intake_cli.run(
        ["traces", "get-metrics", "--timezone", "Mars/Olympus"],
        [httpx.Response(400, json={"detail": "Unknown timezone: Mars/Olympus"})],
    )

    assert result.exit_code == 3
    assert "Bad request: (400) Unknown timezone: Mars/Olympus" in result.stderr


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------


def test_spans_list(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "spans",
            "list",
            "--filter.trace-id",
            "trace-1",
            "--filter.kind",
            "LLM",
            "--filter.model",
            "m",
            "--mode",
            "summary",
            "--page-size",
            "1",
        ],
        [intake_cli.page([SPAN], 1, 2)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{WS}/spans"
    params = dict(recorder.last.url.params)
    assert json.loads(params.pop("filter")) == {"trace_id": "trace-1", "kind": "LLM", "model": "m"}
    assert params == {"mode": "summary", "page_size": "1"}
    assert [item["span_id"] for item in json.loads(result.stdout)["data"]] == ["span-1"]
    assert "More pages" in result.stderr


def test_spans_list_all_pages(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["spans", "list", "--all-pages"],
        [intake_cli.page([SPAN], 1, 2), intake_cli.page([{**SPAN, "span_id": "span-2"}], 2, 2)],
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 2
    assert [item["span_id"] for item in json.loads(result.stdout)["data"]] == ["span-1", "span-2"]


def test_spans_get(intake_cli) -> None:
    result, recorder = intake_cli.run(["spans", "get", "span-1"], [httpx.Response(200, json=SPAN)])

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{WS}/spans/span-1"
    assert json.loads(result.stdout)["span_id"] == "span-1"


def test_spans_get_not_found(intake_cli) -> None:
    result, _ = intake_cli.run(
        ["spans", "get", "nope"], [httpx.Response(404, json={"detail": "Span default/nope not found"})]
    )

    assert result.exit_code == 3
    assert "Not found: (404)" in result.stderr


def test_spans_groups_list_requires_by(intake_cli) -> None:
    result, recorder = intake_cli.run(["spans", "groups", "list"])

    assert result.exit_code == 2
    assert "<BY>" in result.stderr
    assert recorder.requests == []


def test_spans_groups_list(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["spans", "groups", "list", "--by", "trace_id", "--sort", "-started_at", "--filter.session-id", "sess-1"],
        [intake_cli.page([SPAN_GROUP], 1, 1)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{WS}/spans/groups"
    params = dict(recorder.last.url.params)
    assert json.loads(params.pop("filter")) == {"session_id": "sess-1"}
    assert params == {"by": "trace_id", "sort": "-started_at"}
    assert json.loads(result.stdout)["data"][0]["span_count"] == 3
    assert "More pages" not in result.stderr


def test_spans_groups_list_all_pages(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["spans", "groups", "list", "--by", "session_id", "--all-pages"],
        [intake_cli.page([SPAN_GROUP], 1, 2), intake_cli.page([SPAN_GROUP], 2, 2)],
    )

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params)["by"] for r in recorder.requests] == ["session_id", "session_id"]
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    assert len(json.loads(result.stdout)["data"]) == 2


def test_spans_evaluator_results_list_is_not_paginated(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["spans", "evaluator-results", "list", "span-1"], [httpx.Response(200, json=[EVALUATOR_RESULT])]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == f"{WS}/spans/span-1/evaluator-results"
    assert dict(recorder.last.url.params) == {}
    assert json.loads(result.stdout) == [
        {**EVALUATOR_RESULT, "string_value": None, "comment": None, "created_by": None}
    ]


def test_spans_evaluator_results_list_table(intake_cli) -> None:
    result, _ = intake_cli.run(
        ["spans", "evaluator-results", "list", "span-1", "-f", "table"], [httpx.Response(200, json=[EVALUATOR_RESULT])]
    )

    assert result.exit_code == 0, result.output
    assert "faithfulness/v1" in result.stdout


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


def test_sessions_get(intake_cli) -> None:
    result, recorder = intake_cli.run(["sessions", "get", "sess-1"], [httpx.Response(200, json=SESSION)])

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{WS}/sessions/sess-1"
    assert json.loads(result.stdout)["trace_count"] == 1


def test_sessions_get_code_output(intake_cli) -> None:
    result, recorder = intake_cli.run(["sessions", "get", "sess-1", "-f", "code"])

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'response = client.get_session(id="sess-1")' in result.stdout
    assert "NeMoPlatform" not in result.stdout


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------


def test_annotations_create_note(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "annotations",
            "create",
            "--kind",
            "note",
            "--session-id",
            "sess-1",
            "--text",
            "looks good",
            "--span-id",
            "span-1",
        ],
        [httpx.Response(201, json={**ANNOTATION, "span_id": "span-1"})],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{WS}/annotations"
    assert json.loads(recorder.last.content) == {
        "kind": "note",
        "session_id": "sess-1",
        "text": "looks good",
        "span_id": "span-1",
    }
    assert json.loads(result.stdout)["annotation_id"] == "ann-1"


def test_annotations_create_label_from_input_data(intake_cli) -> None:
    payload = {"kind": "label", "session_id": "sess-1", "value_type": "numeric", "value": 4, "name": "helpfulness"}
    result, recorder = intake_cli.run(
        ["annotations", "create", "--input-data", json.dumps(payload)],
        [httpx.Response(201, json={**ANNOTATION, **payload})],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == payload


def test_annotations_create_positional_name_and_exist_ok(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "annotations",
            "create",
            "severity",
            "--kind",
            "label",
            "--session-id",
            "sess-1",
            "--value-type",
            "text",
            "--value",
            "high",
            "--exist-ok",
        ],
        [httpx.Response(201, json=ANNOTATION)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {
        "kind": "label",
        "session_id": "sess-1",
        "value_type": "text",
        "value": "high",
        "name": "severity",
    }


def test_annotations_create_requires_kind_and_session(intake_cli) -> None:
    result, recorder = intake_cli.run(["annotations", "create", "--text", "hi"])

    assert result.exit_code == 2
    assert "--kind" in result.stderr and "--session-id" in result.stderr
    assert recorder.requests == []


def test_annotations_create_invalid_variant_is_usage_error(intake_cli) -> None:
    result, recorder = intake_cli.run(["annotations", "create", "--kind", "note", "--session-id", "sess-1"])

    assert result.exit_code == 2
    assert "Invalid input" in result.stderr
    assert recorder.requests == []


def test_annotations_list(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "annotations",
            "list",
            "--filter.kind",
            "note",
            "--filter.span-id",
            "span-1",
            "--sort",
            "created_at",
            "--page-size",
            "1",
        ],
        [intake_cli.page([ANNOTATION], 1, 2)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{WS}/annotations"
    params = dict(recorder.last.url.params)
    assert json.loads(params.pop("filter")) == {"kind": "note", "span_id": "span-1"}
    assert params == {"sort": "created_at", "page_size": "1"}
    assert json.loads(result.stdout)["data"][0]["text"] == "looks good"
    assert "More pages" in result.stderr


def test_annotations_list_all_pages(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["annotations", "list", "--all-pages"],
        [intake_cli.page([ANNOTATION], 1, 2), intake_cli.page([{**ANNOTATION, "annotation_id": "ann-2"}], 2, 2)],
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 2
    assert [item["annotation_id"] for item in json.loads(result.stdout)["data"]] == ["ann-1", "ann-2"]


def test_annotations_get(intake_cli) -> None:
    result, recorder = intake_cli.run(["annotations", "get", "ann-1"], [httpx.Response(200, json=ANNOTATION)])

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{WS}/annotations/ann-1"
    assert json.loads(result.stdout)["kind"] == "note"


def test_annotations_delete(intake_cli) -> None:
    result, recorder = intake_cli.run(["annotations", "delete", "ann-1"], [httpx.Response(204)])

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == f"{WS}/annotations/ann-1"
    assert "Deleted successfully" in result.stdout


def test_annotations_delete_not_found(intake_cli) -> None:
    result, _ = intake_cli.run(
        ["annotations", "delete", "nope"], [httpx.Response(404, json={"detail": "Annotation default/nope not found"})]
    )

    assert result.exit_code == 3


# ---------------------------------------------------------------------------
# evaluator-results
# ---------------------------------------------------------------------------


def test_evaluator_results_create_numeric(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "evaluator-results",
            "create",
            "faithfulness/v1",
            "--data-type",
            "NUMERIC",
            "--session-id",
            "sess-1",
            "--span-id",
            "span-1",
            "--value",
            "0.9",
            "--comment",
            "ok",
            "--exist-ok",
        ],
        [httpx.Response(201, json=EVALUATOR_RESULT)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{WS}/evaluator-results"
    assert json.loads(recorder.last.content) == {
        "name": "faithfulness/v1",
        "data_type": "NUMERIC",
        "session_id": "sess-1",
        "span_id": "span-1",
        "value": 0.9,
        "comment": "ok",
    }
    assert json.loads(result.stdout)["evaluator_result_id"] == "eval-1"


def test_evaluator_results_create_text_from_input_file(intake_cli, tmp_path) -> None:
    payload = {
        "name": "verdict",
        "data_type": "TEXT",
        "session_id": "sess-1",
        "span_id": "span-1",
        "string_value": "pass",
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))

    result, recorder = intake_cli.run(
        ["evaluator-results", "create", "--input-file", str(path)], [httpx.Response(201, json=EVALUATOR_RESULT)]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == payload


def test_evaluator_results_create_requires_fields(intake_cli) -> None:
    result, recorder = intake_cli.run(["evaluator-results", "create", "faithfulness/v1", "--value", "1"])

    assert result.exit_code == 2
    assert "--data-type" in result.stderr and "--session-id" in result.stderr and "--span-id" in result.stderr
    assert recorder.requests == []


def test_evaluator_results_create_numeric_without_value_is_usage_error(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["evaluator-results", "create", "n", "--data-type", "NUMERIC", "--session-id", "s", "--span-id", "p"]
    )

    assert result.exit_code == 2
    assert "Invalid input" in result.stderr
    assert recorder.requests == []


def test_evaluator_results_list(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "evaluator-results",
            "list",
            "--filter.name",
            "faithfulness/v1",
            "--filter.data-type",
            "NUMERIC",
            "--sort",
            "-value",
        ],
        [intake_cli.page([EVALUATOR_RESULT], 1, 1)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{WS}/evaluator-results"
    params = dict(recorder.last.url.params)
    assert json.loads(params.pop("filter")) == {"name": "faithfulness/v1", "data_type": "NUMERIC"}
    assert params == {"sort": "-value"}
    assert json.loads(result.stdout)["data"][0]["value"] == 0.9


def test_evaluator_results_list_all_pages(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["evaluator-results", "list", "--all-pages"],
        [intake_cli.page([EVALUATOR_RESULT], 1, 2), intake_cli.page([EVALUATOR_RESULT], 2, 2)],
    )

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    assert len(json.loads(result.stdout)["data"]) == 2


def test_evaluator_results_get(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["evaluator-results", "get", "eval-1"], [httpx.Response(200, json=EVALUATOR_RESULT)]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == f"{WS}/evaluator-results/eval-1"
    assert json.loads(result.stdout)["name"] == "faithfulness/v1"


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def test_ingest_atif_create_from_flags(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "ingest",
            "atif",
            "create",
            "--agent",
            '{"name": "bot", "version": "1"}',
            "--schema-version",
            "ATIF-v1.7",
            "--session-id",
            "sess-1",
            "--steps",
            '[{"source": "agent", "step_id": 1, "message": "hi"}]',
        ],
        [httpx.Response(201)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{WS}/ingest/atif"
    assert json.loads(recorder.last.content) == {
        "agent": {"name": "bot", "version": "1"},
        "schema_version": "ATIF-v1.7",
        "session_id": "sess-1",
        "steps": [{"source": "agent", "step_id": 1, "message": "hi"}],
    }
    assert result.stdout.strip() == "null"


def test_ingest_atif_create_forwards_unknown_keys_from_input(intake_cli) -> None:
    payload = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "bot", "version": "1", "model_name": "m"},
        "final_metrics": {"total_steps": 1},
        "custom_top_level": True,
    }
    result, recorder = intake_cli.run(
        ["ingest", "atif", "create", "--input-data", json.dumps(payload)],
        [httpx.Response(422, json={"detail": "Extra inputs are not permitted"})],
    )

    assert result.exit_code == 3
    assert json.loads(recorder.last.content) == payload
    assert "Invalid input: (422)" in result.stderr


def test_ingest_atif_create_requires_agent_and_schema_version(intake_cli) -> None:
    result, recorder = intake_cli.run(["ingest", "atif", "create", "--session-id", "s"])

    assert result.exit_code == 2
    assert "--agent" in result.stderr and "--schema-version" in result.stderr
    assert recorder.requests == []


def test_ingest_chat_completions_create(intake_cli) -> None:
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    response = {"choices": [{"message": {"role": "assistant", "content": "hello"}}], "usage": {"total_tokens": 3}}
    result, recorder = intake_cli.run(
        [
            "ingest",
            "chat-completions",
            "create",
            "--request",
            json.dumps(request),
            "--response",
            json.dumps(response),
            "--session-id",
            "sess-1",
            "--cost-usd",
            "0.5",
            "--provider",
            "openai",
        ],
        [httpx.Response(201, json={"session_id": "sess-1", "span_id": "span-1"})],
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{WS}/ingest/chat-completions"
    assert json.loads(recorder.last.content) == {
        "request": request,
        "response": response,
        "session_id": "sess-1",
        "cost_usd": 0.5,
        "provider": "openai",
    }
    assert json.loads(result.stdout) == {"session_id": "sess-1", "span_id": "span-1"}


def test_ingest_chat_completions_requires_request_and_response(intake_cli) -> None:
    result, recorder = intake_cli.run(["ingest", "chat-completions", "create", "--provider", "x"])

    assert result.exit_code == 2
    assert "--request" in result.stderr and "--response" in result.stderr
    assert recorder.requests == []


def test_ingest_spans_create_from_flags(intake_cli) -> None:
    spans = [{"span_id": "span-1", "trace_id": "trace-1", "started_at": "2026-08-14T00:00:00Z"}]
    result, recorder = intake_cli.run(
        ["ingest", "spans", "create", "--source", "langsmith", "--spans", json.dumps(spans)], [httpx.Response(201)]
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == f"{WS}/ingest/spans"
    assert json.loads(recorder.last.content) == {"source": "langsmith", "spans": spans}


def test_ingest_spans_create_from_stdin(intake_cli) -> None:
    payload = {
        "source": "mlflow",
        "spans": [
            {
                "span_id": "s1",
                "trace_id": "t1",
                "started_at": "2026-08-14T00:00:00Z",
                "kind": "TOOL",
                "attributes": {"tool.name": "search"},
            }
        ],
    }
    result, recorder = intake_cli.run(
        ["ingest", "spans", "create", "--input-file", "-"], [httpx.Response(201)], input=json.dumps(payload)
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == payload


def test_ingest_spans_create_requires_source_and_spans(intake_cli) -> None:
    result, recorder = intake_cli.run(["ingest", "spans", "create"])

    assert result.exit_code == 2
    assert "--source" in result.stderr and "--spans" in result.stderr
    assert recorder.requests == []


def test_ingest_spans_create_retention_error_maps_to_remote_error(intake_cli) -> None:
    spans = [{"span_id": "s1", "trace_id": "t1", "started_at": "2020-01-01T00:00:00Z"}]
    result, _ = intake_cli.run(
        ["ingest", "spans", "create", "--source", "x", "--spans", json.dumps(spans)],
        [httpx.Response(422, json={"detail": "1 span(s) fall outside Intake's retention window"})],
    )

    assert result.exit_code == 3
    assert "Invalid input: (422)" in result.stderr


# ---------------------------------------------------------------------------
# -f code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["traces", "list", "--mode", "summary"], 'client.list_traces(query_params={"mode": "summary"})'),
        (["traces", "get", "trace-1"], 'client.get_trace(id="trace-1")'),
        (["traces", "get-metrics", "--bucket", "hour"], 'client.get_trace_metrics(query_params={"bucket": "hour"})'),
        (["spans", "list"], "client.list_spans()"),
        (["spans", "get", "span-1"], 'client.get_span(span_id="span-1")'),
        (["spans", "groups", "list", "--by", "trace_id"], 'client.list_span_groups(query_params={"by": "trace_id"})'),
        (
            ["spans", "evaluator-results", "list", "span-1"],
            'client.list_evaluator_results_for_span(span_id="span-1")',
        ),
        (["annotations", "list"], "client.list_annotations()"),
        (["annotations", "get", "ann-1"], 'client.get_annotation(annotation_id="ann-1")'),
        (["evaluator-results", "list"], "client.list_evaluator_results()"),
        (["evaluator-results", "get", "eval-1"], 'client.get_evaluator_result(evaluator_result_id="eval-1")'),
    ],
)
def test_code_output_for_reads(intake_cli, args: list[str], expected: str) -> None:
    result, recorder = intake_cli.run([*args, "-f", "code"])

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.intake.client import IntakeClient" in result.stdout
    assert 'client = IntakeClient(base_url="http://test/")' in result.stdout
    assert expected in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_code_output_for_annotation_create_renders_variant(intake_cli) -> None:
    result, recorder = intake_cli.run(
        ["annotations", "create", "--kind", "note", "--session-id", "sess-1", "--text", "hi", "-f", "code"]
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.intake.types import NoteAnnotationInput" in result.stdout
    assert 'body=NoteAnnotationInput(session_id="sess-1", kind="note", text="hi")' in result.stdout


def test_code_output_for_evaluator_result_create(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "evaluator-results",
            "create",
            "n",
            "--data-type",
            "NUMERIC",
            "--session-id",
            "s",
            "--span-id",
            "p",
            "--value",
            "1",
            "-f",
            "code",
        ]
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "EvaluatorResultCreateRequest(" in result.stdout
    assert 'data_type="NUMERIC"' in result.stdout


def test_code_output_for_ingest_spans_renders_iso_timestamps(intake_cli) -> None:
    spans = [{"span_id": "s1", "trace_id": "t1", "started_at": "2026-08-14T00:00:00Z"}]
    result, recorder = intake_cli.run(
        ["ingest", "spans", "create", "--source", "langsmith", "--spans", json.dumps(spans), "-f", "code"]
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "DirectSpansIngestRequest(" in result.stdout
    assert 'started_at="2026-08-14T00:00:00+00:00"' in result.stdout
    assert "datetime.datetime" not in result.stdout


def test_code_output_for_ingest_atif(intake_cli) -> None:
    result, recorder = intake_cli.run(
        [
            "ingest",
            "atif",
            "create",
            "--agent",
            '{"name": "b", "version": "1"}',
            "--schema-version",
            "ATIF-v1.7",
            "-f",
            "code",
        ]
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'AtifCreateRequest({"schema_version": "ATIF-v1.7", "agent": {"name": "b", "version": "1"}})' in result.stdout
