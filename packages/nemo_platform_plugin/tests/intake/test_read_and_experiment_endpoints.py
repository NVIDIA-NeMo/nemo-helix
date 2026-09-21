# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint-shape and wire tests for the Intake read, ingest, annotation, and experiment routes."""

from __future__ import annotations

import json
from typing import Any, get_args, get_origin

import httpx
import pytest
from nemo_platform_plugin.client.errors import ConflictError
from nemo_platform_plugin.client.types import Paginated, PreparedRequest
from nemo_platform_plugin.intake import endpoints
from nemo_platform_plugin.intake.client import IntakeClient
from nemo_platform_plugin.intake.types import (
    ANNOTATION_INPUT_ADAPTER,
    Annotation,
    ChatCompletionsIngestRequest,
    ChatCompletionsIngestResponse,
    DirectSpanInput,
    DirectSpansIngestRequest,
    EvaluatorResult,
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentUpdateRequest,
    FeedbackAnnotationInput,
    LabelAnnotationInput,
    NoteAnnotationInput,
    Session,
    Span,
    SpanGroup,
    Trace,
    TraceMetrics,
)
from pydantic import ValidationError

BASE = "http://test:8000"
WS = "/apis/intake/v2/workspaces/team-a"


def _json_content(prepared: PreparedRequest[Any]) -> object:
    assert isinstance(prepared.content, bytes)
    return json.loads(prepared.content)


def _assert_paginated_model(response_type: object, model_type: type[object]) -> None:
    assert get_origin(response_type) is Paginated
    assert get_args(response_type)[0] is model_type


def _span_input() -> DirectSpanInput:
    return DirectSpanInput.model_validate({"span_id": "s1", "trace_id": "t1", "started_at": "2026-08-14T00:00:00Z"})


def _client(handler) -> IntakeClient:
    return IntakeClient(
        base_url=BASE, workspace="team-a", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


# ---------------------------------------------------------------------------
# Traces / spans / sessions
# ---------------------------------------------------------------------------


def test_get_trace_endpoint_shape() -> None:
    prepared = endpoints.get_trace(workspace="team-a", id="trace-1", query_params={"mode": "summary"})

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/intake/v2/workspaces/{workspace}/traces/{id}"
    assert prepared.path_params == {"workspace": "team-a", "id": "trace-1"}
    assert prepared.query_params == {"mode": "summary"}
    assert prepared.response_type is Trace


def test_get_trace_metrics_endpoint_shape() -> None:
    prepared = endpoints.get_trace_metrics(
        workspace="team-a", query_params={"bucket": "day", "timezone": "UTC", "filter": {"agent_name": "bot"}}
    )

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/intake/v2/workspaces/{workspace}/traces/metrics"
    assert prepared.query_params == {"bucket": "day", "timezone": "UTC", "filter": {"agent_name": "bot"}}
    assert prepared.response_type is TraceMetrics


def test_span_endpoint_shapes() -> None:
    listed = endpoints.list_spans(workspace="team-a", query_params={"page": 2, "filter": {"trace_id": "t-1"}})
    grouped = endpoints.list_span_groups(workspace="team-a", query_params={"by": "trace_id", "sort": "-span_count"})
    single = endpoints.get_span(workspace="team-a", span_id="span-1")

    assert listed.path_template == "/apis/intake/v2/workspaces/{workspace}/spans"
    assert listed.query_params == {"page": 2, "filter": {"trace_id": "t-1"}}
    _assert_paginated_model(listed.response_type, Span)
    assert grouped.path_template == "/apis/intake/v2/workspaces/{workspace}/spans/groups"
    assert grouped.query_params == {"by": "trace_id", "sort": "-span_count"}
    _assert_paginated_model(grouped.response_type, SpanGroup)
    assert single.method == "GET"
    assert single.path_template == "/apis/intake/v2/workspaces/{workspace}/spans/{span_id}"
    assert single.path_params == {"workspace": "team-a", "span_id": "span-1"}
    assert single.response_type is Span


def test_get_session_endpoint_shape() -> None:
    prepared = endpoints.get_session(workspace="team-a", id="sess-1")

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/intake/v2/workspaces/{workspace}/sessions/{id}"
    assert prepared.path_params == {"workspace": "team-a", "id": "sess-1"}
    assert prepared.response_type is Session


def test_get_evaluator_result_endpoint_shape() -> None:
    prepared = endpoints.get_evaluator_result(workspace="team-a", evaluator_result_id="eval-1")

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/intake/v2/workspaces/{workspace}/evaluator-results/{evaluator_result_id}"
    assert prepared.path_params == {"workspace": "team-a", "evaluator_result_id": "eval-1"}
    assert prepared.response_type is EvaluatorResult


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def test_annotation_input_adapter_selects_variant_by_kind() -> None:
    note = ANNOTATION_INPUT_ADAPTER.validate_python({"kind": "note", "session_id": "s", "text": "hi"})
    feedback = ANNOTATION_INPUT_ADAPTER.validate_python({"kind": "feedback", "session_id": "s", "value": "positive"})
    label = ANNOTATION_INPUT_ADAPTER.validate_python(
        {"kind": "label", "session_id": "s", "value_type": "numeric", "value": 4, "name": "helpfulness"}
    )

    assert isinstance(note, NoteAnnotationInput)
    assert isinstance(feedback, FeedbackAnnotationInput)
    assert isinstance(label, LabelAnnotationInput)
    with pytest.raises(ValidationError):
        ANNOTATION_INPUT_ADAPTER.validate_python(
            {"kind": "label", "session_id": "s", "value_type": "numeric", "value": "4"}
        )
    with pytest.raises(ValidationError):
        ANNOTATION_INPUT_ADAPTER.validate_python({"kind": "note", "session_id": "s", "text": "hi", "bogus": 1})


def test_annotation_endpoint_shapes() -> None:
    created = endpoints.create_annotation(
        workspace="team-a", body=NoteAnnotationInput(kind="note", session_id="s", text="hi")
    )
    listed = endpoints.list_annotations(workspace="team-a", query_params={"sort": "-created_at"})
    single = endpoints.get_annotation(workspace="team-a", annotation_id="ann-1")
    deleted = endpoints.delete_annotation(workspace="team-a", annotation_id="ann-1")

    assert created.method == "POST"
    assert created.path_template == "/apis/intake/v2/workspaces/{workspace}/annotations"
    assert _json_content(created) == {"kind": "note", "session_id": "s", "text": "hi"}
    assert created.response_type is Annotation
    _assert_paginated_model(listed.response_type, Annotation)
    assert single.path_template == "/apis/intake/v2/workspaces/{workspace}/annotations/{annotation_id}"
    assert single.path_params == {"workspace": "team-a", "annotation_id": "ann-1"}
    assert deleted.method == "DELETE"
    assert deleted.path_template == single.path_template
    assert deleted.response_type is None


def test_annotation_read_model_keeps_kind_specific_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"{WS}/annotations/ann-1"
        return httpx.Response(
            200,
            request=request,
            json={
                "kind": "label",
                "annotation_id": "ann-1",
                "workspace": "team-a",
                "session_id": "s",
                "value_type": "numeric",
                "value": 4,
                "name": "helpfulness",
                "created_at": "2026-01-01T00:00:00Z",
                "ingested_at": "2026-01-01T00:00:00Z",
            },
        )

    annotation = _client(handler).get_annotation(annotation_id="ann-1").data()

    assert annotation.kind == "label"
    assert annotation.name == "helpfulness"
    assert annotation.model_dump(mode="json")["value"] == 4.0


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def test_ingest_endpoint_shapes() -> None:
    chat = endpoints.create_chat_completion(
        workspace="team-a",
        body=ChatCompletionsIngestRequest(
            request={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            response={"choices": []},
            session_id="s",
        ),
    )
    spans = endpoints.create_spans(
        workspace="team-a",
        body=DirectSpansIngestRequest(
            source="langsmith",
            spans=[_span_input()],
        ),
    )

    assert chat.method == "POST"
    assert chat.path_template == "/apis/intake/v2/workspaces/{workspace}/ingest/chat-completions"
    assert _json_content(chat) == {
        "request": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        "response": {"choices": []},
        "session_id": "s",
    }
    assert chat.response_type is ChatCompletionsIngestResponse
    assert spans.method == "POST"
    assert spans.path_template == "/apis/intake/v2/workspaces/{workspace}/ingest/spans"
    assert _json_content(spans) == {
        "source": "langsmith",
        "spans": [{"span_id": "s1", "trace_id": "t1", "started_at": "2026-08-14T00:00:00Z"}],
    }
    assert spans.response_type is None


def test_create_spans_accepts_empty_201() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"{WS}/ingest/spans"
        return httpx.Response(201, request=request)

    response = _client(handler).create_spans(
        body=DirectSpansIngestRequest(
            source="langsmith",
            spans=[_span_input()],
        )
    )

    assert response.data() is None


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def _experiment_json(name: str = "exp-1") -> dict[str, object]:
    return {
        "id": "id-1",
        "name": name,
        "workspace": "team-a",
        "default_sort": "-created_at",
        "created_at": "2026-01-01T00:00:00Z",
    }


def test_experiment_endpoint_shapes() -> None:
    created = endpoints.create_experiment(
        workspace="team-a", body=ExperimentCreateRequest(name="exp-1", description="d"), exist_ok=True
    )
    listed = endpoints.list_experiments(workspace="team-a", query_params={"sort": "-name", "filter": {"name": "x"}})
    single = endpoints.get_experiment(workspace="team-a", name="exp-1")
    updated = endpoints.update_experiment(
        workspace="team-a", name="exp-1", body=ExperimentUpdateRequest(name="exp-1", summary="s")
    )
    deleted = endpoints.delete_experiment(workspace="team-a", name="exp-1")

    assert created.method == "POST"
    assert created.path_template == "/apis/intake/v2/workspaces/{workspace}/experiments"
    assert _json_content(created) == {"name": "exp-1", "description": "d"}
    assert created.client_options == {"exist_ok": True}
    assert created.on_conflict_get is not None
    assert created.on_conflict_get.path_params == {"workspace": "team-a", "name": "exp-1"}
    assert created.response_type is ExperimentResponse
    _assert_paginated_model(listed.response_type, ExperimentResponse)
    assert listed.query_params == {"sort": "-name", "filter": {"name": "x"}}
    assert single.path_template == "/apis/intake/v2/workspaces/{workspace}/experiments/{name}"
    assert updated.method == "PUT"
    assert updated.path_params == {"workspace": "team-a", "name": "exp-1"}
    assert _json_content(updated) == {"name": "exp-1", "summary": "s"}
    assert deleted.method == "DELETE"
    assert deleted.response_type is None


def test_create_experiment_exist_ok_replays_get_on_conflict() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if request.method == "POST":
            return httpx.Response(409, request=request, json={"detail": "exists"})
        return httpx.Response(200, request=request, json=_experiment_json())

    client = _client(handler)
    existing = client.create_experiment(body=ExperimentCreateRequest(name="exp-1"), exist_ok=True).data()

    assert existing.name == "exp-1"
    assert seen == [f"POST {WS}/experiments", f"GET {WS}/experiments/exp-1"]


def test_create_experiment_without_exist_ok_raises_conflict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, request=request, json={"detail": "exists"})

    with pytest.raises(ConflictError):
        _client(handler).create_experiment(body=ExperimentCreateRequest(name="exp-1"))


def test_list_experiments_serializes_filter_and_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"{WS}/experiments"
        assert json.loads(request.url.params["filter"]) == {"is_favorite": True}
        return httpx.Response(
            200,
            request=request,
            json={
                "data": [_experiment_json()],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )

    page = _client(handler).list_experiments(query_params={"filter": {"is_favorite": True}}).page()

    assert [item.name for item in page.items] == ["exp-1"]
    assert page.metadata["total_results"] == 1
