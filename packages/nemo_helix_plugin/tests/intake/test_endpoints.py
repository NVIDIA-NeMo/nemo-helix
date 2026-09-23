# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any, get_args, get_origin

from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.intake import endpoints
from nemo_helix_plugin.intake.types import (
    AtifCreateRequest,
    EvaluationPatchRequest,
    EvaluationResponse,
    EvaluatorResult,
    EvaluatorResultCreateRequest,
    IngestResponse,
    Trace,
)


def _json_content(prepared: PreparedRequest[Any]) -> object:
    assert isinstance(prepared.content, bytes)
    return json.loads(prepared.content)


def _assert_paginated_model(response_type: object, model_type: type[object]) -> None:
    assert get_origin(response_type) is Paginated
    assert get_args(response_type)[0] is model_type


def test_create_atif_endpoint_shape() -> None:
    prepared = endpoints.create_atif(
        workspace="team-a",
        body=AtifCreateRequest(root={"schema_version": "atif-0.1", "agent": {"name": "agent", "version": "1"}}),
    )

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/intake/v2/workspaces/{workspace}/ingest/atif"
    assert prepared.path_params == {"workspace": "team-a"}
    assert _json_content(prepared) == {
        "schema_version": "atif-0.1",
        "agent": {"name": "agent", "version": "1"},
    }
    assert prepared.content_type == "application/json"
    assert prepared.response_type is None


def test_create_otlp_traces_endpoint_shape() -> None:
    prepared = endpoints.create_otlp_traces(workspace="team-a", content=b"trace-protobuf")

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces"
    assert prepared.path_params == {"workspace": "team-a"}
    assert prepared.content == b"trace-protobuf"
    assert prepared.content_type == "application/x-protobuf"
    assert prepared.response_type is IngestResponse


def test_trace_list_endpoint_shape() -> None:
    prepared = endpoints.list_traces(
        workspace="team-a",
        query_params={"page": 2, "mode": "detailed", "filter": {"session_id": "run-1:trial-1"}},
    )

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/intake/v2/workspaces/{workspace}/traces"
    assert prepared.path_params == {"workspace": "team-a"}
    assert prepared.query_params == {"page": 2, "mode": "detailed", "filter": {"session_id": "run-1:trial-1"}}
    _assert_paginated_model(prepared.response_type, Trace)


def test_evaluator_result_endpoint_shapes() -> None:
    created = endpoints.create_evaluator_result(
        workspace="team-a",
        body=EvaluatorResultCreateRequest(
            span_id="span-1",
            session_id="session-1",
            name="accuracy.score",
            value=1,
            data_type="NUMERIC",
        ),
    )
    listed = endpoints.list_evaluator_results(
        workspace="team-a",
        query_params={"page_size": 20, "filter": {"session_id": "session-1"}},
    )
    by_span = endpoints.list_evaluator_results_for_span(workspace="team-a", span_id="span-1")

    assert created.method == "POST"
    assert created.path_template == "/apis/intake/v2/workspaces/{workspace}/evaluator-results"
    assert _json_content(created) == {
        "span_id": "span-1",
        "session_id": "session-1",
        "name": "accuracy.score",
        "value": 1.0,
        "data_type": "NUMERIC",
    }
    assert created.response_type is EvaluatorResult
    _assert_paginated_model(listed.response_type, EvaluatorResult)
    assert listed.query_params == {"page_size": 20, "filter": {"session_id": "session-1"}}
    assert by_span.path_template == "/apis/intake/v2/workspaces/{workspace}/spans/{span_id}/evaluator-results"
    assert by_span.path_params == {"workspace": "team-a", "span_id": "span-1"}
    assert get_origin(by_span.response_type) is list
    assert get_args(by_span.response_type)[0] is EvaluatorResult


def test_evaluation_endpoint_shapes() -> None:
    retrieved = endpoints.get_evaluation(workspace="team-a", name="eval-1")
    patched = endpoints.patch_evaluation(
        workspace="team-a",
        name="eval-1",
        body=EvaluationPatchRequest(metadata={"eval_duration_sec": "1.0"}),
    )

    assert retrieved.method == "GET"
    assert retrieved.path_template == "/apis/intake/v2/workspaces/{workspace}/evaluations/{name}"
    assert retrieved.path_params == {"workspace": "team-a", "name": "eval-1"}
    assert retrieved.response_type is EvaluationResponse
    assert patched.method == "PATCH"
    assert patched.path_params == {"workspace": "team-a", "name": "eval-1"}
    assert _json_content(patched) == {"metadata": {"eval_duration_sec": "1.0"}}
    assert patched.response_type is EvaluationResponse
