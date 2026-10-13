# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace API filter tests."""

import json
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from nhx.common.api.filter import parse_json_filter
from nhx.common.api.parsed_filter import ParsedFilter
from nhx.intake.spans.api.traces import _trace_filter
from nhx.intake.spans.api.traces_schemas import Trace, TraceFilter
from nhx.intake.spans.domain import IntakeTrace, SpanStatus


def test_trace_filter_maps_public_fields_to_repository_filter():
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    filters = _trace_filter(
        "workspace-a",
        _parsed_filter(
            {
                "id": {"$in": ["trace-a", "trace-b"]},
                "session_id": "session-a",
                "status": "error",
                "started_at": {"$gte": started_at.isoformat()},
                "evaluation_name": "experiment-a",
                "test_case_name": "case-a",
            }
        ),
    )

    assert filters.workspace == "workspace-a"
    assert filters.trace_ids == ["trace-a", "trace-b"]
    assert filters.session_id == "session-a"
    assert filters.status == SpanStatus.ERROR
    assert filters.started_at_gte == started_at
    assert filters.evaluation_name == "experiment-a"
    assert filters.test_case_name == "case-a"


def test_trace_filter_accepts_deprecated_identifier_aliases():
    filters = _trace_filter(
        "workspace-a",
        _parsed_filter(
            {
                "evaluation_name": "experiment-a",
                "evaluation_id": "experiment-a",
                "test_case_name": "case-a",
                "test_case_id": "case-a",
            }
        ),
    )

    assert filters.evaluation_name == "experiment-a"
    assert filters.test_case_name == "case-a"


def test_trace_filter_rejects_conflicting_identifier_aliases():
    with pytest.raises(HTTPException) as exc_info:
        _trace_filter(
            "workspace-a",
            _parsed_filter({"evaluation_name": "experiment-a", "evaluation_id": "experiment-b"}),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Conflicting trace filters for evaluation_name"


def test_trace_filter_schema_exposes_canonical_names_and_deprecated_aliases():
    properties = TraceFilter.model_json_schema()["properties"]

    assert properties["evaluation_name"]["description"] == "Filter by Evaluation name."
    assert properties["test_case_name"]["description"] == "Filter by test case name."
    assert properties["evaluation_id"]["deprecated"] is True
    assert properties["evaluation_id"]["description"] == (
        "Deprecated alias for evaluation_name. Use evaluation_name instead."
    )
    assert properties["test_case_id"]["deprecated"] is True
    assert properties["test_case_id"]["description"] == (
        "Deprecated alias for test_case_name. Use test_case_name instead."
    )
    assert "experiment_id" not in properties


def test_trace_filter_accepts_agent_name():
    filters = _trace_filter("workspace-a", _parsed_filter({"agent_name": "support-bot"}))

    assert filters.agent_name == "support-bot"


def test_trace_filter_schema_exposes_agent_name():
    properties = TraceFilter.model_json_schema()["properties"]

    assert properties["agent_name"]["description"] == "Filter by root-span agent name."
    assert "agent_id" not in properties


def test_trace_filter_applies_no_implicit_time_bound():
    filters = _trace_filter("workspace-a", _parsed_filter({"id": "trace-a"}))

    assert filters.trace_ids == ["trace-a"]
    assert filters.started_at_gte is None
    assert filters.started_at_lte is None


def _parsed_filter(value: dict[str, object]) -> ParsedFilter:
    return ParsedFilter(operation=parse_json_filter(json.dumps(value)))


def test_trace_response_exposes_models_and_providers() -> None:
    trace = IntakeTrace(
        id="trace-a",
        workspace="workspace-a",
        session_id="session-a",
        source_format="atif",
        started_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
        ingested_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
        status=SpanStatus.SUCCESS,
        models=["claude-opus-4-6", "qwen3-next-80b"],
        providers=["anthropic", "openai"],
    )

    response = Trace.from_domain(trace, mode="detailed")

    assert response.models == ["claude-opus-4-6", "qwen3-next-80b"]
    assert response.providers == ["anthropic", "openai"]


def test_trace_response_omits_models_when_the_rollup_did_not_run() -> None:
    # summary mode skips the span-aggregate join, so these stay unset rather
    # than reporting an empty list the caller would read as "no models".
    trace = IntakeTrace(
        id="trace-a",
        workspace="workspace-a",
        session_id="session-a",
        source_format="atif",
        started_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
        ingested_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
        status=SpanStatus.SUCCESS,
    )

    payload = Trace.from_domain(trace, mode="summary").model_dump(exclude_none=True)

    assert "models" not in payload
    assert "providers" not in payload
