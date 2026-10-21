# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the OTLP protobuf builder/exporter and the evaluator-results POST."""

from typing import Any

import pytest
from evaluation.otlp_ingest import (
    export_spans,
    post_evaluator_results,
    trace_id_for,
)

_TRACE_ID = trace_id_for("sess-1")


class _StubResponse:
    def __init__(self, status_code: int, *, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.text = "body"
        self.payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self.payload


def _span(span_id, parent, kind, name, *, attributes=None, status="OK", status_message=None):
    return {
        "span_id": span_id,
        "parent_span_id": parent,
        "name": name,
        "kind": kind,
        "start_ns": 1_700_000_000_000_000_000,
        "end_ns": 1_700_000_000_001_000_000,
        "attributes": {"openinference.span.kind": kind, **(attributes or {})},
        "status": status,
        "status_message": status_message,
    }


class _ExportStub:
    """Captures a single protobuf POST (url, raw body, headers)."""

    def __init__(self, status: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status = status
        self.payload = {"errors": []} if payload is None else payload
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> _StubResponse:
        self.calls.append({"url": url, "content": content, "headers": headers or {}, "json": json})
        return _StubResponse(self.status, payload=self.payload)

    def close(self) -> None:
        pass


def _decode(body: bytes):
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

    req = trace_service_pb2.ExportTraceServiceRequest()
    req.ParseFromString(body)
    return req


def _attr_value(any_value):
    field = any_value.WhichOneof("value")
    return getattr(any_value, field) if field else None


def _attrs(span):
    return {a.key: _attr_value(a.value) for a in span.attributes}


# --- trace id --------------------------------------------------------------


def test_trace_id_for_is_16_bytes_deterministic_nonzero():
    tid = trace_id_for("sess-1")
    assert isinstance(tid, bytes) and len(tid) == 16
    assert any(tid)  # non-zero (Intake rejects all-zero ids)
    assert tid == trace_id_for("sess-1")  # deterministic
    assert tid != trace_id_for("sess-2")


# --- export_spans: protobuf build + POST -----------------------------------


def test_export_spans_posts_protobuf_to_otlp_route():
    stub = _ExportStub()
    spans = [_span("0000000000000001", None, "AGENT", "agent")]
    export_spans("http://x/", "ws", "sess-1", _TRACE_ID, spans, client=stub)
    (call,) = stub.calls
    assert call["url"] == "http://x/apis/intake/v2/workspaces/ws/ingest/otlp/v1/traces"
    assert call["headers"]["Content-Type"] == "application/x-protobuf"
    assert call["content"] is not None and call["json"] is None


def test_export_spans_serializes_ids_parents_kinds():
    stub = _ExportStub()
    spans = [
        _span("0000000000000001", None, "AGENT", "agent"),
        _span("0000000000000002", "0000000000000001", "LLM", "agent-1"),
    ]
    export_spans("http://x", "ws", "sess-1", _TRACE_ID, spans, client=stub)
    req = _decode(stub.calls[0]["content"])
    out = req.resource_spans[0].scope_spans[0].spans
    assert {s.name for s in out} == {"agent", "agent-1"}
    by_name = {s.name: s for s in out}
    assert by_name["agent"].trace_id == _TRACE_ID
    assert by_name["agent"].span_id.hex() == "0000000000000001"
    assert _attrs(by_name["agent"])["openinference.span.kind"] == "AGENT"
    # the LLM turn points at the AGENT root
    assert by_name["agent-1"].parent_span_id.hex() == "0000000000000001"


def test_root_parent_span_id_is_empty_not_zero():
    # Intake rejects an all-zero parent_span_id; a parentless root must leave it unset.
    stub = _ExportStub()
    export_spans("http://x", "ws", "s", _TRACE_ID, [_span("0000000000000001", None, "AGENT", "a")], client=stub)
    root = _decode(stub.calls[0]["content"]).resource_spans[0].scope_spans[0].spans[0]
    assert len(root.parent_span_id) == 0


def test_resource_has_service_name():
    stub = _ExportStub()
    export_spans("http://x", "ws", "s", _TRACE_ID, [_span("0000000000000001", None, "AGENT", "a")], client=stub)
    res = _decode(stub.calls[0]["content"]).resource_spans[0].resource
    names = {a.key: _attr_value(a.value) for a in res.attributes}
    assert names["service.name"] == "nemo-compass-evaluation"


def test_export_spans_typed_attributes_roundtrip():
    stub = _ExportStub()
    attrs = {"input.value": "txt", "score": 1.0, "count": 3, "flag": True}
    export_spans(
        "http://x", "ws", "s", _TRACE_ID, [_span("0000000000000001", None, "TOOL", "t", attributes=attrs)], client=stub
    )
    span = _decode(stub.calls[0]["content"]).resource_spans[0].scope_spans[0].spans[0]
    decoded = _attrs(span)
    assert decoded["input.value"] == "txt"
    assert decoded["score"] == 1.0
    assert decoded["count"] == 3
    assert decoded["flag"] is True


def test_export_spans_error_status_sets_code_2():
    stub = _ExportStub()
    spans = [_span("0000000000000001", None, "TOOL", "boom", status="ERROR", status_message="kaboom")]
    export_spans("http://x", "ws", "s", _TRACE_ID, spans, client=stub)
    span = _decode(stub.calls[0]["content"]).resource_spans[0].scope_spans[0].spans[0]
    assert span.status.code == 2  # STATUS_CODE_ERROR
    assert span.status.message == "kaboom"


def test_export_spans_raises_on_non_2xx():
    with pytest.raises(RuntimeError):
        export_spans(
            "http://x",
            "ws",
            "s",
            _TRACE_ID,
            [_span("0000000000000001", None, "AGENT", "a")],
            client=_ExportStub(status=500),
        )


def test_export_spans_raises_on_errors_array():
    # 2xx but per-span errors reported -> still a failure.
    stub = _ExportStub(status=200, payload={"errors": ["span 01: bad"]})
    with pytest.raises(RuntimeError):
        export_spans("http://x", "ws", "s", _TRACE_ID, [_span("0000000000000001", None, "AGENT", "a")], client=stub)


# --- post_evaluator_results ------------------------------------------------


class _JsonStub:
    def __init__(self, status: int = 201) -> None:
        self.status = status
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(
        self,
        url: str,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> _StubResponse:
        assert json is not None
        self.calls.append((url, json))
        return _StubResponse(self.status)

    def close(self) -> None:
        pass


def test_post_evaluator_results_route_and_body():
    stub = _JsonStub(status=201)
    post_evaluator_results("http://x/", "ws", span_id="sp1", session_id="sess-1", score=1.0, client=stub)
    (url, body) = stub.calls[0]
    assert url == "http://x/apis/intake/v2/workspaces/ws/evaluator-results"
    assert body == {
        "span_id": "sp1",
        "session_id": "sess-1",
        "name": "reward",
        "value": 1.0,
        "data_type": "NUMERIC",
    }


def test_post_evaluator_results_raises_on_error():
    with pytest.raises(RuntimeError):
        post_evaluator_results("http://x", "ws", span_id="s", session_id="z", score=0.0, client=_JsonStub(status=500))
