# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and export OTLP protobuf requests, and post reward rows.

:mod:`evaluation.otlp_build` produces plain span dicts; this module turns them into an
``opentelemetry.proto`` ``ExportTraceServiceRequest`` and POSTs it to Intake's
permissive OTLP route. OTLP ingest does **not** auto-create the queryable
``evaluator_results`` rows the Analyst reads, so :func:`post_evaluator_results`
separately POSTs the reward (mirroring exactly the row the ATIF importer used to
create: ``name="reward"``, ``NUMERIC``, targeting the EVALUATOR span).

The protobuf build mirrors nemo-helix's own
``services/intake/tests/integration/spans/conftest.py::make_otlp_request`` helper,
which is the canonical reference for what the OTLP ingest route accepts.
"""

import hashlib
from typing import Any, Protocol

import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

_SERVICE_NAME = "nemo-insights-evaluation"
_SCOPE_NAME = "evaluation"
_SCOPE_VERSION = "1.0.0"
_STATUS_CODE_ERROR = 2  # opentelemetry.proto.trace.v1.Status.STATUS_CODE_ERROR


class _HTTPResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> dict[str, Any]: ...


class _HTTPClient(Protocol):
    def post(
        self,
        url: str,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> _HTTPResponse: ...

    def close(self) -> None: ...


def trace_id_for(session_id: str) -> bytes:
    """A deterministic, non-zero 16-byte OTLP trace id for a sim's session id."""
    return hashlib.sha256(f"{session_id}:trace".encode()).digest()[:16]


def export_spans(
    base_url: str,
    workspace: str,
    session_id: str,
    trace_id: bytes,
    spans: list[dict],
    *,
    client: httpx.Client | _HTTPClient | None = None,
) -> None:
    """Build an OTLP ``ExportTraceServiceRequest`` from span dicts and POST it.

    Raises ``RuntimeError`` on a non-2xx response **or** a non-empty ``errors``
    array (the route ingests good spans and reports per-span failures in the
    body, so a 2xx with errors is still a failure). ``client`` is injectable for
    tests; when None a short-timeout client is created and closed here.
    """
    export_trace_request(base_url, workspace, _build_request(trace_id, spans), client=client)


def export_trace_request(
    base_url: str,
    workspace: str,
    request: ExportTraceServiceRequest,
    *,
    client: httpx.Client | _HTTPClient | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """POST one OTLP request, raising for HTTP or per-span ingest errors."""
    url = f"{base_url.rstrip('/')}/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces"
    request_headers = {"Content-Type": "application/x-protobuf", **(headers or {})}
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.post(url, content=request.SerializeToString(), headers=request_headers)
    finally:
        if owns_client:
            client.close()
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"OTLP ingest failed ({resp.status_code}): {resp.text}")
    try:
        response_body = resp.json()
    except (AttributeError, ValueError):
        response_body = {}
    errors = response_body.get("errors") or []
    if errors:
        raise RuntimeError(f"OTLP ingest reported {len(errors)} span error(s): {errors}")


def post_evaluator_results(
    base_url: str,
    workspace: str,
    *,
    span_id: str,
    session_id: str,
    score: float,
    client: httpx.Client | _HTTPClient | None = None,
) -> None:
    """POST the reward row the OTLP path doesn't auto-create.

    Reproduces the ATIF importer's row exactly: ``name="reward"``, ``NUMERIC``,
    targeting the EVALUATOR span — so the Analyst reads the reward unchanged.
    Raises ``RuntimeError`` on any non-2xx.
    """
    url = f"{base_url.rstrip('/')}/apis/intake/v2/workspaces/{workspace}/evaluator-results"
    body = {
        "span_id": span_id,
        "session_id": session_id,
        "name": "reward",
        "value": score,
        "data_type": "NUMERIC",
    }
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.post(url, json=body)
    finally:
        if owns_client:
            client.close()
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"evaluator-results post failed ({resp.status_code}): {resp.text}")


def _build_request(trace_id: bytes, spans: list[dict]) -> ExportTraceServiceRequest:
    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    _add_attributes(resource_spans.resource.attributes, {"service.name": _SERVICE_NAME})
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = _SCOPE_NAME
    scope_spans.scope.version = _SCOPE_VERSION
    for spec in spans:
        span = scope_spans.spans.add()
        span.trace_id = trace_id
        span.span_id = bytes.fromhex(spec["span_id"])
        parent = spec.get("parent_span_id")
        if parent:  # leave unset for a root — an all-zero parent_span_id is rejected
            span.parent_span_id = bytes.fromhex(parent)
        span.name = spec["name"]
        span.start_time_unix_nano = int(spec["start_ns"])
        span.end_time_unix_nano = int(spec["end_ns"])
        if spec.get("status") == "ERROR":
            span.status.code = _STATUS_CODE_ERROR
            if spec.get("status_message"):
                span.status.message = spec["status_message"]
        _add_attributes(span.attributes, spec.get("attributes", {}))
    return request


def _add_attributes(attributes, values: dict) -> None:
    for key, value in values.items():
        item = attributes.add()
        item.key = key
        _set_any_value(item.value, value)


def _set_any_value(any_value, value) -> None:
    # bool first: bool is a subclass of int.
    if isinstance(value, bool):
        any_value.bool_value = value
    elif isinstance(value, int):
        any_value.int_value = value
    elif isinstance(value, float):
        any_value.double_value = value
    elif isinstance(value, list):
        for item in value:
            _set_any_value(any_value.array_value.values.add(), item)
    elif isinstance(value, dict):
        _add_attributes(any_value.kvlist_value.values, value)
    else:
        any_value.string_value = str(value)
