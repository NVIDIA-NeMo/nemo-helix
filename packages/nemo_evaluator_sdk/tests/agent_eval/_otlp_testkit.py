# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Fabric OTLP capture tests, used by both runtimes' suites."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from google.protobuf import json_format
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_receiver import PROTOBUF_MEDIA_TYPE
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer import otlp_trace_path
from nemo_evaluator_sdk.values.otlp import parse_resource_spans, resource_spans_from_text
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span


def export(*names: str, padding: int = 0) -> bytes:
    """A serialized export carrying one span per name, optionally padded to a realistic size."""
    attributes = [KeyValue(key="pad", value=AnyValue(string_value="x" * padding))] if padding else []
    spans = [
        Span(
            trace_id=bytes(range(16)),
            span_id=bytes(range(8)),
            name=name,
            start_time_unix_nano=1,
            attributes=attributes,
        )
        for name in names
    ]
    return ExportTraceServiceRequest(
        resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=spans)])]
    ).SerializeToString()


def post(url: str, body: bytes, *, media_type: str = PROTOBUF_MEDIA_TYPE) -> int:
    """POST an export and return the status, including for the error responses under test."""
    request = urllib.request.Request(url, data=body, headers={"Content-Type": media_type}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def span_names(path: Path) -> list[str]:
    """The span names in a folded trace, read through the same reader a metric uses."""
    spans = parse_resource_spans(resource_spans_from_text(path.read_text(encoding="utf-8")))
    return [span.name for rs in spans for ss in rs.scope_spans for span in ss.spans]


def write_answer_trace(evidence_dir: Path, answer: str) -> Path:
    """An OTLP trace at the path both Fabric runtimes capture to, with one agent span carrying ``answer``."""
    span = Span(
        name="agent",
        trace_id=bytes(range(16)),
        span_id=bytes(range(8)),
        attributes=[
            KeyValue(key="openinference.span.kind", value=AnyValue(string_value="AGENT")),
            KeyValue(key="output.value", value=AnyValue(string_value=answer)),
        ],
    )
    request = ExportTraceServiceRequest(resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=[span])])])
    path = otlp_trace_path(evidence_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_format.MessageToJson(request), encoding="utf-8")
    return path
