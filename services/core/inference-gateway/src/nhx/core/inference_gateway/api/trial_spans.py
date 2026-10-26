# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP spans in a leased trial: the trial's root, the gateway's own call spans, and adopted agent spans.

Every span lands in the lease's trace and session. Spans without a parent hang under the trial's root,
so the trial is one tree however many producers report into it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime

from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from nemo_helix_plugin.call_leases.types import CallLease
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, InstrumentationScope, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

SERVICE_NAME = "nemo-inference-gateway"
SESSION_ATTRIBUTE = "gen_ai.conversation.id"
EVALUATION_ATTRIBUTE = "nemo.evaluation.name"
TEST_CASE_ATTRIBUTE = "nemo.test_case.name"
# Marks the gateway's own call spans, whose usage is what the provider reported for the call.
GATEWAY_CALL_ATTRIBUTE = "nemo.inference_gateway.call"

_SCOPE = InstrumentationScope(name="nhx.core.inference_gateway")

AttributeValue = str | bool | int | float | list[str] | Mapping[str, object] | None


def _any_value(value: object) -> AnyValue:
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    if isinstance(value, float):
        return AnyValue(double_value=value)
    if isinstance(value, str):
        return AnyValue(string_value=value)
    if isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
        return AnyValue(array_value=ArrayValue(values=[AnyValue(string_value=item) for item in value]))
    return AnyValue(string_value=json.dumps(value, default=str))


def _set_attribute(attributes: RepeatedCompositeFieldContainer[KeyValue], key: str, value: object) -> None:
    for attribute in attributes:
        if attribute.key == key:
            attribute.value.CopyFrom(_any_value(value))
            return
    attributes.add(key=key, value=_any_value(value))


def _nanos(at: datetime) -> int:
    return int(at.timestamp() * 1_000_000_000)


def _lease_attributes(lease: CallLease) -> dict[str, object]:
    attributes: dict[str, object] = {SESSION_ATTRIBUTE: lease.session_id}
    if lease.evaluation_name:
        attributes[EVALUATION_ATTRIBUTE] = lease.evaluation_name
    if lease.test_case_name:
        attributes[TEST_CASE_ATTRIBUTE] = lease.test_case_name
    return attributes


def resource_spans(spans: Iterable[Span], *, service_name: str = SERVICE_NAME) -> ResourceSpans:
    resource = Resource(attributes=[KeyValue(key="service.name", value=_any_value(service_name))])
    return ResourceSpans(resource=resource, scope_spans=[ScopeSpans(scope=_SCOPE, spans=list(spans))])


def export_request(batch: Iterable[ResourceSpans]) -> bytes:
    return ExportTraceServiceRequest(resource_spans=list(batch)).SerializeToString()


def call_span(
    lease: CallLease,
    *,
    name: str,
    started_at: datetime,
    ended_at: datetime,
    error: bool,
    attributes: Mapping[str, object],
) -> ResourceSpans:
    """One model call the gateway served, as a child of the trial's root."""
    span = Span(
        trace_id=bytes.fromhex(lease.trace_id),
        span_id=os.urandom(8),
        parent_span_id=bytes.fromhex(lease.root_span_id),
        name=name,
        kind=Span.SPAN_KIND_SERVER,
        start_time_unix_nano=_nanos(started_at),
        end_time_unix_nano=_nanos(ended_at),
        status=Status(code=Status.STATUS_CODE_ERROR if error else Status.STATUS_CODE_OK),
    )
    for key, value in {**attributes, GATEWAY_CALL_ATTRIBUTE: True, **_lease_attributes(lease)}.items():
        if value is not None:
            _set_attribute(span.attributes, key, value)
    return resource_spans([span])


def trial_root_span(leases: Iterable[CallLease]) -> ResourceSpans | None:
    """The trial's root, spanning its sandboxes' leases; ``None`` until one of them has ended.

    Its start is the earliest lease's, so every rewrite keeps the same start and replaces the last one.
    """
    leases = list(leases)
    ended = [lease.ended_at for lease in leases if lease.ended_at is not None]
    if not leases or not ended:
        return None
    first = min(leases, key=lambda lease: lease.started_at)
    span = Span(
        trace_id=bytes.fromhex(first.trace_id),
        span_id=bytes.fromhex(first.root_span_id),
        name="trial",
        kind=Span.SPAN_KIND_INTERNAL,
        start_time_unix_nano=_nanos(first.started_at),
        end_time_unix_nano=_nanos(max(ended)),
        status=Status(code=Status.STATUS_CODE_OK),
    )
    for key, value in {**_lease_attributes(first), "gen_ai.operation.name": "invoke_agent"}.items():
        _set_attribute(span.attributes, key, value)
    return resource_spans([span])


def adopt_agent_spans(request: ExportTraceServiceRequest, lease: CallLease) -> list[ResourceSpans]:
    """``request``'s spans moved into the lease's trial: its trace and session, roots under the trial's root.

    The agent chooses nothing about where its spans land: ids it set for the trace or session are replaced,
    and a span claiming the trial root's id is dropped.
    """
    trace_id = bytes.fromhex(lease.trace_id)
    root_span_id = bytes.fromhex(lease.root_span_id)
    attributes = _lease_attributes(lease)
    for batch in request.resource_spans:
        for scope in batch.scope_spans:
            kept = [span for span in scope.spans if span.span_id != root_span_id]
            del scope.spans[:]
            scope.spans.extend(kept)
            for span in scope.spans:
                span.trace_id = trace_id
                if not span.parent_span_id:
                    span.parent_span_id = root_span_id
                for key, value in attributes.items():
                    _set_attribute(span.attributes, key, value)
    return list(request.resource_spans)
