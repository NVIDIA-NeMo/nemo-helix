# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP spans in one trial's trace: the trial's root, and a child span per model call the gateway served it."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from nemo_evals.gateway.tokens import TrialClaims
from nemo_evals.intake.mapping import EVALUATION_NAME_ATTRIBUTE, SESSION_ID_ATTRIBUTE, TEST_CASE_NAME_ATTRIBUTE
from nemo_helix_plugin.intake.span_attributes import GATEWAY_CALL_ATTRIBUTE
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, InstrumentationScope, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

SERVICE_NAME = "nemo-evals"

_SCOPE = InstrumentationScope(name="nemo_evals.gateway.spans")


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


def trial_trace_id(trial: TrialClaims) -> str:
    """The trial's trace, derived from its session so every attempt and every recorder writes the same one."""
    return hashlib.sha256(trial.session.encode()).hexdigest()[:32]


def trial_root_span_id(trial: TrialClaims) -> str:
    return hashlib.sha256(trial.session.encode()).hexdigest()[32:48]


def _nanos(at: datetime) -> int:
    return int(at.timestamp() * 1_000_000_000)


def _trial_attributes(trial: TrialClaims) -> dict[str, object]:
    attributes: dict[str, object] = {SESSION_ID_ATTRIBUTE: trial.session}
    if trial.evaluation:
        attributes[EVALUATION_NAME_ATTRIBUTE] = trial.evaluation
    if trial.task:
        attributes[TEST_CASE_NAME_ATTRIBUTE] = trial.task
    return attributes


def _resource_spans(span: Span, attributes: Mapping[str, object]) -> ResourceSpans:
    for key, value in attributes.items():
        if value is not None:
            span.attributes.add(key=key, value=_any_value(value))
    resource = Resource(attributes=[KeyValue(key="service.name", value=_any_value(SERVICE_NAME))])
    return ResourceSpans(resource=resource, scope_spans=[ScopeSpans(scope=_SCOPE, spans=[span])])


def export_request(batch: Iterable[ResourceSpans]) -> bytes:
    return ExportTraceServiceRequest(resource_spans=list(batch)).SerializeToString()


_TRACEPARENT = re.compile(r"([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}(-.*)?")
MAX_TRACESTATE_LENGTH = 512


@dataclass(frozen=True)
class CallerSpan:
    """The span a caller made its model call in, from the W3C ``traceparent`` it sent."""

    trace_id: str
    span_id: str
    trace_state: str = ""

    @classmethod
    def from_headers(cls, traceparent: str | None, tracestate: str | None = None) -> CallerSpan | None:
        """The caller's span, or ``None`` when ``traceparent`` is missing or isn't a valid W3C value."""
        match = _TRACEPARENT.fullmatch((traceparent or "").strip())
        if match is None:
            return None
        version, trace_id, span_id, rest = match.groups()
        if version == "ff" or (version == "00" and rest) or set(trace_id) == {"0"} or set(span_id) == {"0"}:
            return None
        state = (tracestate or "").strip()
        return cls(trace_id, span_id, state if len(state) <= MAX_TRACESTATE_LENGTH else "")


def call_span(
    trial: TrialClaims,
    *,
    name: str,
    started_at: datetime,
    ended_at: datetime,
    error: bool,
    attributes: Mapping[str, object],
    caller: CallerSpan | None = None,
) -> ResourceSpans:
    """One model call the gateway served the trial: a child of the caller's span, else of the trial's root."""
    span = Span(
        trace_id=bytes.fromhex(caller.trace_id if caller is not None else trial_trace_id(trial)),
        span_id=os.urandom(8),
        parent_span_id=bytes.fromhex(caller.span_id if caller is not None else trial_root_span_id(trial)),
        trace_state=caller.trace_state if caller is not None else "",
        name=name,
        kind=Span.SPAN_KIND_SERVER,
        start_time_unix_nano=_nanos(started_at),
        end_time_unix_nano=_nanos(ended_at),
        status=Status(code=Status.STATUS_CODE_ERROR if error else Status.STATUS_CODE_OK),
    )
    return _resource_spans(
        span,
        {**attributes, GATEWAY_CALL_ATTRIBUTE: True, **_trial_attributes(trial)},
    )


def trial_root_span(trial: TrialClaims, *, started_at: datetime, ended_at: datetime, failed: bool) -> ResourceSpans:
    """The trial's root span, from the start of its first attempt to the end of its last."""
    span = Span(
        trace_id=bytes.fromhex(trial_trace_id(trial)),
        span_id=bytes.fromhex(trial_root_span_id(trial)),
        name="trial",
        kind=Span.SPAN_KIND_INTERNAL,
        start_time_unix_nano=_nanos(started_at),
        end_time_unix_nano=_nanos(ended_at),
        status=Status(code=Status.STATUS_CODE_ERROR if failed else Status.STATUS_CODE_OK),
    )
    return _resource_spans(span, {**_trial_attributes(trial), "gen_ai.operation.name": "invoke_agent"})
