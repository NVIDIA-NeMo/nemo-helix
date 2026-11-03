# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for splitting one trial's OTLP export into requests that fit a byte budget."""

from __future__ import annotations

import pytest
from nemo_evals.intake.mapping import OtlpSpanTooLargeError, split_otlp_request
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, InstrumentationScope, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

_TRACE_ID = bytes.fromhex("0123456789abcdef0123456789abcdef")


def _span(index: int, *, parent: int | None = None, payload_bytes: int = 0) -> Span:
    span = Span(
        trace_id=_TRACE_ID,
        span_id=index.to_bytes(8, "big"),
        name=f"span-{index}",
        start_time_unix_nano=1000 + index,
    )
    if parent is not None:
        span.parent_span_id = parent.to_bytes(8, "big")
    if payload_bytes:
        span.attributes.append(KeyValue(key="payload", value=AnyValue(string_value="x" * payload_bytes)))
    return span


def _resource(name: str, scopes: list[tuple[str, list[Span]]]) -> ResourceSpans:
    return ResourceSpans(
        resource=Resource(attributes=[KeyValue(key="service.name", value=AnyValue(string_value=name))]),
        schema_url=f"https://schema/{name}",
        scope_spans=[
            ScopeSpans(
                scope=InstrumentationScope(name=scope, version="1.0"), schema_url=f"https://scope/{scope}", spans=spans
            )
            for scope, spans in scopes
        ],
    )


def _parse(payload: bytes) -> ExportTraceServiceRequest:
    request = ExportTraceServiceRequest()
    request.ParseFromString(payload)
    return request


def _flatten(requests: list[bytes]) -> list[tuple[ResourceSpans, ScopeSpans, Span]]:
    """Every span in send order, with the resource and scope envelope it arrived under."""
    flat = []
    for payload in requests:
        for resource_spans in _parse(payload).resource_spans:
            for scope_spans in resource_spans.scope_spans:
                flat.extend((resource_spans, scope_spans, span) for span in scope_spans.spans)
    return flat


def _large_trace() -> bytes:
    """A root and three ~2 MiB children: each fits under 4 MiB, together they do not."""
    spans = [_span(1), *(_span(index, parent=1, payload_bytes=2 * 1024 * 1024) for index in (2, 3, 4))]
    return ExportTraceServiceRequest(resource_spans=[_resource("agent", [("scope", spans)])]).SerializeToString()


def test_a_request_that_fits_is_returned_byte_for_byte() -> None:
    payload = _large_trace()

    assert split_otlp_request(payload, max_request_bytes=len(payload)) == [payload]


def test_one_byte_over_the_budget_splits() -> None:
    payload = _large_trace()

    assert len(split_otlp_request(payload, max_request_bytes=len(payload) - 1)) > 1


def test_a_large_trace_splits_under_the_default_budget_preserving_every_span() -> None:
    payload = _large_trace()
    budget = 4 * 1024 * 1024
    assert len(payload) > budget

    requests = split_otlp_request(payload, max_request_bytes=budget)

    assert len(requests) > 1
    assert all(len(request) <= budget for request in requests)
    original = _parse(payload).resource_spans[0]
    flat = _flatten(requests)
    assert [span for _, _, span in flat] == list(original.scope_spans[0].spans)
    for resource_spans, scope_spans, _ in flat:
        assert resource_spans.resource == original.resource
        assert resource_spans.schema_url == original.schema_url
        assert scope_spans.scope == original.scope_spans[0].scope
        assert scope_spans.schema_url == original.scope_spans[0].schema_url


def test_spans_keep_their_order_and_envelopes_across_resources_and_scopes() -> None:
    resources = [
        _resource("a", [("a1", [_span(1), _span(2, parent=1)]), ("a2", [_span(3, parent=1)])]),
        _resource("b", [("b1", [_span(4, parent=1), _span(5, parent=4)])]),
    ]
    payload = ExportTraceServiceRequest(resource_spans=resources).SerializeToString()
    # Small enough to force a split at nearly every span.
    budget = (
        max(len(ExportTraceServiceRequest(resource_spans=[resource]).SerializeToString()) for resource in resources)
        // 2
    )

    requests = split_otlp_request(payload, max_request_bytes=budget)

    assert len(requests) > 2
    assert all(len(request) <= budget for request in requests)
    expected = [
        (resource.resource, resource.schema_url, scope.scope, scope.schema_url, span)
        for resource in resources
        for scope in resource.scope_spans
        for span in scope.spans
    ]
    actual = [
        (resource.resource, resource.schema_url, scope.scope, scope.schema_url, span)
        for resource, scope, span in _flatten(requests)
    ]
    assert actual == expected


def test_each_emitted_request_is_exactly_as_large_as_accounted() -> None:
    # Packing reaches the budget exactly when the size arithmetic matches the real encoding.
    spans = [_span(index, payload_bytes=100 + index) for index in range(1, 40)]
    payload = ExportTraceServiceRequest(resource_spans=[_resource("r", [("s", spans)])]).SerializeToString()
    exact = ExportTraceServiceRequest(resource_spans=[_resource("r", [("s", spans[:5])])]).SerializeToString()

    requests = split_otlp_request(payload, max_request_bytes=len(exact))

    assert requests[0] == exact


def test_size_accounting_is_exact_across_scope_and_resource_boundaries() -> None:
    # The first request opens a second scope and then a second resource; it fills the budget
    # exactly only if each of those transitions is accounted to the byte.
    resources = [
        _resource("a", [("a1", [_span(1, payload_bytes=20_000), _span(2, payload_bytes=129)]), ("a2", [_span(3)])]),
        _resource("b", [("b1", [_span(4, payload_bytes=200), _span(5, payload_bytes=5_000)])]),
    ]
    payload = ExportTraceServiceRequest(resource_spans=resources).SerializeToString()
    first = ExportTraceServiceRequest(
        resource_spans=[
            resources[0],
            _resource("b", [("b1", [resources[1].scope_spans[0].spans[0]])]),
        ]
    ).SerializeToString()

    requests = split_otlp_request(payload, max_request_bytes=len(first))

    assert requests[0] == first
    assert [span.name for _, _, span in _flatten(requests)] == [f"span-{index}" for index in range(1, 6)]


def test_a_span_too_large_for_any_request_fails_without_returning_a_batch() -> None:
    spans = [_span(1), _span(2, parent=1, payload_bytes=1000), _span(3, parent=1, payload_bytes=10_000)]
    payload = ExportTraceServiceRequest(resource_spans=[_resource("r", [("s", spans)])]).SerializeToString()

    with pytest.raises(OtlpSpanTooLargeError, match="0000000000000003"):
        split_otlp_request(payload, max_request_bytes=5_000)


def test_empty_resource_and_scope_entries_carry_nothing_to_send() -> None:
    resources = [
        _resource("empty", [("none", [])]),
        _resource("r", [("s", [_span(1, payload_bytes=100), _span(2, payload_bytes=100)])]),
    ]
    payload = ExportTraceServiceRequest(resource_spans=resources).SerializeToString()

    requests = split_otlp_request(payload, max_request_bytes=len(payload) - 1)

    assert [span.name for _, _, span in _flatten(requests)] == ["span-1", "span-2"]


@pytest.mark.parametrize("budget", [0, -1])
def test_a_non_positive_budget_is_rejected(budget: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        split_otlp_request(b"", max_request_bytes=budget)
