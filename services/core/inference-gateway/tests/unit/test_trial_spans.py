# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A trial's spans share its lease's trace and session, and hang under one root."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from nemo_helix_plugin.call_leases.types import CallLease
from nhx.core.inference_gateway.api.lease_cache import LEASE_LOOKBACK, CallLeaseCache
from nhx.core.inference_gateway.api.trial_spans import adopt_agent_spans, trial_root_span
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

STARTED = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def _lease(name: str = "lease-agent", **overrides: Any) -> CallLease:
    fields: dict[str, Any] = {
        "workspace": "evals",
        "client_address": "10.244.0.7",
        "session_id": "run-1-task-0:hello__abc",
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "root_span_id": "b7ad6b7169203331",
        "role": "agent",
        "started_at": STARTED,
    }
    return CallLease(name=name, **{**fields, **overrides})


def _attributes(span: Span) -> dict[str, str]:
    return {attribute.key: attribute.value.string_value for attribute in span.attributes}


def _spans(batches: list[ResourceSpans]) -> list[Span]:
    return [span for batch in batches for scope in batch.scope_spans for span in scope.spans]


# --- the trial's root -------------------------------------------------------------------------


def test_the_root_waits_for_a_lease_to_end() -> None:
    assert trial_root_span([_lease()]) is None


def test_the_root_spans_every_sandbox_and_keeps_its_first_start() -> None:
    """Re-recorded at each lease's end with the same start, so Intake replaces it instead of adding one."""
    agent = _lease(ended_at=STARTED + timedelta(minutes=5))
    verifier = _lease("lease-verifier", role="verifier", started_at=STARTED + timedelta(minutes=6))
    after_agent = trial_root_span([agent, verifier])
    after_verifier = trial_root_span([agent, verifier.model_copy(update={"ended_at": STARTED + timedelta(minutes=8)})])

    assert after_agent is not None and after_verifier is not None
    [first], [second] = _spans([after_agent]), _spans([after_verifier])
    assert first.span_id == second.span_id == bytes.fromhex(agent.root_span_id)
    assert first.trace_id.hex() == agent.trace_id and not first.parent_span_id
    assert first.start_time_unix_nano == second.start_time_unix_nano
    assert second.end_time_unix_nano - first.end_time_unix_nano == 3 * 60 * 10**9
    assert _attributes(first)["gen_ai.conversation.id"] == agent.session_id


# --- adopting the agent's spans ---------------------------------------------------------------


def _export(*spans: Span, session: str = "agent-chosen-session") -> ExportTraceServiceRequest:
    for span in spans:
        span.attributes.append(KeyValue(key="gen_ai.conversation.id", value=AnyValue(string_value=session)))
    return ExportTraceServiceRequest(resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=list(spans))])])


def test_agent_spans_move_into_the_trial_and_its_roots_hang_under_the_trials() -> None:
    lease = _lease(evaluation_name="run-1")
    root = Span(trace_id=b"\xaa" * 16, span_id=b"\x01" * 8, name="claude_code.interaction")
    child = Span(trace_id=b"\xaa" * 16, span_id=b"\x02" * 8, parent_span_id=b"\x01" * 8, name="claude_code.llm_request")

    adopted = _spans(adopt_agent_spans(_export(root, child), lease))

    assert [span.name for span in adopted] == ["claude_code.interaction", "claude_code.llm_request"]
    assert {span.trace_id.hex() for span in adopted} == {lease.trace_id}
    assert adopted[0].parent_span_id.hex() == lease.root_span_id
    assert adopted[1].parent_span_id == b"\x01" * 8
    for span in adopted:
        assert _attributes(span)["gen_ai.conversation.id"] == lease.session_id
        assert _attributes(span)["nemo.evaluation.name"] == "run-1"


def test_an_agent_span_claiming_the_trials_root_is_dropped() -> None:
    lease = _lease()
    impostor = Span(trace_id=b"\xaa" * 16, span_id=bytes.fromhex(lease.root_span_id), name="fake-root")
    honest = Span(trace_id=b"\xaa" * 16, span_id=b"\x03" * 8, name="tool")

    adopted = _spans(adopt_agent_spans(_export(impostor, honest), lease))

    assert [span.name for span in adopted] == ["tool"]


# --- the lease cache --------------------------------------------------------------------------


def test_the_cache_finds_the_lease_that_held_an_address_at_a_moment() -> None:
    cache = CallLeaseCache()
    first = _lease("lease-old", started_at=STARTED, ended_at=STARTED + timedelta(minutes=10))
    second = _lease("lease-new", started_at=STARTED + timedelta(minutes=20))
    cache.replace([first, second])

    assert cache.lookup("10.244.0.7", STARTED + timedelta(minutes=5)) == first
    assert cache.lookup("10.244.0.7", STARTED + timedelta(minutes=15)) is None
    assert cache.lookup("10.244.0.7", STARTED + timedelta(minutes=25)) == second
    assert cache.lookup("10.244.0.8", STARTED + timedelta(minutes=5)) is None
    assert cache.lookup(None, STARTED) is None


def test_a_refresh_does_not_undo_an_end_this_gateway_already_saw() -> None:
    """The routes update the cache before the entity store's next listing shows the change."""
    cache = CallLeaseCache()
    ended = _lease(started_at=datetime.now(UTC), ended_at=datetime.now(UTC) + timedelta(seconds=1))
    cache.put(ended)

    cache.replace([ended.model_copy(update={"ended_at": None})])

    assert cache.lookup("10.244.0.7", datetime.now(UTC) + timedelta(minutes=5)) is None


def test_a_refresh_forgets_leases_older_than_any_sandbox() -> None:
    cache = CallLeaseCache()
    stale = _lease(started_at=datetime.now(UTC) - LEASE_LOOKBACK - timedelta(hours=1))

    cache.replace([stale])

    assert cache.lookup("10.244.0.7", datetime.now(UTC)) is None


@pytest.mark.parametrize("address", ["10.244.0.7", "fd00::7"])
def test_the_cache_keys_by_the_leased_address(address: str) -> None:
    cache = CallLeaseCache()
    lease = _lease(client_address=address, started_at=datetime.now(UTC) - timedelta(minutes=1))
    cache.put(lease)

    assert cache.lookup(address, datetime.now(UTC)) == lease
