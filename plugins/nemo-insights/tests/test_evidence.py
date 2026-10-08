# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from insight_agent.insight import Insight as CompassInsight
from insight_agent.insight import SpanEvidence as CompassSpanEvidence
from insight_agent.insight import TraceEvidence as CompassTraceEvidence
from nemo_helix_plugin.nooa_model_client import ConfiguredModelClients
from nemo_insights_plugin.analyst import trace_intel
from nemo_insights_plugin.analyst.analyst_backend import LocalAnalystBackend, RemoteAnalystBackend
from nemo_insights_plugin.analyst.result import AnalystResult, InsightUpdate
from nemo_insights_plugin.analyst.trace_intel import load_existing_insights, to_change_set
from nemo_insights_plugin.entities import Insight, InsightStatus
from nemo_insights_plugin.evidence import TraceEvidence, merge_evidence
from nemo_insights_plugin.schema import CreateInsightRequest, UpdateInsightRequest
from trace_ingest.models import Span, SpanKind, Trace, TraceAggregate, TraceSnapshot


def stored(**fields):
    return Insight.model_validate(
        dict(workspace="test", title="Issue", description="Details", agent="target", **fields)
    )


@pytest.mark.parametrize("model", [Insight, CreateInsightRequest])
def test_legacy_evidence_groups_spans_and_keeps_links(model):
    payload = dict(
        workspace="test",
        title="Issue",
        description="Details",
        agent="target",
        trace_refs=["a", "b"],
        trace_links={"a": "https://provider/traces/a"},
        span_refs={"a": ["parent", "nested-child"]},
        span_links={"a": {"nested-child": "https://provider/spans/child"}},
    )
    insight = model.model_validate(payload)
    assert insight.model_dump(exclude_none=True)["evidence"] == [
        {
            "trace_id": "a",
            "url": "https://provider/traces/a",
            "spans": [
                {"span_id": "parent"},
                {"span_id": "nested-child", "url": "https://provider/spans/child"},
            ],
        },
        {"trace_id": "b", "spans": []},
    ]
    assert "trace_refs" not in insight.model_dump()
    assert payload["trace_refs"] == ["a", "b"]


@pytest.mark.parametrize("evidence", [[], [{"trace_id": "new"}]])
def test_explicit_evidence_wins_over_legacy_fields(evidence):
    insight = stored(evidence=evidence, trace_refs=["old"])
    assert [item.trace_id for item in insight.evidence] == [item["trace_id"] for item in evidence]


def test_new_record_round_trip_preserves_optional_links_and_date():
    date = datetime(2026, 10, 7, tzinfo=timezone.utc)
    insight = stored(
        evidence=[{"trace_id": "a", "spans": [{"span_id": "child"}]}, {"trace_id": "b"}], updated_date=date
    )
    reread = Insight.model_validate(insight.model_dump(mode="json"))
    assert reread.evidence == insight.evidence
    assert reread.updated_date == date
    assert UpdateInsightRequest(status=InsightStatus.RESOLVED).model_dump(exclude_unset=True) == {"status": "resolved"}


def test_explicit_legacy_span_parents():
    insight = stored(
        trace_refs=["a"],
        span_refs=[{"trace_id": "a", "span_id": "child"}],
        span_links={"child": "https://provider/child"},
    )
    assert insight.evidence[0].spans[0].span_id == "child"
    assert insight.evidence[0].spans[0].url == "https://provider/child"


def test_merge_retains_links_and_adds_spans_without_mutation():
    previous = [
        TraceEvidence.model_validate({"trace_id": "a", "url": "https://provider/a", "spans": [{"span_id": "parent"}]})
    ]
    added = [
        TraceEvidence.model_validate(
            {"trace_id": "a", "spans": [{"span_id": "child"}, {"span_id": "parent", "url": "https://provider/p"}]}
        )
    ]
    merged = merge_evidence(previous, added)
    assert merged[0].url == "https://provider/a"
    assert [span.span_id for span in merged[0].spans] == ["parent", "child"]
    assert merged[0].spans[0].url == "https://provider/p"
    assert len(previous[0].spans) == 1
    assert previous[0].spans[0].url is None


@pytest.mark.asyncio
async def test_existing_legacy_record_reaches_compass_with_identity_and_date():
    date = datetime(2026, 10, 7, tzinfo=timezone.utc)
    insight = stored(trace_refs=["a", "b"], updated_date=date, name="stable-name")
    insight._id = "stored-id"
    backend = MagicMock(
        list_insights=AsyncMock(return_value=MagicMock(data=[insight], pagination=MagicMock(total_pages=1)))
    )
    existing = await load_existing_insights(backend, workspace="test", agent="target")
    assert len(existing) == 1
    assert existing[0].id == insight.id
    assert existing[0].name == "Issue"
    assert existing[0].description == "Details"
    assert existing[0].updated_date == date
    assert "trace_refs" not in existing[0].model_dump()


def test_reconciliation_carries_new_spans_and_updated_date():
    old = CompassInsight(
        id="id",
        name="Issue",
        description="Details",
        evidence=[CompassTraceEvidence(trace_id="a"), CompassTraceEvidence(trace_id="b")],
    )
    date = datetime(2026, 10, 7, tzinfo=timezone.utc)
    new = old.model_copy(
        update={
            "evidence": [
                type(old.evidence[0]).model_validate({"trace_id": "a", "spans": [{"span_id": "child"}]}),
                old.evidence[1],
            ],
            "updated_date": date,
        }
    )
    result = to_change_set([new], [old], trace_count=2)
    assert result.updated_insights[0].evidence[0].spans[0].span_id == "child"
    assert result.updated_insights[0].updated_date == date


@pytest.mark.asyncio
async def test_nested_span_links_survive_compass_to_platform_adapter(monkeypatch):
    snapshot = TraceSnapshot(
        [
            Trace(
                id="a",
                aggregate=TraceAggregate(),
                source_url="https://provider/a",
                root_spans=[
                    Span(
                        id="root",
                        kind=SpanKind.AGENT,
                        children=[
                            Span(
                                id="child",
                                kind=SpanKind.TOOL,
                                source_url="https://provider/child",
                            )
                        ],
                    )
                ],
            ),
            Trace(id="b", root_spans=[], aggregate=TraceAggregate()),
        ]
    )
    insight = CompassInsight(
        name="Issue",
        description="Details",
        evidence=[
            CompassTraceEvidence(trace_id="a", spans=[CompassSpanEvidence(span_id="child")]),
            CompassTraceEvidence(trace_id="b"),
        ],
    )
    registry = MagicMock()
    registry.names = ["test-stream"]
    registry.analyze = AsyncMock(return_value=MagicMock(problems=["issue"]))
    monkeypatch.setattr(trace_intel, "registered_builtin_streams", MagicMock(return_value=registry))
    compiler = MagicMock(compile_insights=AsyncMock(return_value=[insight]))
    monkeypatch.setattr(trace_intel, "InsightCompilation", MagicMock(return_value=compiler))
    result = await trace_intel.analyze_snapshot(
        snapshot,
        existing=[],
        model_clients=ConfiguredModelClients(default=MagicMock(), fast=MagicMock()),
        ethos=None,
    )
    evidence = result.new_insights[0].evidence
    assert evidence[0].url == "https://provider/a"
    assert evidence[0].spans[0].url == "https://provider/child"
    assert evidence[1].url is None
    assert evidence[1].spans == []


@pytest.mark.asyncio
async def test_local_persistence_converts_legacy_evidence(tmp_path):
    backend = LocalAnalystBackend(client=MagicMock(), path=tmp_path / "insights.yaml")
    backend.store.write_records(
        [dict(id="id", workspace="test", title="Issue", description="Details", agent="target", trace_refs=["a", "b"])]
    )
    date = datetime(2026, 10, 7, tzinfo=timezone.utc)
    await backend.persist_result(
        workspace="test",
        agent="target",
        result=AnalystResult(
            summary="Updated",
            updated_insights=[InsightUpdate(id="id", evidence=[TraceEvidence(trace_id="c")], updated_date=date)],
        ),
    )
    record = backend.store.read_records()[0]
    assert [item["trace_id"] for item in record["evidence"]] == ["a", "b", "c"]
    assert record["title"] == "Issue"
    assert record["updated_date"] == date.isoformat()


@pytest.mark.asyncio
async def test_remote_persistence_merges_evidence_and_preserves_newer_date():
    date = datetime(2026, 10, 7, tzinfo=timezone.utc)
    backend = RemoteAnalystBackend(MagicMock())
    backend._get = AsyncMock(return_value=stored(trace_refs=["a", "b"], updated_date=date))
    backend.client.insights.insights.update = AsyncMock()
    await backend._add_evidence(
        workspace="test", agent="target", insight_id="id", evidence=[TraceEvidence(trace_id="c")]
    )
    assert backend.client.insights.insights.update.await_args is not None
    args = backend.client.insights.insights.update.await_args.kwargs
    assert args["updated_date"] == date
    assert [item.trace_id for item in args["evidence"]] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_direct_compilation_receives_utc_timestamp_and_existing_insights(monkeypatch):
    registry = MagicMock()
    registry.names = ["test-stream"]
    registry.analyze = AsyncMock(return_value=MagicMock(problems=["issue"]))
    monkeypatch.setattr(trace_intel, "registered_builtin_streams", MagicMock(return_value=registry))
    old = CompassInsight(
        id="id",
        name="Issue",
        description="Details",
        evidence=[CompassTraceEvidence(trace_id="a"), CompassTraceEvidence(trace_id="b")],
    )
    compiler = MagicMock(compile_insights=AsyncMock(return_value=[old]))
    monkeypatch.setattr(trace_intel, "InsightCompilation", MagicMock(return_value=compiler))
    snapshot = TraceSnapshot([])
    before = datetime.now(timezone.utc)
    result = await trace_intel.analyze_snapshot(
        snapshot,
        existing=[old],
        model_clients=ConfiguredModelClients(default=MagicMock(), fast=MagicMock()),
        ethos=None,
    )
    timestamp = compiler.compile_insights.await_args.kwargs["run_timestamp"]
    assert timestamp.tzinfo == timezone.utc
    assert before <= timestamp <= datetime.now(timezone.utc)
    assert compiler.compile_insights.await_args.args[1:] == (snapshot, [old])
    assert result.updated_insights == []
