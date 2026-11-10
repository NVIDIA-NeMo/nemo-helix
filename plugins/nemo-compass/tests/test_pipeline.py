# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("nemo-compass requires Python 3.12 or 3.13", allow_module_level=True)

from compass.insight import Insight
from compass.insight import TraceEvidence as CompassTraceEvidence
from compass.traces import TraceSnapshot
from nemo_compass_plugin.analysis import pipeline
from nemo_compass_plugin.analysis.backend import RemoteCompassBackend
from nemo_compass_plugin.analysis.pipeline import (
    BELOW_THRESHOLD_SUMMARY,
    BorrowedModelClient,
    load_existing_insights,
    to_change_set,
)
from nemo_compass_plugin.analysis.result import CompassResult
from nemo_compass_plugin.entities import Insight as StoredInsight
from nemo_compass_plugin.entities import InsightStatus
from nemo_compass_plugin.evidence import TraceEvidence
from nemo_helix_plugin.nooa_model_client import ConfiguredModelClients
from nooa.unifiedllm import UnifiedLLM


def _insight(id="stored-id", **updates):
    refs = updates.pop("trace_refs", ["a", "b"])
    return Insight.model_validate(
        {
            "id": id,
            "name": "Original title",
            "description": "Original description",
            "evidence": [{"trace_id": ref} for ref in refs],
            **updates,
        }
    )


def test_reconciliation_uses_ids_and_preserves_platform_owned_fields():
    original = _insight()
    changed = _insight(name="Renamed by model", description="Rewritten", trace_refs=["a", "b", "c"])
    new = _insight(None, name="New issue", severity="high", confidence="med")
    result = to_change_set([changed, new], [original], trace_count=3)
    assert result.updated_insights[0].id == "stored-id"
    assert result.updated_insights[0].evidence == [TraceEvidence(trace_id="c")]
    assert result.new_insights[0].title == "New issue"
    assert result.new_insights[0].severity == "high"
    assert result.new_insights[0].confidence == "med"


@pytest.mark.parametrize("refs", [["b", "a"], ["a", "a"], ["a", "b", "a"]])
def test_reordered_removed_or_duplicate_refs_are_not_new_evidence(refs):
    result = to_change_set([_insight(trace_refs=refs)], [_insight()], trace_count=3)
    assert not result.updated_insights
    assert "0 existing insights with new evidence" in result.summary


def test_new_evidence_is_unique_and_preserves_generated_order():
    result = to_change_set([_insight(trace_refs=["d", "a", "c", "d"])], [_insight()], trace_count=3)
    assert [entry.trace_id for entry in result.updated_insights[0].evidence] == ["d", "c"]


@pytest.mark.parametrize("returned", [[_insight("unknown")], [_insight(), _insight()]])
def test_unknown_and_duplicate_ids_fail_before_persistence(returned):
    with pytest.raises(ValueError, match="unknown or duplicate"):
        to_change_set(returned, [_insight()], trace_count=3)


def test_omitted_insights_are_not_deleted():
    result = to_change_set([], [_insight()], trace_count=3)
    assert not result.updated_insights
    assert not result.new_insights


@pytest.mark.asyncio
async def test_existing_insights_are_paginated_and_invalid_records_are_skipped(caplog):
    def row(id, **updates):
        return MagicMock(
            id=id,
            title="Issue",
            description="Description",
            evidence=[TraceEvidence(trace_id="a"), TraceEvidence(trace_id="b")],
            updated_date=None,
            severity=None,
            confidence=None,
            status=InsightStatus.OPEN,
            **updates,
        )

    invalid = row("invalid")
    invalid.evidence = []
    resolved = row("resolved")
    resolved.status = InsightStatus.RESOLVED
    backend = MagicMock()
    backend.list_insights = AsyncMock(
        side_effect=[
            MagicMock(data=[row("first"), invalid, resolved], pagination=MagicMock(total_pages=2)),
            MagicMock(data=[row("second")], pagination=MagicMock(total_pages=2)),
        ]
    )
    insights = await load_existing_insights(backend, workspace="test", agent="target")
    assert [item.id for item in insights] == ["first", "second"]
    assert backend.list_insights.await_args is not None
    assert backend.list_insights.await_args.kwargs["page"] == 2
    assert "invalid" in caplog.text


@pytest.mark.asyncio
async def test_streams_cannot_close_the_platform_model_client():
    client = MagicMock(spec=UnifiedLLM)
    client.model = "model"
    client.config = {}
    client.acall = AsyncMock(return_value="response")
    client.aclose = AsyncMock()
    async with BorrowedModelClient(client) as borrowed:
        assert await borrowed.acall([]) == "response"
    client.aclose.assert_not_awaited()
    client.acall.assert_awaited_once()


@pytest.mark.asyncio
async def test_persistence_rechecks_insight_agent():
    backend = object.__new__(RemoteCompassBackend)
    backend._get = AsyncMock(return_value=MagicMock(workspace="test", agent="other"))
    with pytest.raises(ValueError, match="no longer belongs"):
        await backend._add_evidence(
            workspace="test", agent="target", insight_id="id", evidence=[TraceEvidence(trace_id="a")]
        )


@pytest.mark.asyncio
async def test_report_lists_notes_after_insight_lines():
    backend = object.__new__(RemoteCompassBackend)
    backend.mirror = None
    result = CompassResult(summary="Analyzed 2 traces.", notes=["- skipped: ethos-divergence (No ethos document)"])
    report = await backend.persist_result(workspace="test", agent="target", result=result)
    assert report == (
        "Analyzed 2 traces.\n\n- no insights created or updated\n- skipped: ethos-divergence (No ethos document)"
    )


@pytest.mark.asyncio
async def test_existing_legacy_record_reaches_compass_with_identity_and_date():
    date = datetime(2026, 10, 7, tzinfo=timezone.utc)
    insight = StoredInsight.model_validate(
        dict(
            workspace="test",
            title="Issue",
            description="Details",
            agent="target",
            trace_refs=["a", "b"],
            updated_date=date,
            name="stable-name",
            severity="high",
            confidence="med",
        )
    )
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
    assert existing[0].severity == "high"
    assert existing[0].confidence == "med"
    assert "trace_refs" not in existing[0].model_dump()


def test_reconciliation_carries_new_spans_and_updated_date():
    old = Insight(
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
async def test_shared_result_keeps_full_details_and_projects_only_entity_updates(monkeypatch):
    from compass.analysis import AnalysisOptions

    snapshot = TraceSnapshot([])
    existing = [_insight()]
    generated = _insight(None, severity="high")
    serialized = {
        "insights": [generated.model_dump(mode="json")],
        "compass_version": "test-version",
        "effective_options": {"compass_signals": {"tool_issues": False}},
        "validation_rejections": [{"reason": "expected behavior"}],
        "future_field": {"preserve": True},
    }
    full_result = MagicMock(
        insights=[generated],
        notes=["- skipped: tool-issues (disabled)"],
        below_threshold_count=1,
        model_dump=MagicMock(return_value=serialized),
    )
    analyze = AsyncMock(return_value=full_result)
    monkeypatch.setattr(pipeline, "analyze", analyze)
    clients = ConfiguredModelClients(
        default=MagicMock(model="default", config={}), fast=MagicMock(model="fast", config={})
    )
    options = {"compass_signals": {"tool_issues": False}}
    result = await pipeline.analyze_snapshot(
        snapshot, existing=existing, model_clients=clients, ethos="Expected behavior", analysis_options=options
    )
    assert analyze.await_args is not None
    kwargs = analyze.await_args.kwargs
    assert analyze.await_args.args == (snapshot,)
    assert kwargs["existing_insights"] is existing
    assert kwargs["options"] == AnalysisOptions.model_validate(options)
    assert kwargs["llm_factory"]().client is clients.default
    assert kwargs["fast_llm_factory"]().client is clients.fast
    assert result.new_insights[0].title == generated.name
    assert result.analysis_result == serialized
    assert result.notes == full_result.notes
    assert result.summary.endswith(BELOW_THRESHOLD_SUMMARY)


@pytest.mark.asyncio
async def test_latest_compass_signals_skip_empty_snapshot_without_model_calls():
    client = MagicMock()
    client.acall = AsyncMock()
    result = await pipeline.analyze_snapshot(
        TraceSnapshot([]),
        existing=[],
        model_clients=ConfiguredModelClients(default=client, fast=client),
        ethos=None,
    )
    assert result.new_insights == []
    assert result.updated_insights == []
    assert result.notes
    assert any("grader-failure-patterns" in note for note in result.notes)
    client.acall.assert_not_awaited()
