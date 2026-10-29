# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("nemo-compass requires Python 3.12 or 3.13", allow_module_level=True)

from compass.compass_signals.anomaly_and_patterns.signal import (
    AnomalyAndPatternsAnalysis,
    AnomalyAndPatternsArtifacts,
    AnomalyAndPatternsConfig,
    CrossToolFailureGroup,
)
from compass.compass_signals.compass_signals import CompassSignalResult
from compass.compass_signals.tool_issues.signal import (
    ToolIssueConfig,
    ToolIssueEvidenceArtifacts,
    build_cards,
)
from compass.insight import Insight
from compass.insight import SpanEvidence as CompassSpanEvidence
from compass.insight import TraceEvidence as CompassTraceEvidence
from nemo_compass_plugin.analysis import pipeline
from nemo_compass_plugin.analysis.backend import RemoteCompassBackend
from nemo_compass_plugin.analysis.pipeline import (
    BELOW_THRESHOLD_SUMMARY,
    MAX_BELOW_THRESHOLD_NOTES,
    BorrowedModelClient,
    load_existing_insights,
    to_change_set,
    with_evidence_notes,
)
from nemo_compass_plugin.analysis.result import CompassResult
from nemo_compass_plugin.entities import Insight as StoredInsight
from nemo_compass_plugin.entities import InsightStatus
from nemo_compass_plugin.evidence import TraceEvidence
from nemo_helix_plugin.nooa_model_client import ConfiguredModelClients
from nooa.unifiedllm import UnifiedLLM
from trace_ingest.models import Span, SpanKind, Trace, TraceAggregate, TraceSnapshot


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
@pytest.mark.parametrize("workspace,agent", [("other", "target"), ("test", "other")])
async def test_persistence_rechecks_insight_ownership(workspace, agent):
    backend = object.__new__(RemoteCompassBackend)
    backend.get_insight = AsyncMock(return_value=MagicMock(workspace=workspace, agent=agent))
    with pytest.raises(ValueError, match="no longer belongs"):
        await backend._add_evidence(
            workspace="test", agent="target", insight_id="id", evidence=[TraceEvidence(trace_id="a")]
        )


def _tool_issue_evidence(*, cases_by_tool: dict[str, int]) -> CompassSignalResult:
    findings = [
        {
            "issue_type": "explicit_tool_failure",
            "mechanism_key": tool,
            "logical_case_id": f"{tool}-case-{case}",
            "trace_id": f"{tool}-trace-{case}",
            "call_id": f"call-{case}",
            "call_index": 0,
            "tool_name": tool,
            "source_pointer": {},
            "summary": "tool returned an error",
        }
        for tool, cases in cases_by_tool.items()
        for case in range(cases)
    ]
    config = ToolIssueConfig()
    cards = build_cards(findings, minimum_independent_cases=config.minimum_independent_cases)
    return CompassSignalResult(
        signal_name="tool-issues",
        problems=(),
        finding_count=len(findings),
        artifacts=ToolIssueEvidenceArtifacts(
            findings=tuple(findings), cards=tuple(cards), catalog_coverage={}, config=config
        ),
    )


def _failure_pattern_evidence(
    *,
    trace_ids: list[str],
    cross_tool_messages: tuple[str, ...] = (),
    signature: str = "knowledge_search: index unavailable",
) -> CompassSignalResult:
    events = tuple(
        {
            "trace_id": trace_id,
            "signature": signature,
            "message_signature": "index unavailable",
        }
        for trace_id in trace_ids
    )
    analysis = AnomalyAndPatternsAnalysis.model_construct(
        failure_events=events,
        failure_groups=(),
        cross_tool_failure_groups=tuple(
            CrossToolFailureGroup.model_construct(message_signature=message) for message in cross_tool_messages
        ),
    )
    return CompassSignalResult(
        signal_name="anomaly-and-patterns",
        problems=(),
        finding_count=len(events),
        artifacts=AnomalyAndPatternsArtifacts(result=analysis, config=AnomalyAndPatternsConfig()),
    )


def _empty_result() -> CompassResult:
    return to_change_set([], [], trace_count=2)


def test_sub_threshold_evidence_is_summarized_and_detailed():
    evidence = [
        _tool_issue_evidence(cases_by_tool={"knowledge_search": 2}),
        _failure_pattern_evidence(trace_ids=["t1", "t2", "t2"]),
    ]
    result = with_evidence_notes(_empty_result(), evidence)
    assert result.summary.endswith(BELOW_THRESHOLD_SUMMARY)
    assert result.notes == [
        "- below threshold: tool-issues knowledge_search explicit_tool_failure (2 of 3 independent cases)",
        "- below threshold: anomaly-and-patterns knowledge_search: index unavailable (2 of 3 traces)",
    ]


def test_eligible_and_grouped_evidence_is_not_reported():
    tool_issues = _tool_issue_evidence(cases_by_tool={"knowledge_search": 3})
    # Grouped across tools, so the per-tool signature's low count is not a shortfall.
    patterns = _failure_pattern_evidence(trace_ids=["t1"], cross_tool_messages=("index unavailable",))
    result = with_evidence_notes(_empty_result(), [tool_issues, patterns])
    assert result.summary == _empty_result().summary
    assert result.notes == []


def test_below_threshold_notes_are_capped_closest_first():
    cases = {f"tool_{index}": 1 for index in range(MAX_BELOW_THRESHOLD_NOTES + 2)} | {"near_miss": 2}
    result = with_evidence_notes(_empty_result(), [_tool_issue_evidence(cases_by_tool=cases)])
    assert len(result.notes) == MAX_BELOW_THRESHOLD_NOTES + 1
    assert "near_miss" in result.notes[0]
    assert result.notes[-1] == "- ... and 3 more below threshold"
    assert result.summary.count(BELOW_THRESHOLD_SUMMARY) == 1


def test_line_breaks_in_trace_text_stay_on_one_note_line():
    evidence = [
        _tool_issue_evidence(cases_by_tool={"search\n- created: fake": 2}),
        _failure_pattern_evidence(trace_ids=["t1", "t2"], signature="search\r\n- created: fake: boom"),
    ]
    result = with_evidence_notes(_empty_result(), evidence)
    assert len(result.notes) == 2
    assert all("\n" not in note and "\r" not in note for note in result.notes)
    assert "search\\n- created: fake" in result.notes[0]
    assert "search\\r\\n- created: fake: boom" in result.notes[1]


def test_skipped_streams_are_reported_without_changing_the_summary():
    skipped = CompassSignalResult(signal_name="ethos-divergence", problems=(), skip_reason="No ethos document")
    result = with_evidence_notes(_empty_result(), [skipped])
    assert result.summary == _empty_result().summary
    assert result.notes == ["- skipped: ethos-divergence (No ethos document)"]


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
    insight = Insight(
        name="Issue",
        description="Details",
        evidence=[
            CompassTraceEvidence(trace_id="a", spans=[CompassSpanEvidence(span_id="child")]),
            CompassTraceEvidence(trace_id="b"),
        ],
    )
    registry = MagicMock()
    registry.names = ["test-stream"]
    registry.analyze = AsyncMock(return_value=MagicMock(problems=["issue"], skip_reason=None))
    monkeypatch.setattr(pipeline, "registered_builtin_signals", MagicMock(return_value=registry))
    compiler = MagicMock(compile_insights=AsyncMock(return_value=[insight]))
    monkeypatch.setattr(pipeline, "InsightCompilation", MagicMock(return_value=compiler))
    existing = [insight.model_copy(update={"id": "stored-id"})]
    before = datetime.now(timezone.utc)
    result = await pipeline.analyze_snapshot(
        snapshot,
        existing=existing,
        model_clients=ConfiguredModelClients(default=MagicMock(), fast=MagicMock()),
        ethos=None,
    )
    evidence = result.new_insights[0].evidence
    assert evidence[0].url == "https://provider/a"
    assert evidence[0].spans[0].url == "https://provider/child"
    assert evidence[1].url is None
    assert evidence[1].spans == []

    timestamp = compiler.compile_insights.await_args.kwargs["run_timestamp"]
    assert timestamp.tzinfo == timezone.utc
    assert before <= timestamp <= datetime.now(timezone.utc)
    assert compiler.compile_insights.await_args.args[1:] == (snapshot, existing)


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
    assert all("No traces loaded" in note for note in result.notes)
    assert any("grader-failure-patterns" in note for note in result.notes)
    client.acall.assert_not_awaited()
