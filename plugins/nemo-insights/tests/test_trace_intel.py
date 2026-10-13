# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from insight_agent.evidence_streams.anomaly_and_patterns.stream import (
    AnomalyAndPatternsAnalysis,
    AnomalyAndPatternsArtifacts,
    AnomalyAndPatternsConfig,
    CrossToolFailureGroup,
)
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
from insight_agent.evidence_streams.tool_issues.stream import (
    ToolIssueConfig,
    ToolIssueEvidenceArtifacts,
    build_cards,
)
from insight_agent.insight import Insight
from nemo_insights_plugin.analyst.analyst_backend import RemoteAnalystBackend
from nemo_insights_plugin.analyst.result import AnalystResult
from nemo_insights_plugin.analyst.trace_intel import (
    BELOW_THRESHOLD_SUMMARY,
    MAX_BELOW_THRESHOLD_NOTES,
    BorrowedModelClient,
    load_existing_insights,
    to_change_set,
    with_evidence_notes,
)
from nemo_insights_plugin.entities import InsightStatus
from nooa.unifiedllm import UnifiedLLM


def _insight(id="stored-id", **updates):
    return Insight.model_validate(
        {
            "id": id,
            "name": "Original title",
            "description": "Original description",
            "trace_refs": ["a", "b"],
            **updates,
        }
    )


def test_reconciliation_uses_ids_and_preserves_platform_owned_fields():
    original = _insight()
    changed = _insight(name="Renamed by model", description="Rewritten", trace_refs=["a", "b", "c"])
    new = _insight(None, name="New issue")
    result = to_change_set([changed, new], [original], trace_count=3)
    assert result.updated_insights[0].model_dump() == {"id": "stored-id", "trace_refs": ["c"]}
    assert result.new_insights[0].title == "New issue"


@pytest.mark.parametrize("refs", [["b", "a"], ["a", "a"], ["a", "b", "a"]])
def test_reordered_removed_or_duplicate_refs_are_not_new_evidence(refs):
    result = to_change_set([_insight(trace_refs=refs)], [_insight()], trace_count=3)
    assert not result.updated_insights
    assert "0 existing insights with new evidence" in result.summary


def test_new_evidence_is_unique_and_preserves_generated_order():
    result = to_change_set([_insight(trace_refs=["d", "a", "c", "d"])], [_insight()], trace_count=3)
    assert result.updated_insights[0].trace_refs == ["d", "c"]


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
            id=id, title="Issue", description="Description", trace_refs=["a", "b"], status=InsightStatus.OPEN, **updates
        )

    invalid = row("invalid")
    invalid.trace_refs = []
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
    backend = object.__new__(RemoteAnalystBackend)
    backend._get = AsyncMock(return_value=MagicMock(workspace=workspace, agent=agent))
    with pytest.raises(ValueError, match="no longer belongs"):
        await backend._add_trace_refs(workspace="test", agent="target", insight_id="id", trace_refs=["a", "b"])


def _tool_issue_evidence(*, cases_by_tool: dict[str, int]) -> EvidenceStreamResult:
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
    return EvidenceStreamResult(
        stream_name="tool-issues",
        problems=(),
        finding_count=len(findings),
        artifacts=ToolIssueEvidenceArtifacts(
            findings=tuple(findings), cards=tuple(cards), catalog_coverage={}, config=config
        ),
    )


def _failure_pattern_evidence(
    *, trace_ids: list[str], cross_tool_messages: tuple[str, ...] = ()
) -> EvidenceStreamResult:
    events = tuple(
        {
            "trace_id": trace_id,
            "signature": "knowledge_search: index unavailable",
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
    return EvidenceStreamResult(
        stream_name="anomaly-and-patterns",
        problems=(),
        finding_count=len(events),
        artifacts=AnomalyAndPatternsArtifacts(result=analysis, config=AnomalyAndPatternsConfig()),
    )


def _empty_result() -> AnalystResult:
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


def test_skipped_streams_are_reported_without_changing_the_summary():
    skipped = EvidenceStreamResult(stream_name="ethos-divergence", problems=(), skip_reason="No ethos document")
    result = with_evidence_notes(_empty_result(), [skipped])
    assert result.summary == _empty_result().summary
    assert result.notes == ["- skipped: ethos-divergence (No ethos document)"]


@pytest.mark.asyncio
async def test_report_lists_notes_after_insight_lines():
    backend = object.__new__(RemoteAnalystBackend)
    backend.mirror = None
    result = AnalystResult(summary="Analyzed 2 traces.", notes=["- skipped: ethos-divergence (No ethos document)"])
    report = await backend.persist_result(workspace="test", agent="target", result=result)
    assert report == (
        "Analyzed 2 traces.\n\n- no insights created or updated\n- skipped: ethos-divergence (No ethos document)"
    )
