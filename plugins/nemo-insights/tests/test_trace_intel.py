# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from insight_agent.insight import Insight
from nemo_insights_plugin.analyst.analyst_backend import RemoteAnalystBackend
from nemo_insights_plugin.analyst.trace_intel import BorrowedModelClient, load_existing_insights, to_change_set
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
