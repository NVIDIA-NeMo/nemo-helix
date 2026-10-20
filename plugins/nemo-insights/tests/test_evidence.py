# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest
from nemo_insights_plugin.entities import Insight, InsightStatus
from nemo_insights_plugin.evidence import TraceEvidence, merge_evidence
from nemo_insights_plugin.schema import CreateInsightRequest, UpdateInsightRequest


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
    assert insight.severity is None
    assert insight.confidence is None
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


@pytest.mark.parametrize("bad_span", [None, "span", {}, {"trace_id": None}, {"trace_id": []}, {"trace_id": " "}])
def test_legacy_span_without_usable_parent_does_not_hide_valid_evidence(bad_span):
    insight = stored(trace_refs=["a"], span_refs=[bad_span, {"trace_id": "a", "span_id": "valid"}])
    assert insight.evidence[0].spans[0].span_id == "valid"


@pytest.mark.parametrize("grouped", [False, True])
def test_legacy_invalid_span_ids_do_not_hide_valid_evidence(grouped):
    spans = [{}, {"span_id": None}, {"span_id": []}, {"span_id": " "}, {"span_id": "valid"}]
    refs = {"a": spans} if grouped else [{"trace_id": "a", **span} for span in spans]
    insight = stored(span_refs=refs, span_links={"a": {"valid": "https://provider/valid"}})
    assert insight.evidence[0].model_dump(exclude_none=True) == {
        "trace_id": "a",
        "spans": [{"span_id": "valid", "url": "https://provider/valid"}],
    }


@pytest.mark.parametrize("model", [Insight, CreateInsightRequest, UpdateInsightRequest])
def test_explicit_duplicate_evidence_merges_at_model_boundary(model):
    payload = dict(
        workspace="test",
        title="Issue",
        description="Details",
        agent="target",
        evidence=[
            {"trace_id": "a", "url": "https://provider/a", "spans": [{"span_id": "first"}]},
            {"trace_id": "a", "spans": [{"span_id": "second"}, {"span_id": "first", "url": "https://provider/first"}]},
        ],
    )
    insight = model.model_validate(payload)
    assert len(insight.evidence) == 1
    trace = insight.evidence[0]
    assert trace.url == "https://provider/a"
    assert [span.span_id for span in trace.spans] == ["first", "second"]
    assert trace.spans[0].url == "https://provider/first"
    assert len(payload["evidence"]) == 2
