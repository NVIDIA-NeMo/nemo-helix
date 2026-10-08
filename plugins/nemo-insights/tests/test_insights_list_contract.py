# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.entity_client import NemoPaginationInfo, get_entity_client
from nemo_insights_plugin.entities import Insight
from nemo_insights_plugin.evidence import TraceEvidence
from nemo_insights_plugin.service import InsightsService
from nhx.intake.entities.experiments import ExperimentGroup
from nhx.intake.spans.api.dependencies import get_spans_service


def _insight(name: str, entity_id: str) -> Insight:
    insight = Insight(
        name=name,
        workspace="default",
        title=f"Title for {name}",
        agent="test-agent",
        description=f"Description for {name}",
    )
    insight._id = entity_id
    return insight


def _app(entity_client: AsyncMock, spans_service: AsyncMock) -> FastAPI:
    app = FastAPI()
    for spec in InsightsService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    app.dependency_overrides[get_spans_service] = lambda: spans_service
    return app


def test_service_exposes_analysis_runs_without_legacy_job_routes() -> None:
    paths = _app(AsyncMock(), AsyncMock()).openapi()["paths"]
    assert "/v2/workspaces/{workspace}/analysis-runs" in paths
    assert not any("/jobs/" in path for path in paths)


def test_api_returns_converted_evidence_and_persists_span_updates() -> None:
    entity_client = AsyncMock()
    insight = Insight.model_validate(
        {
            "name": "stable-name",
            "workspace": "default",
            "title": "Original title",
            "description": "Original description",
            "agent": "test-agent",
            "trace_refs": ["trace-a"],
            "trace_links": {"trace-a": "https://provider/trace-a"},
        }
    )
    insight._id = "insight-a"
    entity_client.get_by_id.return_value = insight
    entity_client.update.side_effect = lambda item: item
    client = TestClient(_app(entity_client, AsyncMock()))
    response = client.get("/v2/workspaces/default/insights/insight-a")
    assert response.status_code == 200
    assert "trace_refs" not in response.json()
    assert response.json()["evidence"][0]["url"] == "https://provider/trace-a"
    response = client.patch(
        "/v2/workspaces/default/insights/insight-a",
        json={
            "evidence": [{"trace_id": "trace-a", "spans": [{"span_id": "child"}]}],
            "updated_date": "2026-10-07T12:00:00Z",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence"][0]["spans"][0]["span_id"] == "child"
    assert data["updated_date"] == "2026-10-07T12:00:00Z"
    assert (data["id"], data["name"], data["title"], data["description"]) == (
        "insight-a",
        "stable-name",
        "Original title",
        "Original description",
    )


def test_list_insights_enriches_the_page_with_counts_and_last_seen_at() -> None:
    entity_client = AsyncMock()
    spans_service = AsyncMock()
    insights = [
        _insight("first", "insight-a"),
        _insight("second", "insight-b"),
        _insight("third", "insight-c"),
    ]
    insights[0].evidence = [TraceEvidence(trace_id=ref) for ref in ["trace-old", "trace-new"]]
    insights[1].evidence = [TraceEvidence(trace_id="trace-missing")]
    entity_client.list.return_value = SimpleNamespace(
        data=insights,
        pagination=NemoPaginationInfo(
            page=1,
            page_size=20,
            current_page_size=len(insights),
            total_pages=1,
            total_results=len(insights),
        ),
    )
    entity_client.count_by.return_value = {"insight-a": 3}
    latest = datetime(2026, 1, 2, tzinfo=timezone.utc)
    spans_service.latest_trace_started_at_by_group.return_value = {"insight-a": latest}

    response = TestClient(_app(entity_client, spans_service)).get("/v2/workspaces/default/insights")

    assert response.status_code == 200
    assert [(item["id"], item["experiment_group_count"], item["last_seen_at"]) for item in response.json()["data"]] == [
        ("insight-a", 3, "2026-01-02T00:00:00Z"),
        ("insight-b", 0, None),
        ("insight-c", 0, None),
    ]
    entity_client.count_by.assert_awaited_once_with(
        ExperimentGroup,
        "insight_id",
        workspace="default",
        filter_obj={
            "insight_id": {"$in": ["insight-a", "insight-b", "insight-c"]},
            "is_deleted": False,
        },
    )
    spans_service.latest_trace_started_at_by_group.assert_awaited_once_with(
        workspace="default",
        trace_refs_by_group={
            "insight-a": ["trace-old", "trace-new"],
            "insight-b": ["trace-missing"],
            "insight-c": [],
        },
    )
