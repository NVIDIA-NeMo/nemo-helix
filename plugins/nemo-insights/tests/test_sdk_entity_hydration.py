# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for entity metadata returned by plugin SDK resources."""

from nemo_insights_plugin.entities import AnalysisConfig, Insight
from nemo_insights_plugin.sdk_resources._entity import entity_from_response
from nemo_insights_plugin.sdk_resources.analysis_configs import _analysis_config_page_from_response
from nemo_insights_plugin.sdk_resources.insights import _insight_from_response

_METADATA = {
    "id": "entity-123",
    "created_at": "2026-07-13T12:00:00+00:00",
    "created_by": "service:platform",
    "updated_at": "2026-07-13T12:01:00+00:00",
    "updated_by": "user:tester",
    "parent": "entity-parent",
    "db_version": 3,
}


def test_entity_from_response_restores_store_metadata() -> None:
    config = entity_from_response(
        AnalysisConfig,
        {
            **_METADATA,
            "name": "agent-a",
            "workspace": "default",
            "agent": "agent-a",
            "enabled": True,
        },
    )

    assert config.id == "entity-123"
    assert config.created_at is not None and config.created_at.isoformat() == "2026-07-13T12:00:00+00:00"
    assert config.created_by == "service:platform"
    assert config.updated_at is not None and config.updated_at.isoformat() == "2026-07-13T12:01:00+00:00"
    assert config.updated_by == "user:tester"
    assert config.parent == "entity-parent"
    assert config._db_version == 3


def test_analysis_config_page_restores_item_metadata() -> None:
    page = _analysis_config_page_from_response(
        {
            "data": [
                {
                    **_METADATA,
                    "name": "agent-a",
                    "workspace": "default",
                    "agent": "agent-a",
                    "enabled": True,
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "current_page_size": 1,
                "total_pages": 1,
                "total_results": 1,
            },
            "sort": "-created_at",
            "filter": None,
        }
    )

    assert page.data[0].id == "entity-123"
    assert page.data[0].created_by == "service:platform"


def test_insight_response_restores_all_store_metadata() -> None:
    insight = _insight_from_response(
        {
            **_METADATA,
            "name": "insight-a",
            "workspace": "default",
            "title": "Validation insight",
            "agent": "agent-a",
            "description": "description",
            "status": "open",
            "trace_refs": [],
        }
    )

    assert isinstance(insight, Insight)
    assert insight.id == "entity-123"
    assert insight.created_by == "service:platform"
    assert insight.parent == "entity-parent"
