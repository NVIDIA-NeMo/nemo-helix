# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``client.insights.insights`` SDK sub-resource."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_insights_plugin.entities import InsightStatus
from nemo_insights_plugin.sdk_resources.insights import _AsyncInsightResource, _build_update_body, _InsightResource


def test_update_insight_sdk_body_omits_none_fields() -> None:
    empty = _build_update_body(agent=None, description=None, status=None, evidence=None, updated_date=None)
    partial = _build_update_body(
        agent=None,
        description="Updated description",
        status=InsightStatus.RESOLVED,
        evidence=[],
        updated_date=None,
    )

    assert empty.model_dump(mode="json", exclude_unset=True) == {}
    assert partial.model_dump(mode="json", exclude_unset=True) == {
        "description": "Updated description",
        "status": "resolved",
        "evidence": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "ratings", [{}, {"severity": None}, {"confidence": None}, {"severity": "high", "confidence": "med"}]
)
async def test_public_sdk_update_preserves_rating_patch(asynchronous, ratings):
    response = MagicMock()
    response.http_response.json.return_value = {
        "id": "id",
        "title": "Issue",
        "description": "Details",
        "agent": "target",
        "workspace": "test",
    }
    parent = MagicMock()
    parent._client.update_insight = (
        AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    )
    if asynchronous:
        await _AsyncInsightResource(parent).update(workspace="test", insight_id="id", **ratings)
    else:
        _InsightResource(parent).update(workspace="test", insight_id="id", **ratings)
    body = parent._client.update_insight.call_args.kwargs["body"]
    assert body.model_dump(mode="json", exclude_unset=True) == ratings
