# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP client for the Insights plugin APIs used outside the plugin."""

from __future__ import annotations

from abc import abstractmethod

from nemo_insights_plugin.entities import Insight
from nemo_insights_plugin.sdk_resources._entity import entity_from_response
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.endpoint import get
from nemo_platform_plugin.client.method import method

_INSIGHTS_BASE = "/apis/insights/v2/workspaces/{workspace}"


@get(f"{_INSIGHTS_BASE}/insights/{{insight_id}}")
@abstractmethod
def get_insight_endpoint(*, workspace: str | None = None, insight_id: str) -> Insight: ...


class _InsightsMethods:
    get_insight_response = method(get_insight_endpoint)


class InsightsClient(_InsightsMethods, NemoClient):
    """Sync client for the Insights API subset Experimentalist uses."""

    def get_insight(self, *, workspace: str | None = None, insight_id: str) -> Insight:
        response = self.get_insight_response(workspace=workspace, insight_id=insight_id)
        return entity_from_response(Insight, response.http_response.json())


class AsyncInsightsClient(_InsightsMethods, AsyncNemoClient):
    """Async client for the Insights API subset Experimentalist uses."""

    async def get_insight(self, *, workspace: str | None = None, insight_id: str) -> Insight:
        response = await self.get_insight_response(workspace=workspace, insight_id=insight_id)
        return entity_from_response(Insight, response.http_response.json())
