# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK sub-resources for ``Insight`` CRUD.

Mounted as ``client.insights.insights`` (sync) and on the async client.
Each method maps 1:1 onto the FastAPI routes in
:mod:`nemo_insights_plugin.service`.
"""

from typing import Protocol, TypedDict

from nemo_insights_plugin.client import AsyncInsightsClient, InsightsClient
from nemo_insights_plugin.entities import Insight, InsightStatus
from nemo_insights_plugin.schema import (
    CreateInsightRequest,
    InsightPage,
    UpdateInsightRequest,
)
from nemo_insights_plugin.sdk_resources._entity import entity_from_response, hydrate_page
from nemo_insights_plugin.sdk_resources._errors import httpx_status_errors
from nemo_insights_plugin.types import ListInsightsQueryParams


class _InsightUpdatePatch(TypedDict, total=False):
    agent: str
    description: str
    status: InsightStatus
    trace_refs: list[str]


def _insight_from_response(data: dict[str, object]) -> Insight:
    """Parse one insight response body, preserving its store-assigned id."""
    return entity_from_response(Insight, data)


def _page_from_response(data: dict[str, object]) -> InsightPage:
    """Parse a list response, preserving each item's store-assigned id.

    Validated items lose their ids (computed field), so we re-attach them
    positionally from the raw payload, which preserves order.
    """
    page = InsightPage.model_validate(data)
    hydrate_page(page.data, data.get("data"))
    return page


class _ResourceParent(Protocol):
    """The slice of the insights SDK namespace these sub-resources rely on.

    Both the sync and async ``InsightsPluginResource`` satisfy this — typing
    against it (rather than importing the concrete classes from
    :mod:`nemo_insights_plugin.sdk`) avoids an import cycle, since ``sdk``
    imports the resource classes defined here.
    """

    _client: InsightsClient


class _AsyncResourceParent(Protocol):
    _client: AsyncInsightsClient


def _build_create_body(
    *,
    title: str,
    agent: str,
    description: str,
    status: InsightStatus | str,
    trace_refs: list[str] | None,
) -> CreateInsightRequest:
    body = CreateInsightRequest(
        title=title,
        agent=agent,
        description=description,
        status=InsightStatus(status) if isinstance(status, str) else status,
        trace_refs=list(trace_refs or []),
    )
    return body


def _build_update_body(
    *,
    agent: str | None,
    description: str | None,
    status: InsightStatus | str | None,
    trace_refs: list[str] | None,
) -> UpdateInsightRequest:
    update: _InsightUpdatePatch = {}
    if agent is not None:
        update["agent"] = agent
    if description is not None:
        update["description"] = description
    if status is not None:
        update["status"] = InsightStatus(status) if isinstance(status, str) else status
    if trace_refs is not None:
        update["trace_refs"] = trace_refs
    return UpdateInsightRequest(**update)


def _list_params(
    *,
    page: int,
    page_size: int,
    sort: str,
    agent: str | None,
    status: InsightStatus | str | None,
) -> ListInsightsQueryParams:
    params: ListInsightsQueryParams = {"page": page, "page_size": page_size, "sort": sort}
    if agent is not None:
        params["agent"] = agent
    if status is not None:
        params["status"] = status.value if isinstance(status, InsightStatus) else status
    return params


class _InsightResource:
    """Sync ``insights`` sub-resource — five CRUD verbs."""

    def __init__(self, parent: _ResourceParent) -> None:
        self._parent = parent

    def create(
        self,
        *,
        workspace: str,
        title: str,
        agent: str,
        description: str,
        status: InsightStatus | str = InsightStatus.OPEN,
        trace_refs: list[str] | None = None,
    ) -> Insight:
        body = _build_create_body(
            title=title,
            agent=agent,
            description=description,
            status=status,
            trace_refs=trace_refs,
        )
        with httpx_status_errors():
            response = self._client.create_insight(workspace=workspace, body=body)
            return _insight_from_response(response.http_response.json())

    def list_insights(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
        agent: str | None = None,
        status: InsightStatus | str | None = None,
    ) -> InsightPage:
        with httpx_status_errors():
            response = self._client.list_insights(
                workspace=workspace,
                query_params=_list_params(
                    page=page,
                    page_size=page_size,
                    sort=sort,
                    agent=agent,
                    status=status,
                ),
            )
            response.page()
            return _page_from_response(response.http_response.json())

    def get(self, *, workspace: str, insight_id: str) -> Insight:
        with httpx_status_errors():
            response = self._client.get_insight(workspace=workspace, insight_id=insight_id)
            return _insight_from_response(response.http_response.json())

    def update(
        self,
        *,
        workspace: str,
        insight_id: str,
        agent: str | None = None,
        description: str | None = None,
        status: InsightStatus | str | None = None,
        trace_refs: list[str] | None = None,
    ) -> Insight:
        body = _build_update_body(
            agent=agent,
            description=description,
            status=status,
            trace_refs=trace_refs,
        )
        with httpx_status_errors():
            response = self._client.update_insight(workspace=workspace, insight_id=insight_id, body=body)
            return _insight_from_response(response.http_response.json())

    def delete(self, *, workspace: str, insight_id: str) -> None:
        with httpx_status_errors():
            self._client.delete_insight(workspace=workspace, insight_id=insight_id).data()

    @property
    def _client(self) -> InsightsClient:
        return self._parent._client


class _AsyncInsightResource:
    """Async ``insights`` sub-resource — mirrors :class:`_InsightResource`."""

    def __init__(self, parent: _AsyncResourceParent) -> None:
        self._parent = parent

    async def create(
        self,
        *,
        workspace: str,
        title: str,
        agent: str,
        description: str,
        status: InsightStatus | str = InsightStatus.OPEN,
        trace_refs: list[str] | None = None,
    ) -> Insight:
        body = _build_create_body(
            title=title,
            agent=agent,
            description=description,
            status=status,
            trace_refs=trace_refs,
        )
        with httpx_status_errors():
            response = await self._client.create_insight(workspace=workspace, body=body)
            return _insight_from_response(response.http_response.json())

    async def list_insights(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
        agent: str | None = None,
        status: InsightStatus | str | None = None,
    ) -> InsightPage:
        with httpx_status_errors():
            response = await self._client.list_insights(
                workspace=workspace,
                query_params=_list_params(
                    page=page,
                    page_size=page_size,
                    sort=sort,
                    agent=agent,
                    status=status,
                ),
            )
            response.page()
            return _page_from_response(response.http_response.json())

    async def get(self, *, workspace: str, insight_id: str) -> Insight:
        with httpx_status_errors():
            response = await self._client.get_insight(workspace=workspace, insight_id=insight_id)
            return _insight_from_response(response.http_response.json())

    async def update(
        self,
        *,
        workspace: str,
        insight_id: str,
        agent: str | None = None,
        description: str | None = None,
        status: InsightStatus | str | None = None,
        trace_refs: list[str] | None = None,
    ) -> Insight:
        body = _build_update_body(
            agent=agent,
            description=description,
            status=status,
            trace_refs=trace_refs,
        )
        with httpx_status_errors():
            response = await self._client.update_insight(workspace=workspace, insight_id=insight_id, body=body)
            return _insight_from_response(response.http_response.json())

    async def delete(self, *, workspace: str, insight_id: str) -> None:
        with httpx_status_errors():
            (await self._client.delete_insight(workspace=workspace, insight_id=insight_id)).data()

    @property
    def _client(self) -> AsyncInsightsClient:
        return self._parent._client


__all__ = ["_AsyncInsightResource", "_InsightResource"]
