# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK sub-resources for periodic analysis opt-in configs and run status."""

from datetime import datetime
from enum import Enum
from typing import Protocol, TypeAlias, TypedDict

from nemo_insights_plugin.client import AsyncInsightsClient, InsightsClient
from nemo_insights_plugin.entities import (
    AnalysisConfig,
    AnalysisConfigStatus,
    AnalysisRunStatus,
)
from nemo_insights_plugin.schema import (
    AnalysisConfigPage,
    AnalysisRunStatusPage,
    EnableAnalysisConfigRequest,
    UpdateAnalysisConfigRequest,
    UpdateAnalysisRunStatusRequest,
)
from nemo_insights_plugin.sdk_resources._entity import entity_from_response, hydrate_page
from nemo_insights_plugin.sdk_resources._errors import httpx_status_errors
from nemo_insights_plugin.types import ListAnalysisConfigsQueryParams, ListAnalysisRunStatusesQueryParams


class _AnalysisRunStatusPatch(TypedDict, total=False):
    status: AnalysisConfigStatus
    last_successful_run_at: datetime | None
    last_attempted_at: datetime | None
    last_completed_at: datetime | None
    last_submitted_job: str
    last_error: str


class _Unset(Enum):
    VALUE = "unset"


_UNSET = _Unset.VALUE
_NullableDatetimeUpdate: TypeAlias = datetime | None | _Unset


class _ResourceParent(Protocol):
    """The slice of the insights SDK namespace this sub-resource needs."""

    _client: InsightsClient


class _AsyncResourceParent(Protocol):
    _client: AsyncInsightsClient


def _list_params(
    *,
    page: int,
    page_size: int,
    sort: str,
    enabled: bool | None,
) -> ListAnalysisConfigsQueryParams:
    params: ListAnalysisConfigsQueryParams = {"page": page, "page_size": page_size, "sort": sort}
    if enabled is not None:
        params["enabled"] = enabled
    return params


def _build_update_body(
    *,
    enabled: bool | None,
) -> UpdateAnalysisConfigRequest:
    if enabled is None:
        return UpdateAnalysisConfigRequest()
    return UpdateAnalysisConfigRequest(enabled=enabled)


def _build_enable_body(*, default_model: str, fast_model: str) -> EnableAnalysisConfigRequest:
    return EnableAnalysisConfigRequest(default_model=default_model, fast_model=fast_model)


def _build_status_update_body(
    *,
    status: AnalysisConfigStatus | str | None = None,
    last_successful_run_at: _NullableDatetimeUpdate = _UNSET,
    last_attempted_at: _NullableDatetimeUpdate = _UNSET,
    last_completed_at: _NullableDatetimeUpdate = _UNSET,
    last_submitted_job: str | None = None,
    last_error: str | None = None,
) -> UpdateAnalysisRunStatusRequest:
    update: _AnalysisRunStatusPatch = {}
    if status is not None:
        update["status"] = AnalysisConfigStatus(status) if isinstance(status, str) else status
    if not isinstance(last_successful_run_at, _Unset):
        update["last_successful_run_at"] = last_successful_run_at
    if not isinstance(last_attempted_at, _Unset):
        update["last_attempted_at"] = last_attempted_at
    if not isinstance(last_completed_at, _Unset):
        update["last_completed_at"] = last_completed_at
    if last_submitted_job is not None:
        update["last_submitted_job"] = last_submitted_job
    if last_error is not None:
        update["last_error"] = last_error
    return UpdateAnalysisRunStatusRequest(**update)


def _analysis_config_page_from_response(data: dict[str, object]) -> AnalysisConfigPage:
    page = AnalysisConfigPage.model_validate(data)
    hydrate_page(page.data, data.get("data"))
    return page


def _analysis_run_status_page_from_response(data: dict[str, object]) -> AnalysisRunStatusPage:
    page = AnalysisRunStatusPage.model_validate(data)
    hydrate_page(page.data, data.get("data"))
    return page


class _AnalysisConfigResource:
    """Sync ``analysis_configs`` sub-resource."""

    def __init__(self, parent: _ResourceParent) -> None:
        self._parent = parent

    @property
    def _client(self) -> InsightsClient:
        return self._parent._client

    def enable(self, *, workspace: str, agent: str, default_model: str, fast_model: str) -> AnalysisConfig:
        with httpx_status_errors():
            response = self._client.enable_analysis_config(
                workspace=workspace,
                agent=agent,
                body=_build_enable_body(default_model=default_model, fast_model=fast_model),
            )
            return entity_from_response(AnalysisConfig, response.http_response.json())

    def disable(self, *, workspace: str, agent: str) -> AnalysisConfig:
        with httpx_status_errors():
            response = self._client.disable_analysis_config(workspace=workspace, agent=agent)
            return entity_from_response(AnalysisConfig, response.http_response.json())

    def list_configs(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
        enabled: bool | None = None,
    ) -> AnalysisConfigPage:
        with httpx_status_errors():
            response = self._client.list_analysis_configs(
                workspace=workspace,
                query_params=_list_params(page=page, page_size=page_size, sort=sort, enabled=enabled),
            )
            response.page()
            return _analysis_config_page_from_response(response.http_response.json())

    def get(self, *, workspace: str, agent: str) -> AnalysisConfig:
        with httpx_status_errors():
            response = self._client.get_analysis_config(workspace=workspace, agent=agent)
            return entity_from_response(AnalysisConfig, response.http_response.json())

    def update(
        self,
        *,
        workspace: str,
        agent: str,
        enabled: bool | None = None,
    ) -> AnalysisConfig:
        with httpx_status_errors():
            response = self._client.update_analysis_config(
                workspace=workspace,
                agent=agent,
                body=_build_update_body(enabled=enabled),
            )
            return entity_from_response(AnalysisConfig, response.http_response.json())


class _AsyncAnalysisConfigResource:
    """Async ``analysis_configs`` sub-resource."""

    def __init__(self, parent: _AsyncResourceParent) -> None:
        self._parent = parent

    @property
    def _client(self) -> AsyncInsightsClient:
        return self._parent._client

    async def enable(self, *, workspace: str, agent: str, default_model: str, fast_model: str) -> AnalysisConfig:
        with httpx_status_errors():
            response = await self._client.enable_analysis_config(
                workspace=workspace,
                agent=agent,
                body=_build_enable_body(default_model=default_model, fast_model=fast_model),
            )
            return entity_from_response(AnalysisConfig, response.http_response.json())

    async def disable(self, *, workspace: str, agent: str) -> AnalysisConfig:
        with httpx_status_errors():
            response = await self._client.disable_analysis_config(workspace=workspace, agent=agent)
            return entity_from_response(AnalysisConfig, response.http_response.json())

    async def list_configs(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
        enabled: bool | None = None,
    ) -> AnalysisConfigPage:
        with httpx_status_errors():
            response = await self._client.list_analysis_configs(
                workspace=workspace,
                query_params=_list_params(page=page, page_size=page_size, sort=sort, enabled=enabled),
            )
            response.page()
            return _analysis_config_page_from_response(response.http_response.json())

    async def get(self, *, workspace: str, agent: str) -> AnalysisConfig:
        with httpx_status_errors():
            response = await self._client.get_analysis_config(workspace=workspace, agent=agent)
            return entity_from_response(AnalysisConfig, response.http_response.json())

    async def update(
        self,
        *,
        workspace: str,
        agent: str,
        enabled: bool | None = None,
    ) -> AnalysisConfig:
        with httpx_status_errors():
            response = await self._client.update_analysis_config(
                workspace=workspace,
                agent=agent,
                body=_build_update_body(enabled=enabled),
            )
            return entity_from_response(AnalysisConfig, response.http_response.json())


class _AnalysisRunStatusResource:
    """Sync ``analysis_run_statuses`` sub-resource."""

    def __init__(self, parent: _ResourceParent) -> None:
        self._parent = parent

    @property
    def _client(self) -> InsightsClient:
        return self._parent._client

    def list_statuses(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-updated_at",
    ) -> AnalysisRunStatusPage:
        query_params: ListAnalysisRunStatusesQueryParams = {"page": page, "page_size": page_size, "sort": sort}
        with httpx_status_errors():
            response = self._client.list_analysis_run_statuses(
                workspace=workspace,
                query_params=query_params,
            )
            response.page()
            return _analysis_run_status_page_from_response(response.http_response.json())

    def get(self, *, workspace: str, agent: str) -> AnalysisRunStatus:
        with httpx_status_errors():
            response = self._client.get_analysis_run_status(workspace=workspace, agent=agent)
            return entity_from_response(AnalysisRunStatus, response.http_response.json())

    def update(
        self,
        *,
        workspace: str,
        agent: str,
        status: AnalysisConfigStatus | str | None = None,
        last_successful_run_at: _NullableDatetimeUpdate = _UNSET,
        last_attempted_at: _NullableDatetimeUpdate = _UNSET,
        last_completed_at: _NullableDatetimeUpdate = _UNSET,
        last_submitted_job: str | None = None,
        last_error: str | None = None,
    ) -> AnalysisRunStatus:
        with httpx_status_errors():
            response = self._client.update_analysis_run_status(
                workspace=workspace,
                agent=agent,
                body=_build_status_update_body(
                    status=status,
                    last_successful_run_at=last_successful_run_at,
                    last_attempted_at=last_attempted_at,
                    last_completed_at=last_completed_at,
                    last_submitted_job=last_submitted_job,
                    last_error=last_error,
                ),
            )
            return entity_from_response(AnalysisRunStatus, response.http_response.json())


class _AsyncAnalysisRunStatusResource:
    """Async ``analysis_run_statuses`` sub-resource."""

    def __init__(self, parent: _AsyncResourceParent) -> None:
        self._parent = parent

    @property
    def _client(self) -> AsyncInsightsClient:
        return self._parent._client

    async def list_statuses(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-updated_at",
    ) -> AnalysisRunStatusPage:
        query_params: ListAnalysisRunStatusesQueryParams = {"page": page, "page_size": page_size, "sort": sort}
        with httpx_status_errors():
            response = await self._client.list_analysis_run_statuses(
                workspace=workspace,
                query_params=query_params,
            )
            response.page()
            return _analysis_run_status_page_from_response(response.http_response.json())

    async def get(self, *, workspace: str, agent: str) -> AnalysisRunStatus:
        with httpx_status_errors():
            response = await self._client.get_analysis_run_status(workspace=workspace, agent=agent)
            return entity_from_response(AnalysisRunStatus, response.http_response.json())

    async def update(
        self,
        *,
        workspace: str,
        agent: str,
        status: AnalysisConfigStatus | str | None = None,
        last_successful_run_at: _NullableDatetimeUpdate = _UNSET,
        last_attempted_at: _NullableDatetimeUpdate = _UNSET,
        last_completed_at: _NullableDatetimeUpdate = _UNSET,
        last_submitted_job: str | None = None,
        last_error: str | None = None,
    ) -> AnalysisRunStatus:
        with httpx_status_errors():
            response = await self._client.update_analysis_run_status(
                workspace=workspace,
                agent=agent,
                body=_build_status_update_body(
                    status=status,
                    last_successful_run_at=last_successful_run_at,
                    last_attempted_at=last_attempted_at,
                    last_completed_at=last_completed_at,
                    last_submitted_job=last_submitted_job,
                    last_error=last_error,
                ),
            )
            return entity_from_response(AnalysisRunStatus, response.http_response.json())


__all__ = [
    "_AnalysisConfigResource",
    "_AnalysisRunStatusResource",
    "_AsyncAnalysisConfigResource",
    "_AsyncAnalysisRunStatusResource",
]
