# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK sub-resources for ``AuditConfig`` CRUD.

Exposed as ``GarakPluginResource.configs`` and its async twin. Each
method maps 1:1 onto the CLI verbs at ``nemo garak configs <verb>`` and
the FastAPI routes in :mod:`nemo_garak.api.v2.configs`.
"""

from __future__ import annotations

from nemo_garak.api.v2.schemas import CreateAuditConfigRequest, UpdateAuditConfigRequest
from nemo_garak.entities import (
    AuditConfig,
    AuditPluginsData,
    AuditReportData,
    AuditRunData,
    AuditSystemData,
)
from nemo_garak.sdk_resources._parent import AsyncGarakResourceParent, GarakResourceParent


def _build_create_body(
    *,
    name: str,
    description: str | None,
    system: AuditSystemData | None,
    run: AuditRunData | None,
    plugins: AuditPluginsData | None,
    reporting: AuditReportData | None,
) -> dict:
    body = CreateAuditConfigRequest(
        name=name,
        description=description,
        system=system or AuditSystemData(),
        run=run or AuditRunData(),
        plugins=plugins or AuditPluginsData(),
        reporting=reporting or AuditReportData(),
    )
    return body.model_dump(mode="json")


def _build_update_body(
    *,
    description: str | None,
    system: AuditSystemData | None,
    run: AuditRunData | None,
    plugins: AuditPluginsData | None,
    reporting: AuditReportData | None,
) -> dict:
    body = UpdateAuditConfigRequest(
        description=description,
        system=system or AuditSystemData(),
        run=run or AuditRunData(),
        plugins=plugins or AuditPluginsData(),
        reporting=reporting or AuditReportData(),
    )
    return body.model_dump(mode="json")


class _ConfigResource:
    """Sync ``configs`` sub-resource — five CRUD verbs."""

    def __init__(self, parent: GarakResourceParent) -> None:
        self._parent = parent

    def create(
        self,
        *,
        workspace: str,
        name: str,
        description: str | None = None,
        system: AuditSystemData | None = None,
        run: AuditRunData | None = None,
        plugins: AuditPluginsData | None = None,
        reporting: AuditReportData | None = None,
    ) -> AuditConfig:
        body = _build_create_body(
            name=name,
            description=description,
            system=system,
            run=run,
            plugins=plugins,
            reporting=reporting,
        )
        response = self._parent._http_client.post(
            self._parent._url(f"/v2/workspaces/{workspace}/configs"),
            json=body,
        )
        response.raise_for_status()
        return AuditConfig.model_validate(response.json())

    def list(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
    ) -> dict:
        response = self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/configs"),
            params={"page": page, "page_size": page_size, "sort": sort},
        )
        response.raise_for_status()
        return response.json()

    def get(self, *, workspace: str, name: str) -> AuditConfig:
        response = self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
        )
        response.raise_for_status()
        return AuditConfig.model_validate(response.json())

    def update(
        self,
        *,
        workspace: str,
        name: str,
        description: str | None = None,
        system: AuditSystemData | None = None,
        run: AuditRunData | None = None,
        plugins: AuditPluginsData | None = None,
        reporting: AuditReportData | None = None,
    ) -> AuditConfig:
        body = _build_update_body(
            description=description,
            system=system,
            run=run,
            plugins=plugins,
            reporting=reporting,
        )
        response = self._parent._http_client.put(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
            json=body,
        )
        response.raise_for_status()
        return AuditConfig.model_validate(response.json())

    def delete(self, *, workspace: str, name: str) -> None:
        response = self._parent._http_client.delete(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
        )
        response.raise_for_status()


class _AsyncConfigResource:
    """Async ``configs`` sub-resource — mirrors :class:`_ConfigResource`."""

    def __init__(self, parent: AsyncGarakResourceParent) -> None:
        self._parent = parent

    async def create(
        self,
        *,
        workspace: str,
        name: str,
        description: str | None = None,
        system: AuditSystemData | None = None,
        run: AuditRunData | None = None,
        plugins: AuditPluginsData | None = None,
        reporting: AuditReportData | None = None,
    ) -> AuditConfig:
        body = _build_create_body(
            name=name,
            description=description,
            system=system,
            run=run,
            plugins=plugins,
            reporting=reporting,
        )
        response = await self._parent._http_client.post(
            self._parent._url(f"/v2/workspaces/{workspace}/configs"),
            json=body,
        )
        response.raise_for_status()
        return AuditConfig.model_validate(response.json())

    async def list(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
    ) -> dict:
        response = await self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/configs"),
            params={"page": page, "page_size": page_size, "sort": sort},
        )
        response.raise_for_status()
        return response.json()

    async def get(self, *, workspace: str, name: str) -> AuditConfig:
        response = await self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
        )
        response.raise_for_status()
        return AuditConfig.model_validate(response.json())

    async def update(
        self,
        *,
        workspace: str,
        name: str,
        description: str | None = None,
        system: AuditSystemData | None = None,
        run: AuditRunData | None = None,
        plugins: AuditPluginsData | None = None,
        reporting: AuditReportData | None = None,
    ) -> AuditConfig:
        body = _build_update_body(
            description=description,
            system=system,
            run=run,
            plugins=plugins,
            reporting=reporting,
        )
        response = await self._parent._http_client.put(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
            json=body,
        )
        response.raise_for_status()
        return AuditConfig.model_validate(response.json())

    async def delete(self, *, workspace: str, name: str) -> None:
        response = await self._parent._http_client.delete(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
        )
        response.raise_for_status()


__all__ = ["_AsyncConfigResource", "_ConfigResource"]
