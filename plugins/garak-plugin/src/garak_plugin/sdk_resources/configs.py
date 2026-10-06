# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK sub-resources for ``ScanConfig`` CRUD.

Mounted as ``client.garak_plugin.configs`` (sync) and on the async client. Each
method maps 1:1 onto the CLI verbs at ``nemo garak-plugin configs <verb>`` and
the FastAPI routes in :mod:`garak_plugin.api.v2.configs`.
"""

from __future__ import annotations

from garak_plugin.api.v2.schemas import CreateScanConfigRequest, UpdateScanConfigRequest
from garak_plugin.entities import (
    ScanConfig,
    ScanPluginsData,
    ScanReportData,
    ScanRunData,
    ScanSystemData,
)
from garak_plugin.sdk_resources._parent import AsyncGarakPluginResourceParent, GarakPluginResourceParent


def _build_create_body(
    *,
    name: str,
    description: str | None,
    system: ScanSystemData | None,
    run: ScanRunData | None,
    plugins: ScanPluginsData | None,
    reporting: ScanReportData | None,
) -> dict:
    body = CreateScanConfigRequest(
        name=name,
        description=description,
        system=system or ScanSystemData(),
        run=run or ScanRunData(),
        plugins=plugins or ScanPluginsData(),
        reporting=reporting or ScanReportData(),
    )
    return body.model_dump(mode="json")


def _build_update_body(
    *,
    description: str | None,
    system: ScanSystemData | None,
    run: ScanRunData | None,
    plugins: ScanPluginsData | None,
    reporting: ScanReportData | None,
) -> dict:
    body = UpdateScanConfigRequest(
        description=description,
        system=system or ScanSystemData(),
        run=run or ScanRunData(),
        plugins=plugins or ScanPluginsData(),
        reporting=reporting or ScanReportData(),
    )
    return body.model_dump(mode="json")


class _ConfigResource:
    """Sync ``configs`` sub-resource — five CRUD verbs."""

    def __init__(self, parent: GarakPluginResourceParent) -> None:
        self._parent = parent

    def create(
        self,
        *,
        workspace: str,
        name: str,
        description: str | None = None,
        system: ScanSystemData | None = None,
        run: ScanRunData | None = None,
        plugins: ScanPluginsData | None = None,
        reporting: ScanReportData | None = None,
    ) -> ScanConfig:
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
        return ScanConfig.model_validate(response.json())

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

    def get(self, *, workspace: str, name: str) -> ScanConfig:
        response = self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
        )
        response.raise_for_status()
        return ScanConfig.model_validate(response.json())

    def update(
        self,
        *,
        workspace: str,
        name: str,
        description: str | None = None,
        system: ScanSystemData | None = None,
        run: ScanRunData | None = None,
        plugins: ScanPluginsData | None = None,
        reporting: ScanReportData | None = None,
    ) -> ScanConfig:
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
        return ScanConfig.model_validate(response.json())

    def delete(self, *, workspace: str, name: str) -> None:
        response = self._parent._http_client.delete(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
        )
        response.raise_for_status()


class _AsyncConfigResource:
    """Async ``configs`` sub-resource — mirrors :class:`_ConfigResource`."""

    def __init__(self, parent: AsyncGarakPluginResourceParent) -> None:
        self._parent = parent

    async def create(
        self,
        *,
        workspace: str,
        name: str,
        description: str | None = None,
        system: ScanSystemData | None = None,
        run: ScanRunData | None = None,
        plugins: ScanPluginsData | None = None,
        reporting: ScanReportData | None = None,
    ) -> ScanConfig:
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
        return ScanConfig.model_validate(response.json())

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

    async def get(self, *, workspace: str, name: str) -> ScanConfig:
        response = await self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
        )
        response.raise_for_status()
        return ScanConfig.model_validate(response.json())

    async def update(
        self,
        *,
        workspace: str,
        name: str,
        description: str | None = None,
        system: ScanSystemData | None = None,
        run: ScanRunData | None = None,
        plugins: ScanPluginsData | None = None,
        reporting: ScanReportData | None = None,
    ) -> ScanConfig:
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
        return ScanConfig.model_validate(response.json())

    async def delete(self, *, workspace: str, name: str) -> None:
        response = await self._parent._http_client.delete(
            self._parent._url(f"/v2/workspaces/{workspace}/configs/{name}"),
        )
        response.raise_for_status()


__all__ = ["_AsyncConfigResource", "_ConfigResource"]
