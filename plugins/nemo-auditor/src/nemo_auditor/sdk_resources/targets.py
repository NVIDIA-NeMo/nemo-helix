# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK sub-resources for ``AuditTarget`` CRUD.

Exposed as ``AuditorPluginResource.targets`` and its async twin. Each
method maps 1:1 onto the CLI verbs at ``nemo auditor targets <verb>`` and
the FastAPI routes in :mod:`nemo_auditor.api.v2.targets`.
"""

from __future__ import annotations

from typing import Any

from nemo_auditor.api.v2.schemas import CreateAuditTargetRequest, UpdateAuditTargetRequest
from nemo_auditor.entities import AuditTarget
from nemo_auditor.sdk_resources._parent import AsyncAuditorResourceParent, AuditorResourceParent


def _build_create_body(
    *,
    name: str,
    type: str,
    model: str,
    options: dict[str, Any] | None,
    description: str | None,
) -> dict:
    body = CreateAuditTargetRequest(
        name=name,
        description=description,
        type=type,
        model=model,
        options=options or {},
    )
    return body.model_dump(mode="json")


def _build_update_body(
    *,
    type: str,
    model: str,
    options: dict[str, Any] | None,
    description: str | None,
) -> dict:
    body = UpdateAuditTargetRequest(
        description=description,
        type=type,
        model=model,
        options=options or {},
    )
    return body.model_dump(mode="json")


class _TargetResource:
    """Sync ``targets`` sub-resource — five CRUD verbs."""

    def __init__(self, parent: AuditorResourceParent) -> None:
        self._parent = parent

    def create(
        self,
        *,
        workspace: str,
        name: str,
        type: str,
        model: str,
        options: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> AuditTarget:
        body = _build_create_body(
            name=name,
            type=type,
            model=model,
            options=options,
            description=description,
        )
        response = self._parent._http_client.post(
            self._parent._url(f"/v2/workspaces/{workspace}/targets"),
            json=body,
        )
        response.raise_for_status()
        return AuditTarget.model_validate(response.json())

    def list(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
    ) -> dict:
        response = self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/targets"),
            params={"page": page, "page_size": page_size, "sort": sort},
        )
        response.raise_for_status()
        return response.json()

    def get(self, *, workspace: str, name: str) -> AuditTarget:
        response = self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/targets/{name}"),
        )
        response.raise_for_status()
        return AuditTarget.model_validate(response.json())

    def update(
        self,
        *,
        workspace: str,
        name: str,
        type: str,
        model: str,
        options: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> AuditTarget:
        body = _build_update_body(
            type=type,
            model=model,
            options=options,
            description=description,
        )
        response = self._parent._http_client.put(
            self._parent._url(f"/v2/workspaces/{workspace}/targets/{name}"),
            json=body,
        )
        response.raise_for_status()
        return AuditTarget.model_validate(response.json())

    def delete(self, *, workspace: str, name: str) -> None:
        response = self._parent._http_client.delete(
            self._parent._url(f"/v2/workspaces/{workspace}/targets/{name}"),
        )
        response.raise_for_status()


class _AsyncTargetResource:
    """Async ``targets`` sub-resource — mirrors :class:`_TargetResource`."""

    def __init__(self, parent: AsyncAuditorResourceParent) -> None:
        self._parent = parent

    async def create(
        self,
        *,
        workspace: str,
        name: str,
        type: str,
        model: str,
        options: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> AuditTarget:
        body = _build_create_body(
            name=name,
            type=type,
            model=model,
            options=options,
            description=description,
        )
        response = await self._parent._http_client.post(
            self._parent._url(f"/v2/workspaces/{workspace}/targets"),
            json=body,
        )
        response.raise_for_status()
        return AuditTarget.model_validate(response.json())

    async def list(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
    ) -> dict:
        response = await self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/targets"),
            params={"page": page, "page_size": page_size, "sort": sort},
        )
        response.raise_for_status()
        return response.json()

    async def get(self, *, workspace: str, name: str) -> AuditTarget:
        response = await self._parent._http_client.get(
            self._parent._url(f"/v2/workspaces/{workspace}/targets/{name}"),
        )
        response.raise_for_status()
        return AuditTarget.model_validate(response.json())

    async def update(
        self,
        *,
        workspace: str,
        name: str,
        type: str,
        model: str,
        options: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> AuditTarget:
        body = _build_update_body(
            type=type,
            model=model,
            options=options,
            description=description,
        )
        response = await self._parent._http_client.put(
            self._parent._url(f"/v2/workspaces/{workspace}/targets/{name}"),
            json=body,
        )
        response.raise_for_status()
        return AuditTarget.model_validate(response.json())

    async def delete(self, *, workspace: str, name: str) -> None:
        response = await self._parent._http_client.delete(
            self._parent._url(f"/v2/workspaces/{workspace}/targets/{name}"),
        )
        response.raise_for_status()


__all__ = ["_AsyncTargetResource", "_TargetResource"]
