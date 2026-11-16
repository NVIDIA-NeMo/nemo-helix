# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint tests for the job-group API surface."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from nhx.common.entities import DEFAULT_WORKSPACE
from nhx.core.jobs.api.dependencies import dep_dispatcher
from nhx.core.jobs.api.v2.job_groups.endpoints import router as job_group_router
from nhx.core.jobs.app.test_helpers import TestConstants


@pytest_asyncio.fixture()
async def job_group_client(mock_dispatcher) -> AsyncGenerator[AsyncClient, None]:
    from nhx.common.auth.middleware import AuthorizationMiddleware

    app = FastAPI()
    app.add_middleware(AuthorizationMiddleware)

    async def override_get_dispatcher():
        return mock_dispatcher

    app.dependency_overrides[dep_dispatcher] = override_get_dispatcher
    app.include_router(job_group_router, prefix="/apis/jobs")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _create_body(**overrides) -> dict:
    body = {
        "name": "api-group",
        "source": "test-source",
        "template": TestConstants.PLATFORM_SPEC.model_dump(mode="json"),
        "members": [
            {"name": "member-a", "config_patch": {"k": "a"}},
            {"name": "member-b", "config_patch": {"k": "b"}},
        ],
    }
    body.update(overrides)
    return body


def _base(workspace: str = DEFAULT_WORKSPACE) -> str:
    return f"/apis/jobs/v2/workspaces/{workspace}/job-groups"


@pytest.mark.asyncio
async def test_create_group_endpoint_returns_201(job_group_client: AsyncClient):
    response = await job_group_client.post(_base(), json=_create_body())
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "api-group"
    assert body["member_count"] == 2
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_create_group_requires_at_least_one_member(job_group_client: AsyncClient):
    response = await job_group_client.post(_base(), json=_create_body(members=[]))
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_group_rejects_duplicate_member_names(job_group_client: AsyncClient):
    body = _create_body(members=[{"name": "dupe", "config_patch": {}}, {"name": "dupe", "config_patch": {}}])
    response = await job_group_client.post(_base(), json=body)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_duplicate_group_name_conflicts(job_group_client: AsyncClient):
    assert (await job_group_client.post(_base(), json=_create_body())).status_code == 201
    second = await job_group_client.post(_base(), json=_create_body())
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_get_group_endpoint(job_group_client: AsyncClient):
    await job_group_client.post(_base(), json=_create_body())
    response = await job_group_client.get(f"{_base()}/api-group")
    assert response.status_code == 200
    assert response.json()["name"] == "api-group"


@pytest.mark.asyncio
async def test_get_missing_group_returns_404(job_group_client: AsyncClient):
    response = await job_group_client.get(f"{_base()}/missing")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_groups_endpoint(job_group_client: AsyncClient):
    await job_group_client.post(_base(), json=_create_body(name="g1"))
    await job_group_client.post(_base(), json=_create_body(name="g2"))
    response = await job_group_client.get(_base())
    assert response.status_code == 200
    names = {g["name"] for g in response.json()["data"]}
    assert {"g1", "g2"}.issubset(names)


@pytest.mark.asyncio
async def test_update_group_endpoint(job_group_client: AsyncClient):
    await job_group_client.post(_base(), json=_create_body())
    response = await job_group_client.patch(
        f"{_base()}/api-group",
        json={"max_concurrent_jobs": 7, "fail_fast": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["max_concurrent_jobs"] == 7
    assert body["fail_fast"] is True


@pytest.mark.asyncio
async def test_update_missing_group_returns_404(job_group_client: AsyncClient):
    response = await job_group_client.patch(f"{_base()}/missing", json={"description": "x"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_missing_group_returns_404(job_group_client: AsyncClient):
    response = await job_group_client.delete(f"{_base()}/missing")
    assert response.status_code == 404
