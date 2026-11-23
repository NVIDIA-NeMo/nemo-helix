# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import httpx
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nhx.common.entities.client import EntityClient
from nhx.core.jobs.api.dependencies import dep_system_entity_client


def test_system_entity_client_uses_jobs_identity_and_reuses_transport() -> None:
    http_client = AsyncMock(spec=httpx.AsyncClient)
    http_client.headers = {}
    entity_client = EntityClient(
        AsyncEntitiesClient(
            base_url="http://localhost:8080",
            workspace="default",
            default_headers={"X-NHX-Principal-Id": "alice@example.com"},
            http_client=http_client,
        )
    )

    system_client = dep_system_entity_client(entity_client)

    assert system_client is not entity_client
    assert system_client._client._default_headers["X-NHX-Principal-Id"] == "service:jobs"
    assert system_client._client._default_headers["X-NHX-Internal"] == "true"
    assert entity_client._client._default_headers["X-NHX-Principal-Id"] == "alice@example.com"
    assert system_client._client._http is entity_client._client._http


async def test_log_dependency_keeps_request_endpoint_auth_and_transport() -> None:
    from fastapi import Depends, FastAPI
    from nemo_helix_plugin.client.auth import StaticToken
    from nemo_helix_plugin.client.client import AsyncNemoClient
    from nhx.common.jobs.log_client import JobLogsClient, dep_job_logs_client
    from nhx.common.service.dependencies import get_nemo_client

    requests: list[httpx.Request] = []

    def files(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "request-platform"
        assert request.headers["authorization"] == "Bearer request-token"
        assert request.headers["x-request-context"] == "retained"
        return httpx.Response(200, json={"data": [], "total": 0, "next_page": None, "prev_page": None})

    async with httpx.AsyncClient(transport=httpx.MockTransport(files)) as transport:
        client = AsyncNemoClient(
            base_url="https://request-platform",
            http_client=transport,
            auth=StaticToken("request-token"),
            default_headers={"X-Request-Context": "retained"},
        )
        app = FastAPI()
        app.dependency_overrides[get_nemo_client] = lambda: client

        @app.get("/logs")
        async def logs(logs_client: JobLogsClient = Depends(dep_job_logs_client)) -> int:
            return (await logs_client.query_logs("artifacts", "default")).total

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as api:
            response = await api.get("/logs")
        assert response.status_code == 200 and response.json() == 0
        assert len(requests) == 1
        assert requests[0].url.path == "/apis/files/v2/workspaces/default/filesets/artifacts/otlp/v1/logs/query"
        assert not transport.is_closed
