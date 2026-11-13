# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from fastapi import FastAPI
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.jobs.api_factory import job_route_factory
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nhx.common.entities import DEFAULT_WORKSPACE
from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers
from nhx.common.jobs.log_client import JobLogsClient, dep_job_logs_client
from nhx.common.service.dependencies import get_nemo_client
from nhx.core.jobs.api.v2.jobs.endpoints import dep_dispatcher, router
from nhx.core.jobs.api.v2.jobs.schemas import CreateHelixJobRequest
from nhx.core.jobs.app.dispatcher import JobDispatcher
from pydantic import BaseModel


class LogJobInput(BaseModel):
    pass


def compile_log_job(
    workspace: str,
    input_spec: LogJobInput,
    output_spec: LogJobInput,
    entity_client: EntityClient,
    job_name: str | None,
    async_client: AsyncNemoClient,
) -> HelixJobSpec:
    raise AssertionError("Log retrieval must not compile a job")


@pytest.mark.parametrize("factory_route", [False, True], ids=["core", "factory"])
@pytest.mark.parametrize(
    ("downstream", "expected_status", "expected_detail"),
    [
        pytest.param(httpx.Response(400, json={"detail": "Invalid filter"}), 400, "Invalid filter", id="bad-request"),
        pytest.param(httpx.Response(401, json={"detail": "Unauthorized"}), 401, "Unauthorized", id="unauthorized"),
        pytest.param(
            httpx.Response(403, json={"detail": "Forbidden", "code": "permission_denied"}),
            403,
            "Forbidden",
            id="forbidden",
        ),
        pytest.param(httpx.Response(404, json={"detail": "Fileset absent"}), 200, "", id="missing-fileset"),
        pytest.param(httpx.Response(409, json={"detail": "Conflict"}), 409, "Conflict", id="conflict"),
        pytest.param(
            httpx.Response(418, json={"detail": "Client error"}), 418, "Client error", id="unmapped-client-error"
        ),
        pytest.param(
            httpx.Response(422, json={"detail": "Invalid page cursor"}), 422, "Invalid page cursor", id="cursor"
        ),
        pytest.param(httpx.Response(422, json={"detail": "Invalid query"}), 422, "Invalid query", id="validation"),
        pytest.param(
            httpx.Response(429, json={"detail": "Too many requests"}), 429, "Too many requests", id="rate-limit"
        ),
        pytest.param(
            httpx.Response(500, json={"detail": "Internal diagnostic"}),
            500,
            "Failed to query job logs",
            id="server-error",
        ),
        pytest.param(httpx.Response(502, text="Proxy diagnostic"), 500, "Failed to query job logs", id="gateway-error"),
        pytest.param(
            httpx.Response(302, headers={"location": "http://other/internal"}),
            500,
            "Failed to query job logs",
            id="redirect",
        ),
        pytest.param(
            httpx.ConnectError("Transport diagnostic"),
            500,
            "Failed to query job logs",
            id="transport-error",
        ),
        pytest.param(
            httpx.Response(200, json={"invalid": "response"}),
            500,
            "Failed to query job logs",
            id="invalid-response",
        ),
    ],
)
async def test_job_log_http_errors(
    mock_dispatcher: JobDispatcher,
    sample_platform_job_request: CreateHelixJobRequest,
    factory_route: bool,
    downstream: httpx.Response | httpx.TransportError,
    expected_status: int,
    expected_detail: str,
) -> None:
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    requests: list[httpx.Request] = []

    def files_response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if isinstance(downstream, httpx.TransportError):
            raise downstream
        return downstream

    app = FastAPI()
    register_sdk_exception_handlers(app)
    app.include_router(router, prefix="/apis/jobs")
    app.include_router(
        job_route_factory(
            service_name="log-test",
            job_type="LogTest",
            job_input=LogJobInput,
            platform_job_config_compiler=compile_log_job,
        ),
        prefix="/apis/log-test/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[dep_dispatcher] = lambda: mock_dispatcher
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(files_response)) as files_http,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as caller,
    ):
        logs_client = JobLogsClient(AsyncNemoClient(base_url="http://files", http_client=files_http))
        jobs_client = AsyncNemoClient(base_url="http://test", http_client=caller)
        app.dependency_overrides[dep_job_logs_client] = lambda: logs_client
        app.dependency_overrides[get_nemo_client] = lambda: jobs_client
        service = "log-test" if factory_route else "jobs"
        response = await caller.get(f"/apis/{service}/v2/workspaces/{DEFAULT_WORKSPACE}/jobs/{job.name}/logs")

    assert response.status_code == expected_status
    if expected_status == 200:
        assert response.json() == {"data": [], "total": 0, "next_page": None, "prev_page": None}
    elif expected_status == 403:
        assert response.json() == {"detail": expected_detail, "code": "permission_denied"}
    else:
        assert response.json() == {"detail": expected_detail}
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert (
        requests[0].url.path
        == f"/apis/files/v2/workspaces/{DEFAULT_WORKSPACE}/filesets/{job.fileset}/otlp/v1/logs/query"
    )
    assert json.loads(requests[0].content)["filters"] == {"job": job.name, "job_attempt": job.attempt_id}
