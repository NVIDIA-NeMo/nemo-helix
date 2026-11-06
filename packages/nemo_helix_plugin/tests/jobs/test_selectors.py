# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public selector signatures, query composition and sync/async wire parity."""

import inspect
import json
from types import FunctionType
from typing import get_type_hints
from unittest.mock import AsyncMock, create_autospec

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.jobs import endpoints
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.selectors import JobReferenceKind
from nemo_helix_plugin.jobs.types import JobLogsQueryParams


@pytest.mark.parametrize(
    "operation",
    [
        endpoints.get_job,
        endpoints.update_job,
        endpoints.delete_job,
        endpoints.cancel_job,
        endpoints.pause_job,
        endpoints.resume_job,
        endpoints.rerun_job,
        endpoints.update_job_control,
        endpoints.get_job_status,
        endpoints.list_job_attempts,
        endpoints.get_job_attempt,
        endpoints.update_job_attempt_status_details,
        endpoints.list_job_logs,
        endpoints.create_job_result,
        endpoints.list_job_results,
        endpoints.get_job_result,
        endpoints.download_job_result,
        endpoints.list_steps,
        endpoints.get_job_step,
        endpoints.update_job_step_status,
        endpoints.list_attempt_steps,
        endpoints.get_attempt_step,
        endpoints.update_attempt_step_status,
        endpoints.list_job_step_tasks,
        endpoints.get_job_step_task,
        endpoints.list_attempt_step_tasks,
        endpoints.update_attempt_step_task,
        endpoints.get_attempt_step_task,
    ],
)
def test_all_job_operations_export_explicit_signatures(operation: FunctionType):
    # Introspection is the behavior under test: named exports must preserve each signature family.
    original = inspect.signature(operation)
    reference = "job" if "job" in original.parameters else "name"
    for mode in ("id", "name"):
        named = getattr(endpoints, operation.__name__ + "_by_" + mode)
        signature = inspect.signature(named)
        assert "job_reference_kind" not in signature.parameters
        assert signature.parameters["job_" + mode].default is inspect.Parameter.empty
        assert reference not in signature.parameters
        assert set(signature.parameters) == (set(original.parameters) - {reference, "job_reference_kind"}) | {
            "job_" + mode
        }
        assert get_type_hints(named)["return"] == get_type_hints(operation)["return"]


def test_log_query_composition_preserves_fields_and_input():
    query: JobLogsQueryParams = {"page_cursor": "cursor", "tail": 7, "job_reference_kind": JobReferenceKind.NAME}
    before = query.copy()
    request = endpoints.list_job_logs_by_name(job_name="literal-name", workspace="team", query_params=query)
    assert query == before
    assert request.path_params == {"workspace": "team", "name": "literal-name"}
    assert request.query_params == {"page_cursor": "cursor", "tail": 7, "job_reference_kind": "name"}
    with pytest.raises(ValueError, match="Conflicting"):
        endpoints.list_job_logs_by_id(job_id="immutable-id", query_params=query)
    with pytest.raises(ValueError, match="Conflicting"):
        endpoints.list_job_logs(name="reference", query_params=query, job_reference_kind=JobReferenceKind.ID)


def test_omitted_and_invalid_modes_are_checked_before_request_send():
    assert not endpoints.get_job(name="reference").query_params
    assert (
        endpoints.get_job(name="reference", query_params=json.loads('{"job_reference_kind":null}')).query_params == {}
    )
    with pytest.raises(ValueError):
        endpoints.get_job(name="reference", job_reference_kind=json.loads('"invalid"'))
    with pytest.raises(ValueError):
        endpoints.get_job_by_id(job_id="immutable-id", query_params=json.loads('{"job_reference_kind":"invalid"}'))


@pytest.mark.asyncio
async def test_sync_async_clients_send_identical_explicit_requests():
    requests: list[httpx.Request] = []

    def missing(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(404, json={"detail": "Job not found"})

    with httpx.Client(transport=httpx.MockTransport(missing)) as transport:
        sync_client = JobsClient(base_url="https://platform.test", workspace="team", http_client=transport)
        with pytest.raises(NotFoundError):
            sync_client.get_job_by_id(job_id="immutable-id")
    async with httpx.AsyncClient(transport=httpx.MockTransport(missing)) as transport:
        async_client = AsyncJobsClient(base_url="https://platform.test", workspace="team", http_client=transport)
        with pytest.raises(NotFoundError):
            await async_client.get_job_by_id(job_id="immutable-id")
        assert inspect.iscoroutinefunction(async_client.get_job_by_id)
        assert inspect.iscoroutinefunction(inspect.getattr_static(AsyncJobsClient, "get_job_by_id").__wrapped__)
        assert isinstance(create_autospec(AsyncJobsClient).get_job_by_id, AsyncMock)
    assert len(requests) == 2
    assert requests[0].url == requests[1].url
    assert requests[0].url.path == "/apis/jobs/v2/workspaces/team/jobs/immutable-id"
    assert dict(requests[0].url.params) == {"job_reference_kind": "id"}
