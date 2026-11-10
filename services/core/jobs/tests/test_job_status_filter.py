# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job status lists mean membership without changing generic array equality."""

import json

import pytest
from nemo_helix_plugin.api.filter import parse_json_filter
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nhx.common.entities.client import EntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import CreateHelixJobRequest, HelixJobsListFilter
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ({"status": "completed"}, {"completed"}),
        ({"status": "error"}, {"error"}),
        ({"status": "active"}, {"active"}),
        ({"status": "queued"}, {"queued"}),
        ({"status": ["completed"]}, {"completed"}),
        ({"status": ["error"]}, {"error"}),
        ({"status": ["active"]}, {"active"}),
        ({"status": ["queued"]}, {"queued"}),
        ({"status": ["completed", "error"]}, {"completed", "error"}),
        ({"status": {"$in": ["completed", "error"]}}, {"completed", "error"}),
        ({"status": {"$eq": ["completed", "error"]}}, {"completed", "error"}),
        ({"status": []}, set()),
        ({"$or": [{"status": ["completed", "error"]}, {"status": "queued"}]}, {"completed", "error", "queued"}),
        ({"$not": {"status": ["completed", "error"]}}, {"active", "queued"}),
    ],
)
async def test_job_status_filters_match_rows(
    async_client: AsyncNemoClient,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
    condition: dict,
    expected: set[str],
) -> None:
    for status in ("completed", "error", "active", "queued"):
        job = await mock_store.add(
            HelixJob(
                name=f"status-{status}",
                workspace="default",
                source="status-filter-testing",
                fileset="test-fileset",
                platform_spec=sample_platform_job_request.platform_spec,
                status=HelixJobStatus(status),
            )
        )
        attempt = await mock_store.add(
            HelixJobAttempt(
                name=f"attempt-{status}",
                workspace="default",
                job=job.id,
                platform_spec=sample_platform_job_request.platform_spec,
                status=HelixJobStatus(status),
            )
        )
        job.current_attempt_id = attempt.id
        await mock_store.update(job)
    jobs = AsyncJobsClient.from_client(async_client)
    response = await jobs.list_jobs(
        workspace="default",
        query_params={"filter": json.dumps({"$and": [{"source": "status-filter-testing"}, condition]})},
    )
    assert {job.status.value for job in response.page().items} == expected


def test_status_normalization_preserves_other_array_equality() -> None:
    operation = parse_json_filter('{"$and":[{"status":["completed","error"]},{"spec.values":[1,2]}]}')
    translated = HelixJobsListFilter.translate_operation(operation).to_dict()
    assert translated == {
        "$and": [{"data.status": {"$in": ["completed", "error"]}}, {"data.spec.values": {"$eq": [1, 2]}}]
    }
    # The generic parser must retain array equality for other services.
    assert parse_json_filter('{"status":["completed","error"]}').to_dict() == {
        "status": {"$eq": ["completed", "error"]}
    }
