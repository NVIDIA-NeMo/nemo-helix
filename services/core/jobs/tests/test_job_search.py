# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from httpx import AsyncClient
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, ListJobsQueryParams
from nhx.common.entities import DEFAULT_WORKSPACE

# Skip all substring search tests until entity store supports LIKE queries (nhx-oq7)
SUBSTRING_SEARCH_SKIP = pytest.mark.skip(reason="Requires substring search support in entity store (nhx-oq7)")


TEST_PLATFORM_SPEC = HelixJobSpec.model_validate(
    {
        "steps": [
            {
                "name": "step1",
                "executor": {"provider": "cpu", "profile": "default", "container": {"image": "test"}},
            }
        ]
    }
)


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_jobs_by_name(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="training-job-v1",
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="evaluation-job",
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"name": {"$like": "training"}})),
        )
    ).page()
    assert len(response.items) == 1
    assert "training" in response.items[0].name.lower()


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_jobs_by_project(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="job1",
                source="testing",
                project="nlp-project",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="job2",
                source="testing",
                project="vision-project",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"project": {"$like": "nlp"}})),
        )
    ).page()
    assert len(response.items) == 1
    assert response.items[0].project == "nlp-project"


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_jobs_multiple_values_or_logic(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    for name in ["training-job", "evaluation-job", "inference-job"]:
        (
            await jobs.create_job(
                workspace=DEFAULT_WORKSPACE,
                body=CreateHelixJobRequest(
                    name=name,
                    source="testing",
                    spec={},
                    platform_spec=TEST_PLATFORM_SPEC,
                ),
            )
        ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(
                filter=json.dumps({"$or": [{"name": {"$like": "training"}}, {"name": {"$like": "evaluation"}}]})
            ),
        )
    ).page()
    assert len(response.items) == 2
    names = [job.name for job in response.items]
    assert "training-job" in names
    assert "evaluation-job" in names


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_jobs_multiple_fields_and_logic(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    for name, project in [
        ("training-job-nlp", "nlp-project"),
        ("training-job-vision", "vision-project"),
        ("evaluation-job-nlp", "nlp-project"),
    ]:
        (
            await jobs.create_job(
                workspace=DEFAULT_WORKSPACE,
                body=CreateHelixJobRequest(
                    name=name,
                    source="testing",
                    project=project,
                    spec={},
                    platform_spec=TEST_PLATFORM_SPEC,
                ),
            )
        ).data()

    # Search for training jobs in nlp project - should only match training-job-nlp
    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(
                filter=json.dumps({"$and": [{"name": {"$like": "training"}}, {"project": {"$like": "nlp"}}]})
            ),
        )
    ).page()
    assert len(response.items) == 1
    assert response.items[0].name == "training-job-nlp"
    assert response.items[0].project == "nlp-project"


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_jobs_case_insensitive(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="Training-Job-V1",
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"name": {"$like": "training"}})),
        )
    ).page()
    assert len(response.items) == 1
    assert response.items[0].name == "Training-Job-V1"


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_jobs_partial_match(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="my-training-job-v1",
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"name": {"$like": "train"}})),
        )
    ).page()
    assert len(response.items) == 1
    assert "train" in response.items[0].name.lower()


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_combined_with_filter(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    job1 = (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="training-job-1",
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="training-job-2",
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()

    (await jobs.cancel_job(workspace=DEFAULT_WORKSPACE, name=job1.name)).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"source": "testing", "name": "training-job-1"})),
        )
    ).page()
    assert len(response.items) == 1
    assert response.items[0].id == job1.id


@pytest.mark.asyncio
@SUBSTRING_SEARCH_SKIP
async def test_search_no_results(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="training-job",
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"name": {"$like": "nonexistent"}})),
        )
    ).page()
    assert len(response.items) == 0


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_empty_string(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    for name in ["job1", "job2"]:
        (
            await jobs.create_job(
                workspace=DEFAULT_WORKSPACE,
                body=CreateHelixJobRequest(
                    name=name,
                    source="testing",
                    spec={},
                    platform_spec=TEST_PLATFORM_SPEC,
                ),
            )
        ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"name": {"$like": ""}})),
        )
    ).page()
    assert len(response.items) == 2


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_via_http_client(test_client: AsyncClient):
    await test_client.post(
        "/v1/hello-world/jobs",
        json={
            "name": "search-test-job",
            "spec": {"config": {"key": "value"}, "target": "test"},
        },
    )

    response = await test_client.get("/v1/hello-world/jobs?filter[name]=search")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    assert "search" in data["data"][0]["name"]
    assert "filter" in data


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_pagination(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    for i in range(15):
        (
            await jobs.create_job(
                workspace=DEFAULT_WORKSPACE,
                body=CreateHelixJobRequest(
                    name=f"training-job-{i}",
                    source="testing",
                    spec={},
                    platform_spec=TEST_PLATFORM_SPEC,
                ),
            )
        ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(
                page=1,
                page_size=10,
                filter=json.dumps({"name": {"$like": "training"}}),
            ),
        )
    ).page()
    assert len(response.items) == 10
    assert response.metadata["total_results"] == 15

    response_page2 = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(
                page=2,
                page_size=10,
                filter=json.dumps({"name": {"$like": "training"}}),
            ),
        )
    ).page()
    assert len(response_page2.items) == 5


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_underscore_behavior(test_sdk: AsyncNeMoHelix):
    """Test that underscore is treated as a literal character in search (substring matching)."""
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    for name in ["test_job_with_underscore", "test-job-with-dash"]:
        (
            await jobs.create_job(
                workspace=DEFAULT_WORKSPACE,
                body=CreateHelixJobRequest(
                    name=name,
                    source="testing",
                    spec={},
                    platform_spec=TEST_PLATFORM_SPEC,
                ),
            )
        ).data()

    # Underscore is treated as a literal character - only matches jobs containing "_"
    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"name": {"$like": "_"}})),
        )
    ).page()
    assert len(response.items) == 1  # Only the job with underscore matches
    assert response.items[0].name == "test_job_with_underscore"

    # Create a job with a valid name containing special allowed characters
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="job-100-complete",
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()

    # Search for the job using a substring
    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"name": {"$like": "100"}})),
        )
    ).page()
    assert len(response.items) == 1  # Only the job-100-complete job matches


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_long_string(test_sdk: AsyncNeMoHelix):
    # Use a name within the 255 character limit
    long_name = "job-" + "a" * 200  # Total 204 chars, within 255 limit
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name=long_name,
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()

    long_search = "a" * 200
    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"name": {"$like": long_search}})),
        )
    ).page()
    assert len(response.items) == 1
    assert long_search in response.items[0].name


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_result_limit(test_sdk: AsyncNeMoHelix):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    for i in range(150):
        (
            await jobs.create_job(
                workspace=DEFAULT_WORKSPACE,
                body=CreateHelixJobRequest(
                    name=f"batch-job-{i:03d}",
                    source="testing",
                    spec={},
                    platform_spec=TEST_PLATFORM_SPEC,
                ),
            )
        ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(
                page=1,
                page_size=100,
                filter=json.dumps({"name": {"$like": "batch"}}),
            ),
        )
    ).page()
    assert len(response.items) == 100
    assert response.metadata["total_results"] == 150


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_invalid_field(test_client: AsyncClient):
    """Test that invalid search fields are silently ignored (Pydantic extra='allow' behavior)."""
    await test_client.post(
        "/v1/hello-world/jobs",
        json={
            "name": "test-job",
            "spec": {"config": {"key": "value"}, "target": "test"},
        },
    )
    response = await test_client.get("/v1/hello-world/jobs?filter[invalid_field]=test")
    assert response.status_code == 422  # Invalid fields are NOT ignored


@SUBSTRING_SEARCH_SKIP
@pytest.mark.asyncio
async def test_search_special_characters(test_sdk: AsyncNeMoHelix):
    # Use only valid special characters per the pattern ^[\w\-\+.@:]*$
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreateHelixJobRequest(
                name="job-with-special-chars@example.com:8080",
                source="testing",
                spec={},
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()

    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"name": {"$like": "@example"}})),
        )
    ).page()
    assert len(response.items) == 1
    assert "@example" in response.items[0].name
