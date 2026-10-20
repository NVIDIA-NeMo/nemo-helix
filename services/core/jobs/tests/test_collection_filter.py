# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The core API must filter persisted historical specs before counting/pagination."""

import json

import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nhx.common.entities import DEFAULT_WORKSPACE
from nhx.core.jobs.api.v2.jobs.schemas import CreateHelixJobRequest


@pytest.mark.asyncio
async def test_collection_key_presence_filters_before_pagination(
    async_client: AsyncNemoClient, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    client = AsyncJobsClient.from_client(async_client)
    for name, source, spec in [
        ("war-a", "shared", {"war": False}),
        ("war-b", "shared", {"war": True}),
        ("synth", "shared", {"synth": "interactive"}),
        ("ambiguous", "shared", {"war": False, "synth": "interactive"}),
        ("unknown", "shared", {}),
        ("foreign", "other", {"war": False}),
    ]:
        body = sample_platform_job_request.model_copy(update={"name": name, "source": source, "spec": spec})
        await client.create_job(workspace=DEFAULT_WORKSPACE, body=body)
    predicate = {
        "$and": [
            {"source": "shared"},
            {"spec": {"$hasKey": "war"}},
            {"$not": {"spec": {"$hasKey": "synth"}}},
        ]
    }
    names = []
    for page in (1, 2):
        result = (
            await client.list_jobs(
                workspace=DEFAULT_WORKSPACE,
                query_params={
                    "filter": json.dumps(predicate),
                    "page": page,
                    "page_size": 1,
                    "sort": "created_at",
                },
            )
        ).page()
        assert result.metadata["total_results"] == 2
        assert result.metadata["total_pages"] == 2
        names.extend(job.name for job in result.items)
    assert set(names) == {"war-a", "war-b"}
