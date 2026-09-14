# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filtering jobs on a path inside their plugin-defined ``spec``.

``spec`` is a filter namespace, so ``spec.<path>`` reads the spec exactly as
stored — nothing is derived or backfilled. The motivating case is Studio's Model
Evaluations table, which excludes agent-triggered runs by their target format
while keeping offline runs that name no target at all.
"""

import json

import pytest
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.spec import PlatformJobSpec
from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest, ListJobsQueryParams
from nmp.common.entities import DEFAULT_WORKSPACE

TEST_PLATFORM_SPEC = PlatformJobSpec.model_validate(
    {
        "steps": [
            {
                "name": "step1",
                "executor": {
                    "provider": "cpu",
                    "profile": "default",
                    "container": {"image": "test", "command": ["true"]},
                },
            }
        ]
    }
)

SOURCE = "spec-filter-testing"

AGENT_SPEC = {"target": {"format": "generic", "url": "http://agent/v1/chat/completions", "name": "my-agent"}}
MODEL_SPEC = {"target": {"format": "openai", "url": "http://model/v1", "name": "my-model"}}
OFFLINE_SPEC = {"dataset": "ws/rows"}
NOT_AGENT_TRIGGERED = {
    "$or": [
        {"spec.target.format": {"$nin": ["generic", "nemo_agent_toolkit"]}},
        {"spec.target.format": {"$eq": None}},
    ]
}


async def _create(jobs: AsyncJobsClient, name: str, spec: dict) -> None:
    (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreatePlatformJobRequest(
                name=name,
                source=SOURCE,
                spec=spec,
                platform_spec=TEST_PLATFORM_SPEC,
            ),
        )
    ).data()


async def _list(jobs: AsyncJobsClient, condition: dict) -> set[str]:
    response = (
        await jobs.list_jobs(
            workspace=DEFAULT_WORKSPACE,
            query_params=ListJobsQueryParams(filter=json.dumps({"$and": [{"source": SOURCE}, condition]})),
        )
    ).page()
    return {job.name for job in response.items}


@pytest.fixture
async def seeded_jobs(test_sdk: AsyncNeMoPlatform) -> AsyncJobsClient:
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    await _create(jobs, "agent-job", AGENT_SPEC)
    await _create(jobs, "model-job", MODEL_SPEC)
    await _create(jobs, "offline-job", OFFLINE_SPEC)
    return jobs


@pytest.mark.asyncio
async def test_filters_on_a_nested_spec_path(seeded_jobs: AsyncJobsClient):
    assert await _list(seeded_jobs, {"spec.target.format": "generic"}) == {"agent-job"}
    assert await _list(seeded_jobs, {"spec.target.format": "openai"}) == {"model-job"}


@pytest.mark.asyncio
async def test_absent_spec_path_matches_only_eq_null(seeded_jobs: AsyncJobsClient):
    """A job whose spec names no target matches neither membership operator."""
    assert await _list(seeded_jobs, {"spec.target.format": {"$eq": None}}) == {"offline-job"}
    # SQL NULL semantics: a row with no target is neither "in" nor "not in" anything.
    assert await _list(seeded_jobs, {"spec.target.format": {"$nin": ["generic"]}}) == {"model-job"}


@pytest.mark.asyncio
async def test_excluding_agents_keeps_model_and_offline_jobs(seeded_jobs: AsyncJobsClient):
    """Selecting the model formats positively would drop the offline run; this must not."""
    assert await _list(seeded_jobs, NOT_AGENT_TRIGGERED) == {"model-job", "offline-job"}


@pytest.mark.asyncio
async def test_rejects_a_path_outside_a_declared_namespace(seeded_jobs: AsyncJobsClient):
    with pytest.raises(Exception, match="Unknown filter field"):
        await _list(seeded_jobs, {"platform_spec.steps": "step1"})
