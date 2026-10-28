# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The factory's dry-run route: transform and compile as create would, then check, creating nothing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    DryRunChecks,
    GPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    JobRouteOption,
    job_route_factory,
)
from nemo_helix_plugin.jobs.types import JobDryRunCheck
from pydantic import BaseModel
from starlette.testclient import TestClient

_DRY_RUN_URL = "/apis/widgets/v2/workspaces/default/jobs/dry-run"


class _InputSpec(BaseModel):
    label: str


class _OutputSpec(BaseModel):
    label: str
    learning_rate: float


def _to_output(
    input_spec: _InputSpec, workspace: str, entity_client: object, job_name: str | None, sdk: object
) -> _OutputSpec:
    del workspace, entity_client, job_name, sdk
    return _OutputSpec(label=input_spec.label, learning_rate=5e-6)


def _compiler(
    workspace: str,
    original_spec: _InputSpec,
    transformed_spec: _OutputSpec,
    entity_client: object,
    job_name: str | None,
    sdk: object,
) -> HelixJobSpec:
    del workspace, original_spec, transformed_spec, entity_client, job_name, sdk
    executor = GPUExecutionProviderSpec(provider="gpu", profile="default", container=ContainerSpec(image="trainer:1"))
    return HelixJobSpec(steps=[HelixJobStep(name="training", executor=executor, config={})])


def _client(
    *, dry_run_checks: DryRunChecks | None = None, route_options: list[JobRouteOption] | None = None
) -> TestClient:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_InputSpec,
        job_output=_OutputSpec,
        input_to_output=_to_output,
        platform_job_config_compiler=_compiler,
        route_options=route_options or [JobRouteOption.CORE, JobRouteOption.DRY_RUN],
        dry_run_checks=dry_run_checks,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()
    return TestClient(app)


def _jobs_client(*profiles: tuple[str, str]) -> MagicMock:
    jobs = MagicMock()
    jobs.list_execution_profiles = AsyncMock(
        return_value=[SimpleNamespace(provider=provider, profile=profile) for provider, profile in profiles]
    )
    jobs.create_job = AsyncMock()
    return jobs


def test_dry_run_reports_steps_spec_and_checks_without_creating_a_job() -> None:
    seen: list[tuple[object, str]] = []

    async def checks(spec: object, workspace: str, async_client: AsyncNemoClient) -> list[JobDryRunCheck]:
        del async_client
        seen.append((spec, workspace))
        return [JobDryRunCheck(name="dataset rows", passed=True, message="10 rows match.")]

    jobs = _jobs_client(("gpu", "default"), ("cpu", "default"))
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        response = _client(dry_run_checks=checks).post(_DRY_RUN_URL, json={"spec": {"label": "x"}})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["passed"] is True
    assert body["spec"] == {"label": "x", "learning_rate": 5e-6}
    assert body["steps"] == [{"name": "training", "provider": "gpu", "profile": "default", "image": "trainer:1"}]
    assert [check["name"] for check in body["checks"]] == ["execution profile", "dataset rows"]
    assert seen == [(_OutputSpec(label="x", learning_rate=5e-6), "default")]
    jobs.create_job.assert_not_called()


def test_dry_run_fails_on_a_profile_the_platform_does_not_have() -> None:
    jobs = _jobs_client(("cpu", "default"))
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        response = _client().post(_DRY_RUN_URL, json={"spec": {"label": "x"}})

    body = response.json()
    assert body["passed"] is False
    assert body["checks"] == [
        {
            "name": "execution profile",
            "passed": False,
            "message": "Not configured on this platform: gpu/default. Configured: cpu/default.",
        }
    ]


def test_a_failed_plugin_check_fails_the_dry_run() -> None:
    async def checks(spec: object, workspace: str, async_client: AsyncNemoClient) -> list[JobDryRunCheck]:
        del spec, workspace, async_client
        return [JobDryRunCheck(name="dataset rows", passed=False, message="Row 3 has no 'messages'.")]

    jobs = _jobs_client(("gpu", "default"))
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        response = _client(dry_run_checks=checks).post(_DRY_RUN_URL, json={"spec": {"label": "x"}})

    assert response.json()["passed"] is False


def test_dry_run_route_is_only_mounted_when_requested() -> None:
    response = _client(route_options=[JobRouteOption.CORE]).post(_DRY_RUN_URL, json={"spec": {"label": "x"}})
    assert response.status_code == 405
