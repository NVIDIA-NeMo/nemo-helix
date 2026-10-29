# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create with ``?dry_run=true``: transform and compile as create would, then check, creating nothing."""

from __future__ import annotations

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

_CREATE_URL = "/apis/widgets/v2/workspaces/default/jobs"
_DRY_RUN = {"dry_run": "true"}


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


def _create_job_spy() -> MagicMock:
    jobs = MagicMock()
    jobs.create_job = AsyncMock()
    return jobs


def test_a_passing_dry_run_returns_the_unsaved_job_and_its_checks() -> None:
    seen: list[tuple[object, list[str], str]] = []

    async def checks(
        spec: object, platform_spec: HelixJobSpec, workspace: str, async_client: AsyncNemoClient
    ) -> list[JobDryRunCheck]:
        del async_client
        seen.append((spec, [step.name for step in platform_spec.steps], workspace))
        return [JobDryRunCheck(name="dataset rows", passed=True, message="10 rows match.")]

    jobs = _create_job_spy()
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        response = _client(dry_run_checks=checks).post(
            _CREATE_URL, params=_DRY_RUN, json={"name": "widget-a", "spec": {"label": "x"}}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["name"], body["id"], body["status"]) == ("widget-a", None, None)
    assert body["spec"] == {"label": "x", "learning_rate": 5e-6}
    assert body["status_details"] == {"checks": [{"name": "dataset rows", "passed": True, "message": "10 rows match."}]}
    assert seen == [(_OutputSpec(label="x", learning_rate=5e-6), ["training"], "default")]
    jobs.create_job.assert_not_called()


def test_a_failed_check_is_a_422_that_lists_every_check() -> None:
    async def checks(
        spec: object, platform_spec: HelixJobSpec, workspace: str, async_client: AsyncNemoClient
    ) -> list[JobDryRunCheck]:
        del spec, platform_spec, workspace, async_client
        return [
            JobDryRunCheck(name="execution profile", passed=True, message="Configured: gpu/default."),
            JobDryRunCheck(name="dataset rows", passed=False, message="Row 3 has no 'messages'."),
        ]

    jobs = _create_job_spy()
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        response = _client(dry_run_checks=checks).post(_CREATE_URL, params=_DRY_RUN, json={"spec": {"label": "x"}})

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": "Dry run failed: dataset rows. Nothing was created.",
        "checks": [
            {"name": "execution profile", "passed": True, "message": "Configured: gpu/default."},
            {"name": "dataset rows", "passed": False, "message": "Row 3 has no 'messages'."},
        ],
    }
    jobs.create_job.assert_not_called()


def test_without_checks_a_dry_run_still_validates_and_compiles() -> None:
    jobs = _create_job_spy()
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        response = _client().post(_CREATE_URL, params=_DRY_RUN, json={"spec": {"label": "x"}})

    assert response.status_code == 200, response.text
    assert response.json()["status_details"] == {"checks": []}
    jobs.create_job.assert_not_called()


def test_without_the_dry_run_option_the_flag_is_refused_not_ignored() -> None:
    jobs = _create_job_spy()
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        response = _client(route_options=[JobRouteOption.CORE]).post(
            _CREATE_URL, params=_DRY_RUN, json={"spec": {"label": "x"}}
        )

    assert response.status_code == 400
    jobs.create_job.assert_not_called()


def test_the_flag_is_only_in_the_schema_when_the_option_is_on() -> None:
    def create_params(route_options: list[JobRouteOption]) -> list[str]:
        operation = (
            _client(route_options=route_options)
            .get("/openapi.json")
            .json()["paths"]["/apis/widgets/v2/workspaces/{workspace}/jobs"]["post"]
        )
        return [param["name"] for param in operation.get("parameters", []) if param["in"] == "query"]

    assert create_params([JobRouteOption.CORE, JobRouteOption.DRY_RUN]) == ["dry_run"]
    assert create_params([JobRouteOption.CORE]) == []
