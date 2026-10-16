# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A plugin job can be created as the plugin's service, acting for the caller."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client, get_service_nemo_client_factory
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    job_route_factory,
)
from nemo_helix_plugin.jobs.routes import add_job_routes
from pydantic import BaseModel
from starlette.testclient import TestClient

_CALLER_CLIENT = "caller client"


class _WidgetSpec(BaseModel):
    label: str = "widget"


def _compiler(
    workspace: str,
    original_spec: _WidgetSpec,
    transformed_spec: _WidgetSpec,
    entity_client: object,
    job_name: str | None,
    sdk: object,
) -> HelixJobSpec:
    del workspace, original_spec, transformed_spec, entity_client, job_name, sdk
    return HelixJobSpec(
        steps=[
            HelixJobStep(
                name="step",
                executor=CPUExecutionProviderSpec(
                    provider="cpu", profile="default", container=ContainerSpec(image="t")
                ),
                config={},
            )
        ]
    )


class _ServiceWidgetJob(NemoJob):
    name: ClassVar[str] = "widget"
    spec_schema: ClassVar[type[BaseModel] | None] = _WidgetSpec
    runs_as_service: ClassVar[str | None] = "widget-maker"

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    @classmethod
    async def compile(cls, **kwargs: Any) -> HelixJobSpec:
        return _compiler("default", _WidgetSpec(), _WidgetSpec(), None, None, None)


def _create_response() -> MagicMock:
    job = SimpleNamespace(
        id="job-1",
        name="widget-job",
        description=None,
        workspace="default",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        spec={"label": "widget"},
        status="created",
        status_details=None,
        error_details=None,
        ownership=None,
        custom_fields=None,
    )
    response = MagicMock()
    response.data.return_value = job
    return response


def _post_job(router: Any, path: str) -> list[object]:
    """Create a job through *router* and return the client(s) Jobs was called with."""
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: _CALLER_CLIENT
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()
    app.dependency_overrides[get_service_nemo_client_factory] = lambda: lambda service: f"service:{service} client"

    jobs_created_with: list[object] = []

    def from_client(client: object) -> SimpleNamespace:
        async def create_job(*, workspace: str, body: object) -> MagicMock:
            del workspace, body
            jobs_created_with.append(client)
            return _create_response()

        return SimpleNamespace(create_job=create_job)

    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", side_effect=from_client):
        response = TestClient(app).post(f"/apis/widgets/v2/workspaces/default{path}", json={"spec": {}})

    assert response.status_code == 201, response.text
    return jobs_created_with


def test_a_service_job_is_created_as_the_service() -> None:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_WidgetSpec,
        platform_job_config_compiler=_compiler,
        create_as_service="widget-maker",
    )

    assert _post_job(router, "/jobs") == ["service:widget-maker client"]


def test_a_job_is_created_as_the_caller_by_default() -> None:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_WidgetSpec,
        platform_job_config_compiler=_compiler,
    )

    assert _post_job(router, "/jobs") == [_CALLER_CLIENT]


def test_a_nemo_job_that_runs_as_its_service_is_created_as_the_service() -> None:
    router = add_job_routes(_ServiceWidgetJob, service_name="widgets")

    assert _post_job(router, "/jobs/widget") == ["service:widget-maker client"]
