# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A transformer's ``authorize`` and ``defer`` keywords on the job create route."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from nemo_helix_plugin.dependencies import (
    RequestAuthorizer,
    get_entity_client,
    get_nemo_client,
    get_submission_authorizer,
)
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    DeferWrite,
    GPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    JobRouteOption,
    job_route_factory,
)
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from pydantic import BaseModel
from starlette.testclient import TestClient

_CREATE_URL = "/apis/widgets/v2/workspaces/default/jobs"


class _Spec(BaseModel):
    label: str
    published: bool = False


class _Recorder:
    """A transformer that defers one write, and the order of what the route then ran."""

    def __init__(self, *, refuse_compile: bool = False) -> None:
        self.events: list[str] = []
        self.authorize: RequestAuthorizer | None = None
        self.refuse_compile = refuse_compile

    async def to_output(
        self,
        input_spec: _Spec,
        workspace: str,
        entity_client: object,
        job_name: str | None,
        async_client: object,
        *,
        authorize: RequestAuthorizer | None = None,
        defer: DeferWrite | None = None,
    ) -> _Spec:
        del workspace, entity_client, job_name, async_client
        self.authorize = authorize
        self.events.append("transform")

        async def publish(spec: Any) -> Any:
            self.events.append("deferred write")
            return spec.model_copy(update={"published": True})

        assert defer is not None
        defer(publish)
        return input_spec

    def compile(self, *args: object) -> HelixJobSpec:
        del args
        self.events.append("compile")
        if self.refuse_compile:
            raise HelixJobCompilationError("no such profile")
        executor = GPUExecutionProviderSpec(provider="gpu", profile="default", container=ContainerSpec(image="w:1"))
        return HelixJobSpec(steps=[HelixJobStep(name="work", executor=executor, config={})])


def _post(
    recorder: _Recorder, *, authorize: RequestAuthorizer | None = None, dry_run: bool = False
) -> tuple[Any, MagicMock]:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_Spec,
        job_output=_Spec,
        input_to_output=recorder.to_output,
        platform_job_config_compiler=recorder.compile,
        route_options=[JobRouteOption.CORE, JobRouteOption.DRY_RUN],
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()
    if authorize is not None:
        app.dependency_overrides[get_submission_authorizer] = lambda: authorize
    jobs = MagicMock()
    jobs.create_job = AsyncMock(side_effect=RuntimeError("stop after create"))
    jobs.list_execution_profiles = AsyncMock(return_value=[SimpleNamespace(provider="gpu", profile="default")])
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            _CREATE_URL, params={"dry_run": "true"} if dry_run else None, json={"spec": {"label": "x"}}
        )
    return response, jobs


def test_a_deferred_write_runs_after_compile_and_its_spec_is_the_one_created() -> None:
    recorder = _Recorder()

    _, jobs = _post(recorder)

    assert recorder.events == ["transform", "compile", "deferred write"]
    body = jobs.create_job.await_args.kwargs["body"]
    assert body.spec["published"] is True


@pytest.mark.parametrize(
    ("recorder", "dry_run", "status"),
    [(_Recorder(refuse_compile=True), False, 422), (_Recorder(), True, 200)],
    ids=["compile refusal", "dry run"],
)
def test_a_deferred_write_does_not_run_when_no_job_will_be_created(
    recorder: _Recorder, dry_run: bool, status: int
) -> None:
    response, jobs = _post(recorder, dry_run=dry_run)

    assert response.status_code == status, response.text
    assert "deferred write" not in recorder.events
    jobs.create_job.assert_not_called()


def test_the_transformer_gets_the_platform_authorizer_or_none_outside_it() -> None:
    async def authorize(method: str, path: str) -> None:
        del method, path

    with_platform, standalone = _Recorder(), _Recorder()
    _post(with_platform, authorize=authorize)
    _post(standalone)

    assert with_platform.authorize is authorize
    assert standalone.authorize is None
