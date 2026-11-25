# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A job type's ``store_spec`` on the job create route: the spec a new job stores in place of its canonical one."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from nemo_helix_plugin.client.errors import ConflictError, InternalServerError, UnprocessableEntityError
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    GPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    JobRouteOption,
    job_route_factory,
)
from nemo_helix_plugin.jobs.types import StoredJobSpec
from pydantic import BaseModel
from starlette.testclient import TestClient

_CREATE_URL = "/apis/widgets/v2/workspaces/default/jobs"


class _Spec(BaseModel):
    label: str


class _Storer:
    """Stores a smaller spec in place of the canonical one, and records what it stored and discarded."""

    def __init__(
        self,
        *,
        refuse: Exception | None = None,
        spec: dict[str, Any] | None = None,
        discard_error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[dict[str, Any], str, str]] = []
        self.discarded = 0
        self.refuse = refuse
        self.spec = spec
        self.discard_error = discard_error

    async def __call__(self, spec: _Spec, workspace: str, job_name: str, async_client: object) -> StoredJobSpec:
        del async_client
        self.calls.append((spec.model_dump(mode="json"), workspace, job_name))
        if self.refuse is not None:
            raise self.refuse
        stored = self.spec if self.spec is not None else {"label": spec.label, "stored": True}
        return StoredJobSpec(spec=stored, discard=self._discard)

    async def _discard(self) -> None:
        self.discarded += 1
        if self.discard_error is not None:
            raise self.discard_error


def _compile(workspace: str, input_spec: _Spec, spec: _Spec, *args: object) -> HelixJobSpec:
    del workspace, input_spec, args
    executor = GPUExecutionProviderSpec(provider="gpu", profile="default", container=ContainerSpec(image="w:1"))
    return HelixJobSpec(steps=[HelixJobStep(name="work", executor=executor, config={"label": spec.label})])


def _http_error(error: type[Exception], status_code: int) -> Exception:
    return error(httpx.Response(status_code, request=httpx.Request("POST", "http://jobs/apis/jobs/v2")))


def _post(
    storer: _Storer | None,
    *,
    name: str | None = "widget-1",
    dry_run: bool = False,
    create_error: Exception | None = None,
    raise_server_exceptions: bool = False,
) -> tuple[Any, MagicMock]:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_Spec,
        platform_job_config_compiler=_compile,
        route_options=[JobRouteOption.CORE, JobRouteOption.DRY_RUN],
        allow_spec_update=storer is None,
        store_spec=storer,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()
    jobs = MagicMock()
    jobs.with_retry = MagicMock(return_value=jobs)
    # The route's reply needs a whole job record; these tests read what it sent instead.
    jobs.create_job = AsyncMock(side_effect=create_error or RuntimeError("stop after create"))
    jobs.list_execution_profiles = AsyncMock(return_value=[SimpleNamespace(provider="gpu", profile="default")])
    body: dict[str, Any] = {"spec": {"label": "x"}} | ({"name": name} if name is not None else {})
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
        response = client.post(_CREATE_URL, params={"dry_run": "true"} if dry_run else None, json=body)
    return response, jobs


def test_the_job_stores_what_its_type_stores_and_runs_what_was_compiled() -> None:
    storer = _Storer()

    _, jobs = _post(storer)

    assert storer.calls == [({"label": "x"}, "default", "widget-1")]
    body = jobs.create_job.await_args.kwargs["body"]
    assert body.spec == {"label": "x", "stored": True}
    assert body.platform_spec.steps[0].config == {"label": "x"}


def test_an_unnamed_job_is_named_before_its_spec_is_stored() -> None:
    storer = _Storer()

    _, jobs = _post(storer, name=None)

    [(_, _, job_name)] = storer.calls
    assert re.fullmatch(r"widgets-[0-9a-f]{8}", job_name)  # as the Jobs service would name it
    assert jobs.create_job.await_args.kwargs["body"].name == job_name


def test_a_job_type_that_stores_its_spec_as_is_keeps_the_jobs_services_naming_and_retries() -> None:
    _, jobs = _post(None, name=None)

    assert jobs.create_job.await_args.kwargs["body"].name is None
    jobs.with_retry.assert_not_called()


def test_the_create_of_a_job_that_stored_its_spec_is_sent_once() -> None:
    _, jobs = _post(_Storer())

    [(policy,), _] = jobs.with_retry.call_args
    assert policy.max_retries == 0


def test_a_dry_run_stores_nothing() -> None:
    storer = _Storer()

    response, jobs = _post(storer, dry_run=True)

    assert response.status_code == 200, response.text
    assert storer.calls == []
    jobs.create_job.assert_not_called()


@pytest.mark.parametrize(
    ("refusal", "status"),
    [
        (PermissionError("may not write"), 403),
        (ValueError("could not write"), 422),
        (HTTPException(status_code=409, detail="name taken"), 409),
    ],
    ids=["denied", "failed", "own status"],
)
def test_a_spec_that_cannot_be_stored_creates_no_job(refusal: Exception, status: int) -> None:
    response, jobs = _post(_Storer(refuse=refusal))

    assert response.status_code == status, response.text
    jobs.create_job.assert_not_called()


def test_a_stored_spec_the_replies_cannot_render_is_undone_and_creates_no_job() -> None:
    storer = _Storer(spec={"stored": True})  # no `label`, so `_Spec` can't render it

    response, jobs = _post(storer)

    assert response.status_code == 500, response.text
    assert storer.discarded == 1
    jobs.create_job.assert_not_called()


@pytest.mark.parametrize(
    ("create_error", "discarded"),
    [
        (_http_error(UnprocessableEntityError, 422), 1),
        # Sent once, so a 409 refuses this create too, such as for a missing secret or a taken name.
        (_http_error(ConflictError, 409), 1),
        # Neither of these says whether the job exists.
        (_http_error(InternalServerError, 500), 0),
        (RuntimeError("connection lost"), 0),
    ],
    ids=["refused", "conflict", "server error", "unknown"],
)
def test_only_a_refused_job_undoes_what_its_spec_stored(create_error: Exception, discarded: int) -> None:
    storer = _Storer()

    _post(storer, create_error=create_error)

    assert storer.discarded == discarded


def test_a_failing_undo_leaves_the_refusal_to_surface() -> None:
    storer = _Storer(discard_error=RuntimeError("files are down"))

    with pytest.raises(UnprocessableEntityError):
        _post(storer, create_error=_http_error(UnprocessableEntityError, 422), raise_server_exceptions=True)

    assert storer.discarded == 1


def test_storing_a_spec_requires_refusing_spec_updates() -> None:
    with pytest.raises(TypeError, match="requires allow_spec_update=False"):
        job_route_factory(
            service_name="widgets",
            job_type="Widget",
            job_input=_Spec,
            platform_job_config_compiler=_compile,
            store_spec=_Storer(),
        )
