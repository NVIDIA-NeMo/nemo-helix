# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated job routes reply without the job's spec when asked to, and report custom fields in the status."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    JobRouteOption,
    job_route_factory,
)
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from pydantic import BaseModel
from starlette.testclient import TestClient

_JOBS = "/apis/widgets/v2/workspaces/default/jobs"
_CUSTOM_FIELDS = {"client_request_sha256": "abc123"}


class _Spec(BaseModel):
    rows: list[str]


def _compiler(*_args: object) -> HelixJobSpec:
    return HelixJobSpec(
        steps=[
            HelixJobStep(
                name="step",
                executor=CPUExecutionProviderSpec(
                    provider="cpu", profile="default", container=ContainerSpec(image="test")
                ),
                config={},
            )
        ]
    )


def _job(name: str, status: str = "created") -> SimpleNamespace:
    return SimpleNamespace(
        id="job-1",
        attempt_id="attempt-1",
        name=name,
        description=None,
        project=None,
        workspace="default",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        spec={"rows": ["a", "b"]},
        status=status,
        status_details=None,
        error_details=None,
        ownership=None,
        custom_fields=_CUSTOM_FIELDS,
        control=None,
        source="widgets",
    )


def _data(value: object) -> MagicMock:
    response = MagicMock()
    response.data.return_value = value
    return response


class _JobsClient:
    def __init__(self) -> None:
        self.created: CreateHelixJobRequest | None = None
        self.cancelled: str | None = None

    async def create_job(self, *, workspace: str, body: CreateHelixJobRequest) -> MagicMock:
        del workspace
        self.created = body
        return _data(_job(body.name or "generated"))

    async def cancel_job(self, *, name: str, workspace: str) -> MagicMock:
        del workspace
        self.cancelled = name
        return _data(_job(name, status="cancelling"))

    async def list_execution_profiles(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(provider="cpu", profile="default")]

    async def get_job_status(self, *, name: str, workspace: str) -> MagicMock:
        del workspace
        return _data(
            HelixJobStatusResponse(
                id="job-1",
                name=name,
                status=HelixJobStatus.ACTIVE,
                status_details={},
                error_details=None,
                steps=[],
                created_at=datetime(2026, 1, 1),
                updated_at=datetime(2026, 1, 1),
                custom_fields=_CUSTOM_FIELDS,
            )
        )


@pytest.fixture
def jobs() -> _JobsClient:
    return _JobsClient()


@contextmanager
def _serve(jobs: _JobsClient, job_discriminator: str | None = None) -> Iterator[TestClient]:
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_Spec,
        platform_job_config_compiler=_compiler,
        route_options=[JobRouteOption.CORE, JobRouteOption.DRY_RUN],
        job_discriminator=job_discriminator,
    )
    app = FastAPI()
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        yield TestClient(app)


@pytest.fixture
def client(jobs: _JobsClient) -> Iterator[TestClient]:
    with _serve(jobs) as client:
        yield client


def test_create_returns_the_whole_job_by_default(client: TestClient) -> None:
    response = client.post(_JOBS, json={"name": "w1", "spec": {"rows": ["a", "b"]}})

    assert response.status_code == 201, response.text
    assert response.json()["spec"] == {"rows": ["a", "b"]}
    assert "Preference-Applied" not in response.headers


@pytest.mark.parametrize(
    "prefer",
    [
        ["return=minimal"],
        ["RETURN=minimal"],
        ['return="minimal"'],
        ["return = minimal"],
        ["respond-async, return=minimal"],
        ["return=minimal; x=1"],
        ['return=minimal; x="a,b"'],
        # Only the first ``return`` preference counts (RFC 7240 §2), across header lines too.
        ["return=minimal, return=representation"],
        ["respond-async", "return=minimal"],
    ],
)
def test_create_with_return_minimal_replies_without_a_body(
    client: TestClient, jobs: _JobsClient, prefer: list[str]
) -> None:
    response = client.post(
        _JOBS, json={"name": "w1", "spec": {"rows": ["a", "b"]}}, headers=[("Prefer", line) for line in prefer]
    )

    assert response.status_code == 201, response.text
    assert response.content == b""
    assert response.headers["Preference-Applied"] == "return=minimal"
    assert response.headers["Location"] == f"{_JOBS}/w1"
    assert jobs.created is not None


@pytest.mark.parametrize(
    "prefer",
    [
        ["return=representation"],
        ["return=representation, return=minimal"],
        ["return=representation", "return=minimal"],
        # A preference's value is case-sensitive (RFC 7240 §2).
        ["return=Minimal"],
        # Commas inside a quoted string don't separate preferences.
        ['foo="x,return=minimal"'],
        ['foo="x\\",return=minimal"'],
    ],
)
def test_create_not_preferring_return_minimal_returns_the_whole_job(client: TestClient, prefer: list[str]) -> None:
    response = client.post(
        _JOBS, json={"name": "w1", "spec": {"rows": ["a"]}}, headers=[("Prefer", line) for line in prefer]
    )

    assert response.status_code == 201, response.text
    assert response.json()["name"] == "w1"
    assert "Preference-Applied" not in response.headers


@pytest.mark.parametrize("prefer", [[], ["return=minimal"]])
def test_create_fails_closed_on_a_job_outside_the_collection(jobs: _JobsClient, prefer: list[str]) -> None:
    # The jobs service answers with a job whose spec lacks the collection's marker.
    with _serve(jobs, job_discriminator="widget") as client:
        response = client.post(
            _JOBS, json={"name": "w1", "spec": {"rows": ["a"]}}, headers=[("Prefer", line) for line in prefer]
        )

    assert response.status_code == 502, response.text
    assert "outside the Widget collection" in response.json()["detail"]


def test_dry_run_ignores_return_minimal(client: TestClient, jobs: _JobsClient) -> None:
    response = client.post(
        f"{_JOBS}?dry_run=true", json={"name": "w1", "spec": {"rows": ["a"]}}, headers={"Prefer": "return=minimal"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["spec"] == {"rows": ["a"]}
    assert "Preference-Applied" not in response.headers
    assert jobs.created is None


def test_cancel_with_return_minimal_replies_without_a_body(client: TestClient, jobs: _JobsClient) -> None:
    response = client.post(f"{_JOBS}/w1/cancel", headers={"Prefer": "return=minimal"})

    assert response.status_code == 200, response.text
    assert response.content == b""
    assert response.headers["Preference-Applied"] == "return=minimal"
    assert jobs.cancelled == "w1"


def test_cancel_returns_the_whole_job_by_default(client: TestClient) -> None:
    response = client.post(f"{_JOBS}/w1/cancel")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelling"
    assert response.json()["spec"] == {"rows": ["a", "b"]}


def test_status_reports_custom_fields(client: TestClient) -> None:
    response = client.get(f"{_JOBS}/w1/status")

    assert response.status_code == 200, response.text
    assert response.json()["custom_fields"] == _CUSTOM_FIELDS
