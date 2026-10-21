# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP routes for image builds of a task revision, over the real service and an in-memory store."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from nemo_evals.api.dependencies import get_image_build_service
from nemo_evals.api.v2 import image_builds as image_builds_routes
from nemo_evals.entities import TaskEntity
from nemo_evals.image_builds.models import TaskImageBuildEntity, build_entity_name, build_job_name
from nemo_evals.image_builds.runner import TaskImageBuildSpec
from nemo_evals.image_builds.service import BuildJob, ImageBuildForbiddenError, ImageBuildService
from nemo_evals.jobs.task_image_build import task_image_build_job_request
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, InternalServerError, PermissionDeniedError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

_BASE = "/v2/workspaces/default/tasks/sort-task/revisions"


class _Jobs:
    """Jobs that, like the real service, refuse a second Job under a taken name."""

    def __init__(self) -> None:
        self.created: dict[str, CreateHelixJobRequest] = {}
        self.statuses: dict[str, str] = {}
        self.refuse: Exception | None = None

    async def create(self, request: CreateHelixJobRequest) -> None:
        if self.refuse is not None:
            raise self.refuse
        assert request.name is not None
        if request.name in self.created:
            raise ConflictError(httpx.Response(409, json={"detail": "job exists"}))
        self.created[request.name] = request

    async def get(self, name: str) -> BuildJob | None:
        if name not in self.created:
            return None
        return BuildJob(status=self.statuses.get(name, "active"), custom_fields=self.created[name].custom_fields)


@pytest.fixture
def jobs() -> _Jobs:
    return _Jobs()


class _Authorizer:
    def __init__(self) -> None:
        self.allowed = True
        self.checked: list[str] = []

    async def __call__(self, workspace: str) -> None:
        self.checked.append(workspace)
        if not self.allowed:
            raise ImageBuildForbiddenError(f"You may not submit container image builds in workspace {workspace}")


@pytest.fixture
def authorizer() -> _Authorizer:
    return _Authorizer()


@pytest.fixture
def client(entity_store, jobs, authorizer) -> TestClient:
    app = FastAPI()
    app.include_router(image_builds_routes.router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_image_build_service] = lambda: ImageBuildService(entity_store, jobs, authorizer)
    return TestClient(app)


def _store(store_task, **kwargs) -> str:
    return asyncio.run(store_task(**kwargs))


def _set_status(entity_store, digest: str, attempt: int, status: str) -> None:
    async def update() -> None:
        head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
        build = await entity_store.get(
            TaskImageBuildEntity, name=build_entity_name(digest, attempt), workspace="default", parent=head.id
        )
        build.status = status
        await entity_store.update(build)

    asyncio.run(update())


def test_request_starts_a_build_and_its_job(client, store_task, jobs) -> None:
    digest = _store(store_task)

    response = client.post(f"{_BASE}/{digest}/image-builds", json={"allowed_registries": ["docker.io"]})

    assert response.status_code == 202
    body = response.json()
    assert body["id"] == "1"
    assert body["task"] == "default/sort-task"
    assert body["revision"] == digest
    assert body["status"] == "pending"
    assert body["allowed_registries"] == ["docker.io"]
    assert body["job"] == build_job_name("default/sort-task", digest, 1)
    request = jobs.created[body["job"]]
    step = request.platform_spec.steps[0]
    assert step.executor.container.command == ["nemo_evals.tasks.task_image_build"]
    assert step.executor.resources.requests.memory == "128Mi"
    assert request.spec == {"workspace": "default", "task": "sort-task", "revision_digest": digest, "attempt": 1}


def test_a_caller_who_may_not_build_is_forbidden_before_anything_starts(
    client, store_task, jobs, authorizer, entity_store
) -> None:
    """Builder trusts evals' own permissions, so evals must check its caller's before a Job builds as them."""
    digest = _store(store_task)
    authorizer.allowed = False

    response = client.post(f"{_BASE}/{digest}/image-builds")

    assert response.status_code == 403
    assert authorizer.checked == ["default"]
    assert jobs.created == {}
    assert client.get(f"{_BASE}/{digest}/image-builds").json()["data"] == []


async def test_the_build_check_authorizes_builders_submit_route_as_the_caller(entity_store) -> None:
    checked: list[tuple[str, str]] = []

    async def authorize(method: str, path: str) -> None:
        checked.append((method, path))
        raise HTTPException(status_code=403, detail="Access to taskset member denied")

    service = get_image_build_service(
        "team a", entity_store, AsyncNemoClient(base_url="http://platform.test"), authorize
    )

    with pytest.raises(ImageBuildForbiddenError, match="workspace team a"):
        await service.authorize_builds("team a")
    assert checked == [("POST", "/apis/builder/v2/workspaces/team%20a/builds")]


def test_request_by_tag_builds_the_revision_the_tag_names(client, store_task) -> None:
    digest = _store(store_task)

    response = client.post(f"{_BASE}/latest/image-builds")

    assert response.status_code == 202
    assert response.json()["revision"] == digest


def test_repeating_a_request_returns_the_running_build(client, store_task, jobs) -> None:
    digest = _store(store_task)
    first = client.post(f"{_BASE}/{digest}/image-builds").json()

    again = client.post(f"{_BASE}/{digest}/image-builds")

    assert again.status_code == 202
    assert again.json()["id"] == first["id"]
    assert list(jobs.created) == [first["job"]]


def test_a_ready_build_is_returned_rather_than_rebuilt(client, store_task, entity_store) -> None:
    digest = _store(store_task)
    client.post(f"{_BASE}/{digest}/image-builds")
    _set_status(entity_store, digest, 1, "ready")

    again = client.post(f"{_BASE}/{digest}/image-builds")

    assert (again.json()["id"], again.json()["status"]) == ("1", "ready")


def test_after_a_failed_build_the_next_attempt_starts(client, store_task, entity_store, jobs) -> None:
    digest = _store(store_task)
    client.post(f"{_BASE}/{digest}/image-builds")
    _set_status(entity_store, digest, 1, "failed")

    retry = client.post(f"{_BASE}/{digest}/image-builds").json()

    assert retry["id"] == "2"
    assert retry["job"] == build_job_name("default/sort-task", digest, 2)
    assert len(jobs.created) == 2


def test_a_build_whose_job_ended_without_a_result_is_failed_and_retried(client, store_task, entity_store, jobs) -> None:
    """A task-builder pod that dies can't record its failure; the next request must not return it forever."""
    digest = _store(store_task)
    first = client.post(f"{_BASE}/{digest}/image-builds").json()
    jobs.statuses[first["job"]] = "error"

    retry = client.post(f"{_BASE}/{digest}/image-builds").json()

    assert retry["id"] == "2"
    abandoned = client.get(f"{_BASE}/{digest}/image-builds/1").json()
    assert abandoned["status"] == "failed"
    assert abandoned["error"] == f"task-builder job {first['job']} ended (error) without a result"


def test_a_lost_job_is_created_again_on_the_next_request(client, store_task, jobs) -> None:
    digest = _store(store_task)
    jobs.refuse = InternalServerError(httpx.Response(500, json={"detail": "jobs is down"}))
    assert client.post(f"{_BASE}/{digest}/image-builds").status_code == 502

    jobs.refuse = None
    response = client.post(f"{_BASE}/{digest}/image-builds")

    assert response.status_code == 202
    assert response.json()["id"] == "1"
    assert list(jobs.created) == [response.json()["job"]]


def test_a_caller_jobs_refuses_is_forbidden(client, store_task, jobs) -> None:
    digest = _store(store_task)
    jobs.refuse = PermissionDeniedError(httpx.Response(403, json={"detail": "may not create jobs"}))

    response = client.post(f"{_BASE}/{digest}/image-builds")

    assert response.status_code == 403
    assert "may not create jobs" in response.json()["detail"]


def test_a_job_name_held_by_another_job_is_a_conflict(client, store_task, jobs) -> None:
    digest = _store(store_task)
    name = build_job_name("default/sort-task", digest, 1)
    other = TaskImageBuildSpec(workspace="default", task="other-task", revision_digest=digest, attempt=1)
    jobs.created[name] = task_image_build_job_request(other, name=name)

    response = client.post(f"{_BASE}/{digest}/image-builds")

    assert response.status_code == 409


@pytest.mark.parametrize(
    ("task", "detail"),
    [
        ({"harbor": False}, "not a Harbor task"),
        ({"config": {"environment": {"os": "windows"}}}, "windows_unsupported"),
    ],
)
def test_tasks_that_cannot_be_built_are_rejected(client, store_task, task: dict, detail: str) -> None:
    digest = _store(store_task, **task)

    response = client.post(f"{_BASE}/{digest}/image-builds")

    assert response.status_code == 422
    assert detail in response.json()["detail"]


@pytest.mark.parametrize("method", ["post", "get"])
def test_unknown_task_or_revision_is_not_found(client, store_task, method: str) -> None:
    _store(store_task)

    assert getattr(client, method)(f"{_BASE}/{'c' * 64}/image-builds").status_code == 404
    unknown_task = "/v2/workspaces/default/tasks/nope/revisions/latest/image-builds"
    assert getattr(client, method)(unknown_task).status_code == 404


def test_list_and_get_builds(client, store_task, entity_store) -> None:
    digest = _store(store_task)
    client.post(f"{_BASE}/{digest}/image-builds")
    _set_status(entity_store, digest, 1, "failed")
    client.post(f"{_BASE}/{digest}/image-builds")

    listed = client.get(f"{_BASE}/{digest}/image-builds")
    one = client.get(f"{_BASE}/{digest}/image-builds/1")

    assert listed.status_code == 200
    assert [(b["id"], b["status"]) for b in listed.json()["data"]] == [("2", "pending"), ("1", "failed")]
    assert listed.json()["pagination"]["total_results"] == 2
    assert (one.status_code, one.json()["status"]) == (200, "failed")
    assert client.get(f"{_BASE}/{digest}/image-builds/3").status_code == 404
    assert client.get(f"{_BASE}/{digest}/image-builds/latest").status_code == 404


def test_a_build_for_another_platform_is_not_reused(client, store_task, entity_store) -> None:
    digest = _store(store_task)
    client.post(f"{_BASE}/{digest}/image-builds", json={"platform": "linux/amd64"})
    _set_status(entity_store, digest, 1, "ready")

    arm = client.post(f"{_BASE}/{digest}/image-builds", json={"platform": "linux/arm64"}).json()
    amd = client.post(f"{_BASE}/{digest}/image-builds", json={"platform": "linux/amd64"}).json()
    again = client.post(f"{_BASE}/{digest}/image-builds", json={"platform": "linux/arm64"}).json()

    assert (arm["id"], arm["platform"], arm["status"]) == ("2", "linux/arm64", "pending")
    assert (amd["id"], amd["platform"], amd["status"]) == ("1", "linux/amd64", "ready")
    assert again["id"] == "2"


@pytest.mark.parametrize("platform", ["arm64", "linux/ARM64", ""])
def test_a_platform_that_is_not_os_and_architecture_is_rejected(client, store_task, platform: str) -> None:
    digest = _store(store_task)

    response = client.post(f"{_BASE}/{digest}/image-builds", json={"platform": platform})

    assert response.status_code == 422


def test_rebuild_starts_a_new_attempt_over_a_ready_build(client, store_task, entity_store) -> None:
    """A ready build may be for a previous default platform; ``rebuild`` takes the request to Builder again."""
    digest = _store(store_task)
    client.post(f"{_BASE}/{digest}/image-builds")
    _set_status(entity_store, digest, 1, "ready")

    rebuilt = client.post(f"{_BASE}/{digest}/image-builds", json={"rebuild": True}).json()
    again = client.post(f"{_BASE}/{digest}/image-builds", json={"rebuild": True}).json()
    plain = client.post(f"{_BASE}/{digest}/image-builds").json()

    assert (rebuilt["id"], rebuilt["status"]) == ("2", "pending")
    # A rebuild that's still running is the answer: it isn't started twice.
    assert again["id"] == "2" and plain["id"] == "2"
