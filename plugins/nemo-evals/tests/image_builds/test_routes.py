# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP routes for image builds of a task revision, over the real service and an in-memory store."""

from __future__ import annotations

import asyncio
import re

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from nemo_builder_plugin.entities import IN_CLUSTER_BACKEND
from nemo_evals.api.dependencies import get_image_build_service
from nemo_evals.api.v2 import image_builds as image_builds_routes
from nemo_evals.entities import TaskEntity
from nemo_evals.image_builds.models import (
    BuildImage,
    ImageBuildRequest,
    TaskImageBuildEntity,
    build_entity_name,
    build_job_name,
)
from nemo_evals.image_builds.runner import TaskImageBuildSpec
from nemo_evals.image_builds.service import BuildJob, ImageBuildForbiddenError, ImageBuildService
from nemo_evals.jobs.task_image_build import task_image_build_job_request
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, InternalServerError, PermissionDeniedError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

_BASE = "/v2/workspaces/default/tasks/sort-task/revisions"
# The backend of a deployment that builds through a build service outside the cluster.
REMOTE_BACKEND = "remote"
# Every key a build is returned with, in order. Which backend its images must come from is not one of them.
BUILD_KEYS = [
    "id",
    "task",
    "revision",
    "status",
    "job",
    "allowed_registries",
    "platform",
    "images",
    "error",
    "created_at",
    "updated_at",
]
# Every key a build's record was stored with before it could name the backend its images must come from, in order.
RECORD_KEYS = [
    "task",
    "revision_digest",
    "attempt",
    "status",
    "job",
    "allowed_registries",
    "platform",
    "rebuild",
    "images",
    "error",
]


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


def _routes(service: ImageBuildService) -> TestClient:
    app = FastAPI()
    app.include_router(image_builds_routes.router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_image_build_service] = lambda: service
    return TestClient(app)


@pytest.fixture
def client(entity_store, jobs, authorizer) -> TestClient:
    """The routes of a deployment that names no Builder backend: its task images come from the in-cluster one."""
    return _routes(ImageBuildService(entity_store, jobs, authorizer, builder_backend=IN_CLUSTER_BACKEND))


@pytest.fixture
def remote_client(entity_store, jobs, authorizer) -> TestClient:
    """The routes of a deployment whose task images must come from a remote build service, over the same store."""
    return _routes(ImageBuildService(entity_store, jobs, authorizer, builder_backend=REMOTE_BACKEND))


@pytest.fixture
def untold_client(entity_store, jobs, authorizer) -> TestClient:
    """The routes of a deployment whose Builder backend can't be told, as while Builder's section can't be read."""
    return _routes(ImageBuildService(entity_store, jobs, authorizer, builder_backend=None))


def _store(store_task, **kwargs) -> str:
    return asyncio.run(store_task(**kwargs))


def _stored(entity_store, digest: str, attempt: int) -> TaskImageBuildEntity:
    """The record of one attempt, as the task-builder Job reads it."""

    async def read() -> TaskImageBuildEntity:
        head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
        return await entity_store.get(
            TaskImageBuildEntity, name=build_entity_name(digest, attempt), workspace="default", parent=head.id
        )

    return asyncio.run(read())


def _set_status(
    entity_store, digest: str, attempt: int, status: str, *, images: list[BuildImage] | None = None
) -> None:
    """Settle build ``attempt`` at ``status``, with ``images`` when given, as the task-builder Job does."""

    async def update() -> None:
        head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
        build = await entity_store.get(
            TaskImageBuildEntity, name=build_entity_name(digest, attempt), workspace="default", parent=head.id
        )
        build.status = status
        if images is not None:
            build.images = images
        await entity_store.update(build)

    asyncio.run(update())


def _ready_image(
    *, external_image_id: str | None = None, builder_backend: str | None = None, image_type: str = "main"
) -> BuildImage:
    """A ready image, the main one unless ``image_type`` names another; the other two arguments are what a remote
    build service's build adds to it."""
    return BuildImage(
        type=image_type,
        source="dockerfile",
        status="ready",
        builder_image="harbor-env-abc-1",
        build_job="builder-harbor-env-abc-1",
        image_ref="registry.test/default/harbor-env-abc-1@sha256:" + "c" * 64,
        external_image_id=external_image_id,
        builder_backend=builder_backend,
    )


def _returned(client: TestClient, digest: str) -> list[dict]:
    """Build 1's one image as each route returns it: get, list, and a repeated request."""
    builds = [
        client.get(f"{_BASE}/{digest}/image-builds/1").json(),
        client.get(f"{_BASE}/{digest}/image-builds").json()["data"][0],
        client.post(f"{_BASE}/{digest}/image-builds").json(),
    ]
    return [build["images"][0] for build in builds]


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


@pytest.mark.parametrize(
    ("asking", "answer"), [("client", "1"), ("remote_client", "2")], ids=["same-backend", "switched-to-remote"]
)
def test_a_build_its_job_settles_ready_as_it_is_failed_as_abandoned_answers_only_if_its_backend_matches(
    request, store_task, entity_store, jobs, monkeypatch, asking: str, answer: str
) -> None:
    """The request finds the Job ended with no result and fails the build, while the Job's last write lands first.

    The build then reads ready, with images the cluster built: an answer to a deployment that builds in the
    cluster, and not to one that now builds remotely, which starts the next attempt, as for any ready build.
    """
    asking_client = request.getfixturevalue(asking)
    digest = _store(store_task)
    first = asking_client.post(f"{_BASE}/{digest}/image-builds").json()
    jobs.statuses[first["job"]] = "completed"
    real_update = entity_store.update

    async def settled_first(entity, **kwargs):
        if isinstance(entity, TaskImageBuildEntity) and entity.status == "failed" and entity.attempt == 1:
            monkeypatch.setattr(entity_store, "update", real_update)
            # The Job's write, from the record as the request read it: the request's own write is then stale.
            settled = await entity_store.get(
                TaskImageBuildEntity, name=entity.name, workspace=entity.workspace, parent=entity._parent
            )
            settled.status, settled.images = "ready", [_ready_image()]
            await real_update(settled)
        return await real_update(entity, **kwargs)

    monkeypatch.setattr(entity_store, "update", settled_first)

    again = asking_client.post(f"{_BASE}/{digest}/image-builds").json()

    assert again["id"] == answer
    assert asking_client.get(f"{_BASE}/{digest}/image-builds/1").json()["status"] == "ready"


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


@pytest.mark.parametrize("platform", ["arm64", "linux/arm64/v8/x", "linux/x_y", ""])
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


def test_a_platforms_build_older_than_a_page_of_other_builds_is_reused(
    client, store_task, entity_store, monkeypatch
) -> None:
    """A page holds 2 builds here: the arm64 build is on the second page, behind newer builds for other platforms."""
    monkeypatch.setattr("nemo_evals.image_builds.service._LOOKUP_PAGE_SIZE", 2)
    digest = _store(store_task)
    client.post(f"{_BASE}/{digest}/image-builds", json={"platform": "linux/arm64"})
    _set_status(entity_store, digest, 1, "ready")
    for attempt, platform in enumerate(("linux/amd64", "linux/riscv64", "linux/s390x"), start=2):
        client.post(f"{_BASE}/{digest}/image-builds", json={"platform": platform})
        _set_status(entity_store, digest, attempt, "failed")

    again = client.post(f"{_BASE}/{digest}/image-builds", json={"platform": "linux/arm64"}).json()

    assert (again["id"], again["status"]) == ("1", "ready")


def test_a_platform_is_named_as_buildkit_names_it_before_builds_are_looked_up(client, store_task, entity_store) -> None:
    digest = _store(store_task)
    first = client.post(f"{_BASE}/{digest}/image-builds", json={"platform": "linux/arm64"}).json()
    _set_status(entity_store, digest, 1, "ready")

    again = client.post(f"{_BASE}/{digest}/image-builds", json={"platform": "linux/arm64/v8"}).json()

    assert (first["platform"], again["id"], again["status"]) == ("linux/arm64", "1", "ready")


@pytest.mark.parametrize("platform", ["linux/x86_64", "linux/x86-64", "Linux/AARCH64", "linux/arm64/v8"])
def test_the_request_schema_takes_every_spelling_the_route_does(platform: str) -> None:
    """A client validating against the published schema must not refuse what the API accepts."""
    pattern = _string_pattern(ImageBuildRequest.model_json_schema()["properties"]["platform"])

    assert re.fullmatch(pattern, platform)
    assert ImageBuildRequest(platform=platform).platform in ("linux/amd64", "linux/arm64")


def _string_pattern(schema: dict) -> str:
    return next(option["pattern"] for option in schema.get("anyOf", [schema]) if "pattern" in option)


def test_an_image_this_cluster_built_is_returned_key_for_key_as_before(
    client, store_task, entity_store, build_image_keys
) -> None:
    """An older SDK or CLI refuses an image with a key it doesn't know, so an in-cluster build must gain none."""
    digest = _store(store_task)
    client.post(f"{_BASE}/{digest}/image-builds")
    _set_status(entity_store, digest, 1, "ready", images=[_ready_image()])

    for image in _returned(client, digest):
        assert list(image) == build_image_keys


def test_an_image_a_remote_build_service_built_is_returned_with_its_handle_and_backend(
    remote_client, store_task, entity_store
) -> None:
    digest = _store(store_task)
    remote_client.post(f"{_BASE}/{digest}/image-builds")
    _set_status(
        entity_store,
        digest,
        1,
        "ready",
        images=[_ready_image(external_image_id="image-7f3a", builder_backend="remote")],
    )

    for image in _returned(remote_client, digest):
        assert (image["external_image_id"], image["builder_backend"]) == ("image-7f3a", "remote")


def test_the_published_schema_names_the_handle_and_backend_without_requiring_them(client) -> None:
    """Most images have neither, so a client validating against the schema must take a build without them."""
    schema = client.app.openapi()["components"]["schemas"]["BuildImage"]

    assert {"external_image_id", "builder_backend"} <= set(schema["properties"])
    assert not {"external_image_id", "builder_backend"} & set(schema.get("required", []))


@pytest.mark.parametrize(
    ("before", "after", "answer"),
    [
        ("client", "client", "1"),
        ("remote_client", "remote_client", "1"),
        ("client", "remote_client", "2"),
        ("remote_client", "client", "2"),
    ],
    ids=["in-cluster", "remote", "switched-to-remote", "switched-back"],
)
def test_a_ready_build_answers_a_request_only_while_its_images_come_from_the_deployments_backend(
    request, store_task, entity_store, jobs, before: str, after: str, answer: str
) -> None:
    """A deployment that switches its Builder backend needs each task's images built again, by the new one.

    An image the cluster built names no backend, as on every build recorded before an image could name one.
    """
    requested, repeated = request.getfixturevalue(before), request.getfixturevalue(after)
    digest = _store(store_task)
    requested.post(f"{_BASE}/{digest}/image-builds")
    built_by = REMOTE_BACKEND if before == "remote_client" else None
    _set_status(entity_store, digest, 1, "ready", images=[_ready_image(builder_backend=built_by)])

    again = repeated.post(f"{_BASE}/{digest}/image-builds")

    assert again.status_code == 202
    assert (again.json()["id"], again.json()["status"]) == (answer, "ready" if answer == "1" else "pending")
    assert len(jobs.created) == int(answer)
    # The build that was there stays as it was, whichever build answered.
    assert repeated.get(f"{_BASE}/{digest}/image-builds/1").json()["status"] == "ready"


@pytest.mark.parametrize("asking", ["client", "remote_client"], ids=["in-cluster", "remote"])
def test_a_ready_build_with_one_image_another_backend_built_is_no_answer_to_either_deployment(
    request, store_task, entity_store, jobs, asking: str
) -> None:
    """Every image must come from the deployment's backend: a run takes all of a build's images or none.

    The build holds a main image the cluster built and a verifier image a remote build service built. The other
    backend's image is the last one for the in-cluster deployment and the first one for the remote deployment, so
    a rule that reads one end of the list answers one of the two with the build.
    """
    routes = request.getfixturevalue(asking)
    digest = _store(store_task)
    routes.post(f"{_BASE}/{digest}/image-builds")
    _set_status(
        entity_store,
        digest,
        1,
        "ready",
        images=[
            _ready_image(),
            _ready_image(builder_backend=REMOTE_BACKEND, image_type="verifier"),
        ],
    )

    again = routes.post(f"{_BASE}/{digest}/image-builds").json()

    assert (again["id"], again["status"]) == ("2", "pending")
    assert len(jobs.created) == 2


def test_the_attempt_a_remote_deployment_starts_names_the_backend_on_its_record_for_its_job(
    client, remote_client, store_task, entity_store
) -> None:
    """The task-builder Job reads the record, not the platform config file, so the setting travels there."""
    digest = _store(store_task)
    client.post(f"{_BASE}/{digest}/image-builds")
    _set_status(entity_store, digest, 1, "ready", images=[_ready_image()])

    started = remote_client.post(f"{_BASE}/{digest}/image-builds").json()

    record = _stored(entity_store, digest, int(started["id"]))
    assert (record.attempt, record.expected_builder_backend) == (2, REMOTE_BACKEND)
    assert record.expected_backend == REMOTE_BACKEND
    assert record._get_data_fields()["expected_builder_backend"] == REMOTE_BACKEND
    # The build that answers names it nowhere but on its images, once they are built.
    assert list(started) == BUILD_KEYS


def test_a_deployment_that_names_no_backend_stores_and_returns_a_build_key_for_key_as_before(
    client, store_task, entity_store
) -> None:
    """Its images must come from the in-cluster backend, which neither the record nor the answer spells out."""
    digest = _store(store_task)

    started = client.post(f"{_BASE}/{digest}/image-builds").json()

    record = _stored(entity_store, digest, 1)
    assert (record.expected_builder_backend, record.expected_backend) == (None, IN_CLUSTER_BACKEND)
    assert list(record._get_data_fields()) == RECORD_KEYS
    assert list(started) == BUILD_KEYS
    assert list(client.get(f"{_BASE}/{digest}/image-builds/1").json()) == BUILD_KEYS
    assert list(client.get(f"{_BASE}/{digest}/image-builds").json()["data"][0]) == BUILD_KEYS


def test_a_build_still_running_is_returned_whichever_backend_it_was_requested_for(
    client, remote_client, store_task, jobs
) -> None:
    """A task has one attempt at a time. The switch is seen by the request that follows the attempt's end."""
    digest = _store(store_task)
    first = client.post(f"{_BASE}/{digest}/image-builds").json()

    again = remote_client.post(f"{_BASE}/{digest}/image-builds")

    assert (again.status_code, again.json()["id"]) == (202, first["id"])
    assert list(jobs.created) == [first["job"]]


def test_a_rebuild_starts_a_new_attempt_over_a_ready_build_of_the_deployments_backend(
    remote_client, store_task, entity_store
) -> None:
    digest = _store(store_task)
    remote_client.post(f"{_BASE}/{digest}/image-builds")
    _set_status(entity_store, digest, 1, "ready", images=[_ready_image(builder_backend=REMOTE_BACKEND)])

    rebuilt = remote_client.post(f"{_BASE}/{digest}/image-builds", json={"rebuild": True}).json()

    assert (rebuilt["id"], rebuilt["status"]) == ("2", "pending")
    record = _stored(entity_store, digest, 2)
    assert (record.rebuild, record.expected_builder_backend) == (True, REMOTE_BACKEND)


@pytest.mark.parametrize("built_by", ["client", "remote_client"], ids=["in-cluster", "remote"])
def test_while_the_backend_cant_be_told_a_ready_build_answers_whichever_backend_built_it(
    request, untold_client, store_task, entity_store, jobs, built_by: str
) -> None:
    """Builder refuses every submit then, so an attempt started over the build could have nothing built.

    One that failed would be the task's newest build, and a run takes no failed one.
    """
    digest = _store(store_task)
    request.getfixturevalue(built_by).post(f"{_BASE}/{digest}/image-builds")
    backend = REMOTE_BACKEND if built_by == "remote_client" else None
    _set_status(entity_store, digest, 1, "ready", images=[_ready_image(builder_backend=backend)])

    again = untold_client.post(f"{_BASE}/{digest}/image-builds")

    assert (again.status_code, again.json()["id"], again.json()["status"]) == (202, "1", "ready")
    assert len(jobs.created) == 1


def test_while_the_backend_cant_be_told_an_attempt_names_none_and_its_build_answers_the_next_request(
    untold_client, store_task, entity_store, jobs
) -> None:
    """The attempt's Job expects the in-cluster backend, as for a record that could name none: not one attempt a request."""
    digest = _store(store_task)

    started = untold_client.post(f"{_BASE}/{digest}/image-builds").json()

    record = _stored(entity_store, digest, 1)
    assert (started["id"], record.expected_builder_backend, record.expected_backend) == ("1", None, IN_CLUSTER_BACKEND)
    assert list(record._get_data_fields()) == RECORD_KEYS
    _set_status(entity_store, digest, 1, "ready", images=[_ready_image()])
    assert untold_client.post(f"{_BASE}/{digest}/image-builds").json()["id"] == "1"
    assert len(jobs.created) == 1


def test_a_service_given_no_backend_is_one_whose_backend_cant_be_told(entity_store, jobs, authorizer) -> None:
    """As a run's preparer builds it, only to read builds. The routes' service is given the deployment's backend."""
    assert ImageBuildService(entity_store, jobs, authorizer).builder_backend is None
