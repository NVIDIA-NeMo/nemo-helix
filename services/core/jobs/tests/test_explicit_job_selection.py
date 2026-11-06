# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP conformance against the real Jobs dispatcher and isolated Entity Store."""

from unittest.mock import patch

import pytest
from httpx import AsyncClient
from nemo_helix_plugin.jobs.selectors import JobById, JobByName, LegacyJobSelector
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.core.jobs.api.v2.jobs.schemas import CreateHelixJobRequest, UpdateHelixJobRequest
from nhx.core.jobs.app.dispatcher import JobDispatcher
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.entities import HelixJob

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _job(name: str) -> HelixJob:
    return HelixJob(
        name=name,
        workspace="default",
        source="test-source",
        spec={"kind": "test"},
        platform_spec=TestConstants.PLATFORM_SPEC,
        fileset="test-fileset",
    )


async def test_metadata_resolves_collision_without_attempts(test_client: AsyncClient, mock_store: EntityClient):
    with patch(
        "nhx.core.entities.app.repository.sqlalchemy.entity.generate_entity_id", return_value="platform-job-abcdefghijk"
    ):
        original = await mock_store.create(_job("original"))
    collision = await mock_store.create(_job(original.id))
    path = f"/apis/jobs/v2/workspaces/default/jobs/{original.id}"
    for mode, expected in (("id", original), ("name", collision), ("legacy", original)):
        response = await test_client.get(path + "/metadata", params={"job_reference_kind": mode})
        assert response.status_code == 200, response.text
        assert response.json() == {
            "id": expected.id,
            "name": expected.name,
            "workspace": "default",
            "parent_id": None,
            "source": expected.source,
            "spec": expected.spec,
        }
    assert (await test_client.get(path)).status_code == 404  # existing lifecycle contract
    for params in ({}, {"job_reference_kind": "invalid"}):
        assert (await test_client.get(path + "/metadata", params=params)).status_code == 422
    assert (await test_client.get(path, params={"job_reference_kind": "legacy"})).status_code == 422


async def test_missing_and_foreign_ids_never_fall_back(test_client: AsyncClient, mock_store: EntityClient):
    target = await mock_store.create(_job("missing-id"))
    base = "/apis/jobs/v2/workspaces/default/jobs/missing-id"
    assert (await test_client.get(base + "/metadata", params={"job_reference_kind": "id"})).status_code == 404
    assert (await test_client.get(base + "/metadata", params={"job_reference_kind": "name"})).json()["id"] == target.id
    for suffix in ("cancel", "pause", "resume", "rerun"):
        assert (await test_client.post(base + "/" + suffix, params={"job_reference_kind": "id"})).status_code == 404
    assert (await test_client.delete(base, params={"job_reference_kind": "id"})).status_code == 404
    assert (await mock_store.get_by_id(HelixJob, target.id)).name == "missing-id"
    foreign = await test_client.get(
        f"/apis/jobs/v2/workspaces/system/jobs/{target.id}/metadata", params={"job_reference_kind": "id"}
    )
    assert foreign.status_code == 404


async def test_metadata_does_not_load_dangling_attempt(test_client: AsyncClient, mock_store: EntityClient):
    job = _job("dangling")
    job.current_attempt_id = "missing-attempt"
    job = await mock_store.create(job)
    response = await test_client.get(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/metadata", params={"job_reference_kind": "id"}
    )
    assert response.status_code == 200
    assert "auth_context" not in response.json()
    assert "_auth_context" not in response.json()
    assert "platform_spec" not in response.json()
    assert (await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job.id}")).status_code == 404


async def test_parented_identity_and_root_name_scope(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    parent = await mock_store.create(_job("parent"))
    root = await mock_store.create(_job("same-name"))
    member = _job("same-name")
    member._parent = parent.id
    member = await mock_store.create(member)
    identity = await mock_dispatcher.resolve_job(JobById(member.id), workspace="default")
    assert identity is not None and identity.job_id == member.id and identity.parent_id == parent.id
    for selector in (JobByName("same-name"), LegacyJobSelector("same-name")):
        identity = await mock_dispatcher.resolve_job(selector, workspace="default")
        assert identity is not None and identity.job_id == root.id


async def test_update_reread_cannot_target_replacement(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, sample_platform_job_request: CreateHelixJobRequest
):
    created = await mock_dispatcher.create_job(sample_platform_job_request, "default")
    original_resolve = mock_dispatcher._resolve_job
    replacement_id = ""

    async def replace_after_resolving(selector, workspace):
        nonlocal replacement_id
        resolved = await original_resolve(selector, workspace)
        await mock_store.delete_by_id(HelixJob, created.id)
        replacement = await mock_store.create(_job(created.name))
        replacement_id = replacement.id
        return resolved

    with patch.object(mock_dispatcher, "_resolve_job", side_effect=replace_after_resolving):
        result = await mock_dispatcher.update_job(
            JobById(created.id), "default", UpdateHelixJobRequest(description="wrong")
        )
    assert result is None
    assert (await mock_store.get_by_id(HelixJob, replacement_id)).description != "wrong"
    with pytest.raises(EntityNotFoundError):
        await mock_store.get_by_id(HelixJob, created.id)


@pytest.mark.parametrize(
    "method,suffix,body",
    [
        ("GET", "", None),
        ("PATCH", "", {"description": "changed"}),
        ("DELETE", "", None),
        ("POST", "/cancel", None),
        ("POST", "/pause", None),
        ("POST", "/resume", None),
        ("POST", "/rerun", None),
        ("PATCH", "/control", {"pause_ttl_seconds": 1}),
        ("GET", "/status", None),
        ("GET", "/attempts", None),
        ("GET", "/attempts/current", None),
        ("PATCH", "/attempts/current/status-details", {}),
        ("GET", "/logs", None),
        ("GET", "/results", None),
        ("GET", "/results/result", None),
        (
            "POST",
            "/results/result",
            {"attempt_id": "attempt-id", "artifact_url": "default/fs#path", "artifact_storage_type": "fileset"},
        ),
        ("GET", "/results/result/download", None),
        ("GET", "/steps", None),
        ("GET", "/steps/step", None),
        ("PATCH", "/steps/step/status", {"status": "active"}),
        ("GET", "/attempts/current/steps", None),
        ("GET", "/attempts/current/steps/step", None),
        ("PATCH", "/attempts/current/steps/step/status", {"status": "active"}),
        ("GET", "/steps/step/tasks", None),
        ("GET", "/steps/step/tasks/task", None),
        ("GET", "/attempts/current/steps/step/tasks", None),
        ("GET", "/attempts/current/steps/step/tasks/task", None),
        ("PUT", "/attempts/current/steps/step/tasks/task", {"status": "active"}),
    ],
)
async def test_every_job_route_validates_mode_and_rejects_missing_ids(
    test_client: AsyncClient, method: str, suffix: str, body
):
    path = "/apis/jobs/v2/workspaces/default/jobs/missing-id" + suffix
    invalid = await test_client.request(method, path, params={"job_reference_kind": "invalid"}, json=body)
    assert invalid.status_code == 422, invalid.text
    missing = await test_client.request(method, path, params={"job_reference_kind": "id"}, json=body)
    assert missing.status_code == 404, missing.text


async def test_parented_nested_write_and_listing_retain_same_id(
    test_client: AsyncClient,
    mock_store: EntityClient,
    mock_dispatcher: JobDispatcher,
    sample_platform_job_request: CreateHelixJobRequest,
):
    from nemo_helix_plugin.jobs.schemas import HelixJobStatus
    from nhx.core.jobs.entities import HelixJobAttempt

    request = sample_platform_job_request.model_copy(update={"name": "shared"})
    root = await mock_dispatcher.create_job(request, "default")
    parent = await mock_store.create(_job("parent"))
    member = _job("shared")
    member._parent = parent.id
    member = await mock_store.create(member)
    attempt = await mock_store.create(
        HelixJobAttempt(
            name="member-attempt",
            workspace="default",
            job=member.id,
            seq=0,
            status=HelixJobStatus.CREATED,
            spec=member.spec,
            platform_spec=member.platform_spec,
        )
    )
    member.current_attempt_id = attempt.id
    member = await mock_store.update(member)
    await mock_dispatcher._start_attempt(member, attempt)
    step_name = member.platform_spec.steps[0].name
    step = await mock_dispatcher.get_attempt_step_by_name(JobById(member.id), attempt.id, step_name, "default")
    assert step is not None
    path = f"/apis/jobs/v2/workspaces/default/jobs/{member.id}"
    response = await test_client.put(
        path + f"/attempts/{attempt.id}/steps/{step_name}/tasks/task",
        params={"job_reference_kind": "id"},
        json={"status": "active"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["step_id"] == step.id
    root_step = await mock_dispatcher.get_current_job_step_by_name(JobById(root.id), step_name, "default")
    assert root_step is not None and await mock_dispatcher.list_tasks(root_step.id, "default") == []
    page = await test_client.get(path + "/steps", params={"job_reference_kind": "id"})
    assert page.status_code == 200 and [item["id"] for item in page.json()["data"]] == [step.id]
    wrong_chain = await test_client.put(
        path + f"/attempts/{root.attempt_id}/steps/{step_name}/tasks/task",
        params={"job_reference_kind": "id"},
        json={"status": "active"},
    )
    assert wrong_chain.status_code == 404


async def test_generated_guard_uses_metadata_http_and_pins_caller_mutations(
    test_client: AsyncClient,
    mock_store: EntityClient,
    mock_dispatcher: JobDispatcher,
    sample_platform_job_request: CreateHelixJobRequest,
):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from httpx import ASGITransport, Request
    from nemo_helix_plugin.client.client import AsyncNemoClient
    from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client, get_request_authorizer
    from nemo_helix_plugin.jobs.api_factory import job_route_factory
    from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers
    from nhx.core.entities.app.repository.sqlalchemy.entity import generate_entity_id
    from pydantic import BaseModel

    class Spec(BaseModel):
        kind: str

    def compile_spec(*_args: object, **_kwargs: object):
        return TestConstants.PLATFORM_SPEC

    request = sample_platform_job_request.model_copy(
        update={"name": "original", "source": "test-source", "spec": {"kind": "test", "_nemo_job_route": "typed"}}
    )
    with patch(
        "nhx.core.entities.app.repository.sqlalchemy.entity.generate_entity_id",
        side_effect=lambda kind: "platform-job-abcdefghijk" if kind == "platform_job" else generate_entity_id(kind),
    ):
        original = await mock_dispatcher.create_job(request, "default")
    collision = await mock_dispatcher.create_job(request.model_copy(update={"name": original.id}), "default")
    caller = AsyncNemoClient(
        base_url=str(test_client.base_url), http_client=test_client, default_headers={"X-Test-Credential": "caller"}
    )
    service = caller.with_headers({"X-Test-Credential": "service"})
    denied = False
    replace_target = False
    replacement_id = ""
    calls: list[tuple[str, str, str | None, dict[str, str]]] = []

    async def record(req: Request):
        calls.append((req.method, req.url.path, req.headers.get("X-Test-Credential"), dict(req.url.params)))

    test_client.event_hooks["request"].append(record)

    async def authorize(method: str, path: str):
        if denied:
            raise HTTPException(status_code=403, detail="Denied before privileged lookup")
        assert method == "PATCH" and path.startswith("/apis/jobs/v2/workspaces/default/jobs/")

    @asynccontextmanager
    async def ownership_client(*, as_service: str, internal: bool, workspace: str):
        nonlocal replacement_id
        assert (as_service, internal, workspace) == ("test-source", True, "default")
        yield service
        if replace_target:
            await mock_store.delete_by_id(HelixJob, collision.id)
            replacement = await mock_store.create(_job(collision.name))
            replacement_id = replacement.id

    app = FastAPI()
    register_sdk_exception_handlers(app)
    app.include_router(
        job_route_factory(
            service_name="test-source",
            job_type="Typed",
            job_input=Spec,
            platform_job_config_compiler=compile_spec,
            job_discriminator="typed",
        ),
        prefix="/typed/{workspace}",
    )
    app.dependency_overrides[get_nemo_client] = lambda: caller
    app.dependency_overrides[get_entity_client] = lambda: mock_store
    app.dependency_overrides[get_request_authorizer] = lambda: authorize
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://plugin") as plugin:
        with patch("nemo_helix_plugin.jobs.api_factory.get_async_nemo_client", side_effect=ownership_client):
            path = f"/typed/default/jobs/{original.id}"
            response = await plugin.patch(path, params={"job_reference_kind": "name"}, json={"description": "by-name"})
            assert response.status_code == 200, response.text
            assert response.json()["id"] == collision.id
            assert calls[-2][2:] == ("service", {"job_reference_kind": "name"})
            assert calls[-1][2:] == ("caller", {"job_reference_kind": "id"})
            assert (await mock_store.get_by_id(HelixJob, original.id)).description != "by-name"
            response = await plugin.patch(path, json={"description": "legacy-id-first"})
            assert response.status_code == 200 and response.json()["id"] == original.id
            assert calls[-2][3] == {"job_reference_kind": "legacy"}
            denied = True
            before = len(calls)
            assert (await plugin.patch(path, json={"description": "denied"})).status_code == 403
            assert len(calls) == before
            denied = False
            unstarted = _job("not-started")
            unstarted.spec["_nemo_job_route"] = "typed"
            unstarted = await mock_store.create(unstarted)
            response = await plugin.patch(
                f"/typed/default/jobs/{unstarted.id}",
                params={"job_reference_kind": "id"},
                json={"description": "not-run"},
            )
            assert response.status_code == 409, response.text  # metadata succeeds; core lifecycle still rejects
            replace_target = True
            response = await plugin.patch(
                f"/typed/default/jobs/{collision.id}",
                params={"job_reference_kind": "id"},
                json={"description": "wrong-replacement"},
            )
            assert response.status_code == 404, response.text
            assert (await mock_store.get_by_id(HelixJob, replacement_id)).description != "wrong-replacement"


async def test_nested_task_write_fails_if_original_id_disappears_before_locked_reread(
    test_client: AsyncClient,
    mock_store: EntityClient,
    mock_dispatcher: JobDispatcher,
    sample_platform_job_request: CreateHelixJobRequest,
):
    from fastapi import FastAPI
    from httpx import ASGITransport
    from nemo_helix_plugin.jobs.types import HelixJobTaskUpdate
    from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers
    from nhx.core.jobs.entities import HelixJobStep

    transport = test_client._transport
    assert isinstance(transport, ASGITransport) and isinstance(transport.app, FastAPI)
    register_sdk_exception_handlers(transport.app)
    original = await mock_dispatcher.create_job(sample_platform_job_request, "default")
    step_name = original.platform_spec.steps[0].name
    step = await mock_dispatcher.get_attempt_step_by_name(
        JobById(original.id), original.attempt_id, step_name, "default"
    )
    assert step is not None
    locked_write = mock_dispatcher._create_or_update_task_locked
    replacement_id = ""

    async def replace_before_reread(
        job_id: str, task_name: str, workspace: str, update: HelixJobTaskUpdate, selected_step: HelixJobStep
    ):
        nonlocal replacement_id
        assert job_id == original.id
        await mock_store.delete_by_id(HelixJob, original.id)
        replacement_id = (await mock_store.create(_job(original.name))).id
        return await locked_write(job_id, task_name, workspace, update, selected_step)

    with patch.object(mock_dispatcher, "_create_or_update_task_locked", side_effect=replace_before_reread):
        response = await test_client.put(
            f"/apis/jobs/v2/workspaces/default/jobs/{original.id}/attempts/{original.attempt_id}/steps/{step_name}/tasks/task",
            params={"job_reference_kind": "id"},
            json={"status": "active"},
        )
    assert response.status_code == 404, response.text
    assert replacement_id != original.id
    assert await mock_dispatcher.list_tasks(step.id, "default") == []
    assert (await mock_store.get_by_id(HelixJob, replacement_id)).current_attempt_id is None
