# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP outcomes, concurrent acceptance and route registration for taskset runs."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from nemo_evals.api.dependencies import get_taskset_run_service
from nemo_evals.api.v2.taskset_runs import router
from nemo_evals.harbor.remote.preparation import ImagePreparationUnavailableError
from nemo_evals.jobs.agent_spec import registered_agent_source
from nemo_evals.service import EvaluatorPluginService
from nemo_evals.taskset_runs import store as store_module
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_evals.taskset_runs.store import PlanStorageError, TasksetRunStore
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entities.types import Entity
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError

PATH = "/apis/evals/v2/workspaces/default/agent-evaluate/runs"


@pytest.fixture
def run_app(service):
    app = FastAPI()
    app.include_router(router, prefix="/apis/evals/v2/workspaces/{workspace}")
    app.dependency_overrides[get_taskset_run_service] = lambda: service
    app.dependency_overrides[get_effective_principal_id] = lambda: "user:alice"
    return app


@pytest.fixture
async def run_client(run_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=run_app), base_url="http://test") as client:
        yield client


async def test_taskset_run_create_replay_and_conflict(run_client, published, request_body, files_client, files_service):
    first = await run_client.post(PATH, json=request_body)
    assert first.status_code == 201, first.text
    assert first.json()["phase"] == "accepted" and first.json()["result_state"] == "pending"
    assert first.json()["created_by"] == "user:alice"
    assert first.json()["warnings"] == [f"task 0 (task-1): Job {first.json()['run_id']}-task-0"]
    assert not {"plan_ref", "plan_digest", "request_digest", "db_version"} & first.json().keys()
    before = list(files_service.calls)
    replay = await run_client.post(PATH, json=request_body)
    assert replay.status_code == 200 and replay.json() == first.json()
    assert not [call for call in files_service.calls[len(before) :] if call[0] != "download"]
    request_body["timeout_seconds"] = 1
    conflict = await run_client.post(PATH, json=request_body)
    assert conflict.status_code == 409
    assert conflict.headers["x-should-retry"] == "false"
    assert (
        conflict.json()["detail"]
        == f"run {first.json()['run_id']} exists for this idempotency_key with a different request"
    )


async def test_taskset_run_resent_with_other_sandbox_overrides_conflicts(run_client, published, request_body):
    request_body["execution"]["sandbox"] = {"cpus": 2}
    first = await run_client.post(PATH, json=request_body)
    assert first.status_code == 201, first.text
    request_body["execution"]["sandbox"] = {"cpus": 4}
    conflict = await run_client.post(PATH, json=request_body)
    assert conflict.status_code == 409, conflict.text


async def test_taskset_run_sandbox_override_over_the_disk_limit_is_422(run_client, published, request_body):
    request_body["execution"]["sandbox"] = {"storage_mb": 262_144}
    response = await run_client.post(PATH, json=request_body)
    assert response.status_code == 422, response.text
    assert "execution.sandbox.storage_mb=262144 exceeds" in response.json()["detail"]


async def test_taskset_run_timeout_validation_shape_matches_invalid_body(
    run_client, request_body, files_client, files_service, authorizer
):
    invalid = await run_client.post(PATH, json={**request_body, "timeout_seconds": 0})
    over_max = await run_client.post(PATH, json={**request_body, "timeout_seconds": 259_201})
    assert invalid.status_code == over_max.status_code == 422
    for response in (invalid, over_max):
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "timeout_seconds"]
        assert {"type", "loc", "msg", "input"} <= error.keys()
    assert not files_service.calls
    authorizer.assert_not_awaited()


async def test_taskset_run_source_denied(run_client, request_body, authorizer, files_client, files_service):
    authorizer.side_effect = HTTPException(status_code=403, detail="source denied")
    response = await run_client.post(PATH, json=request_body)
    assert response.status_code == 403 and response.json()["detail"] == "source denied"
    assert not files_service.calls


@pytest.mark.parametrize("change", ["missing", "empty", "non-harbor"])
async def test_taskset_run_invalid_source_returns_422(run_client, request_body, published, entity_store, change):
    from nemo_evals.api.fields import TaskRef
    from nemo_evals.api.task_definitions.evaluator import EvaluatorTaskDefinition
    from nemo_evals.entities import TaskEntity, TaskRevisionEntity, TasksetRevisionEntity
    from nemo_evals.revisions import publish_revision

    head, _ = published
    if change == "missing":
        request_body["spec"]["tasks"] = "missing"
    else:
        head.tasks = []
        if change == "non-harbor":
            definition = EvaluatorTaskDefinition(kind="evaluator", intent="hello")
            task = await entity_store.create(TaskEntity(name="row", workspace="default", spec=definition))
            revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
            head.tasks = [TaskRef(f"default/row#{revision.content_hash}")]
        await entity_store.update(head)
        await publish_revision(entity_store, entity_store, head, TasksetRevisionEntity)
    response = await run_client.post(PATH, json=request_body)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("error", [RuntimeError("sensitive outage"), PlanStorageError("secret upload")])
async def test_taskset_run_unexpected_failure_is_sanitized(run_client, service, monkeypatch, request_body, error):
    monkeypatch.setattr(service, "submit", AsyncMock(side_effect=error))
    response = await run_client.post(PATH, json=request_body)
    assert response.status_code == 500 and response.json() == {"detail": "Internal server error"}
    assert "x-idempotency-key" not in response.headers


@pytest.mark.parametrize("different_body", [False, True])
async def test_taskset_run_concurrent_http_acceptance(
    run_client, published, registered_agent, request_body, files_client, files_service, entity_store, different_body
):
    from nemo_evals.taskset_runs.entities import TasksetRunEntity, TasksetRunMemberEntity

    barrier = asyncio.Event()
    uploads = 0

    async def after_upload():
        nonlocal uploads
        # Only plan uploads gate entity creation; agent snapshots precede them.
        _, workspace, name = files_service.calls[-1]
        if "plan.json" not in files_service.filesets[(workspace, name)]:
            return
        uploads += 1
        if uploads == 2:
            barrier.set()
        await barrier.wait()

    files_service.after_upload = after_upload
    second = {**request_body, "timeout_seconds": 1} if different_body else request_body
    async with asyncio.timeout(10):
        responses = await asyncio.gather(run_client.post(PATH, json=request_body), run_client.post(PATH, json=second))
    assert sorted(response.status_code for response in responses) == ([201, 409] if different_body else [200, 201])
    runs = (await entity_store.list(TasksetRunEntity, workspace="default")).data
    assert len(runs) == 1 and len(files_service.filesets) == 2
    plans = [files for (_, name), files in files_service.filesets.items() if "plan.json" in files]
    snapshots = [files for (ws, name), files in files_service.filesets.items() if name.startswith("agent-files-")]
    assert len(plans) == len(snapshots) == 1
    assert uploads == 2
    assert snapshots[0] == {"skills/coding/SKILL.md": b"# Coding instructions\n"}
    assert not (await entity_store.list(TasksetRunMemberEntity, workspace="default")).data


def test_taskset_run_router_is_registered_once():
    routes = [
        route
        for spec in EvaluatorPluginService().get_routers()
        for route in spec.router.routes
        if isinstance(route, APIRoute) and route.path == "/agent-evaluate/runs"
    ]
    assert len(routes) == 2
    assert any(route.methods == {"POST"} for route in routes)
    assert any(route.methods == {"GET"} for route in routes)


@pytest.mark.parametrize("mode", ["recover", "committed-exhausted", "uncommitted-exhausted"])
async def test_taskset_run_lost_create_reply_through_production_store(
    run_client,
    service,
    published,
    registered_agent,
    request_body,
    files_client,
    files_service,
    entity_store,
    monkeypatch,
    mode,
):
    """Exercise retries on the real raw client and retain uncertain resources after exhaustion."""
    posts = []
    healthy = False
    monkeypatch.setattr(
        store_module, "_ACCEPT_RETRY_POLICY", replace(store_module._ACCEPT_RETRY_POLICY, backoff_base=0)
    )

    def response_entity(run):
        return Entity(
            entity_type=run.__entity_type__,
            name=run.name,
            workspace=run.workspace,
            id=run.entity_id,
            created_by=run.created_by,
            created_at=run.created_at,
            updated_at=run.updated_at,
            db_version=run.db_version,
            data=run.model_dump(exclude=run.__base_fields__, exclude_computed_fields=True),
        ).model_dump(mode="json")

    async def request_handler(request):
        if request.method == "POST":
            posts.append(request)
            body = json.loads(request.content)
            if not healthy and mode.endswith("exhausted"):
                if mode == "committed-exhausted" and len(posts) == 1:
                    await entity_store.create(TasksetRunEntity(name=body["name"], workspace="default", **body["data"]))
                return httpx.Response(503, json={"detail": "reply unavailable"})
            try:
                run = await entity_store.create(
                    TasksetRunEntity(name=body["name"], workspace="default", **body["data"])
                )
            except NemoEntityConflictError:
                return httpx.Response(409, json={"detail": "exists"})
            if not healthy and mode == "recover":
                return httpx.Response(503, json={"detail": "lost committed reply"})
            return httpx.Response(201, json=response_entity(run))
        if request.url.path.endswith("/taskset_run"):
            return httpx.Response(
                200,
                json={
                    "data": [],
                    "pagination": {
                        "page": 1,
                        "page_size": 100,
                        "current_page_size": 0,
                        "total_pages": 1,
                        "total_results": 0,
                    },
                },
            )
        try:
            run = await entity_store.get(TasksetRunEntity, request.url.path.rsplit("/", 1)[1], workspace="default")
        except NemoEntityNotFoundError:
            return httpx.Response(404, json={"detail": "missing"})
        return httpx.Response(200, json=response_entity(run))

    async with httpx.AsyncClient(transport=httpx.MockTransport(request_handler)) as http_client:
        async with AsyncNemoClient(base_url="http://entities.test", http_client=http_client) as client:
            service._store = TasksetRunStore(AsyncEntitiesClient.from_client(client), files_client)
            first = await run_client.post(PATH, json=request_body)
            assert first.status_code == (201 if mode == "recover" else 500), first.text
            assert len(files_service.filesets) == 2
            assert len(posts) == (2 if mode == "recover" else 3)
            healthy = True
            second = await run_client.post(PATH, json=request_body)
            assert second.status_code == (201 if mode == "uncommitted-exhausted" else 200), second.text
            runs = (await entity_store.list(TasksetRunEntity, workspace="default")).data
            assert len(runs) == 1
            plan = await service._store.load_plan(runs[0])
            assert plan.taskset_ref == f"default/suite#{published[1].content_hash}"
            source = registered_agent_source(plan.spec.target)
            assert source is not None and source.files is not None
            snapshot_ref = source.files.root
            assert tuple(snapshot_ref.split("/")) in files_service.filesets
            if mode != "uncommitted-exhausted":
                assert len(files_service.filesets) == 2


async def test_taskset_run_missing_task_job_is_retryable_503(run_client, published, request_body, fake_task_jobs):
    fake_task_jobs.refuse = {"run-11f77762178b0e05-task-0"}
    response = await run_client.post(PATH, json=request_body)
    assert response.status_code == 503, response.text
    assert response.headers["x-should-retry"] == "true"
    assert "task 0" in response.json()["detail"]
    assert response.json()["detail"].endswith("retry the same submit with its idempotency_key")
    fake_task_jobs.refuse = set()
    retry = await run_client.post(PATH, json=request_body)
    assert retry.status_code == 200 and len(fake_task_jobs.created) == 1


@pytest.mark.parametrize("key", ["ci-run", None], ids=["keyed", "keyless"])
async def test_intake_failing_before_acceptance_is_a_retryable_503(
    run_client, published, request_body, fake_evaluations, entity_store, key
):
    body = request_body | {"idempotency_key": key, "publication": {"intake": {"experiment": "swe-agent-tuning"}}}
    fake_evaluations.fail = {"experiment"}
    response = await run_client.post(PATH, json=body)
    assert response.status_code == 503, response.text
    # Nothing was accepted yet, so even a keyless retry can't start a second run.
    assert response.headers["x-should-retry"] == "true"
    assert "nothing was accepted" in response.json()["detail"]
    assert not (await entity_store.list(TasksetRunEntity, workspace="default")).data
    fake_evaluations.fail = set()
    assert (await run_client.post(PATH, json=body)).status_code == 201


@pytest.mark.parametrize("key", ["ci-run", None], ids=["keyed", "keyless"])
async def test_an_evaluation_denied_after_acceptance_is_a_nonretryable_403_naming_the_run(
    run_client, published, request_body, fake_evaluations, fake_task_jobs, key
):
    body = request_body | {"idempotency_key": key, "publication": {"intake": {"experiment": "swe-agent-tuning"}}}
    fake_evaluations.fail = {"create-denied"}
    response = await run_client.post(PATH, json=body)
    assert response.status_code == 403, response.text
    assert response.headers["x-should-retry"] == "false"
    used = key or response.headers["x-idempotency-key"]
    detail = response.json()["detail"]
    assert detail.startswith("run run-") and detail.endswith(f"resubmit with idempotency_key={used}")
    assert not fake_task_jobs.created
    fake_evaluations.fail = set()
    retry = await run_client.post(PATH, json=body | {"idempotency_key": used})
    assert retry.status_code == 200, retry.text
    assert detail.startswith(f"run {retry.json()['run_id']}:") and len(fake_task_jobs.created) == 1


async def test_an_unreadable_experiment_is_403(run_client, published, request_body, fake_evaluations):
    fake_evaluations.deny_experiments = True
    body = request_body | {"publication": {"intake": {"experiment": "swe-agent-tuning"}}}
    response = await run_client.post(PATH, json=body)
    assert response.status_code == 403, response.text
    assert "swe-agent-tuning" in response.json()["detail"]


class _Everything(set):
    def __contains__(self, item: object) -> bool:
        return True


async def test_keyless_missing_task_job_503_names_the_generated_key(
    run_client, published, request_body, fake_task_jobs
):
    """The SDK would resend the keyless request and accept a second run, so the 503 names the key instead."""
    fake_task_jobs.refuse = _Everything()
    response = await run_client.post(PATH, json=request_body | {"idempotency_key": None})
    assert response.status_code == 503, response.text
    assert response.headers["x-should-retry"] == "false"
    key = response.headers["x-idempotency-key"]
    assert response.json()["detail"].endswith(f"retry with idempotency_key={key}")
    fake_task_jobs.refuse = set()
    retry = await run_client.post(PATH, json=request_body | {"idempotency_key": key})
    assert retry.status_code == 200, retry.text
    assert retry.json()["idempotency_key"] == key
    assert {job.name for job in fake_task_jobs.created} == {f"{retry.json()['run_id']}-task-0"}


async def test_keyless_failure_after_acceptance_names_the_generated_key(
    run_client, service, published, request_body, entity_store, monkeypatch
):
    monkeypatch.setattr(service._store, "load_plan", AsyncMock(side_effect=RuntimeError("secret read failure")))
    response = await run_client.post(PATH, json=request_body | {"idempotency_key": None})
    assert response.status_code == 500, response.text
    key = response.headers["x-idempotency-key"]
    assert response.json() == {"detail": f"Internal server error; retry with idempotency_key={key}"}
    assert response.headers["x-should-retry"] == "false"
    (run,) = (await entity_store.list(TasksetRunEntity, workspace="default")).data
    assert run.idempotency_key == key


@pytest.mark.parametrize("key", ["ci-run", None])
async def test_an_unavailable_build_service_is_a_retryable_503(run_client, service, monkeypatch, request_body, key):
    """Raised before acceptance, so resending the same request is safe even without a key."""
    error = ImagePreparationUnavailableError("could not start an image build for task task-1: jobs is down")
    monkeypatch.setattr(service, "submit", AsyncMock(side_effect=error))
    response = await run_client.post(PATH, json=request_body | {"idempotency_key": key})
    assert response.status_code == 503 and response.headers["x-should-retry"] == "true"
    assert response.json() == {"detail": str(error)} and "x-idempotency-key" not in response.headers


async def test_keyless_evaluation_failure_names_the_generated_key(
    run_client, published, request_body, fake_evaluations, fake_task_jobs
):
    body = request_body | {"idempotency_key": None, "publication": {"intake": {"experiment": "swe-agent-tuning"}}}
    fake_evaluations.fail = {"create"}
    response = await run_client.post(PATH, json=body)
    assert response.status_code == 503 and response.headers["x-should-retry"] == "false"
    key = response.headers["x-idempotency-key"]
    assert "creating its Intake Evaluation failed" in response.json()["detail"] and not fake_task_jobs.created
    fake_evaluations.fail = set()
    retry = await run_client.post(PATH, json=body | {"idempotency_key": key})
    assert retry.status_code == 200 and len(fake_evaluations.created) == len(fake_task_jobs.created) == 1


async def test_keyless_submits_get_distinct_generated_keys(run_client, published, request_body):
    first, second = [await run_client.post(PATH, json=request_body | {"idempotency_key": None}) for _ in range(2)]
    assert first.status_code == second.status_code == 201
    assert first.json()["run_id"] != second.json()["run_id"]
    assert len({first.json()["idempotency_key"], second.json()["idempotency_key"]}) == 2
