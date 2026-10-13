# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run acceptance races, verified storage and caller-scoped reads."""

import asyncio
import json
import re
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from unittest.mock import Mock, patch

import httpx
import pytest
from nemo_evals.taskset_runs.entities import TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.identifiers import derive_run_id
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_evals.taskset_runs.store import (
    _ACCEPT_RETRY_POLICY,
    PlanDigestMismatchError,
    PlanStorageError,
    RunIdExhausted,
    RunRequestConflict,
    TasksetRunStore,
)
from nemo_helix_plugin.client.errors import ConflictError, NemoTransportError, NotFoundError
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entities.types import Entity, EntityCreateInput
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation
from pydantic import ValidationError


def make_run(acceptance, *, phase=TasksetRunPhase.ACCEPTED) -> TasksetRunEntity:
    return TasksetRunEntity(
        name=acceptance.plan.run_id,
        workspace=acceptance.plan.workspace,
        taskset_ref=acceptance.plan.taskset_ref,
        plan_ref="default/existing-plan#plan.json",
        plan_digest="a" * 64,
        request_digest=acceptance.request_digest,
        idempotency_key=acceptance.idempotency_key,
        accepted_at=acceptance.accepted_at,
        execution_deadline_at=acceptance.execution_deadline_at,
        finalization_deadline_at=acceptance.finalization_deadline_at,
        phase=phase,
    )


async def test_find_absent_and_workspace_isolation(store, entity_store, files_client, acceptance) -> None:
    assert await store.find("default", acceptance.plan.run_id) is None
    run = await entity_store.create(make_run(acceptance))
    assert await store.find("default", run.name) == run
    assert await store.find("other", run.name) is None
    assert not files_client.calls


async def test_find_propagates_other_errors(store, entity_store, monkeypatch) -> None:
    async def fail(*args, **kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(entity_store, "get", fail)
    with pytest.raises(RuntimeError, match="store unavailable"):
        await store.find("default", "missing")


@pytest.mark.parametrize("phase", list(TasksetRunPhase))
async def test_replay_in_every_phase(store, entity_store, files_client, acceptance, phase) -> None:
    run = await entity_store.create(make_run(acceptance, phase=phase))
    assert await store.replay(run.workspace, run.name, run.request_digest) == run
    with pytest.raises(RunRequestConflict) as exc:
        await store.replay(run.workspace, run.name, "different")
    assert exc.value.run_id == run.name
    assert await store.replay("other", run.name, run.request_digest) is None
    assert not files_client.calls


def test_from_clients_owns_retry_policy(files_client) -> None:
    entities_client = Mock(spec=AsyncEntitiesClient)
    clone = entities_client.with_retry.return_value
    store = TasksetRunStore.from_clients(entities_client, files_client)
    entities_client.with_retry.assert_called_once_with(_ACCEPT_RETRY_POLICY)
    assert isinstance(store.entity_client, EntityClient)
    assert store.entity_client._client is clone
    assert _ACCEPT_RETRY_POLICY.retryable_status_codes == (408, 429)
    assert _ACCEPT_RETRY_POLICY.retry_all_server_errors == PLATFORM_DEFAULT_RETRY_POLICY.retry_all_server_errors
    assert _ACCEPT_RETRY_POLICY.max_retries == PLATFORM_DEFAULT_RETRY_POLICY.max_retries
    assert PLATFORM_DEFAULT_RETRY_POLICY.retryable_status_codes == (408, 409, 429)


async def test_first_accept(store, entity_store, files_client, acceptance) -> None:
    accepted = await store.accept(acceptance)
    assert accepted.created is True
    run = accepted.run
    assert len(entity_store.entities) == 1
    assert run.name == acceptance.plan.run_id
    assert run.workspace == acceptance.plan.workspace
    assert run.phase is TasksetRunPhase.ACCEPTED
    assert run.result_state.value == "pending"
    assert run.request_digest == acceptance.request_digest
    assert run.idempotency_key == acceptance.idempotency_key
    assert run.taskset_ref == acceptance.plan.taskset_ref
    assert (run.accepted_at, run.execution_deadline_at, run.finalization_deadline_at) == (
        acceptance.accepted_at,
        acceptance.execution_deadline_at,
        acceptance.finalization_deadline_at,
    )
    assert len(files_client.filesets) == 1
    (workspace, name), content = next(iter(files_client.filesets.items()))
    assert re.fullmatch(rf"{run.name}-plan-[0-9a-f]{{6}}", name)
    assert run.plan_ref == f"{workspace}/{name}#plan.json"
    assert run.plan_digest == sha256(content["plan.json"]).hexdigest()
    assert TasksetRunPlan.model_validate_json(content["plan.json"]) == acceptance.plan


@pytest.mark.parametrize("operation", ["create", "upload"])
async def test_plan_write_failure(store, entity_store, files_client, acceptance, operation) -> None:
    failure = ConflictError(httpx.Response(409, request=httpx.Request("POST", "http://files.invalid")))
    files_client.errors[operation] = failure
    with pytest.raises(PlanStorageError) as exc:
        await store.accept(acceptance)
    assert exc.value.__cause__ is failure
    assert not entity_store.entities
    assert not files_client.filesets
    assert [op for op, _, _ in files_client.calls] == (
        ["create"] if operation == "create" else ["create", "upload", "delete"]
    )


async def test_upload_cleanup_failure_preserves_cause(store, entity_store, files_client, acceptance, caplog) -> None:
    failure = RuntimeError("upload failed")
    files_client.errors.update(upload=failure, delete=RuntimeError("cleanup failed"))
    with pytest.raises(PlanStorageError) as exc:
        await store.accept(acceptance)
    assert exc.value.__cause__ is failure
    assert not entity_store.entities
    assert len(files_client.filesets) == 1
    assert "cleanup" in caplog.text.lower()


async def test_accept_plan_fileset_lost_reply(entity_store, files_client, acceptance) -> None:
    post_count = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        assert request.method == "POST"
        post_count += 1
        body = CreateFilesetRequest.model_validate_json(request.content)
        if post_count == 1:
            await files_client.create_fileset(workspace="default", body=body)
            raise httpx.ReadTimeout("reply lost", request=request)
        return httpx.Response(409, headers={"x-should-retry": "false"}, json={"detail": "exists"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        real_files_client = AsyncFilesClient(
            base_url="http://files.invalid",
            http_client=http_client,
            retry=replace(PLATFORM_DEFAULT_RETRY_POLICY, backoff_base=0),
        )
        store = TasksetRunStore(entity_store, real_files_client)
        with pytest.raises(PlanStorageError) as exc:
            await store.accept(acceptance)
    assert isinstance(exc.value.__cause__, ConflictError)
    assert post_count == 2
    assert not entity_store.entities
    assert len(files_client.filesets) == 1
    assert all(op != "delete" for op, _, _ in files_client.calls)


async def test_load_plan_downloads_and_validates_every_call(store, files_client, acceptance, monkeypatch) -> None:
    accepted = await store.accept(acceptance)
    validation = Mock(wraps=TasksetRunPlan.model_validate_json)
    monkeypatch.setattr(TasksetRunPlan, "model_validate_json", validation)
    first = await store.load_plan(accepted.run)
    first.spec.labels["mutated"] = "yes"
    first.spec.tasks.clear()
    second = await store.load_plan(accepted.run)
    assert second == acceptance.plan
    assert len([op for op, _, _ in files_client.calls if op == "download"]) == 2
    assert validation.call_count == 2


@pytest.mark.parametrize("loaded_before", [False, True])
@pytest.mark.parametrize("valid_json", [False, True])
async def test_plan_tampering_detected_before_validation(
    store, files_client, acceptance, loaded_before, valid_json
) -> None:
    run = (await store.accept(acceptance)).run
    if loaded_before:
        await store.load_plan(run)
    content = next(iter(files_client.filesets.values()))
    if valid_json:
        changed = json.loads(content["plan.json"])
        changed["max_failed_tasks"] = 10
        content["plan.json"] = json.dumps(changed).encode()
    else:
        content["plan.json"] = b"not json"
    with pytest.raises(PlanDigestMismatchError) as exc:
        await store.load_plan(run)
    assert run.plan_digest in str(exc.value)
    assert sha256(content["plan.json"]).hexdigest() in str(exc.value)
    content["plan.json"] = acceptance.plan.model_dump_json().encode()
    assert await store.load_plan(run) == acceptance.plan


async def test_plan_corrupt_but_matching_digest_fails_validation(store, files_client, acceptance) -> None:
    run = (await store.accept(acceptance)).run
    next(iter(files_client.filesets.values()))["plan.json"] = b"not json"
    run.plan_digest = sha256(b"not json").hexdigest()
    for _ in range(2):
        with pytest.raises(ValidationError):
            await store.load_plan(run)


async def test_plan_download_error_after_successful_load(store, files_client, acceptance) -> None:
    run = (await store.accept(acceptance)).run
    await store.load_plan(run)
    failure = NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://files.invalid")))
    files_client.errors["download"] = failure
    with pytest.raises(NotFoundError) as exc:
        await store.load_plan(run)
    assert exc.value is failure


@pytest.mark.parametrize("ref", ["unqualified#plan.json", "default/plan", "default/plan#", "default/#plan.json"])
async def test_plan_malformed_reference(store, files_client, acceptance, ref) -> None:
    run = make_run(acceptance)
    run.plan_ref = ref
    with pytest.raises(ValueError):
        await store.load_plan(run)
    assert not files_client.calls


def with_snapshot(acceptance, files_client, name):
    files_client.filesets[(acceptance.plan.workspace, name)] = {"agent.yaml": b"config"}
    spec = acceptance.plan.spec.model_copy(deep=True)
    # Keep request ownership visible in the plan without inventing a new producer.
    spec.labels["snapshot"] = f"{acceptance.plan.workspace}/{name}"
    return replace(
        acceptance,
        plan=acceptance.plan.model_copy(update={"spec": spec}),
        discard_on_loss=(f"{acceptance.plan.workspace}/{name}",),
    )


def plan_filesets(files_client):
    return {key: value for key, value in files_client.filesets.items() if "-plan-" in key[1]}


async def test_accept_overlapping_submits(store, entity_store, files_client, acceptance) -> None:
    a = with_snapshot(acceptance, files_client, "snapshot-a")
    b = with_snapshot(
        replace(
            acceptance,
            accepted_at=acceptance.accepted_at + timedelta(seconds=1),
            execution_deadline_at=acceptance.execution_deadline_at + timedelta(seconds=1),
            finalization_deadline_at=acceptance.finalization_deadline_at + timedelta(seconds=1),
        ),
        files_client,
        "snapshot-b",
    )
    files_client.upload_barrier = asyncio.Event()
    first, second = await asyncio.wait_for(asyncio.gather(store.accept(a), store.accept(b)), timeout=5)
    assert sorted([first.created, second.created]) == [False, True]
    assert first.run == second.run
    assert len(entity_store.entities) == len(plan_filesets(files_client)) == 1
    winner, loser = (a, b) if first.created else (b, a)
    assert first.run.accepted_at == winner.accepted_at
    assert first.run.execution_deadline_at == winner.execution_deadline_at
    assert first.run.finalization_deadline_at == winner.finalization_deadline_at
    assert await store.load_plan(first.run) == winner.plan
    assert ("default", winner.discard_on_loss[0].split("/")[1]) in files_client.filesets
    assert ("default", loser.discard_on_loss[0].split("/")[1]) not in files_client.filesets


async def test_accept_same_key_different_digest(store, entity_store, files_client, acceptance) -> None:
    winner = with_snapshot(acceptance, files_client, "snapshot-winner")
    run = (await store.accept(winner)).run
    original = entity_store.entities.copy()
    loser = with_snapshot(replace(acceptance, request_digest="different"), files_client, "snapshot-loser")
    with pytest.raises(RunRequestConflict) as exc:
        await store.accept(loser)
    assert exc.value.run_id == run.name
    assert entity_store.entities == original
    assert len(plan_filesets(files_client)) == 1
    assert ("default", "snapshot-winner") in files_client.filesets
    assert ("default", "snapshot-loser") not in files_client.filesets
    assert await store.load_plan(run) == winner.plan


@pytest.mark.parametrize(
    "phase", [TasksetRunPhase.COMPLETED, TasksetRunPhase.FAILED, TasksetRunPhase.CANCELLED, TasksetRunPhase.EXPIRED]
)
async def test_accept_finished_run_keeps_key(store, entity_store, files_client, acceptance, phase) -> None:
    run = (await store.accept(acceptance)).run
    run.phase = phase
    run = await entity_store.update(run)
    retry = with_snapshot(
        replace(acceptance, accepted_at=acceptance.accepted_at + timedelta(days=1)), files_client, "retry-snapshot"
    )
    replay = await store.accept(retry)
    assert replay.created is False
    assert replay.run == run
    assert len(plan_filesets(files_client)) == 1
    assert ("default", "retry-snapshot") not in files_client.filesets
    assert await store.load_plan(replay.run) == acceptance.plan


async def test_accept_same_key_different_scope(store, entity_store, files_client, acceptance) -> None:
    runs = []
    for workspace, caller in [("default", "user:alice"), ("default", "user:bob"), ("other", "user:alice")]:
        entity_store.caller = caller
        plan = acceptance.plan.model_copy(
            update={"workspace": workspace, "run_id": derive_run_id(workspace, caller, "key")}
        )
        accepted = await store.accept(replace(acceptance, plan=plan))
        assert accepted.created is True
        assert accepted.run.created_by == caller
        runs.append(accepted.run.name)
    assert len(set(runs)) == len(entity_store.entities) == len(plan_filesets(files_client)) == 3


@pytest.mark.parametrize("key", ["key", None])
async def test_accept_lost_reply(store, entity_store, files_client, acceptance, monkeypatch, key) -> None:
    acceptance = with_snapshot(replace(acceptance, idempotency_key=key), files_client, "snapshot")
    original_create = entity_store.create

    async def lost_reply(entity):
        await original_create(entity)
        raise NemoEntityConflictError("retry saw committed create")

    monkeypatch.setattr(entity_store, "create", lost_reply)
    accepted = await store.accept(acceptance)
    assert accepted.created is True
    assert accepted.run.name == acceptance.plan.run_id
    assert len(entity_store.entities) == len(plan_filesets(files_client)) == 1
    assert all(op != "delete" for op, _, _ in files_client.calls)
    assert ("default", "snapshot") in files_client.filesets


async def test_accept_keyless_collision(store, entity_store, files_client, acceptance, monkeypatch) -> None:
    existing = await entity_store.create(make_run(acceptance))
    acceptance = with_snapshot(replace(acceptance, idempotency_key=None), files_client, "snapshot")
    monkeypatch.setattr("nemo_evals.taskset_runs.store.random_run_id", lambda: "run-1111111111111111")
    accepted = await store.accept(acceptance)
    assert accepted.created is True
    assert accepted.run.name == "run-1111111111111111"
    assert await store.find(existing.workspace, existing.name) == existing
    stored_plan = await store.load_plan(accepted.run)
    assert stored_plan.run_id == accepted.run.name
    assert stored_plan.spec == acceptance.plan.spec
    assert len(entity_store.entities) == 2
    assert len(plan_filesets(files_client)) == 1
    assert ("default", "snapshot") in files_client.filesets
    assert len([op for op, _, _ in files_client.calls if op == "delete"]) == 1


async def test_accept_two_keyless_collisions(store, entity_store, files_client, acceptance, monkeypatch) -> None:
    existing = await entity_store.create(make_run(acceptance))
    second = make_run(acceptance)
    second.name = "run-1111111111111111"
    second = await entity_store.create(second)
    monkeypatch.setattr("nemo_evals.taskset_runs.store.random_run_id", lambda: second.name)
    create = Mock(wraps=entity_store.create)
    monkeypatch.setattr(entity_store, "create", create)
    acceptance = with_snapshot(replace(acceptance, idempotency_key=None), files_client, "snapshot")
    with pytest.raises(RunIdExhausted):
        await store.accept(acceptance)
    assert create.call_count == 2
    assert not files_client.filesets
    assert len(entity_store.entities) == 2
    assert await store.find(existing.workspace, existing.name) == existing
    assert await store.find(second.workspace, second.name) == second


@pytest.mark.parametrize("different_digest", [False, True])
async def test_accept_cleanup_failure(store, files_client, acceptance, caplog, different_digest) -> None:
    existing = (await store.accept(acceptance)).run
    loser = with_snapshot(
        replace(acceptance, request_digest="different" if different_digest else acceptance.request_digest),
        files_client,
        "snapshot-a",
    )
    files_client.filesets[("default", "snapshot-b")] = {}
    loser = replace(loser, discard_on_loss=(*loser.discard_on_loss, "default/snapshot-b"))
    # Fail each plan deletion and the first snapshot; the second must still be attempted.
    original_delete = files_client.delete_fileset

    async def fail_plan_delete(**kwargs):
        if "-plan-" in kwargs["name"]:
            files_client.calls.append(("delete", kwargs["workspace"], kwargs["name"]))
            raise RuntimeError("plan cleanup failed")
        return await original_delete(**kwargs)

    files_client.delete_errors.add("snapshot-a")
    with patch.object(files_client, "delete_fileset", fail_plan_delete):
        if different_digest:
            with pytest.raises(RunRequestConflict):
                await store.accept(loser)
        else:
            replay = await store.accept(loser)
            assert replay.run == existing
            assert replay.created is False
    assert ("default", "snapshot-b") not in files_client.filesets
    assert "snapshot-a" in caplog.text
    assert "plan" in caplog.text


async def test_accept_unknown_create_outcome(store, entity_store, files_client, acceptance, monkeypatch) -> None:
    original_create = entity_store.create
    failure = NemoTransportError(httpx.ReadTimeout("unknown outcome"))

    async def unknown(entity):
        await original_create(entity)
        raise failure

    monkeypatch.setattr(entity_store, "create", unknown)
    acceptance = with_snapshot(acceptance, files_client, "snapshot")
    with pytest.raises(NemoTransportError) as exc:
        await store.accept(acceptance)
    assert exc.value is failure
    assert len(entity_store.entities) == len(plan_filesets(files_client)) == 1
    assert all(op != "delete" for op, _, _ in files_client.calls)
    assert ("default", "snapshot") in files_client.filesets


@pytest.mark.parametrize("error_type", [RuntimeError, NemoEntityNotFoundError], ids=["unavailable", "not-found"])
async def test_accept_conflict_lookup_failure(
    store, entity_store, files_client, acceptance, monkeypatch, error_type: type[Exception]
) -> None:
    await entity_store.create(make_run(acceptance))
    failure = error_type("lookup failed")

    async def fail_lookup(*args, **kwargs):
        raise failure

    monkeypatch.setattr(entity_store, "get", fail_lookup)
    acceptance = with_snapshot(acceptance, files_client, "snapshot")
    with pytest.raises(error_type, match="lookup failed") as exc:
        await store.accept(acceptance)
    assert exc.value is failure
    assert len(plan_filesets(files_client)) == 1
    assert all(op != "delete" for op, _, _ in files_client.calls)
    assert ("default", "snapshot") in files_client.filesets


@pytest.mark.parametrize("outcome", ["conflict", "timeout", "503"])
async def test_accept_real_entity_client_retries(entity_store, files_client, acceptance, monkeypatch, outcome) -> None:
    monkeypatch.setattr(
        "nemo_evals.taskset_runs.store._ACCEPT_RETRY_POLICY", replace(_ACCEPT_RETRY_POLICY, backoff_base=0)
    )
    if outcome == "conflict":
        existing = await entity_store.create(make_run(acceptance))
    calls = []

    async def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "POST":
            body = EntityCreateInput.model_validate_json(request.content)
            entity = TasksetRunEntity.model_validate({"name": body.name, "workspace": "default", **body.data})
            try:
                await entity_store.create(entity)
            except NemoEntityConflictError:
                # Deliberately no retry header: the factory policy itself must suppress 409.
                return httpx.Response(409, json={"detail": "exists"})
            if outcome == "timeout":
                raise httpx.ReadTimeout("reply lost", request=request)
            assert outcome == "503"
            return httpx.Response(503, json={"detail": "reply lost"})
        assert request.method == "GET"
        run = await entity_store.get(TasksetRunEntity, request.url.path.split("/")[-1], workspace="default")
        dto = Entity(
            id=run.id,
            name=run.name,
            workspace=run.workspace,
            entity_type=run.__entity_type__,
            parent=run.parent,
            project=run.project,
            created_at=run.created_at,
            updated_at=run.updated_at,
            created_by=run.created_by,
            db_version=run.db_version,
            data=run.model_dump(mode="json", exclude=run.__base_fields__, exclude_computed_fields=True),
        )
        return httpx.Response(200, json=dto.model_dump(mode="json"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        raw_client = AsyncEntitiesClient(base_url="http://entities.invalid", http_client=http_client)
        store = TasksetRunStore.from_clients(raw_client, files_client)
        acceptance = with_snapshot(acceptance, files_client, "snapshot")
        accepted = await store.accept(acceptance)
    assert calls == (["POST", "GET"] if outcome == "conflict" else ["POST", "POST", "GET"])
    assert accepted.created is (outcome != "conflict")
    assert accepted.run.created_by == "user:alice"
    assert accepted.run.id
    assert len(entity_store.entities) == 1
    if outcome == "conflict":
        assert accepted.run == existing
        assert not files_client.filesets
    else:
        assert len(plan_filesets(files_client)) == 1
        assert ("default", "snapshot") in files_client.filesets
        assert all(op != "delete" for op, _, _ in files_client.calls)


@pytest.mark.parametrize("taskset_ref", [None, "default/suite#3"])
async def test_list_active_filters_and_paginates(
    store, entity_store, files_client, acceptance, monkeypatch, taskset_ref
) -> None:
    expected = []
    for workspace in ("default", "other"):
        for caller in ("user:alice", "user:bob"):
            entity_store.caller = caller
            for phase in TasksetRunPhase:
                for revision in ("default/suite#3", "default/suite#4"):
                    # >100 matches for one caller/revision, interleaved with excluded records.
                    for index in range(51 if phase in (TasksetRunPhase.ACCEPTED, TasksetRunPhase.RUNNING) else 1):
                        run = make_run(acceptance, phase=phase)
                        run.workspace = workspace
                        run.taskset_ref = revision
                        run.name = f"{caller.split(':')[1]}-{phase.value}-{revision[-1]}-{index:03d}"
                        await entity_store.create(run)
                        if (
                            workspace == "default"
                            and caller == "user:alice"
                            and phase in (TasksetRunPhase.ACCEPTED, TasksetRunPhase.RUNNING)
                            and (taskset_ref is None or revision == taskset_ref)
                        ):
                            expected.append(run.name)
    query = Mock(wraps=entity_store.list)
    monkeypatch.setattr(entity_store, "list", query)
    runs = await store.list_active("default", "user:alice", taskset_ref)
    assert [run.name for run in runs] == sorted(expected)
    assert query.call_count == (3 if taskset_ref is None else 2)
    for page, call in enumerate(query.call_args_list, start=1):
        assert call.args == (TasksetRunEntity,)
        assert call.kwargs == {
            "workspace": "default",
            "sort": "name",
            "page": page,
            "filter_operation": LogicalOperation(
                operator=FilterOperator.AND,
                operations=[
                    ComparisonOperation(field="created_by", operator=FilterOperator.EQ, value="user:alice"),
                    ComparisonOperation(field="data.phase", operator=FilterOperator.IN, value=["accepted", "running"]),
                    *(
                        [ComparisonOperation(field="data.taskset_ref", operator=FilterOperator.EQ, value=taskset_ref)]
                        if taskset_ref is not None
                        else []
                    ),
                ],
            ),
        }
    assert not files_client.calls


async def test_list_active_empty_and_errors(store, entity_store, monkeypatch) -> None:
    assert await store.list_active("default", "user:alice") == []

    async def fail(*args, **kwargs):
        raise RuntimeError("list unavailable")

    monkeypatch.setattr(entity_store, "list", fail)
    with pytest.raises(RuntimeError, match="list unavailable"):
        await store.list_active("default", "user:alice")
