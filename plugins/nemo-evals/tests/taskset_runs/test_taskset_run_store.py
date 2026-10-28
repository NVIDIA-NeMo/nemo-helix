# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run acceptance races, verified storage and caller-scoped reads."""

import asyncio
import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import Mock

import httpx
import pytest
from nemo_evals.taskset_runs.entities import MemberPhase, TasksetRunEntity, TasksetRunMemberEntity, TasksetRunPhase
from nemo_evals.taskset_runs.identifiers import derive_run_id
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_evals.taskset_runs.store import (
    _ACCEPT_RETRY_POLICY,
    CANCEL_WRITE_ATTEMPTS,
    CancelContended,
    PlanDigestMismatchError,
    PlanStorageError,
    RunAlreadyFinished,
    RunNotFound,
    RunRequestConflict,
    TasksetRunStore,
)
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.client.errors import ConflictError, NemoTransportError, NotFoundError
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entities.types import Entity, EntityCreateInput
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from pydantic import ValidationError


def make_run(acceptance, *, phase=TasksetRunPhase.ACCEPTED) -> TasksetRunEntity:
    """Build an unsaved record from acceptance inputs without uploading a real plan.

    Args:
        acceptance: Plan identity, request digest/key, and fixed timestamps to reuse.
        phase: Lifecycle phase for seeded lookup, replay, and pagination scenarios.

    Returns:
        A run with a placeholder plan binding, ready to seed Entity Store; callers
        needing readable plan bytes must use the real acceptance path instead.
    """
    return TasksetRunEntity(
        name=acceptance.plan.run_id,
        workspace=acceptance.plan.workspace,
        taskset_ref=acceptance.plan.taskset_ref,
        plan_ref="default/existing-plan#plan.json",
        plan_digest="a" * 64,
        request_digest=acceptance.request_digest,
        idempotency_key=acceptance.idempotency_key,
        accepted_at=acceptance.accepted_at,
        timeout_seconds=acceptance.timeout_seconds,
        execution_deadline_at=acceptance.execution_deadline_at,
        finalization_deadline_at=acceptance.finalization_deadline_at,
        phase=phase,
    )


async def test_find_absent_and_workspace_isolation(store, entity_store, files_service, acceptance) -> None:
    """Verify workspace-scoped reads return a seeded record or None without Files work.

    The store, entity/Files fakes, and acceptance fixture supply an initially empty
    namespace; assertions distinguish missing records from the same name elsewhere.
    """
    assert await store.find("default", acceptance.plan.run_id) is None
    run = await entity_store.create(make_run(acceptance))
    assert await store.find("default", run.name) == run
    assert await store.find("other", run.name) is None
    assert not files_service.calls


async def test_find_propagates_other_errors(store, entity_store, monkeypatch) -> None:
    """Inject an entity lookup outage and require find to propagate it instead of None.

    The store/entity fixtures and monkeypatch isolate this failure so storage
    unavailability cannot be mistaken for permission to accept a fresh run.
    """

    async def fail(*args, **kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(entity_store, "get", fail)
    with pytest.raises(RuntimeError, match="store unavailable"):
        await store.find("default", "missing")


@pytest.mark.parametrize("phase", list(TasksetRunPhase))
async def test_replay_in_every_phase(store, entity_store, files_service, acceptance, phase) -> None:
    """Verify each parametrized phase replays only the original logical request.

    Store and fake-client fixtures seed the acceptance record in ``phase``;
    assertions require equal-digest reuse, mismatched-digest conflict, workspace
    isolation, and zero Files calls so retries do not restart or re-resolve runs.
    """
    run = await entity_store.create(make_run(acceptance, phase=phase))
    assert await store.replay(run.workspace, run.name, run.request_digest) == run
    with pytest.raises(RunRequestConflict) as exc:
        await store.replay(run.workspace, run.name, "different")
    assert exc.value.run_id == run.name
    assert await store.replay("other", run.name, run.request_digest) is None
    assert not files_service.calls


def test_constructor_owns_retry_policy(files_client) -> None:
    """Verify the Files fixture and mocked raw client produce the production adapter.

    Assertions require a retry clone that suppresses ordinary 409 retries while
    preserving transient retries and the original platform policy, ensuring
    acceptance owns conflict settlement without changing other client behavior.
    """
    entities_client = Mock(spec=AsyncEntitiesClient)
    clone = entities_client.with_retry.return_value
    store = TasksetRunStore(entities_client, files_client)
    entities_client.with_retry.assert_called_once_with(_ACCEPT_RETRY_POLICY)
    assert isinstance(store.entity_client, EntityClient)
    assert store.entity_client._client is clone
    assert _ACCEPT_RETRY_POLICY.retryable_status_codes == (408, 429)
    assert _ACCEPT_RETRY_POLICY.retry_all_server_errors == PLATFORM_DEFAULT_RETRY_POLICY.retry_all_server_errors
    assert _ACCEPT_RETRY_POLICY.max_retries == PLATFORM_DEFAULT_RETRY_POLICY.max_retries
    assert PLATFORM_DEFAULT_RETRY_POLICY.retryable_status_codes == (408, 409, 429)


async def test_first_accept(store, entity_store, files_service, acceptance) -> None:
    """Verify fresh acceptance binds the fixture's resolved inputs to exact stored bytes.

    Store and fake clients begin empty. Assertions require one accepted/pending
    entity and a uniquely named plan fileset whose ref, digest, JSON, and original
    deadlines match, establishing the normal acceptance contract for failure tests.
    """
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
    assert len(files_service.filesets) == 1
    (workspace, name), content = next(iter(files_service.filesets.items()))
    assert re.fullmatch(rf"{run.name}-plan-[0-9a-f]{{6}}", name)
    assert run.plan_ref == f"{workspace}/{name}#plan.json"
    assert run.plan_digest == sha256(content["plan.json"]).hexdigest()
    assert TasksetRunPlan.model_validate_json(content["plan.json"]) == acceptance.plan


@pytest.mark.parametrize("operation", ["create", "upload"])
async def test_plan_write_failure(store, entity_store, files_service, acceptance, operation) -> None:
    """Inject the parametrized create/upload failure and prevent run creation.

    Store, client fakes, and acceptance supply the request; assertions require
    PlanStorageError to retain the injected cause and cleanup only after a
    successful fileset create, preserving the storage-ownership boundary.
    """
    failure = ConflictError(httpx.Response(409, request=httpx.Request("POST", "http://files.invalid")))
    files_service.failures[operation] = failure
    with pytest.raises(PlanStorageError) as exc:
        await store.accept(acceptance)
    assert exc.value.__cause__ is failure
    assert not entity_store.entities
    assert not files_service.filesets
    assert [op for op, _, _ in files_service.calls] == (
        ["create"] if operation == "create" else ["create", "upload", "delete"]
    )


async def test_upload_cleanup_failure_preserves_cause(store, entity_store, files_service, acceptance, caplog) -> None:
    """Inject upload and rollback failures without replacing the original write error.

    Store/client/acceptance fixtures perform the failing write, and caplog captures
    cleanup diagnostics. Assertions require no run, the original chained cause,
    and a logged leaked fileset so cleanup cannot hide the acceptance failure.
    """
    failure = RuntimeError("upload failed")
    files_service.failures.update(upload=failure, delete=RuntimeError("cleanup failed"))
    with pytest.raises(PlanStorageError) as exc:
        await store.accept(acceptance)
    assert exc.value.__cause__ is failure
    assert not entity_store.entities
    assert len(files_service.filesets) == 1
    assert "cleanup" in caplog.text.lower()


async def test_accept_plan_fileset_lost_reply(entity_store, files_service, files_client, acceptance) -> None:
    """Model a committed fileset create with a lost reply through the real Files client.

    Entity/Files fakes and acceptance provide backing state and input. HTTP retries
    must yield PlanStorageError on conflict while retaining the ambiguous fileset
    and creating no run, because a failed create reply does not prove ownership.
    """
    post_count = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        """Commit the first POST then time out; return nonretryable 409 on its retry.

        The input is the real client's HTTP request; responses and the injected
        timeout reproduce an ambiguous Files create without network access.
        """
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
        store = TasksetRunStore._with_entity_adapter(entity_store, real_files_client)
        with pytest.raises(PlanStorageError) as exc:
            await store.accept(acceptance)
    assert isinstance(exc.value.__cause__, ConflictError)
    assert post_count == 2
    assert not entity_store.entities
    assert len(files_service.filesets) == 1
    assert all(op != "delete" for op, _, _ in files_service.calls)


async def test_load_plan_downloads_and_validates_every_call(store, files_service, acceptance, monkeypatch) -> None:
    """Verify repeated reads return fresh nested models rather than cached mutable state.

    Store/Files/acceptance fixtures create a plan and monkeypatch counts validation.
    Mutating the first result must leave the second equal to the accepted input,
    with two downloads and validations, preserving byte checks on every read.
    """
    accepted = await store.accept(acceptance)
    validation = Mock(wraps=TasksetRunPlan.model_validate_json)
    monkeypatch.setattr(TasksetRunPlan, "model_validate_json", validation)
    first = await store.load_plan(accepted.run)
    first.spec.labels["mutated"] = "yes"
    first.spec.tasks.clear()
    second = await store.load_plan(accepted.run)
    assert second == acceptance.plan
    assert len([op for op, _, _ in files_service.calls if op == "download"]) == 2
    assert validation.call_count == 2


@pytest.mark.parametrize("loaded_before", [False, True])
@pytest.mark.parametrize("valid_json", [False, True])
async def test_plan_tampering_detected_before_validation(
    store, files_service, acceptance, loaded_before, valid_json
) -> None:
    """Require digest rejection for tampered bytes even after a prior successful read.

    Store/Files/acceptance fixtures supply a stored plan; ``loaded_before`` controls
    a prior read and ``valid_json`` selects valid or corrupt replacement bytes.
    Assertions require the mismatch diagnostics and successful recovery after
    restoring bytes, proving neither valid JSON nor earlier reads bypass integrity.
    """
    run = (await store.accept(acceptance)).run
    if loaded_before:
        await store.load_plan(run)
    content = next(iter(files_service.filesets.values()))
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


async def test_plan_corrupt_but_matching_digest_fails_validation(store, files_service, acceptance) -> None:
    """Verify matching hashes cannot make corrupt plan JSON a valid resolved model.

    Store/Files/acceptance fixtures supply a run, then both bytes and digest are
    replaced with a corrupt payload. Repeated reads must raise ValidationError,
    demonstrating that integrity checks and schema validation are both necessary.
    """
    run = (await store.accept(acceptance)).run
    next(iter(files_service.filesets.values()))["plan.json"] = b"not json"
    run.plan_digest = sha256(b"not json").hexdigest()
    for _ in range(2):
        with pytest.raises(ValidationError):
            await store.load_plan(run)


async def test_plan_download_error_after_successful_load(store, files_service, acceptance) -> None:
    """Require a later Files error to propagate despite an earlier successful read.

    Store/Files/acceptance fixtures first load a valid plan, then inject not-found.
    The exact injected error must escape, preventing cached data from hiding loss
    of the authoritative file.
    """
    run = (await store.accept(acceptance)).run
    await store.load_plan(run)
    failure = NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://files.invalid")))
    files_service.failures["download"] = failure
    with pytest.raises(NotFoundError) as exc:
        await store.load_plan(run)
    assert exc.value is failure


@pytest.mark.parametrize("ref", ["unqualified#plan.json", "default/plan", "default/plan#", "default/#plan.json"])
async def test_plan_malformed_reference(store, files_service, acceptance, ref) -> None:
    """Reject each parametrized malformed plan ref before making any Files request.

    Store/Files/acceptance fixtures supply a record with ``ref`` substituted;
    assertions require ValueError and an empty call history so incomplete or
    unqualified references cannot resolve to unintended storage.
    """
    run = make_run(acceptance)
    run.plan_ref = ref
    with pytest.raises(ValueError):
        await store.load_plan(run)
    assert not files_service.calls


def with_snapshot(acceptance, files_service, name):
    """Create an owned snapshot and return acceptance inputs that reference it.

    Args:
        acceptance: Request whose nested spec is copied to avoid fixture mutation.
        files_service: In-memory Files service in which to seed the snapshot payload.
        name: Unique fileset name owned exclusively by this request.

    Returns:
        Replacement acceptance with the snapshot ref in its spec labels and
        discard list, making winner/loser ownership visible in cleanup assertions.
    """
    files_service.filesets[(acceptance.plan.workspace, name)] = {"agent.yaml": b"config"}
    spec = acceptance.plan.spec.model_copy(deep=True)
    # Keep request ownership visible in the plan without inventing a new producer.
    spec.labels["snapshot"] = f"{acceptance.plan.workspace}/{name}"
    return replace(
        acceptance,
        plan=acceptance.plan.model_copy(update={"spec": spec}),
        discard_on_loss=(f"{acceptance.plan.workspace}/{name}",),
    )


def plan_filesets(files_service):
    """Select the Files fake's plan filesets so assertions exclude request snapshots."""
    return {key: value for key, value in files_service.filesets.items() if "-plan-" in key[1]}


async def test_accept_overlapping_submits(store, entity_store, files_service, acceptance) -> None:
    """Race identical keyed requests and retain only the winning plan and snapshot.

    Store/client/acceptance fixtures supply two owned snapshots and distinct times;
    a Files barrier overlaps their uploads. Assertions require one committed run,
    identical returned records, winner deadlines/bytes, and loser cleanup, proving
    named creation makes the acceptance decision without a process-local lock.
    """
    a = with_snapshot(acceptance, files_service, "snapshot-a")
    b = with_snapshot(
        replace(
            acceptance,
            accepted_at=acceptance.accepted_at + timedelta(seconds=1),
            execution_deadline_at=acceptance.execution_deadline_at + timedelta(seconds=1),
            finalization_deadline_at=acceptance.finalization_deadline_at + timedelta(seconds=1),
        ),
        files_service,
        "snapshot-b",
    )
    uploads_stored = 0
    both_stored = asyncio.Event()

    async def hold_until_both_uploads_are_stored() -> None:
        """Let both plans reach entity creation together, as the old upload barrier did."""
        nonlocal uploads_stored
        uploads_stored += 1
        if uploads_stored == 2:
            both_stored.set()
        await both_stored.wait()

    files_service.after_upload = hold_until_both_uploads_are_stored
    first, second = await asyncio.wait_for(asyncio.gather(store.accept(a), store.accept(b)), timeout=5)
    assert sorted([first.created, second.created]) == [False, True]
    assert first.run == second.run
    assert len(entity_store.entities) == len(plan_filesets(files_service)) == 1
    winner, loser = (a, b) if first.created else (b, a)
    assert first.run.accepted_at == winner.accepted_at
    assert first.run.execution_deadline_at == winner.execution_deadline_at
    assert first.run.finalization_deadline_at == winner.finalization_deadline_at
    assert await store.load_plan(first.run) == winner.plan
    assert ("default", winner.discard_on_loss[0].split("/")[1]) in files_service.filesets
    assert ("default", loser.discard_on_loss[0].split("/")[1]) not in files_service.filesets


async def test_accept_same_key_different_digest(store, entity_store, files_service, acceptance) -> None:
    """Reject a different logical request reusing a key without altering its winner.

    Store/client/acceptance fixtures create the winner and an independently owned
    loser snapshot. Assertions require RunRequestConflict, unchanged winner data
    and readable plan, and cleanup confined to the losing request's storage.
    """
    winner = with_snapshot(acceptance, files_service, "snapshot-winner")
    run = (await store.accept(winner)).run
    original = entity_store.entities.copy()
    loser = with_snapshot(replace(acceptance, request_digest="different"), files_service, "snapshot-loser")
    with pytest.raises(RunRequestConflict) as exc:
        await store.accept(loser)
    assert exc.value.run_id == run.name
    assert entity_store.entities == original
    assert len(plan_filesets(files_service)) == 1
    assert ("default", "snapshot-winner") in files_service.filesets
    assert ("default", "snapshot-loser") not in files_service.filesets
    assert await store.load_plan(run) == winner.plan


@pytest.mark.parametrize(
    "phase", [TasksetRunPhase.COMPLETED, TasksetRunPhase.FAILED, TasksetRunPhase.CANCELLED, TasksetRunPhase.EXPIRED]
)
async def test_accept_finished_run_keeps_key(store, entity_store, files_service, acceptance, phase) -> None:
    """Verify a key remains bound when the seeded run reaches each terminal phase.

    Store/client/acceptance fixtures and ``phase`` supply a finished run followed
    by a retry with later times and new storage. Assertions require created=False,
    the original record/plan, and discarded retry storage so completion cannot
    restart execution or renew the original acceptance timestamps.
    """
    run = (await store.accept(acceptance)).run
    run.phase = phase
    run = await entity_store.update(run)
    retry = with_snapshot(
        replace(acceptance, accepted_at=acceptance.accepted_at + timedelta(days=1)), files_service, "retry-snapshot"
    )
    replay = await store.accept(retry)
    assert replay.created is False
    assert replay.run == run
    assert len(plan_filesets(files_service)) == 1
    assert ("default", "retry-snapshot") not in files_service.filesets
    assert await store.load_plan(replay.run) == acceptance.plan


async def test_accept_same_key_different_scope(store, entity_store, files_service, acceptance) -> None:
    """Accept the same key independently for different caller/workspace scope inputs.

    Store/client/acceptance fixtures vary authenticated creator and workspace;
    assertions require three distinct names, records, plans, and correct creator
    metadata so unrelated callers do not share an idempotency binding.
    """
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
    assert len(set(runs)) == len(entity_store.entities) == len(plan_filesets(files_service)) == 3


async def test_accept_lost_reply(store, entity_store, files_service, acceptance, monkeypatch) -> None:
    """Recover this attempt's committed create.

    Store/client/acceptance fixtures own a snapshot, and monkeypatch makes create
    commit before reporting conflict. Assertions require created=True and no
    deletion, proving own-plan equality recovers the lost reply before loser
    cleanup can run.
    """
    acceptance = with_snapshot(acceptance, files_service, "snapshot")
    original_create = entity_store.create

    async def lost_reply(entity):
        await original_create(entity)
        raise NemoEntityConflictError("retry saw committed create")

    monkeypatch.setattr(entity_store, "create", lost_reply)
    accepted = await store.accept(acceptance)
    assert accepted.created is True
    assert accepted.run.name == acceptance.plan.run_id
    assert len(entity_store.entities) == len(plan_filesets(files_service)) == 1
    assert all(op != "delete" for op, _, _ in files_service.calls)
    assert ("default", "snapshot") in files_service.filesets


@pytest.mark.parametrize("different_digest", [False, True])
async def test_accept_cleanup_failure(store, files_service, acceptance, caplog, different_digest) -> None:
    """Keep conflict/replay outcomes authoritative when multiple cleanup steps fail.

    Store/Files/acceptance fixtures supply a winner and loser-owned resources;
    ``different_digest`` selects conflict or replay and caplog captures failures.
    Assertions require the correct outcome and deletion of a later snapshot even
    after plan/earlier-snapshot failures, with leaked resources reported in logs.
    """
    existing = (await store.accept(acceptance)).run
    loser = with_snapshot(
        replace(acceptance, request_digest="different" if different_digest else acceptance.request_digest),
        files_service,
        "snapshot-a",
    )
    files_service.filesets[("default", "snapshot-b")] = {}
    loser = replace(loser, discard_on_loss=(*loser.discard_on_loss, "default/snapshot-b"))
    # Fail each plan deletion and the first snapshot; the second must still be attempted.
    files_service.fail_when = lambda operation, workspace, name: (
        RuntimeError(f"cannot delete {workspace}/{name}")
        if operation == "delete" and ("-plan-" in name or name == "snapshot-a")
        else None
    )
    if different_digest:
        with pytest.raises(RunRequestConflict):
            await store.accept(loser)
    else:
        replay = await store.accept(loser)
        assert replay.run == existing
        assert replay.created is False
    assert ("default", "snapshot-b") not in files_service.filesets
    assert "snapshot-a" in caplog.text
    assert "plan" in caplog.text


async def test_accept_unknown_create_outcome(store, entity_store, files_service, acceptance, monkeypatch) -> None:
    """Preserve potentially referenced storage when create reports an unknown outcome.

    Store/client/acceptance fixtures own a plan and snapshot; monkeypatch commits
    the entity then raises a transport error. Assertions require the exact error
    and zero cleanup, protecting a committed run whose reply was never observed.
    """
    original_create = entity_store.create
    failure = NemoTransportError(httpx.ReadTimeout("unknown outcome"))

    async def unknown(entity):
        await original_create(entity)
        raise failure

    monkeypatch.setattr(entity_store, "create", unknown)
    acceptance = with_snapshot(acceptance, files_service, "snapshot")
    with pytest.raises(NemoTransportError) as exc:
        await store.accept(acceptance)
    assert exc.value is failure
    assert len(entity_store.entities) == len(plan_filesets(files_service)) == 1
    assert all(op != "delete" for op, _, _ in files_service.calls)
    assert ("default", "snapshot") in files_service.filesets


@pytest.mark.parametrize("error_type", [RuntimeError, NemoEntityNotFoundError], ids=["unavailable", "not-found"])
async def test_accept_conflict_lookup_failure(
    store, entity_store, files_service, acceptance, monkeypatch, error_type: type[Exception]
) -> None:
    """Preserve request resources when a conflict lookup cannot identify the winner.

    Store/client/acceptance fixtures seed a collision; monkeypatch injects the
    parametrized outage or not-found ``error_type`` during lookup. Assertions
    require that exact failure and zero deletion because even not-found cannot
    distinguish a deleted winner from ambiguous create/lookup observations.
    """
    await entity_store.create(make_run(acceptance))
    failure = error_type("lookup failed")

    async def fail_lookup(*args, **kwargs):
        raise failure

    monkeypatch.setattr(entity_store, "get", fail_lookup)
    acceptance = with_snapshot(acceptance, files_service, "snapshot")
    with pytest.raises(error_type, match="lookup failed") as exc:
        await store.accept(acceptance)
    assert exc.value is failure
    assert len(plan_filesets(files_service)) == 1
    assert all(op != "delete" for op, _, _ in files_service.calls)
    assert ("default", "snapshot") in files_service.filesets


@pytest.mark.parametrize("outcome", ["conflict", "timeout", "503"])
async def test_accept_real_entity_client_retries(
    entity_store, files_service, files_client, acceptance, monkeypatch, outcome
) -> None:
    """Exercise production retries for conflict and lost timeout/503 replies over HTTP.

    Entity/Files/acceptance fixtures back the mock transport; ``outcome`` selects
    the response sequence and monkeypatch removes backoff. Assertions require
    immediate 409 settlement, transient retries, authoritative creator metadata,
    and cleanup only for a different winning plan, testing the real client policy.
    """
    monkeypatch.setattr(
        "nemo_evals.taskset_runs.store._ACCEPT_RETRY_POLICY", replace(_ACCEPT_RETRY_POLICY, backoff_base=0)
    )
    if outcome == "conflict":
        existing = await entity_store.create(make_run(acceptance))
    calls = []

    async def handle(request: httpx.Request) -> httpx.Response:
        """Translate real client requests into fake persistence and controlled replies.

        POST parses entity-create input, commits it, then returns a conflict/503
        or raises a timeout according to the scenario. GET returns an Entity DTO
        with response metadata, letting retry and deserialization behavior run
        through the production client without an external service.
        """
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
        store = TasksetRunStore(raw_client, files_client)
        acceptance = with_snapshot(acceptance, files_service, "snapshot")
        accepted = await store.accept(acceptance)
    assert calls == (["POST", "GET"] if outcome == "conflict" else ["POST", "POST", "GET"])
    assert accepted.created is (outcome != "conflict")
    assert accepted.run.created_by == "user:alice"
    assert accepted.run.id
    assert len(entity_store.entities) == 1
    if outcome == "conflict":
        assert accepted.run == existing
        assert not files_service.filesets
    else:
        assert len(plan_filesets(files_service)) == 1
        assert ("default", "snapshot") in files_service.filesets
        assert all(op != "delete" for op, _, _ in files_service.calls)


@pytest.mark.parametrize("taskset_ref", [None, "default/suite#3"])
async def test_list_active_filters_and_paginates(
    store, entity_store, files_service, acceptance, monkeypatch, taskset_ref
) -> None:
    """Verify active-run filtering remains scoped and sorted across multiple pages.

    Store/client/acceptance fixtures seed mixed workspaces, callers, phases, and
    revisions; ``taskset_ref`` selects one revision or all, while monkeypatch
    records queries. Assertions require every matching name, consistent filters
    on each page, and no Files access, protecting overlap detection beyond page one.
    """
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
    assert [run.name for run in runs] == list(reversed(expected))
    assert query.call_count == (3 if taskset_ref is None else 2)
    for page, call in enumerate(query.call_args_list, start=1):
        assert call.args == (TasksetRunEntity,)
        assert call.kwargs == {
            "workspace": "default",
            "sort": "-created_at",
            "page": page,
            "page_size": 100,
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
    assert not files_service.calls


async def test_list_active_empty_and_errors(store, entity_store, monkeypatch) -> None:
    """Distinguish an empty active-run result from a failed Entity Store list request.

    Empty store/entity fixtures must yield [], then monkeypatch injects an outage
    that must propagate, preventing a false claim that no active runs exist.
    """
    assert await store.list_active("default", "user:alice") == []

    async def fail(*args, **kwargs):
        raise RuntimeError("list unavailable")

    monkeypatch.setattr(entity_store, "list", fail)
    with pytest.raises(RuntimeError, match="list unavailable"):
        await store.list_active("default", "user:alice")


@pytest.mark.parametrize("caller", [None, "user:alice", "user:bob"])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("taskset_ref", [None, "default/suite#3", "default/suite#4"])
async def test_list_runs_independent_filters_before_pagination(
    store, entity_store, files_service, acceptance, monkeypatch, caller, active, taskset_ref
):
    """Interleaved creators, phases, revisions and workspaces must not distort filtered totals or pages."""
    expected = []
    for index, (owner, phase, ref) in enumerate(
        (owner, phase, ref)
        for owner in ["user:alice", "user:bob"]
        for phase in TasksetRunPhase
        for ref in ["default/suite#3", "default/suite#4"]
    ):
        for workspace in ["default", "other"]:
            entity_store.caller = owner
            run = make_run(acceptance, phase=phase)
            run.name, run.workspace, run.taskset_ref = f"run-{index}", workspace, ref
            await entity_store.create(run)
            if (
                workspace == "default"
                and (caller is None or owner == caller)
                and (not active or phase.value in ["accepted", "running"])
                and (taskset_ref is None or ref == taskset_ref)
            ):
                expected.append(run.name)
    query = Mock(wraps=entity_store.list)
    monkeypatch.setattr(entity_store, "list", query)
    page = await store.list_runs("default", caller=caller, active=active, taskset_ref=taskset_ref, page=2, page_size=3)
    assert [run.name for run in page.data] == list(reversed(expected))[3:6]
    assert page.pagination.model_dump() == {
        "page": 2,
        "page_size": 3,
        "current_page_size": len(page.data),
        "total_pages": (len(expected) + 2) // 3,
        "total_results": len(expected),
    }
    conditions: list[FilterOperation] = []
    if caller is not None:
        conditions.append(ComparisonOperation(field="created_by", operator=FilterOperator.EQ, value=caller))
    if active:
        conditions.append(
            ComparisonOperation(field="data.phase", operator=FilterOperator.IN, value=["accepted", "running"])
        )
    if taskset_ref is not None:
        conditions.append(ComparisonOperation(field="data.taskset_ref", operator=FilterOperator.EQ, value=taskset_ref))
    query.assert_called_once_with(
        TasksetRunEntity,
        workspace="default",
        sort="-created_at",
        page=2,
        page_size=3,
        filter_operation=LogicalOperation(operator=FilterOperator.AND, operations=conditions) if conditions else None,
    )
    assert page.sort == "-created_at"
    assert not files_service.calls


async def test_list_runs_later_pages_empty_pages_and_failures(
    store, entity_store, files_service, acceptance, monkeypatch
):
    """More than 100 matching rows remain pageable; empty results differ from storage failures."""
    for index in range(150):
        for owner in ["user:alice", "user:bob"]:
            entity_store.caller = owner
            run = make_run(acceptance)
            run.name = f"{owner.split(':')[1]}-{index}"
            await entity_store.create(run)
    page = await store.list_runs("default", caller="user:alice", page=3, page_size=50)
    assert [run.name for run in page.data] == [f"alice-{index}" for index in range(49, -1, -1)]
    assert page.pagination.model_dump() == {
        "page": 3,
        "page_size": 50,
        "current_page_size": 50,
        "total_pages": 3,
        "total_results": 150,
    }
    outside = await store.list_runs("default", caller="user:alice", page=4, page_size=50)
    assert outside.data == [] and outside.pagination.total_results == 150
    assert outside.pagination.total_pages == 3 and outside.pagination.current_page_size == 0
    empty = await store.list_runs("other")
    assert empty.data == [] and empty.pagination.total_pages == empty.pagination.total_results == 0
    monkeypatch.setattr(entity_store, "list", Mock(side_effect=RuntimeError("list unavailable")))
    with pytest.raises(RuntimeError, match="list unavailable"):
        await store.list_runs("default")
    assert not files_service.calls


@pytest.mark.parametrize("phase", [None, *MemberPhase])
async def test_list_members_parent_workspace_phase_and_numeric_pages(
    store, entity_store, files_service, acceptance, monkeypatch, phase
):
    """Identical member names under different parents must never mix, even on filtered pages."""
    parents = []
    for name, workspace in [("selected", "default"), ("other", "default"), ("selected", "other")]:
        run = make_run(acceptance)
        run.name, run.workspace = name, workspace
        parents.append(await entity_store.create(run))
    for run in parents:
        for index in [100, 0, 10, 2, 1]:
            member = TasksetRunMemberEntity(
                name=f"task-{index}",
                workspace=run.workspace,
                task_index=index,
                phase=phase or MemberPhase.REGISTERED,
                task_job=f"{run.name}-job-{index}",
            )
            member._parent = run.id
            await entity_store.create(member)
    if phase is not None:
        excluded = TasksetRunMemberEntity(
            name="task-3",
            workspace="default",
            task_index=3,
            phase=MemberPhase.FAILED if phase != MemberPhase.FAILED else MemberPhase.SUCCEEDED,
        )
        excluded._parent = parents[0].id
        await entity_store.create(excluded)
    query = Mock(wraps=entity_store.list)
    monkeypatch.setattr(entity_store, "list", query)
    result = await store.list_members(parents[0], phase=phase, page=2, page_size=2)
    assert [member.task_index for member in result.data] == [2, 10]
    assert all(member._parent == parents[0].id for member in result.data)
    assert [member.task_job for member in result.data] == ["selected-job-2", "selected-job-10"]
    assert result.pagination.model_dump() == {
        "page": 2,
        "page_size": 2,
        "current_page_size": 2,
        "total_pages": 3,
        "total_results": 5,
    }
    conditions: list[FilterOperation] = [
        ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=parents[0].id)
    ]
    if phase is not None:
        conditions.append(ComparisonOperation(field="data.phase", operator=FilterOperator.EQ, value=phase))
    query.assert_called_once_with(
        TasksetRunMemberEntity,
        workspace="default",
        sort="task_index",
        page=2,
        page_size=2,
        filter_operation=LogicalOperation(operator=FilterOperator.AND, operations=conditions),
    )
    assert result.sort == "task_index"
    assert not files_service.calls


async def test_list_members_empty_pages_and_failure(store, entity_store, acceptance, monkeypatch):
    """A persisted run with no members has zero pages, while a failed list remains an error."""
    run = await entity_store.create(make_run(acceptance))
    empty = await store.list_members(run, phase=None, page=2, page_size=1)
    assert empty.data == []
    assert empty.pagination.model_dump() == {
        "page": 2,
        "page_size": 1,
        "current_page_size": 0,
        "total_pages": 0,
        "total_results": 0,
    }
    monkeypatch.setattr(entity_store, "list", Mock(side_effect=RuntimeError("members unavailable")))
    with pytest.raises(RuntimeError, match="members unavailable"):
        await store.list_members(run, phase=None, page=1, page_size=100)


async def test_list_members_typed_client_wire_filters_and_sort(files_client, files_service, acceptance):
    """The production client must send parent scope and numeric data.task_index ordering to Entity Store."""
    run = make_run(acceptance)
    run._id = "taskset_run-parent"
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [],
                "pagination": {
                    "page": 3,
                    "page_size": 2,
                    "current_page_size": 0,
                    "total_pages": 0,
                    "total_results": 0,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        store = TasksetRunStore(
            AsyncEntitiesClient(base_url="http://entities.test", http_client=http_client), files_client
        )
        page = await store.list_members(run, phase=MemberPhase.FAILED, page=3, page_size=2)
    assert len(requests) == 1 and page.data == []
    request = requests[0]
    assert request.url.path == "/apis/entities/v2/workspaces/default/entities/taskset_run_member"
    assert request.url.params["sort"] == "data.task_index"
    assert request.url.params["page"] == "3" and request.url.params["page_size"] == "2"
    assert json.loads(request.url.params["filter"]) == {
        "$and": [{"parent": {"$eq": "taskset_run-parent"}}, {"data.phase": {"$eq": "failed"}}],
    }
    assert not files_service.calls


# ---- cancel intent ---------------------------------------------------------------------------------------

CANCEL_AT = datetime(2026, 10, 15, 12, 30, tzinfo=UTC)


@pytest.mark.parametrize("phase", [TasksetRunPhase.ACCEPTED, TasksetRunPhase.RUNNING])
async def test_request_cancel_records_only_the_intent(store, entity_store, acceptance, phase) -> None:
    run = await entity_store.create(make_run(acceptance, phase=phase))
    cancelled = await store.request_cancel(run.workspace, run.name, now=CANCEL_AT)
    assert cancelled.cancel_requested_at == CANCEL_AT and cancelled.phase is phase
    stored = await entity_store.get(TasksetRunEntity, name=run.name, workspace=run.workspace)
    changed = {"cancel_requested_at", "db_version", "updated_at"}
    assert stored.model_dump(exclude=changed) == run.model_dump(exclude=changed)
    assert stored.cancel_requested_at == CANCEL_AT


async def test_request_cancel_again_keeps_the_first_time(store, entity_store, acceptance) -> None:
    run = await entity_store.create(make_run(acceptance))
    await store.request_cancel(run.workspace, run.name, now=CANCEL_AT)
    again = await store.request_cancel(run.workspace, run.name, now=CANCEL_AT + timedelta(minutes=1))
    assert again.cancel_requested_at == CANCEL_AT and again.db_version == 1


@pytest.mark.parametrize("phase", [TasksetRunPhase.COMPLETED, TasksetRunPhase.FAILED, TasksetRunPhase.EXPIRED])
async def test_request_cancel_of_a_finished_run_is_refused(store, entity_store, acceptance, phase) -> None:
    run = await entity_store.create(make_run(acceptance, phase=phase))
    with pytest.raises(RunAlreadyFinished, match=f"already finished: {phase.value}"):
        await store.request_cancel(run.workspace, run.name, now=CANCEL_AT)
    assert (await entity_store.get(TasksetRunEntity, name=run.name, workspace=run.workspace)) == run


async def test_request_cancel_of_a_run_the_controller_cancelled_returns_it(store, entity_store, acceptance) -> None:
    run = make_run(acceptance, phase=TasksetRunPhase.CANCELLED).model_copy(update={"cancel_requested_at": CANCEL_AT})
    run = await entity_store.create(run)
    assert await store.request_cancel(run.workspace, run.name, now=CANCEL_AT + timedelta(hours=1)) == run


async def test_request_cancel_of_an_unknown_run(store, entity_store, acceptance) -> None:
    await entity_store.create(make_run(acceptance))
    with pytest.raises(RunNotFound, match="other/"):
        await store.request_cancel("other", acceptance.plan.run_id, now=CANCEL_AT)


async def test_request_cancel_rereads_after_a_concurrent_write(store, entity_store, acceptance, monkeypatch) -> None:
    """The controller's pass writes between the read and the update: one re-read, and its fields are kept."""
    run = await entity_store.create(make_run(acceptance))
    real_update = entity_store.update
    written: list[int | None] = []

    async def racing(entity, **kwargs):
        written.append(entity.db_version)  # the version the read returned
        if len(written) == 1:
            current = await entity_store.get(TasksetRunEntity, name=run.name, workspace=run.workspace)
            await real_update(current.model_copy(update={"phase": TasksetRunPhase.RUNNING}))
        return await real_update(entity, **kwargs)

    monkeypatch.setattr(entity_store, "update", racing)
    cancelled = await store.request_cancel(run.workspace, run.name, now=CANCEL_AT)
    assert written == [0, 1]
    assert cancelled.phase is TasksetRunPhase.RUNNING and cancelled.cancel_requested_at == CANCEL_AT


async def test_request_cancel_gives_up_when_every_write_conflicts(store, entity_store, acceptance, monkeypatch) -> None:
    run = await entity_store.create(make_run(acceptance))
    update = Mock(side_effect=NemoEntityConflictError("stale"))

    async def conflicting(entity, **kwargs):
        return update(entity)

    monkeypatch.setattr(entity_store, "update", conflicting)
    with pytest.raises(CancelContended):
        await store.request_cancel(run.workspace, run.name, now=CANCEL_AT)
    assert update.call_count == CANCEL_WRITE_ATTEMPTS
    assert (await entity_store.get(TasksetRunEntity, name=run.name, workspace=run.workspace)) == run


async def test_members_are_created_once_queued_and_task_job_and_job_id_are_write_once(
    store, entity_store, acceptance
) -> None:
    run = await entity_store.create(make_run(acceptance))
    member = await store.create_member(run, 3)
    assert member is not None and (member.name, member.task_index, member.phase) == ("task-3", 3, "image_ready")
    assert member.parent == run.id
    assert await store.create_member(run, 3) is None  # exists = done

    intent = await store.update_member(member, task_job=f"{run.name}-task-3")
    with pytest.raises(NemoEntityConflictError):
        await store.update_member(member, task_job=f"{run.name}-task-3")  # stale version
    with pytest.raises(ValueError, match="write-once"):
        await store.update_member(intent, task_job="other")
    with pytest.raises(ValueError, match="registered without its Job's id"):
        await store.update_member(intent, phase=MemberPhase.REGISTERED)
    registered = await store.update_member(
        intent, phase=MemberPhase.REGISTERED, job_id="job-1", job_status=HelixJobStatus.ACTIVE
    )
    with pytest.raises(ValueError, match="job_id is write-once"):
        await store.update_member(registered, job_id="job-2")
    done = await store.update_member(registered, phase=MemberPhase.SUCCEEDED, bundle_ref="ref")
    with pytest.raises(ValueError, match="illegal taskset run member transition"):
        await store.update_member(done, phase=MemberPhase.FAILED)
    assert [m.phase for m in await store.list_all_members(run)] == [MemberPhase.SUCCEEDED]


async def test_set_state_checks_transitions(store, entity_store, acceptance) -> None:
    run = await entity_store.create(make_run(acceptance))
    # Its first slot writes running before any Job, so an accepted run never completes without it.
    with pytest.raises(ValueError, match="illegal taskset run transition"):
        await store.set_state(run, phase=TasksetRunPhase.COMPLETED)
    assert (await store.set_state(run, phase=TasksetRunPhase.EXPIRED)).phase is TasksetRunPhase.EXPIRED
    running = await entity_store.create(
        make_run(acceptance, phase=TasksetRunPhase.RUNNING).model_copy(update={"name": "r2"})
    )
    with pytest.raises(ValueError, match="illegal taskset run transition"):
        await store.set_state(running, phase=TasksetRunPhase.ACCEPTED)
    assert (await store.set_state(running, phase=TasksetRunPhase.RUNNING)).phase is TasksetRunPhase.RUNNING  # no-op


async def test_start_counts_both_deadlines_from_the_first_slot_in_one_versioned_write(
    store, entity_store, acceptance
) -> None:
    run = await entity_store.create(make_run(acceptance))
    now = acceptance.accepted_at + timedelta(minutes=50)
    started = await store.start(run, now=now)
    assert (started.phase, started.started_at) == (TasksetRunPhase.RUNNING, now)
    assert started.execution_deadline_at == now + timedelta(seconds=acceptance.timeout_seconds)
    grace = acceptance.finalization_deadline_at - acceptance.execution_deadline_at
    assert started.finalization_deadline_at == started.execution_deadline_at + grace  # submit's grace, kept
    assert (await store.find(run.workspace, run.name)) == started
    with pytest.raises(NemoEntityConflictError):  # a stale copy: another writer (here, the start) went first
        await store.start(run, now=now + timedelta(minutes=1))


async def test_open_runs_repeated_across_pages_are_listed_once(store, entity_store, acceptance, monkeypatch) -> None:
    run = await entity_store.create(make_run(acceptance))
    other = await entity_store.create(make_run(acceptance).model_copy(update={"name": "other"}))
    real_list = entity_store.list

    async def shifted(*args, **kwargs):
        page = await real_list(*args, **{**kwargs, "page": 1})  # a finished run shifted page 2 back onto page 1
        return page.model_copy(update={"pagination": page.pagination.model_copy(update={"total_pages": 2})})

    monkeypatch.setattr(entity_store, "list", shifted)
    assert [r.name for r in await store.list_open_runs(page_size=1000)] == [run.name, other.name]


async def test_accept_stores_the_callers_full_principal_for_the_controller(store, entity_store, acceptance) -> None:
    principal = AuthContext(
        principal_id="user:bob",
        principal_email="bob@ac9.test",
        principal_groups=["ac9-users"],
        principal_account_id="a1",
    )
    run = (await store.accept(replace(acceptance, auth_context=principal))).run
    stored = await entity_store.get(TasksetRunEntity, name=run.name, workspace=run.workspace)
    assert stored.auth_context == principal
    assert "_auth_context" in stored._get_data_fields()  # persisted with the record
    assert "auth_context" not in stored.model_dump()  # never part of the status DTO's source
    assert (await store.set_state(stored, phase=TasksetRunPhase.RUNNING)).auth_context == principal


async def test_accept_without_a_principal_stores_none(store, acceptance) -> None:
    assert (await store.accept(acceptance)).run.auth_context is None
