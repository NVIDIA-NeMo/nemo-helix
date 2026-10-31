# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller authorization, revision pinning and taskset run submission."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nemo_evals.api.fields import AgentRef, TasksetRef
from nemo_evals.api.service.taskset_run_service import TasksetRunResolutionError, pin_taskset_ref
from nemo_evals.api.taskset_run_schemas import TasksetRunInput
from nemo_evals.entities import TasksetRevisionEntity
from nemo_evals.filesets import FilesetRef
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import (
    HarborBuiltinAgentSource,
    HarborModelName,
    HarborRunnerTarget,
    RegisteredAgentSource,
)
from nemo_evals.revisions import RevisionContentMismatchError, publish_revision
from nemo_evals.source_authorization import remember_allowed
from nemo_evals.taskset_runs.entities import (
    MemberPhase,
    ResultState,
    TasksetRunEntity,
    TasksetRunMemberEntity,
    TasksetRunPhase,
)
from nemo_evals.taskset_runs.store import PlanStorageError, RunRequestConflict
from nemo_evals.taskset_runs.submission import TimeoutAboveMaximumError
from nemo_helix_plugin.jobs.schemas import HelixJobStatus

NOW = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "reference,expected",
    [
        ("suite#stable", "/apis/evals/v2/workspaces/default/tasksets/suite"),
        ("source/suite#latest", "/apis/evals/v2/workspaces/source/tasksets/suite"),
    ],
)
async def test_source_authorization_uses_qualified_workspace(service, authorizer, request_body, reference, expected):
    request_body["spec"]["tasks"] = reference
    await service._authorize(TasksetRunInput.model_validate(request_body), "default")
    authorizer.assert_awaited_once_with("GET", expected)


async def test_registered_agent_requires_source_read(service, authorizer, request_body):
    request_body["spec"]["target"]["source"] = {"agent": "agents/coder"}
    await service._authorize(TasksetRunInput.model_validate(request_body), "default")
    assert authorizer.await_args_list[0].args == ("GET", "/apis/evals/v2/workspaces/default/tasksets/suite")
    assert authorizer.await_args_list[1].args == ("GET", "/apis/agents/v2/workspaces/agents/agents/coder")


async def test_denied_source_read_propagates(service, authorizer, request_body, files_client, files_service):
    authorizer.side_effect = HTTPException(status_code=403, detail="source denied")
    with pytest.raises(HTTPException) as exc:
        await service._authorize(TasksetRunInput.model_validate(request_body), "default")
    assert exc.value.status_code == 403
    assert not files_service.calls


@pytest.mark.parametrize("fragment", ["", "#latest", "#stable", "digest"])
async def test_pin_uses_full_verified_content_digest(entity_store, published, fragment):
    _, revision = published
    fragment = f"#{revision.content_hash}" if fragment == "digest" else fragment
    pinned = await pin_taskset_ref(entity_store, TasksetRef(f"suite{fragment}"), "default")
    assert pinned.root == f"default/suite#{revision.content_hash}"


async def test_pin_retains_cross_workspace_source(entity_store, published):
    _, revision = published
    pinned = await pin_taskset_ref(entity_store, TasksetRef("default/suite#stable"), "destination")
    assert pinned.root == f"default/suite#{revision.content_hash}"


@pytest.mark.parametrize("reference", ["missing", "suite#missing", "suite#" + "f" * 64])
async def test_pin_missing_source_is_resolution_error(entity_store, published, reference):
    with pytest.raises(TasksetRunResolutionError):
        await pin_taskset_ref(entity_store, TasksetRef(reference), "default")


async def test_pin_tampered_revision_is_integrity_error(entity_store, published):
    _, revision = published
    revision.description = "tampered"
    await entity_store.update(revision)
    with pytest.raises(RevisionContentMismatchError):
        await pin_taskset_ref(entity_store, TasksetRef("suite"), "default")


async def test_accept_freezes_resolved_harbor_content(
    service, published, request_body, run_store, entity_store, files_client, files_service
):
    request_body["spec"].update(max_concurrent_tasks=8, fail_fast=True)
    result = await service.submit(
        TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice"
    )
    assert result.created is True
    assert result.status.created_by == "user:alice"
    assert result.status.phase == TasksetRunPhase.ACCEPTED
    assert result.status.result_state == ResultState.PENDING
    assert result.status.accepted_at == NOW
    # Until its first slot the run waits at most the queue timeout; its budget counts from that slot.
    assert result.status.timeout_seconds == 86_400
    assert result.status.started_at is None
    assert result.status.execution_deadline_at == NOW + timedelta(seconds=172_800)
    assert result.status.finalization_deadline_at == NOW + timedelta(seconds=172_800 + 1_800)
    run = await run_store.find("default", result.status.run_id)
    plan = await run_store.load_plan(run)
    assert plan.taskset_ref == f"default/suite#{published[1].content_hash}"
    assert plan.spec.tasks[0].spec.kind == "harbor"
    assert plan.spec.tasks[0].spec.provenance.revision_digest
    assert plan.spec.max_concurrent_tasks == 8 and plan.spec.fail_fast is True
    assert plan.sandbox_policy.profile == "harbor-standard"
    assert plan.sandbox_policy.connection.domain == "opensandbox.opensandbox-system.svc.cluster.local"
    assert plan.dispatcher_profile == "local-k8s" and plan.dispatcher_image.endswith("1" * 64)
    assert result.status.warnings == []
    assert len(files_service.filesets) == 1
    assert not (await entity_store.list(TasksetRunMemberEntity, workspace="default")).data


async def test_accept_defaults_dispatcher_image_to_platform_nhx_tasks(
    service, published, request_body, run_store, run_config, sandbox_platform
):
    run_config.runs.harbor.dispatcher_image = None
    result = await service.submit(
        TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice"
    )
    plan = await run_store.load_plan(await run_store.find("default", result.status.run_id))
    assert plan.dispatcher_image == f"{sandbox_platform.image_registry}/nhx-tasks:{sandbox_platform.image_tag}"


async def test_authorized_replay_preserves_original_run_without_resolution(
    service, published, request_body, monkeypatch, files_client, files_service, entity_store
):
    request = TasksetRunInput.model_validate(request_body)
    first = await service.submit(request, workspace="default", caller="user:alice")
    run = await service._store.find("default", first.status.run_id)
    run.phase = TasksetRunPhase.COMPLETED
    await entity_store.update(run)
    resolve = AsyncMock(side_effect=AssertionError("retry resolved again"))
    monkeypatch.setattr(AgentEvalJob, "to_spec", resolve)
    before = list(files_service.calls)
    replay = await service.submit(request, workspace="default", caller="user:alice")
    assert replay.created is False
    assert replay.status.run_id == first.status.run_id
    assert replay.status.accepted_at == first.status.accepted_at
    assert replay.status.phase == TasksetRunPhase.COMPLETED
    assert replay.status.warnings == []  # a replay returns the stored status only
    assert not [call for call in files_service.calls[len(before) :] if call[0] != "download"]


async def test_conflicting_retry_fails_before_resolution(service, published, request_body, monkeypatch):
    await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice")
    request_body["timeout_seconds"] = 100
    monkeypatch.setattr(AgentEvalJob, "to_spec", AsyncMock(side_effect=AssertionError("conflict resolved")))
    with pytest.raises(RunRequestConflict):
        await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice")


async def test_timeout_above_maximum_precedes_authorization_and_io(
    service, request_body, authorizer, files_client, files_service
):
    request_body["timeout_seconds"] = 259_201
    with pytest.raises(TimeoutAboveMaximumError):
        await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice")
    authorizer.assert_not_awaited()
    assert not files_service.calls


async def test_retry_rechecks_revoked_source_access(service, published, request_body, authorizer):
    request = TasksetRunInput.model_validate(request_body)
    await service.submit(request, workspace="default", caller="user:alice")
    authorizer.side_effect = HTTPException(status_code=403, detail="revoked")
    with pytest.raises(HTTPException) as exc:
        await service.submit(request, workspace="default", caller="user:alice")
    assert exc.value.status_code == 403


async def test_retry_rechecks_lowered_operator_maximum(service, published, request_body):
    request_body["timeout_seconds"] = 100
    request = TasksetRunInput.model_validate(request_body)
    await service.submit(request, workspace="default", caller="user:alice")
    service._config.runs.max_timeout_seconds = 99
    with pytest.raises(TimeoutAboveMaximumError):
        await service.submit(request, workspace="default", caller="user:alice")


async def test_submit_needs_a_key_from_the_route(service, published, request_body):
    request = TasksetRunInput.model_validate(request_body | {"idempotency_key": None})
    with pytest.raises(ValueError, match="idempotency_key"):
        await service.submit(request, workspace="default", caller="user:alice")


async def test_requests_with_new_keys_create_distinct_runs_and_warn(service, published, request_body):
    first, second = [
        await service.submit(
            TasksetRunInput.model_validate(request_body | {"idempotency_key": key}),
            workspace="default",
            caller="user:alice",
        )
        for key in ("first", "second")
    ]
    assert first.created and second.created
    assert first.status.run_id != second.status.run_id
    assert second.status.warnings == [f"{first.status.run_id} is already running {first.status.taskset_ref}"]


async def test_warning_failure_preserves_acceptance(service, published, request_body, monkeypatch, caplog):
    monkeypatch.setattr(service._store, "list_active", AsyncMock(side_effect=RuntimeError("listing failed")))
    outcome = await service.submit(
        TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice"
    )
    assert outcome.created
    assert outcome.status.warnings == []
    assert "listing failed" in caplog.text


async def test_republication_after_pin_does_not_change_resolved_members(
    service, published, request_body, monkeypatch, entity_store
):
    head, revision = published
    resolve = AgentEvalJob.to_spec

    async def republish(input_spec, **kwargs):
        assert input_spec.tasks.root == f"default/suite#{revision.content_hash}"
        head.tasks = []
        await entity_store.update(head)
        await publish_revision(entity_store, entity_store, head, TasksetRevisionEntity)
        return await resolve(input_spec, **kwargs)

    monkeypatch.setattr(AgentEvalJob, "to_spec", republish)
    result = await service.submit(
        TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice"
    )
    run = await service._store.find("default", result.status.run_id)
    plan = await service._store.load_plan(run)
    assert plan.taskset_ref == f"default/suite#{revision.content_hash}"
    assert len(plan.spec.tasks) == 1


@pytest.mark.parametrize(
    "failure,expected",
    [
        (ValueError("bad input"), TasksetRunResolutionError),
        (PermissionError("agent denied"), HTTPException),
        (RuntimeError("outage"), RuntimeError),
    ],
)
async def test_resolution_errors_keep_request_permission_and_outage_distinct(
    service, published, request_body, monkeypatch, failure, expected
):
    monkeypatch.setattr(AgentEvalJob, "to_spec", AsyncMock(side_effect=failure))
    with pytest.raises(expected):
        await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice")


@pytest.mark.parametrize("stage,kept", [("clock", False), ("plan", False), ("ambiguous", True)])
async def test_snapshot_cleanup_requires_proven_precreate_failure(
    service, published, request_body, monkeypatch, files_client, files_service, stage, kept
):
    from nemo_evals.api.fields import AgentRef
    from nemo_evals.filesets import FilesetRef
    from nemo_evals.jobs.agent_spec import HarborRunnerTarget, RegisteredAgentSource

    request = TasksetRunInput.model_validate(request_body)
    pinned, spec = await service._resolve(request, "default")
    snapshot = FilesetRef("agents/request-snapshot")
    files_service.filesets[("agents", "request-snapshot")] = {"skill.txt": b"skill"}
    spec.target = HarborRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef("agents/coder"), files=snapshot),
        agent_kwargs={"fabric_config": {"name": "coder"}},
    )
    monkeypatch.setattr(service, "_resolve", AsyncMock(return_value=(pinned, spec)))
    error = RuntimeError("original failure")
    if stage == "clock":
        service._clock = lambda: (_ for _ in ()).throw(error)
    else:
        failure = PlanStorageError("plan upload failed") if stage == "plan" else error
        monkeypatch.setattr(service._store, "accept", AsyncMock(side_effect=failure))
    with pytest.raises(
        PlanStorageError if stage == "plan" else RuntimeError, match="plan upload failed|original failure"
    ):
        await service.submit(request, workspace="default", caller="user:alice")
    assert (("agents", "request-snapshot") in files_service.filesets) is kept


async def test_snapshot_cleanup_failure_preserves_original_error(
    service,
    published,
    registered_agent,
    request_body,
    files_client,
    files_service,
    caplog,
):
    original = RuntimeError("acceptance clock failed")
    service._clock = lambda: (_ for _ in ()).throw(original)
    files_service.failures["delete"] = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError) as exc:
        await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice")
    assert exc.value is original
    assert len(files_service.filesets) == 1
    assert "cleanup failed" in caplog.text


async def test_submit_freezes_agent_and_inline_metric_without_compiling(
    service,
    published,
    registered_agent,
    request_body,
    entity_store,
    monkeypatch,
):
    from nemo_evals.api.fields import MetricInline, TaskRef
    from nemo_evals.entities import TaskEntity, TaskRevisionEntity
    from nemo_evals.shared.metric_bundles.bundles import bundle_metric
    from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
    from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric

    task = await entity_store.get(TaskEntity, "task-1", workspace="default")
    bundle = bundle_metric(ExactMatchMetric(reference="a", candidate="a"), InlineMetricBundlePackager())
    task.spec.metrics = [MetricInline.model_validate(bundle.model_dump(mode="json"))]
    await entity_store.update(task)
    revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    head, _ = published
    head.tasks = [TaskRef(f"default/task-1#{revision.content_hash}")]
    await entity_store.update(head)
    await publish_revision(entity_store, entity_store, head, TasksetRevisionEntity)
    compile_job = AsyncMock(side_effect=AssertionError("submit compiled a Job"))
    monkeypatch.setattr(AgentEvalJob, "compile", compile_job)
    result = await service.submit(
        TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice"
    )
    run = await service._store.find("default", result.status.run_id)
    plan = await service._store.load_plan(run)
    assert plan.spec.tasks[0].spec.metrics[0].payload.digest == bundle.payload.digest
    assert (
        plan.spec.target.agent_kwargs["fabric_config"]["harness"]["adapter_id"] == "nvidia.fabric.langchain.deepagents"
    )
    assert plan.spec.target.source.agent.root == "agents/coder"
    compile_job.assert_not_awaited()


def _deny(*paths: str):
    """Return an authorizer side effect that denies only the given nested source paths."""

    async def check(method: str, path: str) -> None:
        if path in paths:
            raise HTTPException(status_code=403, detail=f"denied {path}")

    return check


async def _republish_task_metrics(entity_store, published, metrics) -> None:
    """Give the published member task new metrics and republish the taskset to pin that revision."""
    from nemo_evals.api.fields import TaskRef
    from nemo_evals.entities import TaskEntity, TaskRevisionEntity

    task = await entity_store.get(TaskEntity, "task-1", workspace="default")
    task.spec.metrics = metrics
    await entity_store.update(task)
    revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    head, _ = published
    head.tasks = [TaskRef(f"default/task-1#{revision.content_hash}")]
    await entity_store.update(head)
    await publish_revision(entity_store, entity_store, head, TasksetRevisionEntity)


async def test_member_task_revoked_after_publication_is_rejected_before_snapshot_or_plan(
    service, published, registered_agent, request_body, authorizer, entity_store, files_service
):
    """The publisher could read the task; the submitter cannot, so nothing is copied into the run."""
    authorizer.side_effect = _deny("/apis/evals/v2/workspaces/default/tasks/task-1")
    with pytest.raises(HTTPException) as exc:
        await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:bob")
    assert exc.value.status_code == 403
    assert not files_service.filesets
    assert not (await entity_store.list(TasksetRunEntity, workspace="default")).data


async def test_unreadable_metric_is_rejected_before_lookup(service, published, request_body, authorizer, entity_store):
    """A denied caller gets 403 even for a metric that does not exist, so existence is not revealed."""
    from nemo_evals.api.fields import MetricRef

    await _republish_task_metrics(entity_store, published, [MetricRef("private/judge")])
    authorizer.side_effect = _deny("/apis/evals/v2/workspaces/private/metrics/judge")
    with pytest.raises(HTTPException) as exc:
        await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:bob")
    assert exc.value.status_code == 403
    assert not (await entity_store.list(TasksetRunEntity, workspace="default")).data


async def test_unreadable_metric_model_is_rejected_before_resolution(
    service, published, request_body, authorizer, entity_store
):
    from nemo_evals.api.fields import MetricInline
    from nemo_evals.shared.metric_bundles.bundles import bundle_metric
    from nemo_evals.shared.metric_bundles.hybrid import HybridMetricBundlePackager
    from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
    from nhx_evals_sdk.values import ModelRef
    from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore

    judge = LLMJudgeMetric(
        model=ModelRef("private/judge"),
        scores=[RangeScore(name="quality", minimum=0, maximum=1, parser=JSONScoreParser(json_path="quality"))],
    )
    bundle = bundle_metric(judge, HybridMetricBundlePackager())
    await _republish_task_metrics(entity_store, published, [MetricInline.model_validate_json(bundle.model_dump_json())])
    authorizer.side_effect = _deny(
        "/apis/models/v2/workspaces/private/models/judge",
        "/apis/inference-gateway/v2/workspaces/private/virtual-models/judge",
    )
    with pytest.raises(HTTPException) as exc:
        await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:bob")
    assert exc.value.status_code == 403
    assert not (await entity_store.list(TasksetRunEntity, workspace="default")).data


async def test_readable_nested_sources_are_checked_as_caller(service, published, request_body, authorizer):
    result = await service.submit(
        TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice"
    )
    assert result.created is True
    paths = [call.args[1] for call in authorizer.await_args_list]
    assert "/apis/evals/v2/workspaces/default/tasks/task-1" in paths


async def test_repeated_source_is_checked_once_but_denials_are_not_remembered():
    """Tasks sharing one metric need one check; a denied path is checked again on the next request."""
    authorize = AsyncMock()
    check = remember_allowed(authorize)
    await check("GET", "/metrics/judge")
    await check("GET", "/metrics/judge")
    assert authorize.await_count == 1
    authorize.side_effect = HTTPException(status_code=403)
    for _ in range(2):
        with pytest.raises(HTTPException):
            await check("GET", "/metrics/other")
    assert authorize.await_count == 3


async def test_get_reads_stored_state_without_plan_or_source_resolution(
    service, run_store, seed_run, files_service, authorizer, monkeypatch
):
    """One stored run read preserves terminal/cancellation/result state even with an unavailable plan."""
    run = await seed_run(
        phase=TasksetRunPhase.EXPIRED,
        result_state=ResultState.READY,
        cancel_requested_at=NOW,
        error="execution expired",
        idempotency_key="original-key",
    )
    find = AsyncMock(wraps=run_store.find)
    monkeypatch.setattr(run_store, "find", find)
    monkeypatch.setattr(run_store, "load_plan", AsyncMock(side_effect=AssertionError("plan read")))
    status = await service.get("default", run.name)
    find.assert_awaited_once_with("default", run.name)
    assert status.model_dump(mode="json") == {
        "run_id": "run-selected",
        "workspace": "default",
        "taskset_ref": "default/suite#" + "a" * 64,
        "phase": "expired",
        "result_state": "ready",
        "accepted_at": "2026-10-14T12:00:00Z",
        "timeout_seconds": 3600,
        "started_at": None,
        "execution_deadline_at": "2026-10-14T13:00:00Z",
        "finalization_deadline_at": "2026-10-14T14:00:00Z",
        "cancel_requested_at": "2026-10-14T12:00:00Z",
        "error": "execution expired",
        "created_by": "user:alice",
        "idempotency_key": "original-key",
        "progress": None,  # no members yet
    }
    assert not files_service.calls
    authorizer.assert_not_awaited()


async def test_get_counts_progress_from_every_member_page_without_jobs(
    service, run_store, entity_store, seed_run, fake_task_jobs, monkeypatch
):
    """``progress`` counts the run's members, across pages, on each read; Jobs is never called."""
    run = await seed_run(phase=TasksetRunPhase.RUNNING)
    states = [
        (MemberPhase.IMAGE_READY, None, None),  # queued
        (MemberPhase.IMAGE_READY, "intent", None),  # a create intent: holds a slot, not queued
        (MemberPhase.REGISTERED, "job", HelixJobStatus.ACTIVE),
        (MemberPhase.REGISTERED, "job", HelixJobStatus.ACTIVE),
        (MemberPhase.REGISTERED, "job", HelixJobStatus.PENDING),
        (MemberPhase.SUCCEEDED, "job", HelixJobStatus.COMPLETED),
        (MemberPhase.FAILED, "job", HelixJobStatus.ERROR),
    ]
    for index, (phase, job, job_status) in enumerate(states):
        member = TasksetRunMemberEntity(
            name=f"task-{index}",
            workspace="default",
            task_index=index,
            phase=phase,
            task_job=f"{run.name}-task-{index}" if job else None,
            job_status=job_status,
        )
        member._parent = run.id
        await entity_store.create(member)
    original = run_store.list_members

    async def small_pages(run, *, phase, page, page_size):  # the store reads pages of 1,000; force several
        return await original(run, phase=phase, page=page, page_size=3)

    monkeypatch.setattr(run_store, "list_members", small_pages)
    monkeypatch.setattr(fake_task_jobs, "is_subprocess_profile", AsyncMock(side_effect=AssertionError("Jobs call")))
    status = await service.get("default", run.name)
    assert status.progress is not None
    assert status.progress.model_dump(mode="json") == {
        "tasks": 7,
        "members": {"image_ready": 2, "registered": 3, "succeeded": 1, "failed": 1},
        "queued": 1,
        "task_jobs": {"active": 2, "pending": 1},
    }


async def test_get_missing_and_cross_workspace_returns_none(service, seed_run):
    """Only absent records produce None, including a name that exists in another workspace."""
    await seed_run()
    assert await service.get("default", "unknown") is None
    assert await service.get("other", "run-selected") is None


@pytest.mark.parametrize(
    "active,mine,expected",
    [
        (False, False, ["bob-active", "alice-done", "alice-active"]),
        (True, False, ["bob-active", "alice-active"]),
        (False, True, ["alice-done", "alice-active"]),
        (True, True, ["alice-active"]),
    ],
)
async def test_list_runs_service_preserves_filtering_and_page_metadata(
    service, run_store, seed_run, authorizer, files_service, monkeypatch, active, mine, expected
):
    """Public mine/active filters stay independent and retain the store's complete page metadata."""
    await seed_run("alice-active")
    await seed_run("alice-done", phase=TasksetRunPhase.COMPLETED)
    await seed_run("bob-active", caller="user:bob")
    await seed_run("outside", workspace="other")
    query = AsyncMock(wraps=run_store.list_runs)
    monkeypatch.setattr(run_store, "list_runs", query)
    result = await service.list_runs("default", caller="user:alice", active=active, mine=mine, page=1, page_size=2)
    query.assert_awaited_once_with(
        "default",
        caller="user:alice" if mine else None,
        active=active,
        page=1,
        page_size=2,
    )
    assert [run.run_id for run in result.data] == expected[:2]
    assert result.pagination.model_dump() == {
        "page": 1,
        "page_size": 2,
        "current_page_size": min(2, len(expected)),
        "total_pages": (len(expected) + 1) // 2,
        "total_results": len(expected),
    }
    assert result.sort == "-created_at" and result.filter is None
    assert not files_service.calls
    authorizer.assert_not_awaited()


async def test_list_runs_mine_normalizes_headerless_development_caller(service, seed_run):
    """Headerless mine requests select the same stable creator used by submit."""
    await seed_run("development-run", caller="user:development")
    await seed_run("alice-run")
    result = await service.list_runs("default", caller="", mine=True)
    assert [run.run_id for run in result.data] == ["development-run"]


async def test_list_tasks_distinguishes_missing_parent_from_empty_page(service, run_store, seed_run, monkeypatch):
    """Unknown parents return None without querying children; existing empty parents return a page."""
    run = await seed_run()
    members = AsyncMock(wraps=run_store.list_members)
    monkeypatch.setattr(run_store, "list_members", members)
    assert await service.list_tasks("other", run.name, phase=None, page=1, page_size=100) is None
    members.assert_not_awaited()
    result = await service.list_tasks("default", run.name, phase=None, page=1, page_size=100)
    assert result.data == [] and result.pagination.total_results == result.pagination.total_pages == 0
    members.assert_awaited_once_with(run, phase=None, page=1, page_size=100)


async def test_list_tasks_preserves_stored_task_details_and_page_metadata(
    service, run_store, entity_store, seed_run, files_service, authorizer, monkeypatch
):
    """Task projection retains Job/error details and filtered pagination without reading Jobs or plans."""
    run = await seed_run()
    for index in [100, 2, 10]:
        member = TasksetRunMemberEntity(
            name=f"task-{index}",
            workspace="default",
            task_index=index,
            phase=MemberPhase.FAILED,
            task_job=f"recorded-job-{index}",
            job_status=HelixJobStatus.ERROR,
            error="saved diagnostic",
        )
        member._parent = run.id
        await entity_store.create(member)
    find = AsyncMock(wraps=run_store.find)
    members = AsyncMock(wraps=run_store.list_members)
    monkeypatch.setattr(run_store, "find", find)
    monkeypatch.setattr(run_store, "list_members", members)
    monkeypatch.setattr(run_store, "load_plan", AsyncMock(side_effect=AssertionError("plan read")))
    result = await service.list_tasks("default", run.name, phase=MemberPhase.FAILED, page=2, page_size=1)
    assert [task.model_dump(mode="json") for task in result.data] == [
        {
            "task": "task-10",
            "task_index": 10,
            "phase": "failed",
            "job": "recorded-job-10",
            "job_status": "error",
            "error": "saved diagnostic",
            "publication": None,
        }
    ]
    assert result.pagination.model_dump() == {
        "page": 2,
        "page_size": 1,
        "current_page_size": 1,
        "total_pages": 3,
        "total_results": 3,
    }
    assert result.sort == "task_index" and result.filter is None
    find.assert_awaited_once_with("default", run.name)
    members.assert_awaited_once_with(run, phase=MemberPhase.FAILED, page=2, page_size=1)
    assert not files_service.calls
    authorizer.assert_not_awaited()


@pytest.mark.parametrize("operation", ["get", "list_runs", "list_tasks"])
async def test_read_service_propagates_storage_failures(service, seed_run, entity_store, monkeypatch, operation):
    """Unexpected storage failures propagate rather than becoming missing runs or empty lists."""
    await seed_run()
    target = "get" if operation == "get" else "list"
    monkeypatch.setattr(entity_store, target, AsyncMock(side_effect=RuntimeError("store unavailable")))
    with pytest.raises(RuntimeError, match="store unavailable"):
        if operation == "get":
            await service.get("default", "run-selected")
        elif operation == "list_runs":
            await service.list_runs("default", caller="user:alice")
        else:
            await service.list_tasks("default", "run-selected", phase=None, page=1, page_size=100)


async def _no_network_spec(service, request, target):
    pinned, spec = await service._resolve(request, "default")
    task = spec.tasks[0]
    config = {**task.spec.config, "environment": {**task.spec.config["environment"], "network_mode": "no-network"}}
    spec.tasks = [task.model_copy(update={"spec": task.spec.model_copy(update={"config": config})})]
    spec.target = target
    return pinned, spec


def _enforce_closed_book(service) -> None:
    service._config.runs.harbor.sandbox_profiles["harbor-standard"].enforce_closed_book = True


async def test_no_network_tasks_without_a_closed_book_route_are_refused_before_acceptance(
    service, published, request_body, monkeypatch, entity_store
):
    _enforce_closed_book(service)
    request = TasksetRunInput.model_validate(request_body)
    target = HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model=HarborModelName(name="gpt-5")))
    monkeypatch.setattr(service, "_resolve", AsyncMock(return_value=await _no_network_spec(service, request, target)))

    with pytest.raises(TasksetRunResolutionError, match=r"closed_book_required: no-network tasks \['task-1'\]"):
        await service.submit(request, workspace="default", caller="user:alice")
    assert not (await entity_store.list(TasksetRunEntity, workspace="default")).data


async def test_no_network_tasks_are_accepted_without_a_closed_book_route_by_default(
    service, published, request_body, monkeypatch
):
    request = TasksetRunInput.model_validate(request_body)
    target = HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex", model=HarborModelName(name="gpt-5")))
    monkeypatch.setattr(service, "_resolve", AsyncMock(return_value=await _no_network_spec(service, request, target)))

    outcome = await service.submit(request, workspace="default", caller="user:alice")

    assert outcome.created is True


async def test_a_refused_registered_agent_does_not_leave_its_files_snapshot(
    service, published, request_body, monkeypatch, files_service
):
    _enforce_closed_book(service)
    request = TasksetRunInput.model_validate(request_body)
    snapshot = FilesetRef("agents/request-snapshot")
    files_service.filesets[("agents", "request-snapshot")] = {"skill.txt": b"skill"}
    target = HarborRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef("agents/coder"), files=snapshot),
        agent_kwargs={"fabric_config": {"name": "coder"}},
    )
    monkeypatch.setattr(service, "_resolve", AsyncMock(return_value=await _no_network_spec(service, request, target)))

    with pytest.raises(TasksetRunResolutionError, match="closed_book_required"):
        await service.submit(request, workspace="default", caller="user:alice")
    assert ("agents", "request-snapshot") not in files_service.filesets
