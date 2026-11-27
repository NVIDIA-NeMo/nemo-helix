# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import Response
from nemo_helix_plugin.client.errors import InternalServerError, PermissionDeniedError
from nemo_helix_plugin.entities.types import EntityUpdate
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSecretEnvironmentVariableRef
from nemo_helix_plugin.jobs.types import HelixJobStepWithContext
from nhx.common.api.parsed_filter import ParsedFilter
from nhx.common.auth import AuthContext
from nhx.common.entities import SyncEntityClient
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import HelixJobSortField
from nhx.core.jobs.app.constants import STORAGE_RECLAIMED_AT
from nhx.core.jobs.app.dispatcher import (
    JobDispatcher,
    JobOperationConflictError,
    JobSecretValidationError,
)
from nhx.core.jobs.controllers import job_groups
from nhx.core.jobs.controllers.backends.exceptions import AdmissionDeferred, SchedulingDeferred
from nhx.core.jobs.controllers.job_group_admission import JobGroupAdmission
from nhx.core.jobs.controllers.job_groups import JobGroupReconciler, member_request
from nhx.core.jobs.entities import (
    HelixJob,
    HelixJobAttempt,
    HelixJobStep,
)
from nhx.core.jobs.job_groups import (
    HelixJobGroup,
    JobGroupControllerState,
    JobGroupMember,
    JobGroupMemberOutcome,
    JobGroupStatus,
    JobGroupSubmission,
)


@pytest.fixture
def reconciler(mock_dispatcher: JobDispatcher, monkeypatch: pytest.MonkeyPatch) -> JobGroupReconciler:
    @asynccontextmanager
    async def delegated_dispatcher(group: HelixJobGroup, store: EntityClient) -> AsyncGenerator[JobDispatcher]:
        assert group.auth_context is not None
        assert group.auth_context.principal_id == "submitter"
        yield mock_dispatcher

    controller = JobGroupReconciler()
    monkeypatch.setattr(controller, "delegated_dispatcher", delegated_dispatcher)
    return controller


@pytest.fixture
async def group(mock_store, sample_platform_job_request) -> HelixJobGroup:
    return await mock_store.create(
        HelixJobGroup(
            name="reconcile-group",
            workspace="default",
            submission=JobGroupSubmission(
                source="jobs-tests",
                template=sample_platform_job_request.platform_spec,
                spec={"value": "submitted"},
                members=[JobGroupMember(config_patch={"seed": 42}), JobGroupMember()],
                max_concurrent_jobs=1,
                default_timeout_seconds=42,
            ),
            controller_state=JobGroupControllerState(),
        ).with_auth_context(AuthContext(principal_id="submitter"))
    )


async def reconcile(controller: JobGroupReconciler, dispatcher: JobDispatcher, group: HelixJobGroup) -> HelixJobGroup:
    await controller.reconcile_group(group, dispatcher)
    return await dispatcher.store.get_by_id(HelixJobGroup, group.id)


@pytest.mark.parametrize("current_grant", [False, True])
async def test_deferred_authorization_uses_current_grants_without_saved_groups(
    group: HelixJobGroup, mock_store: EntityClient, monkeypatch: pytest.MonkeyPatch, current_grant: bool
) -> None:
    saved = AuthContext(
        principal_id="service:submitter",
        principal_groups=["old-actor-group"],
        principal_on_behalf_of="submitter",
        principal_on_behalf_of_groups=["old-subject-group"],
    )
    group.with_auth_context(saved)
    permission = AsyncMock(return_value=current_grant)
    auth = Mock(return_value=Mock(auth_enabled=True, principal=saved.to_principal(), check_permissions=permission))
    monkeypatch.setattr(job_groups, "AuthClient", auth)
    monkeypatch.setattr(job_groups, "get_auth_config", Mock())
    client = Mock(side_effect=RuntimeError("delegated client reached"))
    monkeypatch.setattr(job_groups, "get_async_nemo_client", client)
    error = RuntimeError if current_grant else JobOperationConflictError

    with pytest.raises(error):
        async with JobGroupReconciler().delegated_dispatcher(group, mock_store):
            pytest.fail("dispatch must not reach member creation in this test")

    permission.assert_awaited_once_with(group.workspace, subject=["jobs.create"])
    principal = auth.call_args.kwargs["principal"]
    assert principal.id == saved.principal_id
    assert principal.on_behalf_of == saved.principal_on_behalf_of
    assert not principal.groups
    assert not principal.on_behalf_of_groups
    assert group.auth_context == saved
    if current_grant:
        assert client.call_args.kwargs["on_behalf_of"] is principal
    else:
        client.assert_not_called()


async def test_controller_skips_unsupported_group_and_advances_page(
    reconciler: JobGroupReconciler,
    mock_store: EntityClient,
    group: HelixJobGroup,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    entities_client = mock_store._client
    unsupported = (
        await entities_client.update_entity_by_id(
            workspace=group.workspace,
            entity_type=HelixJobGroup.__entity_type__,
            entity_id=group.id,
            body=EntityUpdate(
                data={"controller_state": {"schema_version": 99, "status": "active"}},
                expected_db_version=group.db_version,
            ),
        )
    ).data()
    groups = [
        await mock_store.create(
            HelixJobGroup(
                workspace=group.workspace,
                name=name,
                submission=group.submission,
                controller_state=JobGroupControllerState(status=status),
            )
        )
        for name, status in (("same-page", JobGroupStatus.PENDING), ("next-page", JobGroupStatus.CANCELLING))
    ]

    @asynccontextmanager
    async def test_client(**kwargs):
        yield entities_client

    monkeypatch.setattr(job_groups, "get_async_nemo_client", test_client)
    monkeypatch.setattr(job_groups, "PAGE_SIZE", 2)
    with caplog.at_level("ERROR"):
        await reconciler.step(stopped=lambda: False)
        assert reconciler.page == 2
        same_page = await mock_store.get_by_id(HelixJobGroup, groups[0].id)
        assert same_page.controller_state.status == JobGroupStatus.ACTIVE
        assert same_page.member_state(0).phase == "preparing"
        await reconciler.step(stopped=lambda: False)

    assert any(
        record.levelname == "ERROR" and group.id in record.message and "schema version 99" in record.message
        for record in caplog.records
    )
    assert reconciler.page == 1
    assert (
        await mock_store.get_by_id(HelixJobGroup, groups[1].id)
    ).controller_state.status == JobGroupStatus.INCOMPLETE
    saved = (await entities_client.get_entity_by_id(entity_id=group.id)).data()
    assert saved.db_version == unsupported.db_version


async def test_group_recovers_store_assigned_job_id_after_lost_write(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert member.phase == "preparing"
    assert member.job_id is None
    assert (await store.list(HelixJob)).data == []

    save = store.update

    async def lose_group_write(entity):
        if isinstance(entity, HelixJobGroup):
            raise EntityConflictError("lost JobGroup progress write")
        return await save(entity)

    with monkeypatch.context() as patch:
        patch.setattr(store, "update", lose_group_write)
        with pytest.raises(EntityConflictError):
            await reconcile(reconciler, mock_dispatcher, group)

    group = await store.get_by_id(HelixJobGroup, group.id)
    assert group.member_state(0).job_id is None
    [job] = (await store.list(HelixJob, workspace=group.workspace)).data
    assert job.parent == group.id
    with pytest.raises(EntityNotFoundError):
        await store.get(HelixJob, job.name, workspace=group.workspace)
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert [item.id for item in (await store.list(HelixJob)).data] == [job.id]
    assert group.member_state(0).job_id == job.id
    assert job.platform_spec.steps[0].config["seed"] == 42
    assert job.platform_spec.steps[0].lifecycle.staleness_timeout_seconds == 42
    assert group.member_state(1).phase == "pending"


async def test_name_recovery_only_adopts_a_job_under_its_group_parent(
    reconciler: JobGroupReconciler, mock_dispatcher: JobDispatcher, group: HelixJobGroup
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    other = await store.create(
        HelixJob(
            name=group.member_job_name(0),
            workspace=group.workspace,
            source=group.submission.source,
            fileset="unused",
            platform_spec=group.submission.template,
        )
    )

    group = await reconcile(reconciler, mock_dispatcher, group)

    member = group.member_state(0)
    assert member.job_id and member.job_id != other.id
    owned = await store.get(HelixJob, other.name, workspace=group.workspace, parent=group.id)
    assert owned.id == member.job_id
    assert (await store.get(HelixJob, other.name, workspace=group.workspace)).id == other.id
    assert (await store.list(HelixJobAttempt)).data == []


async def test_first_pass_recovers_an_insert_before_making_the_attempt_runnable(
    reconciler: JobGroupReconciler, mock_dispatcher: JobDispatcher, group: HelixJobGroup
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert member.attempt_name is not None
    assert (await store.list(HelixJobAttempt)).data == []
    group = await reconcile(reconciler, mock_dispatcher, group)
    job_id = group.member_state(0).job_id
    assert job_id is not None
    job = await store.get_by_id(HelixJob, job_id)
    request = member_request(group, group.submission.members[0], job.name)
    prepared = await mock_dispatcher.prepare_group_attempt(job, request, member.attempt_name)

    assert job.current_attempt_id is None
    assert not await mock_dispatcher.has_first_attempt_step(prepared)
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.member_state(0).attempt_id == prepared.id
    assert not await mock_dispatcher.has_first_attempt_step(prepared)
    group = await reconcile(reconciler, mock_dispatcher, group)

    assert await mock_dispatcher.has_first_attempt_step(prepared)
    assert (await store.get_by_id(HelixJob, job.id)).current_attempt_id == prepared.id
    assert (await mock_dispatcher.get_job(job.id, job.workspace)).id == job.id
    steps = await mock_dispatcher.list_attempt_steps(
        job.id,
        prepared.id,
        job.workspace,
        ParsedFilter(None),
        page=1,
        page_size=10,
        sort=HelixJobSortField.CREATED_AT_ASC,
    )
    assert steps is not None
    [step] = steps[0]
    assert step.job == step.job_id == job.id
    assert [item.id for item in (await store.list(HelixJobAttempt)).data] == [prepared.id]


async def test_large_group_only_claims_preparation_needed_to_fill_capacity(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
) -> None:
    store = mock_dispatcher.store
    group.submission.members = [JobGroupMember() for _ in range(1000)]
    group.submission.max_concurrent_jobs = 2
    group = await store.update(group)

    group = await reconcile(reconciler, mock_dispatcher, group)

    assert len(group.controller_state.members_in_flight(len(group.submission.members))) == 2
    assert group.controller_state.occupied_slots() == 0
    assert (await store.list(HelixJob)).data == []


async def test_scheduler_reserves_slots_only_for_admitted_attempts(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    first = group.member_state(0)
    assert first.phase == "preparing"
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    first = group.member_state(0)
    assert first.phase == "tracking"
    assert first.job_id and first.attempt_id
    job = await store.get_by_id(HelixJob, first.job_id)
    attempt = await store.get_by_id(HelixJobAttempt, first.attempt_id)
    step = HelixJobStepWithContext(
        id="step",
        job="first",
        job_id=first.job_id,
        attempt_id=attempt.id,
        fileset="artifacts",
        workspace=group.workspace,
        name="first",
    )
    job = job.model_copy(update={"name": "renamed-job"})
    sync_store = Mock(spec=SyncEntityClient)
    admission = JobGroupAdmission(sync_store)
    sync_store.get_by_id.side_effect = [job, group]
    sync_store.find_one.return_value = attempt
    admission.reserve(step, job_id=first.job_id, current_attempt_id=attempt.id)
    assert sync_store.get_by_id.call_args.args == (HelixJobGroup, group.id)
    sync_store.find_one.assert_called_once_with(
        HelixJobAttempt,
        workspace=job.workspace,
        filter_str=json.dumps({"id": attempt.id, "parent": job.id}),
    )
    reserved = sync_store.update.call_args.args[0]
    assert reserved.member_state(0).phase == "admitted"
    group = await store.update(reserved)

    sync_store.reset_mock()
    sync_store.get_by_id.side_effect = [job, group]
    sync_store.find_one.return_value = attempt
    JobGroupAdmission(sync_store).reserve(
        step.model_copy(update={"id": "next-step", "name": "next-step"}),
        job_id=first.job_id,
        current_attempt_id=attempt.id,
    )
    sync_store.update.assert_not_called()

    standalone = await store.create(
        HelixJobAttempt(
            name="att-standalone",
            job=job.id,
            workspace=job.workspace,
            seq=1,
            platform_spec=job.platform_spec,
        )
    )
    standalone_job = job.model_copy(update={"current_attempt_id": standalone.id})
    sync_store.reset_mock()
    sync_store.get_by_id.side_effect = [standalone_job, group]
    sync_store.find_one.return_value = standalone
    standalone_step = step.model_copy(update={"attempt_id": standalone.id})
    admission.reserve(standalone_step, job_id=first.job_id, current_attempt_id=standalone.id)
    sync_store.update.assert_not_called()

    attempt.status = HelixJobStatus.COMPLETED
    await store.update(attempt)
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.member_state(0).phase == "completed"
    assert group.member_state(1).phase == "preparing"


async def test_missing_member_job_releases_its_slot(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    first = group.member_state(0)
    assert first.job_id and first.attempt_id
    assert first.job_id
    group.controller_state.admit_attempt(first.job_id, first.attempt_id, limit=1)
    group = await store.update(group)
    await store.delete_by_id(HelixJob, first.job_id)

    group = await reconcile(reconciler, mock_dispatcher, group)

    first = group.member_state(0)
    assert first.outcome == JobGroupMemberOutcome.ERROR
    assert group.controller_state.occupied_slots() == 0
    assert group.member_state(1).phase == "preparing"


async def test_scheduler_checks_capacity_and_group_version_before_dispatch(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
) -> None:
    store = mock_dispatcher.store
    group.submission.max_concurrent_jobs = 2
    group = await store.update(group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    first = group.member_state(0)
    second = group.member_state(1)
    assert first.attempt_id and second.attempt_id and second.job_id
    spare_capacity = group.model_copy(deep=True)
    group.submission.max_concurrent_jobs = 1
    spare_capacity.submission.max_concurrent_jobs = 1
    assert first.job_id
    group.controller_state.admit_attempt(first.job_id, first.attempt_id, limit=1)
    job = await store.get_by_id(HelixJob, second.job_id)
    attempt = await store.get_by_id(HelixJobAttempt, second.attempt_id)
    step = HelixJobStepWithContext(
        id="step",
        job="second",
        job_id=second.job_id,
        attempt_id=attempt.id,
        fileset="artifacts",
        workspace=group.workspace,
        name="second",
    )
    sync_store = Mock(spec=SyncEntityClient)
    sync_store.get_by_id.side_effect = [job, group]
    sync_store.find_one.return_value = attempt
    admission = JobGroupAdmission(sync_store)
    with pytest.raises(AdmissionDeferred, match="capacity"):
        admission.reserve(step, job_id=second.job_id, current_attempt_id=attempt.id)
    sync_store.update.assert_not_called()

    group = spare_capacity
    found = await store.find_one(
        HelixJobAttempt,
        workspace=job.workspace,
        filter_str=json.dumps({"id": attempt.id, "parent": job.id}),
    )
    assert found.id == attempt.id
    with pytest.raises(EntityNotFoundError):
        await store.find_one(
            HelixJobAttempt,
            workspace=job.workspace,
            filter_str=json.dumps({"id": attempt.id, "parent": first.job_id}),
        )

    earlier_attempt = await store.create(
        HelixJobAttempt(
            name="att-older",
            job=job.id,
            workspace=job.workspace,
            platform_spec=job.platform_spec,
            status=HelixJobStatus.CANCELLED,
        )
    )
    old_job = job.model_copy(update={"current_attempt_id": earlier_attempt.id})
    old_step = step.model_copy(update={"attempt_id": earlier_attempt.id})
    sync_store.get_by_id.side_effect = [old_job, group]
    sync_store.find_one.return_value = earlier_attempt
    with pytest.raises(SchedulingDeferred, match="has settled"):
        admission.reserve(old_step, job_id=second.job_id, current_attempt_id=earlier_attempt.id)
    sync_store.update.assert_not_called()

    sync_store.get_by_id.side_effect = [job, group]
    sync_store.find_one.return_value = attempt
    sync_store.update.side_effect = EntityConflictError("Cancel won the version check")
    with pytest.raises(SchedulingDeferred, match="confirm"):
        admission.reserve(step, job_id=second.job_id, current_attempt_id=attempt.id)

    group.controller_state.status = JobGroupStatus.CANCELLING
    sync_store.get_by_id.side_effect = [job, group]
    sync_store.find_one.return_value = attempt
    sync_store.update.reset_mock()
    with pytest.raises(SchedulingDeferred, match="no longer accepting"):
        admission.reserve(step, job_id=second.job_id, current_attempt_id=attempt.id)
    sync_store.update.assert_not_called()


@pytest.mark.parametrize("attempt_inserted", [False, True])
async def test_cancel_drains_a_late_job_insert(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
    attempt_inserted: bool,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    group.controller_state.status = JobGroupStatus.CANCELLING
    group = await store.update(group)

    late = await store.create(
        HelixJob(
            name=group.member_job_name(0),
            workspace=group.workspace,
            source=group.submission.source,
            fileset="unused",
            platform_spec=group.submission.template,
            status=HelixJobStatus.CREATION_PENDING,
        ).with_group_parent(group.id)
    )
    if attempt_inserted:
        reserved = group.member_state(0).attempt_name
        assert reserved is not None
        await store.create(
            HelixJobAttempt(name=reserved, job=late.id, workspace=late.workspace, platform_spec=late.platform_spec)
        )
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.controller_state.status == JobGroupStatus.CANCELLING
    assert (await store.get_by_id(HelixJob, late.id)).status == HelixJobStatus.DELETION_PENDING
    assert (await store.list(HelixJobStep)).data == []

    await store.delete_by_id(HelixJob, late.id)
    assert (await store.list(HelixJobAttempt)).data == []
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.controller_state.status == JobGroupStatus.INCOMPLETE
    assert {group.member_state(index).outcome for index in range(len(group.submission.members))} == {
        JobGroupMemberOutcome.CANCELLED
    }


def claim_first_for_retry(group: HelixJobGroup) -> None:
    state = group.controller_state
    state.finish_member(0, JobGroupMemberOutcome.ERROR)
    state.finish_member(1, JobGroupMemberOutcome.CANCELLED)
    assert state.settle(len(group.submission.members))
    assert state.retry_unsuccessful_members()
    assert group.member_state(0).attempt_id is None
    state.claim_member(0, "att-reserved-retry")


async def test_interrupted_initial_publication_resumes_from_recorded_attempt(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    pending = (await store.list(HelixJob)).data[0]
    initial = (await store.list(HelixJobAttempt)).data[0]
    assert pending.status == HelixJobStatus.CREATION_PENDING
    assert group.member_state(0).attempt_id == initial.id
    assert not await mock_dispatcher.has_first_attempt_step(initial)
    save = store.update

    async def lose_publication(entity):
        if isinstance(entity, HelixJob) and entity.status == HelixJobStatus.CREATED:
            raise EntityConflictError("lost Job publication")
        return await save(entity)

    with monkeypatch.context() as patch:
        patch.setattr(store, "update", lose_publication)
        group = await reconcile(reconciler, mock_dispatcher, group)
        assert not await mock_dispatcher.has_first_attempt_step(initial)

    group = await store.get_by_id(HelixJobGroup, group.id)
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.member_state(0).attempt_id == initial.id
    assert [job.id for job in (await store.list(HelixJob)).data] == [pending.id]
    assert [attempt.id for attempt in (await store.list(HelixJobAttempt)).data] == [initial.id]

    saved_job = await store.get_by_id(HelixJob, pending.id)
    sync_store = Mock(spec=SyncEntityClient)
    sync_store.get_by_id.side_effect = [saved_job, group]
    sync_store.find_one.return_value = initial
    step = HelixJobStepWithContext(
        id="step",
        job=saved_job.name,
        job_id=saved_job.id,
        attempt_id=initial.id,
        fileset=saved_job.fileset,
        workspace=group.workspace,
        name="first",
    )
    JobGroupAdmission(sync_store).reserve(step, job_id=saved_job.id, current_attempt_id=initial.id)
    sync_store.update.assert_called_once_with(group)
    assert group.controller_state.occupied_slots() == 1


async def test_transient_secret_lookup_does_not_trigger_fail_fast(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group.submission.fail_fast = True
    group.submission.template.steps[0].environment = [
        HelixJobEnvironmentVariable(name="TOKEN", from_secret=HelixJobSecretEnvironmentVariableRef(name="secret"))
    ]
    store = mock_dispatcher.store
    group = await store.update(group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    lookup = AsyncMock(side_effect=InternalServerError(Response(503)))
    monkeypatch.setattr(mock_dispatcher.secrets, "get_secret", lookup)

    group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert group.controller_state.status == JobGroupStatus.ACTIVE
    assert member.outcome is None and member.phase == "preparing"
    assert (await store.list(HelixJob)).data == []

    lookup.side_effect = None
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.member_state(0).attempt_id is not None


async def test_fileset_denial_waits_for_job_cleanup_before_retry(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    mock_files_client,
    group: HelixJobGroup,
) -> None:
    store = mock_dispatcher.store
    group.submission.members = group.submission.members[:1]
    group = await store.update(group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    mock_files_client.create_fileset.side_effect = PermissionDeniedError(Response(403))

    group = await reconcile(reconciler, mock_dispatcher, group)
    failed = group.member_state(0)
    assert failed.job_id and failed.reason
    assert failed.outcome is None
    assert not group.controller_state.status.terminal
    old_fileset = mock_files_client.create_fileset.call_args.kwargs["body"].name

    group = await reconcile(reconciler, mock_dispatcher, group)
    job = await store.get_by_id(HelixJob, failed.job_id)
    assert job.status == HelixJobStatus.DELETION_PENDING
    assert not group.controller_state.status.terminal
    await store.delete_by_id(HelixJob, job.id)

    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.controller_state.status == JobGroupStatus.INCOMPLETE
    assert group.member_state(0).reason == failed.reason
    assert group.controller_state.retry_unsuccessful_members()
    group = await store.update(group)
    mock_files_client.create_fileset.side_effect = None

    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.member_state(0).job_id
    assert group.member_state(0).job_id != job.id
    assert mock_files_client.create_fileset.call_args.kwargs["body"].name != old_fileset


async def test_retryable_member_error_leaves_peers_progressing(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group.submission.max_concurrent_jobs = 2
    group = await mock_dispatcher.store.update(group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    create_member = mock_dispatcher.create_group_member
    first_name = group.member_job_name(0)

    async def create(request, *args, **kwargs):
        if request.name == first_name:
            raise EntityConflictError("temporary failure")
        return await create_member(request, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(mock_dispatcher, "create_group_member", create)
        progress = Mock()
        await reconciler.reconcile_group(group, mock_dispatcher, on_progress=progress)
        assert progress.call_count == 2

    group = await mock_dispatcher.store.get_by_id(HelixJobGroup, group.id)
    assert group.member_state(0).phase == "preparing"
    assert group.member_state(1).job_id
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.member_state(0).job_id


async def test_retry_recovers_exact_attempt_and_preserves_submission(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert member.job_id and member.attempt_id and group.auth_context
    job = await store.get_by_id(HelixJob, member.job_id)
    original_attempt = await store.get_by_id(HelixJobAttempt, member.attempt_id)
    original_attempt.status = HelixJobStatus.ERROR
    await store.update(original_attempt)
    job.spec = {"value": "independent edit"}
    job = await store.update(job)

    claim_first_for_retry(group)
    group = await store.update(group)

    request = member_request(group, group.submission.members[0], group.member_job_name(0))
    assert group.auth_context is not None
    name = group.member_state(0).attempt_name
    assert name is not None
    attempt = await mock_dispatcher.prepare_group_attempt(job, request, name)
    monkeypatch.setattr(
        reconciler,
        "delegated_dispatcher",
        Mock(side_effect=JobOperationConflictError("Submitter can no longer create Jobs.")),
    )
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.member_state(0).attempt_id == attempt.id
    assert group.member_state(0).outcome is None
    attempts = (await store.list(HelixJobAttempt, filter_obj={"job": job.id})).data
    assert {item.id for item in attempts} == {original_attempt.id, attempt.id}
    assert attempt.spec == {"value": "submitted"}
    steps = (await store.list(HelixJobStep, filter_obj={"attempt_id": attempt.id})).data
    assert steps == []
    await reconcile(reconciler, mock_dispatcher, group)
    assert await mock_dispatcher.has_first_attempt_step(attempt)


async def test_retry_recovers_a_published_attempt_before_its_first_step(
    reconciler: JobGroupReconciler, mock_dispatcher: JobDispatcher, group: HelixJobGroup
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert member.job_id and member.attempt_id
    job = await store.get_by_id(HelixJob, member.job_id)
    claim_first_for_retry(group)
    original = await store.get_by_id(HelixJobAttempt, member.attempt_id)
    original.status = HelixJobStatus.ERROR
    await store.update(original)
    group = await store.update(group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    retry_id = group.member_state(0).attempt_id
    assert retry_id is not None and retry_id != original.id
    retry = await store.get_by_id(HelixJobAttempt, retry_id)
    job.current_attempt_id = retry.id
    await store.update(job)

    group = await reconcile(reconciler, mock_dispatcher, group)

    assert group.member_state(0).attempt_id == retry.id
    assert await mock_dispatcher.has_first_attempt_step(retry)
    attempts = (await store.list(HelixJobAttempt, filter_obj={"job": job.id})).data
    assert {attempt.id for attempt in attempts} == {member.attempt_id, retry.id}


async def test_fail_fast_stops_unclaimed_members_after_definitive_failure(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def invalid_reference(*args, **kwargs):
        raise JobSecretValidationError("Reference is no longer accessible")

    monkeypatch.setattr(mock_dispatcher, "create_group_member", invalid_reference)
    group.submission.fail_fast = True
    group = await mock_dispatcher.store.update(group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.controller_state.status == JobGroupStatus.CANCELLING
    assert group.member_state(0).outcome == JobGroupMemberOutcome.ERROR

    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.controller_state.status == JobGroupStatus.INCOMPLETE
    assert group.member_state(1).outcome == JobGroupMemberOutcome.CANCELLED
    assert (await mock_dispatcher.store.list(HelixJob)).data == []


async def test_cancel_settles_an_unpublished_initial_attempt(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
) -> None:
    store = mock_dispatcher.store
    for _ in range(3):
        group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert member.job_id is not None and member.attempt_id is not None
    pending = await store.get_by_id(HelixJob, member.job_id)
    assert pending.status == HelixJobStatus.CREATION_PENDING and pending.current_attempt_id is None

    group.controller_state.status = JobGroupStatus.CANCELLING
    group = await store.update(group)
    for _ in range(3):
        group = await reconcile(reconciler, mock_dispatcher, group)

    assert group.controller_state.status == JobGroupStatus.INCOMPLETE
    assert group.member_state(0).outcome == JobGroupMemberOutcome.CANCELLED
    attempt = await store.get_by_id(HelixJobAttempt, member.attempt_id)
    assert attempt.status == HelixJobStatus.CANCELLED
    assert await mock_dispatcher.has_first_attempt_step(attempt)
    job = await store.get_by_id(HelixJob, pending.id)
    assert job.status == HelixJobStatus.CANCELLED and job.current_attempt_id == attempt.id


async def test_cancel_fences_first_step_of_a_prepared_retry(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert member.job_id and member.attempt_id
    original = await store.get_by_id(HelixJobAttempt, member.attempt_id)
    original.status = HelixJobStatus.ERROR
    await store.update(original)
    claim_first_for_retry(group)
    group = await store.update(group)
    attempt = await store.create(
        HelixJobAttempt(
            name="att-reserved-retry",
            workspace=group.workspace,
            job=member.job_id,
            seq=1,
            platform_spec=original.platform_spec,
        )
    )
    group.controller_state.status = JobGroupStatus.CANCELLING
    group = await store.update(group)

    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    assert group.controller_state.status == JobGroupStatus.INCOMPLETE
    assert group.member_state(0).attempt_id == attempt.id
    assert group.member_state(0).outcome == JobGroupMemberOutcome.CANCELLED
    job = await store.get_by_id(HelixJob, member.job_id)
    assert job.current_attempt_id == attempt.id
    steps = (await store.list(HelixJobStep, filter_obj={"attempt_id": attempt.id})).data
    assert len(steps) == 1 and steps[0].status == HelixJobStatus.CANCELLED


async def test_delete_waits_for_owned_jobs_and_preserves_unrelated_jobs(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
    sample_platform_job_request,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert member.job_id and member.attempt_id
    unrelated = await mock_dispatcher.create_job(sample_platform_job_request, group.workspace)
    attempt = await store.get_by_id(HelixJobAttempt, member.attempt_id)
    attempt.status = HelixJobStatus.COMPLETED
    await store.update(attempt)
    for step in (await store.list(HelixJobStep, filter_obj={"attempt_id": attempt.id})).data:
        step.status = HelixJobStatus.COMPLETED
        await store.update(step)

    group.controller_state.status = JobGroupStatus.DELETION_PENDING
    group = await store.update(group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    owned = await store.get_by_id(HelixJob, member.job_id)
    assert owned.status == HelixJobStatus.DELETION_PENDING
    assert (await store.get_by_id(HelixJob, unrelated.id)).status == HelixJobStatus.CREATED

    await store.delete_by_id(HelixJob, owned.id, expected_db_version=owned.db_version)
    await reconciler.reconcile_group(group, mock_dispatcher)
    with pytest.raises(EntityNotFoundError):
        await store.get_by_id(HelixJobGroup, group.id)
    assert (await store.get_by_id(HelixJob, unrelated.id)).id == unrelated.id


async def test_retry_reports_reclaimed_member_storage(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert member.job_id and member.attempt_id
    attempt = await store.get_by_id(HelixJobAttempt, member.attempt_id)
    attempt.status = HelixJobStatus.CANCELLED
    attempt.status_details[STORAGE_RECLAIMED_AT] = "2026-11-24T00:00:00+00:00"
    await store.update(attempt)

    claim_first_for_retry(group)
    group = await store.update(group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    failed = group.member_state(0)
    assert failed.outcome == JobGroupMemberOutcome.ERROR
    assert failed.reason == "The previous Attempt's storage was reclaimed."
    assert [item.id for item in (await store.list(HelixJobAttempt, filter_obj={"job": member.job_id})).data] == [
        attempt.id
    ]


async def test_retry_can_follow_an_attempt_retired_by_the_recovery_sweep(
    reconciler: JobGroupReconciler,
    mock_dispatcher: JobDispatcher,
    group: HelixJobGroup,
) -> None:
    store = mock_dispatcher.store
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    member = group.member_state(0)
    assert member.job_id and member.attempt_id and group.auth_context
    original = await store.get_by_id(HelixJobAttempt, member.attempt_id)
    original.status = HelixJobStatus.ERROR
    await store.update(original)
    claim_first_for_retry(group)
    group = await store.update(group)
    retired = await store.create(
        HelixJobAttempt(
            name="att-reserved-retry",
            workspace=group.workspace,
            job=member.job_id,
            seq=1,
            platform_spec=original.platform_spec,
            status=HelixJobStatus.ERROR,
        )
    )

    group = await reconcile(reconciler, mock_dispatcher, group)
    group = await reconcile(reconciler, mock_dispatcher, group)
    job = await store.get_by_id(HelixJob, member.job_id)
    assert job.current_attempt_id == retired.id
    assert job.status == HelixJobStatus.ERROR
    assert group.member_state(0).attempt_id == retired.id
