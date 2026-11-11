# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatcher-level tests for job groups: CRUD, bulk-create expansion, timeout propagation, status aggregation."""

import asyncio

import pytest
from nemo_helix_plugin.jobs.schemas import JobGroupStatus
from nhx.common.entities import DEFAULT_WORKSPACE, EntityClient
from nhx.common.entities.client import EntityNotFoundError
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import (
    CreateJobGroupRequest,
    JobGroupMember,
    UpdateJobGroupRequest,
)
from nhx.core.jobs.app.dispatcher import (
    JobAlreadyExistsError,
    JobDeletionConflictError,
    JobDispatcher,
    JobOperationConflictError,
)
from nhx.core.jobs.app.schemas import HelixJobSpec, HelixJobStepSpec, StepLifecycle
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt, HelixJobStep, JobGroup


def _template(steps: int = 1) -> HelixJobSpec:
    return HelixJobSpec(
        steps=[
            HelixJobStepSpec(
                name=f"step-{i}",
                executor=TestConstants.TEST_EXECUTOR,
                config={"shared": "x"},
                lifecycle=StepLifecycle(staleness_timeout_seconds=5),
            )
            for i in range(steps)
        ]
    )


def _create_request(**overrides) -> CreateJobGroupRequest:
    defaults: dict = dict(
        name="my-group",
        source="test-source",
        template=_template(),
        members=[
            JobGroupMember(name="member-a", config_patch={"k": "a"}),
            JobGroupMember(name="member-b", config_patch={"k": "b"}),
        ],
    )
    defaults.update(overrides)
    return CreateJobGroupRequest(**defaults)


async def _members_of(dispatcher: JobDispatcher, group_id: str) -> list[HelixJob]:
    return await dispatcher._list_group_members(group_id, DEFAULT_WORKSPACE)


async def _set_member_status(store: EntityClient, member: HelixJob, status: HelixJobStatus) -> None:
    """Drive a member's current attempt (and its steps) to a status so aggregation/deletion can be exercised."""
    assert member.current_attempt_id is not None
    attempt = await store.get_by_id(HelixJobAttempt, member.current_attempt_id)
    attempt.status = status
    await store.update(attempt)
    # The real delete guard checks step terminality too, so move the attempt's steps in lockstep.
    steps = await store.list(
        HelixJobStep, filter_obj={"attempt_id": attempt.id}, workspace=member.workspace, page_size=1000
    )
    for step in steps.data:
        step.status = status
        await store.update(step)


async def _fail_member_via_rollup(dispatcher: JobDispatcher, store: EntityClient, member: HelixJob) -> None:
    """Drive a member to ERROR through the dispatcher roll-up path (not a direct write).

    Fail-fast rides the member status write, so the trigger must go through
    ``update_job_status_from_step`` -- the same path a controller uses -- rather
    than flipping the attempt/step rows directly like :func:`_set_member_status`.
    """
    assert member.current_attempt_id is not None
    steps = await store.list(
        HelixJobStep, filter_obj={"attempt_id": member.current_attempt_id}, workspace=member.workspace, page_size=1000
    )
    assert steps.data, "member has no step to drive to ERROR"
    await dispatcher.update_job_status_from_step(steps.data[0], HelixJobStatus.ERROR)


async def _member_status(store: EntityClient, member: HelixJob) -> HelixJobStatus:
    """Read a member's current effective status from its attempt."""
    fresh = await store.get_by_id(HelixJob, member.id)
    assert fresh.current_attempt_id is not None
    return (await store.get_by_id(HelixJobAttempt, fresh.current_attempt_id)).status


async def _drive_member_via_rollup(
    dispatcher: JobDispatcher, store: EntityClient, member: HelixJob, status: HelixJobStatus
) -> None:
    """Drive a member to ``status`` through the dispatcher roll-up path (not a direct write).

    Group status persistence rides the member status write, so a test must go through
    ``update_job_status_from_step`` (the controller's path) rather than flipping rows directly.
    The default template is single-step. COMPLETED is not reachable from CREATED in one hop
    (the step state machine requires CREATED -> ACTIVE -> COMPLETED), so walk through ACTIVE
    first; ERROR/ACTIVE/CANCELLED are each reachable directly from CREATED.
    """
    assert member.current_attempt_id is not None
    steps = await store.list(
        HelixJobStep, filter_obj={"attempt_id": member.current_attempt_id}, workspace=member.workspace, page_size=1000
    )
    assert steps.data, "member has no step to drive"
    step = steps.data[0]
    if status == HelixJobStatus.COMPLETED and step.status == HelixJobStatus.CREATED:
        await dispatcher.update_job_status_from_step(step, HelixJobStatus.ACTIVE)
        step = await store.get_by_id(HelixJobStep, step.id)
    await dispatcher.update_job_status_from_step(step, status)


async def _group_status(dispatcher: JobDispatcher, store: EntityClient, group_id: str) -> JobGroupStatus:
    """Read the group's PERSISTED status straight off the entity (what GET returns)."""
    return (await store.get_by_id(JobGroup, group_id)).status


def _failfast_request(fail_fast: bool, member_count: int = 3) -> CreateJobGroupRequest:
    return _create_request(
        fail_fast=fail_fast,
        members=[JobGroupMember(name=f"member-{i}", config_patch={"k": str(i)}) for i in range(member_count)],
    )


# ---------------------------------------------------------------------------
# Create + expansion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_group_expands_members_into_parented_jobs(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)

    assert group.name == "my-group"
    assert group.member_count == 2
    assert group.status == JobGroupStatus.PENDING

    members = await _members_of(mock_dispatcher, group.id)
    assert len(members) == 2
    # Every member is parented to the group via the Entity Store parent relationship.
    assert {m.parent for m in members} == {group.id}
    assert {m.name for m in members} == {"member-a", "member-b"}


@pytest.mark.asyncio
async def test_bulk_create_merges_template_with_per_member_patch(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    members = {m.name: m for m in await _members_of(mock_dispatcher, group.id)}

    # Shared template config is preserved and the per-member patch is merged on top.
    assert members["member-a"].platform_spec.steps[0].config == {"shared": "x", "k": "a"}
    assert members["member-b"].platform_spec.steps[0].config == {"shared": "x", "k": "b"}


@pytest.mark.asyncio
async def test_create_group_auto_generates_name_when_omitted(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_create_request(name=None), DEFAULT_WORKSPACE)
    assert group.name
    assert group.member_count == 2


@pytest.mark.asyncio
async def test_create_duplicate_group_name_raises(mock_dispatcher, mock_store):
    await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    with pytest.raises(JobAlreadyExistsError):
        await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)


# ---------------------------------------------------------------------------
# Timeout propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_group_default_timeout_propagates_to_member_step_lifecycle(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_create_request(default_timeout_seconds=120), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    for member in members:
        assert member.platform_spec.steps[0].lifecycle.staleness_timeout_seconds == 120


@pytest.mark.asyncio
async def test_member_timeout_override_wins_over_group_default(mock_dispatcher, mock_store):
    req = _create_request(
        default_timeout_seconds=120,
        members=[
            JobGroupMember(name="default-member", config_patch={}),
            JobGroupMember(name="override-member", config_patch={}, timeout_seconds=30),
        ],
    )
    group = await mock_dispatcher.create_job_group(req, DEFAULT_WORKSPACE)
    members = {m.name: m for m in await _members_of(mock_dispatcher, group.id)}

    assert members["default-member"].platform_spec.steps[0].lifecycle.staleness_timeout_seconds == 120
    assert members["override-member"].platform_spec.steps[0].lifecycle.staleness_timeout_seconds == 30


@pytest.mark.asyncio
async def test_no_group_timeout_leaves_template_value(mock_dispatcher, mock_store):
    # Template ships staleness_timeout_seconds=5; with no group default or override, it is left in place.
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    for member in members:
        assert member.platform_spec.steps[0].lifecycle.staleness_timeout_seconds == 5


# ---------------------------------------------------------------------------
# Persisted status (set at create; read off the stored field, not re-derived)
# ---------------------------------------------------------------------------
#
# The group status is a DENORMALIZED field persisted on the entity, not recomputed
# per read. It is set at create time from the expanded members, and kept current as
# members change by the status roll-up (see the roll-up tests below), so there is
# deliberately NO re-derive-on-read here. The pure derivation function
# (aggregate_job_group_status) is unit-tested across every member-state combination in
# test_job_groups_helpers.py.


@pytest.mark.asyncio
async def test_create_persists_pending_status_from_fresh_members(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    # Freshly-expanded members are CREATED, so the group's initial persisted status is PENDING.
    assert group.status == JobGroupStatus.PENDING
    stored = await mock_store.get_by_id(JobGroup, group.id)
    assert stored.status == JobGroupStatus.PENDING


@pytest.mark.asyncio
async def test_get_reads_persisted_status_without_re_deriving_from_members(mock_dispatcher, mock_store):
    """get_job_group returns the STORED status; poking member states does not change it at this layer.

    This pins the denormalization contract: the GET path does not recompute the group status
    from live members on read (that fan-out is what persisting the field avoids). The status
    roll-up (exercised by the roll-up tests below) is what re-persists the status when members
    change.
    """
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    await _set_member_status(mock_store, members[0], HelixJobStatus.ACTIVE)

    refreshed = await mock_dispatcher.get_job_group(group.id, DEFAULT_WORKSPACE)
    # Still the persisted PENDING — the read does NOT re-derive RUNNING from the poked member.
    assert refreshed.status == JobGroupStatus.PENDING


@pytest.mark.asyncio
async def test_get_reflects_an_externally_persisted_status(mock_dispatcher, mock_store):
    """When the stored status is updated (as the roll-up writer will do), get reflects it."""
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    stored = await mock_store.get_by_id(JobGroup, group.id)
    stored.status = JobGroupStatus.FAILED
    await mock_store.update(stored)

    refreshed = await mock_dispatcher.get_job_group(group.id, DEFAULT_WORKSPACE)
    assert refreshed.status == JobGroupStatus.FAILED
    assert refreshed.member_count == 2


# ---------------------------------------------------------------------------
# Get / list / update / delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_group_by_name_and_id(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    by_name = await mock_dispatcher.get_job_group("my-group", DEFAULT_WORKSPACE)
    by_id = await mock_dispatcher.get_job_group(group.id, DEFAULT_WORKSPACE)
    assert by_name.id == group.id
    assert by_id.id == group.id


@pytest.mark.asyncio
async def test_get_missing_group_returns_none(mock_dispatcher, mock_store):
    assert await mock_dispatcher.get_job_group("nope", DEFAULT_WORKSPACE) is None


@pytest.mark.asyncio
async def test_list_groups_returns_created_groups(mock_dispatcher, mock_store):
    await mock_dispatcher.create_job_group(_create_request(name="group-1"), DEFAULT_WORKSPACE)
    await mock_dispatcher.create_job_group(_create_request(name="group-2"), DEFAULT_WORKSPACE)

    groups, total = await mock_dispatcher.list_job_groups(DEFAULT_WORKSPACE, limit=10, offset=0)
    names = {g.name for g in groups}
    assert {"group-1", "group-2"}.issubset(names)
    assert total >= 2


@pytest.mark.asyncio
async def test_update_group_mutable_fields(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(
        _create_request(max_concurrent_jobs=2, fail_fast=False), DEFAULT_WORKSPACE
    )
    updated = await mock_dispatcher.update_job_group(
        group.id,
        DEFAULT_WORKSPACE,
        UpdateJobGroupRequest(max_concurrent_jobs=5, fail_fast=True, description="now described"),
    )
    assert updated.max_concurrent_jobs == 5
    assert updated.fail_fast is True
    assert updated.description == "now described"


@pytest.mark.asyncio
async def test_update_missing_group_returns_none(mock_dispatcher, mock_store):
    result = await mock_dispatcher.update_job_group("nope", DEFAULT_WORKSPACE, UpdateJobGroupRequest(description="x"))
    assert result is None


@pytest.mark.asyncio
async def test_delete_group_removes_group_and_members(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    # Drive members terminal so job deletion is allowed.
    for member in members:
        await _set_member_status(mock_store, member, HelixJobStatus.COMPLETED)

    deleted = await mock_dispatcher.delete_job_group(group.id, DEFAULT_WORKSPACE)
    assert deleted is True
    assert await mock_dispatcher.get_job_group(group.id, DEFAULT_WORKSPACE) is None
    with pytest.raises(EntityNotFoundError):
        await mock_store.get_by_id(JobGroup, group.id)
    # Members are gone too.
    assert await _members_of(mock_dispatcher, group.id) == []
    # Each member is individually deleted (not merely unreachable): the member-delete path ran.
    for member in members:
        with pytest.raises(EntityNotFoundError):
            await mock_store.get_by_id(HelixJob, member.id)


@pytest.mark.asyncio
async def test_delete_group_refuses_atomically_while_a_member_is_non_terminal(mock_dispatcher, mock_store):
    """A non-terminal member blocks group deletion, and NOTHING is deleted (atomic refusal).

    The first member is terminal and the second is not; a non-atomic loop would delete the
    first member before hitting the conflict on the second, leaving a corrupted partial group.
    The up-front pre-check must refuse before deleting anything.
    """
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    await _set_member_status(mock_store, members[0], HelixJobStatus.COMPLETED)
    await _set_member_status(mock_store, members[1], HelixJobStatus.ACTIVE)

    with pytest.raises(JobDeletionConflictError):
        await mock_dispatcher.delete_job_group(group.id, DEFAULT_WORKSPACE)

    # The group survives, and so does EVERY member -- including the terminal one that a
    # non-atomic delete would have removed before hitting the conflict.
    assert await mock_dispatcher.get_job_group(group.id, DEFAULT_WORKSPACE) is not None
    for member in members:
        assert await mock_store.get_by_id(HelixJob, member.id) is not None


@pytest.mark.asyncio
async def test_delete_missing_group_returns_false(mock_dispatcher, mock_store):
    assert await mock_dispatcher.delete_job_group("nope", DEFAULT_WORKSPACE) is False


# ---------------------------------------------------------------------------
# Individual member ops by id: a group member's name is unique only within its
# parent group, so the terminal-op re-fetch must be by the member's global id.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerun_member_job_by_id_succeeds(mock_dispatcher, mock_store):
    """rerun_job reruns a failed group member addressed by its id.

    A member's name is unique only within its parent group, so a re-fetch by
    workspace-scoped name would miss it. The rerun re-fetches by the member's
    immutable global id instead, which resolves the parented member exactly.
    """
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    member = (await _members_of(mock_dispatcher, group.id))[0]
    await _fail_member_via_rollup(mock_dispatcher, mock_store, member)
    assert await _member_status(mock_store, member) == HelixJobStatus.ERROR

    result = await mock_dispatcher.rerun_job(member.id, DEFAULT_WORKSPACE)

    assert result is not None
    # A fresh attempt is created back in CREATED (non-terminal), not a delete-and-recreate.
    assert await _member_status(mock_store, member) == HelixJobStatus.CREATED
    attempts = await mock_store.list(HelixJobAttempt, filter_obj={"job": member.id})
    assert len(attempts.data) == 2


@pytest.mark.asyncio
async def test_delete_member_job_by_id_succeeds(mock_dispatcher, mock_store):
    """delete_job deletes a terminal group member addressed by its id (parent-scoped name re-fetch not needed)."""
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    member = (await _members_of(mock_dispatcher, group.id))[0]
    await _set_member_status(mock_store, member, HelixJobStatus.COMPLETED)

    deleted = await mock_dispatcher.delete_job(member.id, DEFAULT_WORKSPACE)

    assert deleted is True
    with pytest.raises(EntityNotFoundError):
        await mock_store.get_by_id(HelixJob, member.id)


# ---------------------------------------------------------------------------
# fail_fast roll-up: a member reaching ERROR sweeps its siblings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_fast_cancels_non_terminal_siblings_when_a_member_errors(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=True), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)

    await _fail_member_via_rollup(mock_dispatcher, mock_store, members[0])

    # The failed member stays ERROR; the siblings are swept into cancellation.
    assert await _member_status(mock_store, members[0]) == HelixJobStatus.ERROR
    for sibling in members[1:]:
        assert await _member_status(mock_store, sibling) in (
            HelixJobStatus.CANCELLING,
            HelixJobStatus.CANCELLED,
        )


@pytest.mark.asyncio
async def test_fail_fast_disabled_leaves_siblings_untouched(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=False), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)

    await _fail_member_via_rollup(mock_dispatcher, mock_store, members[0])

    assert await _member_status(mock_store, members[0]) == HelixJobStatus.ERROR
    for sibling in members[1:]:
        assert await _member_status(mock_store, sibling) == HelixJobStatus.CREATED


@pytest.mark.asyncio
async def test_member_cancelled_not_error_does_not_trigger_fail_fast(mock_dispatcher, mock_store):
    """A user cancelling one member must NOT sweep the group -- only ERROR triggers fail_fast."""
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=True), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)

    # Cancel one member through the real roll-up path.
    steps = await mock_store.list(
        HelixJobStep,
        filter_obj={"attempt_id": members[0].current_attempt_id},
        workspace=DEFAULT_WORKSPACE,
        page_size=1000,
    )
    await mock_dispatcher.update_job_status_from_step(steps.data[0], HelixJobStatus.CANCELLED)

    assert await _member_status(mock_store, members[0]) == HelixJobStatus.CANCELLED
    for sibling in members[1:]:
        assert await _member_status(mock_store, sibling) == HelixJobStatus.CREATED


@pytest.mark.asyncio
async def test_fail_fast_leaves_already_terminal_siblings_alone(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=True), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    # One sibling already completed before the failure.
    await _set_member_status(mock_store, members[1], HelixJobStatus.COMPLETED)

    await _fail_member_via_rollup(mock_dispatcher, mock_store, members[0])

    assert await _member_status(mock_store, members[0]) == HelixJobStatus.ERROR
    # The completed sibling is untouched; the still-running one is swept.
    assert await _member_status(mock_store, members[1]) == HelixJobStatus.COMPLETED
    assert await _member_status(mock_store, members[2]) in (
        HelixJobStatus.CANCELLING,
        HelixJobStatus.CANCELLED,
    )


@pytest.mark.asyncio
async def test_fail_fast_is_idempotent_across_repeated_member_error_writes(mock_dispatcher, mock_store):
    """A replayed member-ERROR write must not re-fan-out cancels on siblings already cancelling."""
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=True), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)

    await _fail_member_via_rollup(mock_dispatcher, mock_store, members[0])
    siblings_after_first = {m.id: await _member_status(mock_store, m) for m in members[1:]}

    # Replay the same member's ERROR write (as a controller retry would). The
    # transition is now a no-op (already ERROR), so fail_fast must not fire again
    # and the siblings' cancellation states are untouched.
    steps = await mock_store.list(
        HelixJobStep,
        filter_obj={"attempt_id": (await mock_store.get_by_id(HelixJob, members[0].id)).current_attempt_id},
        workspace=DEFAULT_WORKSPACE,
        page_size=1000,
    )
    await mock_dispatcher.update_job_status_from_step(steps.data[0], HelixJobStatus.ERROR)

    assert await _member_status(mock_store, members[0]) == HelixJobStatus.ERROR
    for sibling in members[1:]:
        assert await _member_status(mock_store, sibling) == siblings_after_first[sibling.id]


@pytest.mark.asyncio
async def test_cancel_group_members_helper_skips_trigger_and_terminal(mock_dispatcher, mock_store):
    """The shared fan-out helper is reusable: it cancels only cancellable members and honors the exclude."""
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=True), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    await _set_member_status(mock_store, members[2], HelixJobStatus.COMPLETED)

    cancelled = await mock_dispatcher.cancel_group_members(group.id, DEFAULT_WORKSPACE, exclude_job_id=members[0].id)

    # Excluded member not cancelled, already-terminal member skipped; only member[1] swept.
    assert cancelled == [members[1].id]
    assert await _member_status(mock_store, members[0]) == HelixJobStatus.CREATED
    assert await _member_status(mock_store, members[2]) == HelixJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_fail_fast_concurrent_same_group_failures_do_not_deadlock(mock_dispatcher, mock_store):
    """Two members of a fail_fast group failing concurrently must not deadlock.

    The sibling fan-out runs after the failed member's own lock is released, so
    task A (holding no lock during its sweep) can take member B's lock while task
    B takes member A's -- there is no lock-while-holding-lock to form an AB-BA
    cycle. Regression guard for that fix; a reintroduced in-lock sweep would hang
    here (and trip the test timeout).
    """
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=True, member_count=4), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)

    # Drive two members into ERROR concurrently through the real roll-up path.
    await asyncio.wait_for(
        asyncio.gather(
            _fail_member_via_rollup(mock_dispatcher, mock_store, members[0]),
            _fail_member_via_rollup(mock_dispatcher, mock_store, members[1]),
        ),
        timeout=20,
    )

    # Both triggers ended in ERROR; every other member was swept into cancellation.
    assert await _member_status(mock_store, members[0]) == HelixJobStatus.ERROR
    assert await _member_status(mock_store, members[1]) == HelixJobStatus.ERROR
    for sibling in members[2:]:
        assert await _member_status(mock_store, sibling) in (
            HelixJobStatus.CANCELLING,
            HelixJobStatus.CANCELLED,
        )


@pytest.mark.asyncio
async def test_fail_fast_skips_sweep_when_group_already_terminal(mock_dispatcher, mock_store):
    """If every other member already settled before the error write, the aggregate-terminal guard skips the sweep."""
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=True, member_count=2), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    # The only sibling already completed, so once members[0] errors every member is terminal.
    await _set_member_status(mock_store, members[1], HelixJobStatus.COMPLETED)

    await _fail_member_via_rollup(mock_dispatcher, mock_store, members[0])

    # Nothing to cancel: the completed sibling is left exactly as it was.
    assert await _member_status(mock_store, members[0]) == HelixJobStatus.ERROR
    assert await _member_status(mock_store, members[1]) == HelixJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_fail_fast_re_reads_group_flag_so_clearing_it_before_the_sweep_disables_it(mock_dispatcher, mock_store):
    """The post-lock sweep re-reads fail_fast from fresh state; clearing it before the error write disables the sweep."""
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=True), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    # Turn fail_fast off after creation but before any member errors.
    await mock_dispatcher.update_job_group(group.id, DEFAULT_WORKSPACE, UpdateJobGroupRequest(fail_fast=False))

    await _fail_member_via_rollup(mock_dispatcher, mock_store, members[0])

    assert await _member_status(mock_store, members[0]) == HelixJobStatus.ERROR
    for sibling in members[1:]:
        assert await _member_status(mock_store, sibling) == HelixJobStatus.CREATED


# ---------------------------------------------------------------------------
# Group status roll-up: member transitions persist the group's denormalized status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_group_rolls_up_to_running_when_a_member_goes_active(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=False), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.PENDING

    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[0], HelixJobStatus.ACTIVE)

    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.RUNNING


@pytest.mark.asyncio
async def test_group_rolls_up_to_completed_when_the_last_member_completes(mock_dispatcher, mock_store):
    """The headline case: all members complete -> the group's persisted status becomes COMPLETED."""
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=3), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)

    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[0], HelixJobStatus.COMPLETED)
    # Not everyone is done yet -> still RUNNING.
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.RUNNING
    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[1], HelixJobStatus.COMPLETED)
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.RUNNING

    # Last member completes -> group flips to COMPLETED, persisted on the entity.
    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[2], HelixJobStatus.COMPLETED)
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.COMPLETED
    # And a GET reads exactly that stored value.
    assert (await mock_dispatcher.get_job_group(group.id, DEFAULT_WORKSPACE)).status == JobGroupStatus.COMPLETED


@pytest.mark.asyncio
async def test_group_rolls_up_to_failed_when_a_member_errors_and_rest_terminal(mock_dispatcher, mock_store):
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=2), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)

    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[0], HelixJobStatus.COMPLETED)
    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[1], HelixJobStatus.ERROR)

    # One errored, all terminal -> FAILED (persisted).
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.FAILED


@pytest.mark.asyncio
async def test_group_rollup_does_not_write_in_flight_cancelling(mock_dispatcher, mock_store):
    """A single member cancelled while others still run must NOT flip the whole group.

    Non-terminal CANCELLING is the in-flight cancel intent that the separate group-cancellation
    work owns top-down. With one member ACTIVE and one CANCELLED the live aggregate is CANCELLING,
    but this bottom-up roll-up must not write it -- the stored status stays at the last non-cancel
    roll-up (RUNNING).
    """
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=2), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)
    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[0], HelixJobStatus.ACTIVE)

    # Cancel one member directly through the roll-up path (as a single-member cancel would).
    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[1], HelixJobStatus.CANCELLED)

    # One ACTIVE + one CANCELLED -> aggregate CANCELLING -> NOT written. Stays RUNNING.
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.RUNNING


@pytest.mark.asyncio
async def test_group_rolls_up_to_cancelled_when_every_member_is_cancelled(mock_dispatcher, mock_store):
    """Terminal CANCELLED IS rolled up: a group whose every member ended cancelled is cancelled.

    Distinct from the in-flight CANCELLING intent (owned by the separate group-cancellation work):
    once every member has settled into a
    terminal CANCELLED state with no error, the group is a genuine settled end state -- the same
    class as COMPLETED/FAILED -- so this layer persists it.
    """
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=2), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)

    for member in members:
        await _drive_member_via_rollup(mock_dispatcher, mock_store, member, HelixJobStatus.CANCELLED)

    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.CANCELLED
    assert (await mock_dispatcher.get_job_group(group.id, DEFAULT_WORKSPACE)).status == JobGroupStatus.CANCELLED


@pytest.mark.asyncio
async def test_fail_fast_persists_failed_group_status_after_sweep(mock_dispatcher, mock_store):
    """Fail_fast + persistence together: a member errors, siblings get cancelled, group persists FAILED.

    The erroring member's own roll-up persists the group status from members; because the error
    makes the aggregate FAILED once the swept siblings settle, the stored status lands on FAILED.
    """
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=True, member_count=3), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)

    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[0], HelixJobStatus.ERROR)

    # Siblings swept to cancellation by fail_fast.
    for sibling in members[1:]:
        assert await _member_status(mock_store, sibling) in (HelixJobStatus.CANCELLING, HelixJobStatus.CANCELLED)
    # The persisted group status reflects the error (not left at PENDING/RUNNING).
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.FAILED


# -- Group cancellation: the cancel endpoint writes a single top-down intent, and the
# -- per-member teardown fans out off the request path and settles the group to cancelled.


@pytest.mark.asyncio
async def test_cancel_group_writes_cancelling_intent_in_a_single_write(mock_dispatcher, mock_store):
    """Cancelling a group records the group's own CANCELLING status without touching members.

    The caller signals intent once: cancel_job_group writes the group status and returns. The
    members are not swept here (that is the endpoint's off-path background fan-out); so right
    after the intent write the members are still in their pre-cancel state.
    """
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=3), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)
    for member in members:
        await _drive_member_via_rollup(mock_dispatcher, mock_store, member, HelixJobStatus.ACTIVE)

    response = await mock_dispatcher.cancel_job_group(group.id, DEFAULT_WORKSPACE)

    assert response is not None
    assert response.status == JobGroupStatus.CANCELLING
    stored = await mock_store.get_by_id(JobGroup, group.id)
    assert stored.status == JobGroupStatus.CANCELLING
    # Intent write alone does not cancel members; the teardown fan-out is separate.
    for member in members:
        assert await _member_status(mock_store, member) == HelixJobStatus.ACTIVE


@pytest.mark.asyncio
async def test_cancel_group_sweep_cancels_non_terminal_members(mock_dispatcher, mock_store):
    """The teardown sweep cancels every non-terminal member through the per-job cancel path."""
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=3), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)
    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[0], HelixJobStatus.ACTIVE)
    # members[1], members[2] remain in CREATED (never dispatched).

    await mock_dispatcher.cancel_job_group(group.id, DEFAULT_WORKSPACE)
    cancelled = await mock_dispatcher.cancel_group_members(group.id, DEFAULT_WORKSPACE)

    assert set(cancelled) == {m.id for m in members}
    for member in members:
        assert await _member_status(mock_store, member) in (HelixJobStatus.CANCELLING, HelixJobStatus.CANCELLED)


@pytest.mark.asyncio
async def test_cancel_group_cancels_not_yet_dispatched_members_without_dispatch(mock_dispatcher, mock_store):
    """Members still in CREATED (not yet started) go straight to CANCELLED, never dispatched."""
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=2), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)
    # Both members are freshly created (CREATED) -- nothing dispatched.

    await mock_dispatcher.cancel_job_group(group.id, DEFAULT_WORKSPACE)
    await mock_dispatcher.cancel_group_members(group.id, DEFAULT_WORKSPACE)

    for member in members:
        assert await _member_status(mock_store, member) == HelixJobStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_group_settles_to_cancelled_once_all_members_terminal(mock_dispatcher, mock_store):
    """Full lifecycle: cancel intent -> sweep -> members terminal -> group rolls up to CANCELLED.

    The group-status roll-up (which settles terminal CANCELLED) rides the member status write,
    so driving each swept member to its terminal CANCELLED through the roll-up path lands the
    stored group status on CANCELLED.
    """
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=2), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)

    await mock_dispatcher.cancel_job_group(group.id, DEFAULT_WORKSPACE)
    # Drive each member to the terminal CANCELLED through the roll-up path (what the controller
    # does as the backend finishes tearing each member down).
    for member in members:
        await _drive_member_via_rollup(mock_dispatcher, mock_store, member, HelixJobStatus.CANCELLED)

    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.CANCELLED
    assert (await mock_dispatcher.get_job_group(group.id, DEFAULT_WORKSPACE)).status == JobGroupStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_group_is_idempotent_for_an_already_cancelling_group(mock_dispatcher, mock_store):
    """Re-cancelling a group that is already cancelling is a no-op on the stored status."""
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=2), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)
    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[0], HelixJobStatus.ACTIVE)

    first = await mock_dispatcher.cancel_job_group(group.id, DEFAULT_WORKSPACE)
    second = await mock_dispatcher.cancel_job_group(group.id, DEFAULT_WORKSPACE)

    assert first.status == JobGroupStatus.CANCELLING
    assert second.status == JobGroupStatus.CANCELLING
    stored = await mock_store.get_by_id(JobGroup, group.id)
    assert stored.status == JobGroupStatus.CANCELLING


@pytest.mark.asyncio
async def test_cancel_group_refuses_a_group_that_has_already_finished(mock_dispatcher, mock_store):
    """A group that has settled into a terminal state is not reopened by a cancel request."""
    group = await mock_dispatcher.create_job_group(
        _failfast_request(fail_fast=False, member_count=2), DEFAULT_WORKSPACE
    )
    members = await _members_of(mock_dispatcher, group.id)
    for member in members:
        await _drive_member_via_rollup(mock_dispatcher, mock_store, member, HelixJobStatus.COMPLETED)
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.COMPLETED

    with pytest.raises(JobOperationConflictError):
        await mock_dispatcher.cancel_job_group(group.id, DEFAULT_WORKSPACE)

    # The terminal status is untouched.
    stored = await mock_store.get_by_id(JobGroup, group.id)
    assert stored.status == JobGroupStatus.COMPLETED


@pytest.mark.asyncio
async def test_cancel_missing_group_returns_none(mock_dispatcher):
    assert await mock_dispatcher.cancel_job_group("does-not-exist", DEFAULT_WORKSPACE) is None


@pytest.mark.asyncio
async def test_member_rollup_does_not_overwrite_a_stored_cancelling_intent(mock_dispatcher, mock_store):
    """A stored top-down CANCELLING intent survives a benign non-terminal member roll-up.

    After cancel writes the group's CANCELLING intent but before the teardown sweep moves any
    member, a member can still transition on its own (e.g. a controller moves it CREATED ->
    ACTIVE). That roll-up computes a non-terminal RUNNING -- which must NOT clobber the stored
    CANCELLING, or the group would report ``running`` while a cancel is in flight. Only a
    terminal roll-up may move the group off CANCELLING.
    """
    group = await mock_dispatcher.create_job_group(_failfast_request(fail_fast=False, member_count=2), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)

    await mock_dispatcher.cancel_job_group(group.id, DEFAULT_WORKSPACE)
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.CANCELLING

    # A member transitions non-terminally in the window before the sweep reaches it.
    await _drive_member_via_rollup(mock_dispatcher, mock_store, members[0], HelixJobStatus.ACTIVE)

    # The stored cancel intent is preserved, not overwritten with RUNNING.
    assert await _group_status(mock_dispatcher, mock_store, group.id) == JobGroupStatus.CANCELLING
