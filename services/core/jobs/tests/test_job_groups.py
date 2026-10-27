# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatcher-level tests for job groups: CRUD, bulk-create expansion, timeout propagation, status aggregation."""

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
from nhx.core.jobs.app.dispatcher import JobAlreadyExistsError, JobDeletionConflictError, JobDispatcher
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
# per read. 1431 sets it at create time from the expanded members; keeping it current
# as members change is the controller group roll-up's job (a later ticket), so there is
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

    This pins the denormalization contract: 1431 does not recompute the group status from
    live members on read (that fan-out is what persisting the field avoids). The controller
    roll-up — a later ticket — is what re-persists the status when members change.
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
async def test_delete_group_refuses_while_a_member_is_non_terminal(mock_dispatcher, mock_store):
    """A member still running must block group deletion with the same guard as deleting that job directly."""
    group = await mock_dispatcher.create_job_group(_create_request(), DEFAULT_WORKSPACE)
    members = await _members_of(mock_dispatcher, group.id)
    await _set_member_status(mock_store, members[0], HelixJobStatus.COMPLETED)
    await _set_member_status(mock_store, members[1], HelixJobStatus.ACTIVE)

    with pytest.raises(JobDeletionConflictError):
        await mock_dispatcher.delete_job_group(group.id, DEFAULT_WORKSPACE)
    # The group and its members survive the refused delete.
    assert await mock_dispatcher.get_job_group(group.id, DEFAULT_WORKSPACE) is not None


@pytest.mark.asyncio
async def test_delete_missing_group_returns_false(mock_dispatcher, mock_store):
    assert await mock_dispatcher.delete_job_group("nope", DEFAULT_WORKSPACE) is False
