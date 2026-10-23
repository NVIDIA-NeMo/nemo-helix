# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-4 tests for the Evaluator Run Controller: the real services answer what the unit-test fakes assume.

Each test talks to a real ``nemo services run`` platform, with auth off (``subprocess_platform``) and on
(``auth_subprocess_platform``). The controller itself is not started: these tests drive the real
``TasksetRunStore`` and ``JobsServiceRunJobs`` through clients that carry exactly the headers the controller's
``_service_client`` builds (the platform's own client provider makes them; only the base URL differs).

- The undelegated ``service:evals`` client: every record write, Job read and cancel.
- The creating client, from ``on_behalf_of(run)``: the run's stored ``AuthContext`` (id, email, groups, aliases).

No real task Job runs: task Jobs need a Kubernetes dispatcher profile. A trivial ``python -c 'sleep'`` Job on
the harness's ``subprocess/default`` profile stands in for one. Submit itself needs that profile too, so the
group-only submit check (M1 story 6) goes through the read routes, which use the same ``get_taskset_run_service``
dependency and its full-principal delegation.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from nemo_evals.taskset_runs.controller import NO_RETRY, JobsServiceRunJobs, on_behalf_of
from nemo_evals.taskset_runs.entities import MemberPhase, ResultState, TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.store import TasksetRunStore
from nemo_helix_plugin.auth import AuthContext, RuntimePrincipal
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.client_provider import get_async_nemo_client, get_forwarding_headers
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, CreateWorkspaceRequest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


@asynccontextmanager
async def _controller_client(
    base_url: str, delegate: str | RuntimePrincipal | None = None
) -> AsyncGenerator[AsyncNemoClient]:
    """The controller's client (``as_service="evals", internal=True``), aimed at ``base_url``.

    ``delegate=None`` is the undelegated service client; anything else is a creating client.
    """
    async with get_async_nemo_client(as_service="evals", internal=True, on_behalf_of=delegate) as template:
        headers = get_forwarding_headers(template)
    async with AsyncNemoClient(base_url=base_url, default_headers=headers, retry=NO_RETRY) as client:
        yield client


def _store(client: AsyncNemoClient) -> TasksetRunStore:
    return TasksetRunStore(AsyncEntitiesClient.from_client(client), AsyncFilesClient.from_client(client))


#: Sets up workspaces and bindings (users can't create workspaces; unknown service names are refused).
ADMIN = "service:platform"


def _admin(base_url: str) -> AsyncWorkspacesClient:
    headers = {"X-NHX-Principal-Id": ADMIN, "X-NHX-Actor-Aliases": ADMIN, "X-NHX-Internal": "true"}
    return AsyncWorkspacesClient(base_url=base_url, default_headers=headers, retry=NO_RETRY)


async def _new_workspace(base_url: str, prefix: str) -> str:
    """A workspace with no service bindings, like one a user owns.

    Its creator is bound ``Admin``, and any explicit binding replaces a service principal's default
    ``ServiceSystem`` role there (``common.rego`` ``effective_roles``): left in place, ``service:platform``
    (Jobs' own internal caller) would be refused in it. So a user takes over as Admin. Auth off binds no one.
    """
    name = f"{prefix}-{uuid.uuid4().hex[:8]}"
    async with _admin(base_url) as admin:
        await admin.create_workspace(body=CreateWorkspaceRequest(name=name))
        members = (await admin.list_workspace_members(workspace=name)).data()
        if ADMIN in {member.principal for member in members.data}:
            await admin.create_workspace_member(
                workspace=name,
                body=CreateWorkspaceMemberRequest(principal="ws-owner@example.com", roles=["Admin"]),
                query_params={"wait_role_propagation": False},
            )
            await admin.delete_workspace_member(
                workspace=name, principal_id=ADMIN, query_params={"wait_role_propagation": True}
            )
    return name


def _run(workspace: str, auth_context: AuthContext | None = None) -> TasksetRunEntity:
    """An accepted run record; the controller under test never loads its plan."""
    now = datetime.now(UTC)
    name = f"run-{uuid.uuid4().hex[:16]}"
    return TasksetRunEntity(
        name=name,
        workspace=workspace,
        plan_ref=f"{workspace}/{name}-plan#plan.json",
        plan_digest="0" * 64,
        taskset_ref=f"{workspace}/suite#1",
        request_digest="digest",
        accepted_at=now,
        timeout_seconds=3600,
        execution_deadline_at=now + timedelta(hours=1),
        finalization_deadline_at=now + timedelta(hours=2),
    ).with_auth_context(auth_context)


def _sleep_job(name: str) -> CreateHelixJobRequest:
    """A trivial Job on the harness's ``subprocess/default`` profile that stays active until cancelled."""
    return CreateHelixJobRequest.model_validate(
        {
            "name": name,
            "source": "taskset-run-controller-test",
            "spec": {},
            "platform_spec": {
                "steps": [
                    {
                        "name": "sleep",
                        "executor": {
                            "provider": "subprocess",
                            "profile": "default",
                            "command": ["python", "-c", "import time; time.sleep(300)"],
                        },
                    }
                ]
            },
        }
    )


@dataclass(frozen=True)
class GroupOnlyWorkspace:
    """A fresh workspace whose only binding for the user is their group's (``default`` binds ``*``)."""

    workspace: str
    group: str
    user: AuthContext  # what submit stores on the run: id, email, groups, aliases


_SHARED: dict[str, GroupOnlyWorkspace] = {}


async def _shared_group_only_workspace(base_url: str) -> GroupOnlyWorkspace:
    """One group-only workspace per platform for the tests that only read it: each setup waits ~30 s for roles."""
    if base_url not in _SHARED:
        _SHARED[base_url] = await _group_only_workspace(base_url)
    return _SHARED[base_url]


async def _group_only_workspace(base_url: str) -> GroupOnlyWorkspace:
    suffix = uuid.uuid4().hex[:8]
    workspace = await _new_workspace(base_url, "ctl-group")
    group = f"evals-team-{suffix}"
    email = f"alice-{suffix}@example.com"
    async with _admin(base_url) as admin:
        await admin.create_workspace_member(
            workspace=workspace,
            body=CreateWorkspaceMemberRequest(principal=group, roles=["Editor"]),
            query_params={"wait_role_propagation": True},
        )
        members = (await admin.list_workspace_members(workspace=workspace)).data()
    assert email not in {member.principal for member in members.data}, "the user must reach it only through the group"
    user = AuthContext(
        principal_id=email, principal_email=email, principal_groups=[group], principal_authz_aliases=[email]
    )
    return GroupOnlyWorkspace(workspace=workspace, group=group, user=user)


async def _check_store_semantics(base_url: str, workspace: str) -> None:
    """The Entity Store semantics ``TasksetRunStore`` relies on, as the evals service."""
    async with _controller_client(base_url) as service:
        store = _store(service)
        run = await store.entity_client.create(_run(workspace))

        # A named create under its parent: the first makes the member, a repeat is a 409, which reads as None.
        first = await store.create_member(run, 0)
        assert first is not None
        assert (first.name, first.task_index, first.phase, first.parent) == (
            "task-0",
            0,
            MemberPhase.IMAGE_READY,
            run.id,
        )
        assert await store.create_member(run, 0) is None
        # Uniqueness is per parent: another run in the same workspace gets its own task-0.
        other = await store.entity_client.create(_run(workspace))
        assert await store.create_member(other, 0) is not None

        # A versioned write from a stale copy is refused.
        await store.update_member(first, task_job=f"{run.name}-task-0", create_sent_at=datetime.now(UTC))
        with pytest.raises(NemoEntityConflictError):
            await store.update_member(first, phase=MemberPhase.FAILED, error="stale write")

        # Members list in numeric task_index order, not by name or creation time.
        for index in (10, 2, 1):
            assert await store.create_member(run, index) is not None
        members = await store.list_all_members(run)
        assert [member.task_index for member in members] == [0, 1, 2, 10]
        assert members[0].task_job == f"{run.name}-task-0" and members[0].phase is MemberPhase.IMAGE_READY

        # Run writes are versioned too, and an illegal transition never reaches the store.
        running = await store.set_state(run, phase=TasksetRunPhase.RUNNING)
        with pytest.raises(NemoEntityConflictError):
            await store.set_state(run, phase=TasksetRunPhase.FAILED, error="stale write")
        with pytest.raises(ValueError, match="illegal taskset run transition"):
            await store.set_state(running, phase=TasksetRunPhase.ACCEPTED)
        found = await store.find(workspace, run.name)
        assert found is not None and found.phase is TasksetRunPhase.RUNNING

        # The controller's cross-workspace scan sees both open runs.
        open_runs = {(open_run.workspace, open_run.name) for open_run in await store.list_open_runs()}
        assert {(workspace, run.name), (workspace, other.name)} <= open_runs


async def test_store_semantics_auth_off(subprocess_platform: str) -> None:
    await _check_store_semantics(subprocess_platform, await _new_workspace(subprocess_platform, "ctl-store"))


async def test_store_semantics_auth_on(auth_subprocess_platform: str) -> None:
    space = await _shared_group_only_workspace(auth_subprocess_platform)
    await _check_store_semantics(auth_subprocess_platform, space.workspace)


async def test_auth_context_snapshot_is_service_only_and_survives_service_writes(auth_subprocess_platform: str) -> None:
    async with _controller_client(auth_subprocess_platform) as service:
        store = _store(service)
        space = await _shared_group_only_workspace(auth_subprocess_platform)
        created = await store.entity_client.create(_run(space.workspace, space.user))

        stored = await store.find(space.workspace, created.name)
        assert stored is not None and stored.auth_context == space.user

        async with _controller_client(auth_subprocess_platform, on_behalf_of(stored)) as delegated:
            seen = await _store(delegated).find(space.workspace, created.name)
        assert seen is not None and seen.auth_context is None

        # A cancel-style versioned write and the controller's state write, both through the service store.
        cancel_requested = await store.entity_client.update(
            stored.model_copy(update={"cancel_requested_at": datetime.now(UTC)})
        )
        await store.set_state(cancel_requested, phase=TasksetRunPhase.CANCELLED, result_state=ResultState.NONE)

        reread = await store.find(space.workspace, created.name)
        assert reread is not None
        assert (reread.phase, reread.cancel_requested_at is not None) == (TasksetRunPhase.CANCELLED, True)
        assert reread.auth_context == space.user


async def _group_run(
    base_url: str, service: AsyncNemoClient, *, own_workspace: bool = False
) -> tuple[GroupOnlyWorkspace, TasksetRunEntity]:
    """A run with the group-only user's snapshot, read back as the controller reads it (service store).

    ``own_workspace`` for a test that removes the group's binding; the others share one workspace.
    """
    space = await (_group_only_workspace(base_url) if own_workspace else _shared_group_only_workspace(base_url))
    created = await _store(service).entity_client.create(_run(space.workspace, space.user))
    run = await _store(service).find(space.workspace, created.name)
    assert run is not None
    return space, run


async def _unbind_group(base_url: str, space: GroupOnlyWorkspace) -> None:
    async with _admin(base_url) as admin:
        await admin.delete_workspace_member(
            workspace=space.workspace, principal_id=space.group, query_params={"wait_role_propagation": True}
        )


async def _cancel_and_wait(reads: JobsServiceRunJobs, name: str) -> None:
    """GET and cancel ``name`` the way the controller does, then wait for Jobs to take the cancel."""
    assert await reads.get(name) is not None
    await reads.cancel(name)
    status = None
    for _ in range(60):
        current = await reads.get(name)
        assert current is not None
        status = current.status
        if status in (HelixJobStatus.CANCELLING, HelixJobStatus.CANCELLED):
            return
        await asyncio.sleep(1)
    raise AssertionError(f"Job {name} still {status} a minute after its cancel")


async def test_group_only_creator_creates_records_through_full_principal_only(auth_subprocess_platform: str) -> None:
    async with _controller_client(auth_subprocess_platform) as service:
        space, run = await _group_run(auth_subprocess_platform, service)
        principal = on_behalf_of(run)
        assert not isinstance(principal, str) and list(principal.groups) == [space.group]
        async with (
            _controller_client(auth_subprocess_platform, principal) as creating,
            _controller_client(auth_subprocess_platform, space.user.principal_id) as id_only,
        ):
            assert await _store(creating).create_member(run, 0) is not None
            # Delegating the id alone loses the group, and with it the workspace.
            with pytest.raises(PermissionDeniedError):
                await _store(id_only).create_member(run, 1)


async def test_service_client_gets_and_cancels_jobs_without_any_binding(auth_subprocess_platform: str) -> None:
    """The controller's GET and cancel need no binding: the group is gone and ``service:evals`` never had one.

    The Job is created by the undelegated client, so this holds whether or not Jobs accepts delegated creates
    (see ``test_a_creators_job_is_created_for_them_then_read_and_cancelled_as_the_service``).
    """
    async with _controller_client(auth_subprocess_platform) as service:
        space, run = await _group_run(auth_subprocess_platform, service, own_workspace=True)
        task_job = f"{run.name}-task-0"
        assert await JobsServiceRunJobs(service, service, space.workspace).create(_sleep_job(task_job)) is not None
        await _unbind_group(auth_subprocess_platform, space)
        await _cancel_and_wait(JobsServiceRunJobs(service, None, space.workspace), task_job)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Jobs fix pending: with auth on, Jobs checks its system records as the caller, so every create made for "
        "a user answers 409, and a forbidden delegated create answers 422, not 403. Passes once Jobs acts as "
        "service:jobs there; then drop this marker"
    ),
)
async def test_a_creators_job_is_created_for_them_then_read_and_cancelled_as_the_service(
    auth_subprocess_platform: str,
) -> None:
    """A task Job created for a group-only creator, then read and cancelled after the creator loses access.

    Without the Jobs fix, Jobs answers ``409 Could not confirm that storage of an earlier job ... was reclaimed``
    to any create made for a user: its pending-reclaim check lists the ``system`` workspace as the caller. The
    controller reads that 409 as a name conflict: GET 404, then ``job_name_conflict`` once past grace.
    """
    async with _controller_client(auth_subprocess_platform) as service:
        space, run = await _group_run(auth_subprocess_platform, service, own_workspace=True)
        task_job = f"{run.name}-task-0"
        async with _controller_client(auth_subprocess_platform, on_behalf_of(run)) as creating:
            jobs = JobsServiceRunJobs(service, creating, space.workspace)
            job = await jobs.create(_sleep_job(task_job))
            assert job is not None and job.name == task_job
            await _unbind_group(auth_subprocess_platform, space)
            # The creator lost access: new creates are refused (the controller's job_forbidden) ...
            with pytest.raises(PermissionDeniedError):
                await jobs.create(_sleep_job(f"{run.name}-task-1"))
        # ... while the undelegated service client still reads and cancels the creator's Job.
        await _cancel_and_wait(JobsServiceRunJobs(service, None, space.workspace), task_job)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Jobs fix pending: with auth on, Jobs checks its system records as the caller, so every create made for "
        "a user answers 409, and a forbidden delegated create answers 422, not 403. Passes once Jobs acts as "
        "service:jobs there; then drop this marker"
    ),
)
async def test_a_forbidden_delegated_job_create_is_a_403(auth_subprocess_platform: str) -> None:
    """A Job create for a principal without access to the workspace is a 403, the controller's ``job_forbidden``.

    Without the Jobs fix, Jobs answers ``422 Not allowed to list entities in workspace '<ws>'`` (the delegated
    user's workspace check inside Jobs), which the controller files as ``job_spec_invalid``.
    """
    async with _controller_client(auth_subprocess_platform) as service:
        space, run = await _group_run(auth_subprocess_platform, service)
        async with _controller_client(auth_subprocess_platform, space.user.principal_id) as id_only:
            with pytest.raises(PermissionDeniedError):
                await JobsServiceRunJobs(service, id_only, space.workspace).create(_sleep_job(f"{run.name}-task-0"))


async def test_submit_dependency_delegates_group_only_caller(auth_subprocess_platform: str) -> None:
    """``get_taskset_run_service`` with a group-only caller reads the workspace's run records (M1 story 6).

    Submit needs a Kubernetes dispatcher profile, so this goes through the run read routes, which build their
    service with the same dependency: an id-only delegation would not find the run.
    """
    async with _controller_client(auth_subprocess_platform) as service:
        store = _store(service)
        space = await _shared_group_only_workspace(auth_subprocess_platform)
        run = await store.entity_client.create(_run(space.workspace, space.user))
        await store.create_member(run, 0)

    user = space.user
    headers = {
        "X-NHX-Principal-Id": user.principal_id,
        "X-NHX-Principal-Email": user.principal_email or "",
        "X-NHX-Principal-Groups": ",".join(user.principal_groups),
        "X-NHX-Actor-Aliases": ",".join(user.principal_authz_aliases),
    }
    runs_path = f"/apis/evals/v2/workspaces/{space.workspace}/agent-evaluate/runs"
    async with httpx.AsyncClient(base_url=auth_subprocess_platform, headers=headers, timeout=30) as http:
        status = await http.get(f"{runs_path}/{run.name}")
        assert status.status_code == 200, status.text
        assert status.json()["run_id"] == run.name
        assert "auth_context" not in status.text and space.group not in status.text

        tasks = await http.get(f"{runs_path}/{run.name}/tasks")
        assert tasks.status_code == 200, tasks.text
        assert [task["task_index"] for task in tasks.json()["data"]] == [0]

        listed = await http.get(runs_path)
        assert listed.status_code == 200, listed.text
        assert run.name in [item["run_id"] for item in listed.json()["data"]]  # the workspace is shared

        # Without the group the same caller is refused.
        outsider = await http.get(f"{runs_path}/{run.name}", headers={"X-NHX-Principal-Groups": ""})
        assert outsider.status_code == 403, outsider.text
