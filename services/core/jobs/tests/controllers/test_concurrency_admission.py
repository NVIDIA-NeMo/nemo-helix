# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import httpx
from nemo_helix_plugin.client.errors import NemoTransportError
from nemo_helix_plugin.workspaces.types import WorkspaceLimits
from nhx.common.auth.models import AuthContext
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import HelixJobStepWithContext
from nhx.core.jobs.controllers.backends import JobUpdate
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.concurrency_admission import ConcurrencyAdmission, round_robin_by_user
from nhx.core.jobs.controllers.scheduler import JobScheduler
from nhx.core.jobs.entities import JobsUserConcurrencyLimit
from pytest import fixture

from services.core.jobs.tests.controllers.client_mocks import paginated_response

EXECUTOR = ("cpu", "default")


@fixture
def test_step_created(test_step_pending: HelixJobStepWithContext) -> HelixJobStepWithContext:
    return test_step_pending.model_copy(update={"status": HelixJobStatus.CREATED})


def _step(
    template: HelixJobStepWithContext,
    name: str,
    *,
    user: str | None = "alice",
    workspace: str = "default",
    status: HelixJobStatus = HelixJobStatus.CREATED,
) -> HelixJobStepWithContext:
    return template.model_copy(
        update={
            "id": name,
            "name": name,
            "workspace": workspace,
            "status": status,
            "auth_context": AuthContext(principal_id=user) if user else None,
        }
    )


def _workspaces_client(limits: dict[str, int] | None = None) -> MagicMock:
    """Workspaces client whose workspaces carry ``limits`` (name -> max_concurrent_jobs); others are unset."""
    client = MagicMock()

    def get_workspace(*, name: str):
        cap = (limits or {}).get(name)
        response = MagicMock()
        response.data.return_value.limits = WorkspaceLimits(max_concurrent_jobs=cap) if cap is not None else None
        return response

    client.get_workspace.side_effect = get_workspace
    return client


def _entities_client(overrides: dict[str, int] | None = None) -> MagicMock:
    """System entity client whose single page lists a ``JobsUserConcurrencyLimit`` per override."""
    client = MagicMock()
    entities = [
        JobsUserConcurrencyLimit(
            name=JobsUserConcurrencyLimit.name_for_user(user),
            workspace="system",
            user_id=user,
            max_concurrent_jobs=limit,
        )
        for user, limit in (overrides or {}).items()
    ]
    client.list.return_value = MagicMock(data=entities, pagination=MagicMock(total_pages=1))
    return client


def _admission(
    jobs_client,
    in_flight: list[HelixJobStepWithContext],
    *,
    executor: int = 100,
    workspace: int = 100,
    user: int = 100,
    workspace_caps: dict[str, int] | None = None,
    overrides: dict[str, int] | None = None,
    system_entities: MagicMock | None = None,
) -> ConcurrencyAdmission:
    jobs_client.list_steps.return_value = paginated_response(in_flight)
    admission = ConcurrencyAdmission(
        jobs_client,
        _workspaces_client(workspace_caps),
        executor_limits={EXECUTOR: executor},
        max_active_jobs_per_workspace=workspace,
        max_active_jobs_per_user=user,
        system_entities=system_entities if system_entities is not None else _entities_client(overrides),
    )
    admission.begin_pass()
    return admission


def test_admits_under_every_limit(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", status=HelixJobStatus.ACTIVE)]
    admission = _admission(mock_jobs_client, running, executor=2, workspace=2, user=2)

    assert admission.check(_step(test_step_created, "new")).admitted


def test_defers_at_executor_limit(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", user="bob", workspace="other", status=HelixJobStatus.ACTIVE)]
    admission = _admission(mock_jobs_client, running, executor=1)

    decision = admission.check(_step(test_step_created, "new"))

    assert not decision.admitted
    assert decision.reason is not None and "executor" in decision.reason


def test_defers_at_workspace_limit(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", user="bob", status=HelixJobStatus.PENDING)]
    admission = _admission(mock_jobs_client, running, workspace=1)

    decision = admission.check(_step(test_step_created, "new"))

    assert not decision.admitted
    assert decision.reason is not None and "workspace" in decision.reason


def test_defers_at_user_limit(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", workspace="other", status=HelixJobStatus.ACTIVE)]
    admission = _admission(mock_jobs_client, running, user=1)

    decision = admission.check(_step(test_step_created, "new"))

    assert not decision.admitted
    assert decision.reason is not None and "user" in decision.reason
    # A different user in the same workspace is unaffected.
    assert admission.check(_step(test_step_created, "other-user", user="bob")).admitted


def test_defers_then_admits_as_capacity_frees(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", status=HelixJobStatus.ACTIVE)]
    admission = _admission(mock_jobs_client, running, user=1)
    waiting = _step(test_step_created, "waiting")
    assert not admission.check(waiting).admitted

    mock_jobs_client.list_steps.return_value = paginated_response([])
    admission.begin_pass()

    assert admission.check(waiting).admitted


def test_non_in_flight_statuses_are_requested(mock_jobs_client, test_step_created):
    _admission(mock_jobs_client, [])

    query = mock_jobs_client.list_steps.call_args.kwargs["query_params"]
    assert query["filter[status]"] == "pending,active,cancelling,pausing"


def test_admitted_steps_count_against_later_steps_in_same_pass(mock_jobs_client, test_step_created):
    admission = _admission(mock_jobs_client, [], workspace=1)

    assert admission.check(_step(test_step_created, "first")).admitted
    assert not admission.check(_step(test_step_created, "second", user="bob")).admitted


def test_step_without_principal_skips_user_limit(mock_jobs_client, test_step_created):
    admission = _admission(mock_jobs_client, [], user=1)

    assert admission.check(_step(test_step_created, "a", user=None)).admitted
    assert admission.check(_step(test_step_created, "b", user=None)).admitted


def test_unconfigured_executor_is_not_capped(mock_jobs_client, test_step_created):
    admission = _admission(mock_jobs_client, [])
    admission._executor_limits.clear()

    assert admission.check(_step(test_step_created, "new")).admitted


def test_workspace_limit_uses_default_for_every_workspace(mock_jobs_client, test_step_created):
    admission = _admission(mock_jobs_client, [], workspace=7)

    assert admission.workspace_limit("default") == 7
    assert admission.workspace_limit("anything-else") == 7


def test_workspace_limit_prefers_configured_value(mock_jobs_client, test_step_created):
    admission = _admission(mock_jobs_client, [], workspace=7, workspace_caps={"capped": 2})

    assert admission.workspace_limit("capped") == 2
    assert admission.workspace_limit("other") == 7


def test_defers_at_configured_workspace_cap_below_default(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", status=HelixJobStatus.ACTIVE)]
    admission = _admission(mock_jobs_client, running, workspace=100, workspace_caps={"default": 1})

    decision = admission.check(_step(test_step_created, "new"))

    assert not decision.admitted
    assert decision.reason is not None and "workspace" in decision.reason


def test_defers_when_workspace_limits_cannot_be_read(mock_jobs_client, test_step_created):
    request = httpx.Request("GET", "http://localhost/apis/entities/v2/workspaces/default")
    admission = _admission(mock_jobs_client, [])
    admission._workspaces.get_workspace.side_effect = NemoTransportError(httpx.ConnectError("refused", request=request))

    assert not admission.check(_step(test_step_created, "new")).admitted


def test_failed_workspace_lookup_is_not_repeated_within_a_pass_but_retried_next_pass(
    mock_jobs_client, test_step_created
):
    request = httpx.Request("GET", "http://localhost/apis/entities/v2/workspaces/default")
    admission = _admission(mock_jobs_client, [])
    get_workspace = admission._workspaces.get_workspace
    healthy = get_workspace.side_effect
    get_workspace.side_effect = NemoTransportError(httpx.ConnectError("refused", request=request))

    assert not admission.check(_step(test_step_created, "n1")).admitted
    assert not admission.check(_step(test_step_created, "n2")).admitted
    assert get_workspace.call_count == 1

    get_workspace.side_effect = healthy
    admission.begin_pass()

    assert admission.check(_step(test_step_created, "n3")).admitted
    assert get_workspace.call_count == 2


def test_defers_when_load_cannot_be_counted(mock_jobs_client, test_step_created):
    request = httpx.Request("GET", "http://localhost/apis/jobs/v2/workspaces/-/jobs/-/steps")
    mock_jobs_client.list_steps.side_effect = NemoTransportError(httpx.ConnectError("refused", request=request))
    admission = ConcurrencyAdmission(
        mock_jobs_client,
        _workspaces_client(),
        executor_limits={EXECUTOR: 1},
        max_active_jobs_per_workspace=1,
        max_active_jobs_per_user=1,
    )
    admission.begin_pass()

    assert not admission.check(_step(test_step_created, "new")).admitted


def test_round_robin_interleaves_users_and_keeps_fifo_within_user(test_step_created):
    steps = [
        _step(test_step_created, "a1", user="alice"),
        _step(test_step_created, "a2", user="alice"),
        _step(test_step_created, "a3", user="alice"),
        _step(test_step_created, "b1", user="bob"),
        _step(test_step_created, "n1", user=None),
        _step(test_step_created, "b2", user="bob"),
    ]

    assert [s.name for s in round_robin_by_user(steps)] == ["a1", "b1", "n1", "a2", "b2", "a3"]


def test_scheduler_shares_scarce_capacity_across_users(
    backend_registry: BackendRegistry, mock_nemo_client, mock_jobs_client, test_step_created
):
    alice = [_step(test_step_created, f"a{i}", user="alice") for i in (1, 2, 3)]
    bob = [_step(test_step_created, "b1", user="bob")]
    admission = ConcurrencyAdmission(
        mock_jobs_client,
        _workspaces_client(),
        executor_limits={EXECUTOR: 2},
        max_active_jobs_per_workspace=100,
        max_active_jobs_per_user=100,
    )
    scheduler = JobScheduler(backend_registry, mock_nemo_client, admission=admission)
    # Calls in order: the schedulable queue; the in-flight recount for the running gauge; the
    # admission snapshot of in-flight work (none).
    mock_jobs_client.list_steps.side_effect = [
        paginated_response(alice + bob),
        paginated_response([]),
        paginated_response([]),
    ]

    with patch.object(
        scheduler, "schedule_step", return_value=JobUpdate(status=HelixJobStatus.PENDING)
    ) as schedule_step:
        scheduler.step()

    assert [c.args[0].name for c in schedule_step.call_args_list] == ["a1", "b1"]
    statuses = {
        c.kwargs["name"]: c.kwargs["body"].status for c in mock_jobs_client.update_job_step_status.call_args_list
    }
    assert statuses == {
        "a1": HelixJobStatus.PENDING,
        "b1": HelixJobStatus.PENDING,
        "a2": HelixJobStatus.QUEUED,
        "a3": HelixJobStatus.QUEUED,
    }


def test_override_above_default_admits(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", workspace="other", status=HelixJobStatus.ACTIVE)]
    admission = _admission(mock_jobs_client, running, user=1, overrides={"alice": 3})

    assert admission.check(_step(test_step_created, "new")).admitted


def test_override_below_default_defers(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", workspace="other", status=HelixJobStatus.ACTIVE)]
    admission = _admission(mock_jobs_client, running, user=5, overrides={"alice": 1})

    decision = admission.check(_step(test_step_created, "new"))

    assert not decision.admitted
    assert decision.reason is not None and "custom limit" in decision.reason and "1/1" in decision.reason


def test_user_without_override_uses_default(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", user="bob", workspace="other", status=HelixJobStatus.ACTIVE)]
    admission = _admission(mock_jobs_client, running, user=1, overrides={"alice": 3})

    assert not admission.check(_step(test_step_created, "new", user="bob")).admitted


def test_override_changes_apply_on_next_pass(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", workspace="other", status=HelixJobStatus.ACTIVE)]
    entities = _entities_client({})
    admission = _admission(mock_jobs_client, running, user=1, system_entities=entities)
    waiting = _step(test_step_created, "waiting")
    assert not admission.check(waiting).admitted

    entities.list.return_value = _entities_client({"alice": 2}).list.return_value
    admission.begin_pass()

    assert admission.check(waiting).admitted


def test_keeps_last_known_overrides_when_listing_fails(mock_jobs_client, test_step_created):
    running = [_step(test_step_created, "r1", workspace="other", status=HelixJobStatus.ACTIVE)]
    entities = _entities_client({"alice": 3})
    admission = _admission(mock_jobs_client, running, user=1, system_entities=entities)

    entities.list.side_effect = RuntimeError("entities down")
    admission.begin_pass()

    assert admission.check(_step(test_step_created, "new")).admitted


def test_overrides_cost_one_list_per_pass_regardless_of_queue_size(mock_jobs_client, test_step_created):
    entities = _entities_client({"alice": 10_000})
    admission = _admission(mock_jobs_client, [], user=1, system_entities=entities)

    for i in range(500):
        admission.check(_step(test_step_created, f"s{i}"))

    assert entities.list.call_count == 1


def test_defers_until_overrides_have_loaded_once(mock_jobs_client, test_step_created):
    entities = _entities_client({"alice": 1})
    entities.list.side_effect = RuntimeError("entities down")
    admission = _admission(mock_jobs_client, [], user=100, system_entities=entities)

    decision = admission.check(_step(test_step_created, "new"))

    assert not decision.admitted
    assert decision.reason is not None and "per-user limits" in decision.reason

    entities.list.side_effect = None
    entities.list.return_value = _entities_client({"alice": 1}).list.return_value
    admission.begin_pass()

    assert admission.check(_step(test_step_created, "new")).admitted
