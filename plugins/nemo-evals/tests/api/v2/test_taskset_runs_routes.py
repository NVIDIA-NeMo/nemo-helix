# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored run/status/task HTTP contracts, query validation and real authorization."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from nemo_evals.api.dependencies import get_service_run_store, get_taskset_run_service
from nemo_evals.api.v2.taskset_runs import router
from nemo_evals.service import EvaluatorPluginService
from nemo_evals.taskset_runs.entities import MemberPhase, ResultState, TasksetRunMemberEntity, TasksetRunPhase
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nhx.core.entities.service import EntitiesService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client

PATH = "/apis/evals/v2/workspaces/default/agent-evaluate/runs"


@pytest.fixture
def read_app(service, run_store):
    """Mount the shared router over the real service/store and a controlled caller."""
    app = FastAPI()
    app.include_router(router, prefix="/apis/evals/v2/workspaces/{workspace}")
    app.dependency_overrides[get_taskset_run_service] = lambda: service
    app.dependency_overrides[get_service_run_store] = lambda: run_store
    app.dependency_overrides[get_effective_principal_id] = lambda: "user:alice"
    return app


@pytest.fixture
async def read_client(read_app):
    """Yield a network-free HTTP client for the mounted run routes."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=read_app), base_url="http://test") as client:
        yield client


async def test_get_run_returns_all_stored_status_fields(read_client, seed_run, files_service, authorizer):
    """A populated record returns every public field without exposing its plan or persistence metadata."""
    from datetime import datetime, timezone

    await seed_run(
        phase=TasksetRunPhase.EXPIRED,
        result_state=ResultState.READY,
        cancel_requested_at=datetime(2026, 10, 14, 12, tzinfo=timezone.utc),
        error="execution expired",
        idempotency_key="original-key",
    )
    response = await read_client.get(f"{PATH}/run-selected")
    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-selected",
        "workspace": "default",
        "taskset_ref": "default/suite#" + "a" * 64,
        "phase": "expired",
        "result_state": "ready",
        "accepted_at": "2026-10-14T12:00:00Z",
        "timeout_seconds": 3600,
        "execution_deadline_at": "2026-10-14T13:00:00Z",
        "finalization_deadline_at": "2026-10-14T14:00:00Z",
        "cancel_requested_at": "2026-10-14T12:00:00Z",
        "error": "execution expired",
        "created_by": "user:alice",
        "idempotency_key": "original-key",
    }
    assert not files_service.calls
    authorizer.assert_not_awaited()


@pytest.mark.parametrize("suffix", ["unknown", "unknown/tasks", "run-selected", "run-selected/tasks"])
async def test_get_unknown_or_cross_workspace_run_returns_404(read_client, seed_run, suffix):
    """Neither single-run nor task reads disclose a record from another workspace."""
    await seed_run(workspace="other")
    response = await read_client.get(f"{PATH}/{suffix}")
    assert response.status_code == 404


@pytest.mark.parametrize(
    "active,mine,expected",
    [
        (False, False, ["bob-active", "alice-done", "alice-active"]),
        (True, False, ["bob-active", "alice-active"]),
        (False, True, ["alice-done", "alice-active"]),
        (True, True, ["alice-active"]),
    ],
)
async def test_run_lists_filter_before_paging_and_omit_nulls(
    read_client, seed_run, files_service, active, mine, expected
):
    """Every public filter combination produces newest-first pages and totals from the full filtered set."""
    await seed_run("alice-active")
    await seed_run("alice-done", phase=TasksetRunPhase.COMPLETED)
    await seed_run("bob-active", caller="user:bob", phase=TasksetRunPhase.RUNNING)
    await seed_run("outside", workspace="other")
    for page in range(1, len(expected) + 2):
        response = await read_client.get(PATH, params={"active": active, "mine": mine, "page": page, "page_size": 1})
        assert response.status_code == 200
        body = response.json()
        assert [run["run_id"] for run in body["data"]] == expected[page - 1 : page]
        assert body["pagination"] == {
            "page": page,
            "page_size": 1,
            "current_page_size": int(page <= len(expected)),
            "total_pages": len(expected),
            "total_results": len(expected),
        }
        assert body["sort"] == "-created_at"
        for run in body["data"]:
            assert (
                not {"warnings", "cancel_requested_at", "error", "idempotency_key", "plan_ref", "db_version"}
                & run.keys()
            )
    assert not files_service.calls


async def test_run_list_defaults_are_workspace_wide_and_trailing_slash_works(read_client, seed_run):
    """Omitting mine/active includes all workspace creators and the redirected slash URL behaves identically."""
    await seed_run("alice")
    await seed_run("bob", caller="user:bob", phase=TasksetRunPhase.FAILED)
    response = await read_client.get(PATH)
    assert response.status_code == 200
    assert [run["run_id"] for run in response.json()["data"]] == ["bob", "alice"]
    assert response.json()["pagination"] == {
        "page": 1,
        "page_size": 100,
        "current_page_size": 2,
        "total_pages": 1,
        "total_results": 2,
    }
    redirected = await read_client.get(PATH + "/", follow_redirects=True)
    assert redirected.status_code == 200 and redirected.json() == response.json()


async def test_m1_submission_has_empty_task_page(read_client, published, request_body):
    """The real submit path creates an accepted run whose task collection is empty until M4."""
    submitted = await read_client.post(PATH, json=request_body)
    assert submitted.status_code == 201
    response = await read_client.get(f"{PATH}/{submitted.json()['run_id']}/tasks")
    assert response.status_code == 200
    assert response.json() == {
        "data": [],
        "pagination": {
            "page": 1,
            "page_size": 100,
            "current_page_size": 0,
            "total_pages": 0,
            "total_results": 0,
        },
        "sort": "task_index",
    }


async def test_task_pages_are_numeric_parent_scoped_and_preserve_details(
    read_client, seed_run, entity_store, files_service
):
    """Stored members with identical names across runs stay isolated and order by their numeric index."""
    selected = await seed_run()
    other = await seed_run("run-other")
    for run in [selected, other]:
        for index, phase in [(100, MemberPhase.FAILED), (2, MemberPhase.FAILED), (10, MemberPhase.REGISTERED)]:
            member = TasksetRunMemberEntity(
                name=f"task-{index}",
                workspace="default",
                task_index=index,
                phase=phase,
                task_job=f"{run.name}-task-{index}" if index != 10 else None,
                error="verifier exited 1" if phase == MemberPhase.FAILED else None,
            )
            member._parent = run.id
            await entity_store.create(member)
    response = await read_client.get(f"{PATH}/run-selected/tasks", params={"page": 2, "page_size": 1})
    assert response.status_code == 200
    assert response.json() == {
        "data": [{"task": "task-10", "task_index": 10, "phase": "registered"}],
        "pagination": {"page": 2, "page_size": 1, "current_page_size": 1, "total_pages": 3, "total_results": 3},
        "sort": "task_index",
    }
    failed = await read_client.get(f"{PATH}/run-selected/tasks", params={"phase": "failed"})
    assert failed.status_code == 200
    assert failed.json()["data"] == [
        {
            "task": "task-2",
            "task_index": 2,
            "phase": "failed",
            "job": "run-selected-task-2",
            "error": "verifier exited 1",
        },
        {
            "task": "task-100",
            "task_index": 100,
            "phase": "failed",
            "job": "run-selected-task-100",
            "error": "verifier exited 1",
        },
    ]
    assert failed.json()["pagination"]["total_results"] == 2
    assert not files_service.calls


@pytest.mark.parametrize("phase", list(MemberPhase))
async def test_every_member_phase_is_a_valid_query(read_client, seed_run, entity_store, phase):
    """Every persisted phase can be selected through the public query enum."""
    run = await seed_run()
    member = TasksetRunMemberEntity(name="task-0", workspace="default", task_index=0, phase=phase)
    member._parent = run.id
    await entity_store.create(member)
    response = await read_client.get(f"{PATH}/run-selected/tasks", params={"phase": phase.value})
    assert response.status_code == 200
    assert response.json()["data"] == [{"task": "task-0", "task_index": 0, "phase": phase.value}]


@pytest.mark.parametrize("suffix", ["", "/run-selected/tasks"])
@pytest.mark.parametrize("params", [{"page": 0}, {"page_size": 0}, {"page_size": 1001}])
async def test_both_list_routes_validate_pagination(read_client, seed_run, suffix, params):
    """Invalid pagination bounds fail with 422 before querying the store."""
    await seed_run()
    response = await read_client.get(PATH + suffix, params=params)
    assert response.status_code == 422


@pytest.mark.parametrize("params", [{"active": "bogus"}, {"mine": "bogus"}])
async def test_run_lists_reject_malformed_booleans(read_client, params):
    """Invalid filters are rejected rather than silently changing the selected runs."""
    assert (await read_client.get(PATH, params=params)).status_code == 422


async def test_task_lists_reject_unknown_phase(read_client, seed_run):
    """Unknown phases cannot turn a filtered list into an unfiltered response."""
    await seed_run()
    assert (await read_client.get(f"{PATH}/run-selected/tasks", params={"phase": "bogus"})).status_code == 422


@pytest.mark.parametrize("suffix", ["", "/run-selected/tasks"])
@pytest.mark.parametrize("page_size", [1, 1000])
async def test_both_list_routes_accept_pagination_boundaries(read_client, seed_run, suffix, page_size):
    """The documented minimum and maximum page sizes remain valid."""
    await seed_run()
    response = await read_client.get(PATH + suffix, params={"page": 1, "page_size": page_size})
    assert response.status_code == 200 and response.json()["pagination"]["page_size"] == page_size


@pytest.mark.parametrize("suffix,operation", [("", "list"), ("/run-selected", "get"), ("/run-selected/tasks", "list")])
async def test_read_routes_sanitize_unexpected_storage_errors(
    read_client, seed_run, entity_store, monkeypatch, suffix, operation
):
    """Storage outages produce a generic 500 instead of leaking exception details or returning an empty collection."""
    await seed_run()
    monkeypatch.setattr(entity_store, operation, AsyncMock(side_effect=RuntimeError("private database failure")))
    response = await read_client.get(PATH + suffix)
    assert response.status_code == 500 and response.json() == {"detail": "Internal server error"}


async def test_status_cost_is_one_run_read_with_twenty_thousand_members(
    read_client, seed_run, entity_store, run_store, files_service, monkeypatch
):
    """Status work stays constant with 20 runs of 1,000 tasks; this is a query budget, not a latency benchmark."""
    for run_index in range(20):
        run = await seed_run(f"run-{run_index}")
        for task_index in range(1000):
            member = TasksetRunMemberEntity(
                name=f"task-{task_index}",
                workspace="default",
                task_index=task_index,
                phase=MemberPhase.REGISTERED,
            )
            member._parent = run.id
            await entity_store.create(member)
    get = AsyncMock(wraps=entity_store.get)
    monkeypatch.setattr(entity_store, "get", get)
    monkeypatch.setattr(entity_store, "list", AsyncMock(side_effect=AssertionError("status joined members")))
    monkeypatch.setattr(run_store, "load_plan", AsyncMock(side_effect=AssertionError("status loaded plan")))
    response = await read_client.get(f"{PATH}/run-19")
    assert response.status_code == 200 and response.json()["run_id"] == "run-19"
    assert get.await_count == 1 and not files_service.calls


class RunReadTestService(NemoServiceAdapter):
    """Mount the evaluator plugin unchanged in the real platform authorization harness."""

    __test__ = False

    def __init__(self) -> None:
        """Create the adapter for evaluator routes; takes no input and returns None."""
        super().__init__(EvaluatorPluginService())


def test_real_authorization_denies_missing_read_list_and_direct_service(tmp_path, service, entity_store, monkeypatch):
    """Real middleware rejects each denied caller before any Entity Store operation."""
    get = AsyncMock(side_effect=AssertionError("unauthorized get"))
    listing = AsyncMock(side_effect=AssertionError("unauthorized list"))
    monkeypatch.setattr(entity_store, "get", get)
    monkeypatch.setattr(entity_store, "list", listing)
    with create_test_client(
        RunReadTestService,
        EntitiesService,
        client_type=ClientContext,
        auth_enabled=True,
        tmp_dir=tmp_path,
        workspaces=["private"],
        projects=[],
        auth_bundle_cache_seconds=0,
        dependency_overrides={get_taskset_run_service: lambda: service},
    ) as ctx:
        for principal in ["no-run-permissions@example.com", "service:integration-test"]:
            for suffix in ["", "/run-selected", "/run-selected/tasks"]:
                response = ctx.test_client.get(
                    PATH.replace("/default/", "/private/") + suffix, headers={"X-NHX-Principal-Id": principal}
                )
                assert response.status_code == 403, response.text
    get.assert_not_awaited()
    listing.assert_not_awaited()


async def test_auth_disabled_mine_uses_effective_development_creator(read_client, read_app, seed_run):
    """An empty effective principal selects the same development creator that submit stamps."""
    read_app.dependency_overrides[get_effective_principal_id] = lambda: ""
    await seed_run("dev-run", caller="user:development")
    await seed_run("alice-run")
    response = await read_client.get(PATH, params={"mine": True})
    assert response.status_code == 200
    assert [run["run_id"] for run in response.json()["data"]] == ["dev-run"]


# ---- cancel ----------------------------------------------------------------------------------------------

CANCEL_AT = datetime(2026, 10, 14, 12, 30, tzinfo=timezone.utc)


@pytest.fixture
def cancel_clock(monkeypatch):
    """Pin the cancel route's clock; returns a setter for a later request's time."""

    def at(when: datetime) -> None:
        monkeypatch.setattr("nemo_evals.api.v2.taskset_runs.utc_now", lambda: when)

    at(CANCEL_AT)
    return at


async def test_cancel_records_the_intent_and_the_run_stays_active(read_client, seed_run, cancel_clock):
    """Only ``cancel_requested_at`` changes; the run stays accepted, and listed as active, until the controller acts."""
    await seed_run()
    response = await read_client.post(f"{PATH}/run-selected/cancel")
    assert response.status_code == 200
    assert response.json()["cancel_requested_at"] == "2026-10-14T12:30:00Z"
    assert response.json()["phase"] == "accepted"
    assert (await read_client.get(f"{PATH}/run-selected")).json()["cancel_requested_at"] == "2026-10-14T12:30:00Z"
    active = await read_client.get(PATH, params={"active": True})
    assert [run["run_id"] for run in active.json()["data"]] == ["run-selected"]


async def test_cancel_again_returns_the_first_request_time(read_client, seed_run, cancel_clock):
    await seed_run()
    first = await read_client.post(f"{PATH}/run-selected/cancel")
    cancel_clock(CANCEL_AT + timedelta(minutes=1))
    again = await read_client.post(f"{PATH}/run-selected/cancel")
    assert again.status_code == 200 and again.json() == first.json()


@pytest.mark.parametrize("phase", [TasksetRunPhase.COMPLETED, TasksetRunPhase.FAILED, TasksetRunPhase.EXPIRED])
async def test_cancel_of_a_finished_run_is_a_409_not_to_retry(read_client, seed_run, cancel_clock, phase):
    await seed_run(phase=phase)
    response = await read_client.post(f"{PATH}/run-selected/cancel")
    assert response.status_code == 409
    assert response.json()["detail"] == f"run run-selected already finished: {phase.value}"
    assert response.headers["x-should-retry"] == "false"
    assert "cancel_requested_at" not in (await read_client.get(f"{PATH}/run-selected")).json()


async def test_cancel_of_an_unknown_or_cross_workspace_run_is_a_404(read_client, seed_run, cancel_clock):
    await seed_run(workspace="other")
    for run_id in ["unknown", "run-selected"]:
        assert (await read_client.post(f"{PATH}/{run_id}/cancel")).status_code == 404


async def test_cancel_that_keeps_losing_to_the_controller_is_a_503_to_retry(
    read_client, seed_run, entity_store, cancel_clock, monkeypatch
):
    await seed_run()
    monkeypatch.setattr(entity_store, "update", AsyncMock(side_effect=NemoEntityConflictError("stale")))
    response = await read_client.post(f"{PATH}/run-selected/cancel")
    assert response.status_code == 503 and response.headers["x-should-retry"] == "true"


def test_real_authorization_denies_cancel_without_the_permission(
    tmp_path, service, run_store, entity_store, monkeypatch
):
    """A caller without ``evaluator.taskset-runs.cancel``, or a service caller, is refused before any read or write."""
    get = AsyncMock(side_effect=AssertionError("unauthorized get"))
    update = AsyncMock(side_effect=AssertionError("unauthorized update"))
    monkeypatch.setattr(entity_store, "get", get)
    monkeypatch.setattr(entity_store, "update", update)
    with create_test_client(
        RunReadTestService,
        EntitiesService,
        client_type=ClientContext,
        auth_enabled=True,
        tmp_dir=tmp_path,
        workspaces=["private"],
        projects=[],
        auth_bundle_cache_seconds=0,
        dependency_overrides={get_taskset_run_service: lambda: service, get_service_run_store: lambda: run_store},
    ) as ctx:
        for principal in ["no-run-permissions@example.com", "service:integration-test"]:
            response = ctx.test_client.post(
                PATH.replace("/default/", "/private/") + "/run-selected/cancel",
                headers={"X-NHX-Principal-Id": principal},
            )
            assert response.status_code == 403, response.text
    get.assert_not_awaited()
    update.assert_not_awaited()
