# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from nemo_helix_plugin.entities.types import EntityCreateInput
from nhx.common.api.utils import tweak_spec
from nhx.common.auth import AuthClient, Principal, get_auth_client
from nhx.common.entities.client import EntityClient
from nhx.core.jobs.api.v2.job_groups import endpoints
from nhx.core.jobs.controllers.backends.subprocess import (
    SubprocessJobExecutionProfile,
    SubprocessJobExecutionProfileConfig,
)
from nhx.core.jobs.entities import HelixJob
from nhx.core.jobs.job_groups import (
    HelixJobGroup,
    JobGroupControllerState,
    JobGroupMemberOutcome,
    JobGroupMemberState,
    JobGroupStatus,
    group_member_job_name,
)

GROUPS = "/v2/workspaces/default/job-groups"
SUBMISSION = {
    "name": "batch",
    "source": "training",
    "template": {
        "steps": [
            {
                "name": "run",
                "executor": {"provider": "subprocess", "profile": "local", "command": ["true"]},
            }
        ]
    },
    "members": [{}, {"config_patch": {"seed": 42}}],
    "max_concurrent_jobs": 1,
}


@pytest.fixture
def auth() -> AsyncMock:
    client = AsyncMock(spec=AuthClient)
    client.auth_enabled = True
    client.principal = Principal(id="submitter")
    client.check_permissions.return_value = True
    return client


@pytest.fixture
async def groups_client(
    mock_store: EntityClient, auth: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    monkeypatch.setattr(
        endpoints,
        "profiles",
        [SubprocessJobExecutionProfile(profile="local", config=SubprocessJobExecutionProfileConfig())],
    )
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[endpoints.get_job_group_store] = lambda: mock_store
    app.dependency_overrides[get_auth_client] = lambda: auth
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_submission_saves_only_the_group(groups_client: AsyncClient, mock_store: EntityClient) -> None:
    response = await groups_client.post(GROUPS, json=SUBMISSION)

    assert response.status_code == 201, response.text
    data = response.json()
    assert data["member_count"] == 2
    assert data["status"] == "pending"
    group = await mock_store.get_by_id(HelixJobGroup, data["id"])
    assert group.controller_state.schema_version == 7
    assert group.auth_context is not None
    assert group.auth_context.principal_id == "submitter"
    assert group.submission.members[1].config_patch == {"seed": 42}
    raw = (await mock_store._client.get_entity_by_id(entity_id=group.id)).data()
    assert all("name" not in member for member in raw.data["submission"]["members"])
    assert (await mock_store.list(HelixJob, workspace="default")).data == []
    assert (await groups_client.get(f"{GROUPS}/batch")).json()["id"] == group.id
    assert (await groups_client.post(GROUPS, json=SUBMISSION)).status_code == 409

    page = await groups_client.get(f"{GROUPS}/{group.id}/members", params={"page": 2, "page_size": 1})
    assert page.status_code == 200
    assert page.json()["data"] == [
        {"index": 1, "job_id": None, "attempt_id": None, "status": "pending", "reason": None}
    ]
    assert page.json()["pagination"]["total_results"] == 2


async def test_submission_rejects_invalid_workspace(groups_client: AsyncClient, auth: AsyncMock) -> None:
    auth.auth_enabled = False

    response = await groups_client.post(GROUPS.replace("/default/", "/bad!/"), json=SUBMISSION)

    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["path", "workspace"]


async def test_list_members_reads_only_the_group(
    groups_client: AsyncClient, mock_store: EntityClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ["completed", "running", "preparing", "retrying", "failed"]
    submission = {**SUBMISSION, "members": [{} for _ in names]}
    created = await groups_client.post(GROUPS, json=submission)
    assert created.status_code == 201, created.text
    group = await mock_store.get_by_id(HelixJobGroup, created.json()["id"])
    state = group.controller_state
    for index, name in enumerate(names):
        state.claim_member(index, f"att-{name}")
        state.record_job(index, f"job-{name}")
        state.record_attempt(index, f"job-{name}", f"attempt-{name}")
        outcome = JobGroupMemberOutcome.COMPLETED if name == "completed" else JobGroupMemberOutcome.ERROR
        state.finish_member(index, outcome)
    assert state.settle(len(names))
    state.retry_unsuccessful_members()
    for index in (1, 2, 4):
        name = names[index]
        state.claim_member(index, f"att-{name}")
    state.record_attempt(1, "job-running", "new-running")
    state.record_attempt(4, "job-failed", "new-failed")
    state.finish_member(4, JobGroupMemberOutcome.ERROR, "Job failed")
    await mock_store.update(group)
    read = AsyncMock(wraps=mock_store.get_by_id)
    monkeypatch.setattr(mock_store, "get_by_id", read)

    response = await groups_client.get(f"{GROUPS}/{group.id}/members")

    assert response.status_code == 200, response.text
    read.assert_awaited_once_with(HelixJobGroup, group.id)
    rows = response.json()["data"]
    assert [row["index"] for row in rows] == list(range(len(names)))
    assert [row["job_id"] for row in rows] == [f"job-{name}" for name in names]
    assert [row["attempt_id"] for row in rows] == ["attempt-completed", "new-running", None, None, "new-failed"]
    assert [row["status"] for row in rows] == ["completed", "pending", "pending", "pending", "error"]
    assert rows[-1]["reason"] == "Job failed"


@pytest.mark.parametrize(
    "progress",
    [
        {"phase": "pending", "attempt_id": "stale"},
        {"phase": "tracking"},
        {"phase": "admitted"},
        {"phase": "completed"},
        {"phase": "unsuccessful", "outcome": "completed"},
    ],
)
def test_member_state_rejects_invalid_progress(progress: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        JobGroupMemberState.model_validate({"job_id": "job-id", "progress": progress})


def test_abandoned_creation_recovers_an_unrecorded_job_by_name() -> None:
    state = JobGroupControllerState()
    state.claim_member(0, "attempt-name")
    group_id = "platform-job-group-example"
    name = group_member_job_name(group_id, 0)

    assert not state.record_abandoned_creation("platform-job-other", job_name="different-member", group_id=group_id)
    assert state.record_abandoned_creation("platform-job-created", job_name=name, group_id=group_id)
    member = state.member(0)
    assert member.job_id == "platform-job-created"
    assert member.reason == "Job creation did not finish before the creation deadline."
    assert set(state.model_dump(mode="json")["members"]) == {"0"}
    assert not state.record_abandoned_creation("platform-job-other", job_name=name, group_id=group_id)
    assert not state.record_abandoned_creation("platform-job-created", job_name="renamed-job", group_id=group_id)


@pytest.mark.parametrize(
    "environment",
    [
        {"name": "NHX_PRINCIPAL", "value": "reserved"},
        {"name": "VALID_NAME", "from_secret": {"name": "bad//reference"}},
    ],
)
async def test_submission_rejects_invalid_environment(
    groups_client: AsyncClient, mock_store: EntityClient, environment: dict[str, str | dict[str, str]]
) -> None:
    submission = deepcopy(SUBMISSION)
    submission["template"]["steps"][0]["environment"] = [environment]

    assert (await groups_client.post(GROUPS, json=submission)).status_code == 422
    assert (await mock_store.list(HelixJobGroup)).data == []


@pytest.mark.parametrize(("sort", "sources"), [("source", ["a", "z"]), ("-source", ["z", "a"])])
async def test_list_sorts_by_submission_source(groups_client: AsyncClient, sort: str, sources: list[str]) -> None:
    for source in ("z", "a"):
        response = await groups_client.post(GROUPS, json={**SUBMISSION, "name": f"batch-{source}", "source": source})
        assert response.status_code == 201

    response = await groups_client.get(GROUPS, params={"sort": sort})
    assert response.status_code == 200
    assert [group["source"] for group in response.json()["data"]] == sources


async def test_list_pagination_counts_only_readable_groups(
    groups_client: AsyncClient, mock_store: EntityClient
) -> None:
    created = await groups_client.post(GROUPS, json=SUBMISSION)
    assert created.status_code == 201, created.text
    group = await mock_store.get_by_id(HelixJobGroup, created.json()["id"])
    await mock_store._client.create_entity(
        workspace="default",
        entity_type=HelixJobGroup.__entity_type__,
        body=EntityCreateInput(
            name="newer-schema",
            data={
                "submission": {**group.submission.model_dump(mode="json"), "source": "ahead"},
                "controller_state": {"schema_version": 99},
            },
        ),
    )

    for page, page_size, names, total_pages in [(1, 2, ["batch"], 1), (1, 1, [], 2), (2, 1, ["batch"], 2)]:
        response = await groups_client.get(GROUPS, params={"sort": "source", "page": page, "page_size": page_size})

        assert response.status_code == 200, response.text
        body = response.json()
        assert [row["name"] for row in body["data"]] == names
        assert body["pagination"] == {
            "page": page,
            "page_size": page_size,
            "current_page_size": len(names),
            "total_pages": total_pages,
            "total_results": 2,
        }


async def test_cancel_and_metadata_use_the_same_persisted_state(
    groups_client: AsyncClient, mock_store: EntityClient
) -> None:
    created = await groups_client.post(GROUPS, json=SUBMISSION)
    path = f"{GROUPS}/{created.json()['id']}"

    assert (await groups_client.delete(path)).status_code == 409
    assert (await groups_client.post(f"{path}/cancel")).json()["status"] == "cancelling"
    group = await mock_store.get(HelixJobGroup, name="batch", workspace="default")

    assert (await groups_client.post(f"{path}/cancel")).status_code == 202
    assert (await mock_store.get_by_id(HelixJobGroup, group.id)).db_version == group.db_version
    assert (await groups_client.patch(path, json={"fail_fast": True})).status_code == 422
    assert (await groups_client.patch(path, json={"controller_state": {}})).status_code == 422
    assert (await groups_client.patch(path, json={"description": "new"})).json()["status"] == "cancelling"

    updated = await mock_store.get_by_id(HelixJobGroup, group.id)
    assert updated.description == "new"
    assert updated.controller_state == group.controller_state
    assert updated.auth_context == group.auth_context
    assert (await groups_client.delete(path)).status_code == 409


async def test_retry_selects_only_unsuccessful_members(groups_client: AsyncClient, mock_store: EntityClient) -> None:
    created = await groups_client.post(GROUPS, json=SUBMISSION)
    group = await mock_store.get_by_id(HelixJobGroup, created.json()["id"])
    state = group.controller_state
    state.claim_member(0, "att-first")
    state.record_job(0, "job-id")
    state.record_attempt(0, "job-id", "attempt-id")
    state.finish_member(0, JobGroupMemberOutcome.COMPLETED)
    state.finish_member(1, JobGroupMemberOutcome.CANCELLED)
    assert state.settle(len(group.submission.members))
    await mock_store.update(group)
    path = f"{GROUPS}/{group.id}"

    assert (await groups_client.post(f"{path}/retry")).json()["status"] == "pending"
    assert (await groups_client.post(f"{path}/retry")).status_code == 409
    assert (await groups_client.delete(path)).status_code == 409

    state = (await mock_store.get_by_id(HelixJobGroup, group.id)).controller_state
    assert state.schema_version == 7
    assert state.retry_revision == 1
    assert state.member(0) == group.member_state(0)
    assert state.member(1).phase == "pending"
    assert state.member(1).attempt_id is None


async def test_delete_records_intent_and_closes_the_group(groups_client: AsyncClient, mock_store: EntityClient) -> None:
    created = await groups_client.post(GROUPS, json=SUBMISSION)
    group = await mock_store.get_by_id(HelixJobGroup, created.json()["id"])
    group.controller_state.status = JobGroupStatus.INCOMPLETE
    await mock_store.update(group)
    path = f"{GROUPS}/{group.id}"

    assert (await groups_client.delete(path)).json()["status"] == "deletion_pending"
    assert (await groups_client.post(f"{path}/retry")).status_code == 409
    assert (await groups_client.post(f"{path}/cancel")).status_code == 409
    assert (await groups_client.patch(path, json={"description": "new"})).status_code == 409
    assert (await groups_client.delete(path)).status_code == 202
    assert (await mock_store.get_by_id(HelixJobGroup, group.id)).controller_state.status == (
        JobGroupStatus.DELETION_PENDING
    )


async def test_delegated_submission_requires_subject_permission(groups_client: AsyncClient, auth: AsyncMock) -> None:
    auth.principal = Principal(id="service:training", on_behalf_of="submitter")
    auth.check_permissions.return_value = False

    assert (await groups_client.post(GROUPS, json=SUBMISSION)).status_code == 403
    auth.check_permissions.assert_awaited_once_with("default", subject=["jobs.create"])


@pytest.mark.parametrize("winner", [JobGroupStatus.DELETION_PENDING, JobGroupStatus.PENDING])
async def test_retry_and_delete_cannot_both_commit(
    groups_client: AsyncClient,
    mock_store: EntityClient,
    monkeypatch: pytest.MonkeyPatch,
    winner: JobGroupStatus,
) -> None:
    created = await groups_client.post(GROUPS, json=SUBMISSION)
    group = await mock_store.get_by_id(HelixJobGroup, created.json()["id"])
    group.controller_state.status = JobGroupStatus.INCOMPLETE
    group.controller_state.finish_member(0, JobGroupMemberOutcome.ERROR)
    await mock_store.update(group)
    path = f"{GROUPS}/{group.id}"

    commit = mock_store.update
    losing_write_reached = asyncio.Event()
    winner_committed = asyncio.Event()

    async def ordered_commit(candidate: HelixJobGroup) -> HelixJobGroup:
        if candidate.controller_state.status == winner:
            await losing_write_reached.wait()
            updated = await commit(candidate)
            winner_committed.set()
            return updated
        losing_write_reached.set()
        await winner_committed.wait()
        return await commit(candidate)

    monkeypatch.setattr(mock_store, "update", ordered_commit)
    retry, delete = await asyncio.wait_for(
        asyncio.gather(groups_client.post(f"{path}/retry"), groups_client.delete(path)), timeout=5
    )
    expected = (202, 409) if winner == JobGroupStatus.PENDING else (409, 202)
    assert (retry.status_code, delete.status_code) == expected
    assert (await mock_store.get_by_id(HelixJobGroup, group.id)).controller_state.status == winner


def test_job_group_openapi_exposes_only_the_submission_and_action_contract() -> None:
    app = FastAPI()
    app.include_router(endpoints.router)
    schema = app.openapi()
    base = "/v2/workspaces/{workspace}/job-groups"
    paths = schema["paths"]
    assert set(paths[base]) == {"get", "post"}
    assert set(paths[base + "/{identifier}"]) == {"get", "patch", "delete"}
    assert set(paths[base + "/{identifier}/members"]) == {"get"}
    for action in ("cancel", "retry"):
        assert "202" in paths[base + "/{identifier}/" + action]["post"]["responses"]
    assert "controller_state" not in schema["components"]["schemas"]["CreateJobGroupRequest"]["properties"]
    assert set(schema["components"]["schemas"]["UpdateJobGroupRequest"]["properties"]) == {
        "description",
        "custom_fields",
    }


def test_exported_job_group_schemas_keep_null_types() -> None:
    app = FastAPI()
    app.include_router(endpoints.router)
    schemas = tweak_spec(app.openapi())["components"]["schemas"]
    response = schemas["JobGroupResponse"]
    types = {
        "project": "string",
        "description": "string",
        "custom_fields": "object",
        "ownership": "object",
        "max_concurrent_jobs": "integer",
        "default_timeout_seconds": "integer",
        "created_at": "string",
        "updated_at": "string",
    }
    for field, value_type in types.items():
        assert field in response["required"]
        assert response["properties"][field]["type"] == [value_type, "null"]

    for model in ("UpdateJobGroupRequest", "CreateJobGroupRequest"):
        properties = schemas[model]["properties"]
        assert properties["description"]["type"] == ["string", "null"]
        assert properties["custom_fields"]["type"] == ["object", "null"]

    member = schemas["JobGroupMember"]["properties"]
    assert "name" not in member
    response_member = schemas["JobGroupMemberResponse"]
    assert "name" not in response_member["properties"]
    assert "index" in response_member["required"]
    assert response_member["properties"]["index"]["type"] == "integer"
    timeout = member["timeout_seconds"]
    assert timeout["type"] == ["integer", "null"]
    assert timeout["minimum"] == 0
    for field in ("job_id", "attempt_id", "reason"):
        assert schemas["JobGroupMemberResponse"]["properties"][field]["type"] == ["string", "null"]
