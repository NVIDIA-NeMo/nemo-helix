# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.common.auth import AuthContext, Principal
from nhx.common.auth.workload_delegations import (
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
    OPAQUE_DOCKER_PROOF_PREFIX,
    InvalidWorkloadProofTokenError,
    SyncWorkloadDelegationStore,
    WorkloadDelegationAlreadyActiveError,
    WorkloadDelegationConflictError,
    WorkloadDelegationEntity,
    WorkloadDelegationError,
    WorkloadDelegationScope,
    WorkloadDelegationStore,
    WorkloadDelegationValidationError,
    create_opaque_docker_proof_token,
    docker_delegation_name,
    docker_deployment_delegation_name,
    docker_workload_delegation_name,
    kubernetes_pod_uid_delegation_name,
    parse_opaque_docker_proof_token,
    reference_delegation_name,
    verify_opaque_docker_proof_token_hash,
)
from nhx.common.entities import (
    SYSTEM_WORKSPACE,
    EntityClient,
    EntityConflictError,
    EntityNotFoundError,
    ListResponse,
    PaginationInfo,
)
from pydantic import ValidationError


def _expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


def _entity(**overrides) -> WorkloadDelegationEntity:
    values = {
        "name": docker_delegation_name(
            workload_workspace="default",
            job_id="job-123",
            attempt_id="attempt-1",
            step_id="step-a",
        ),
        "workspace": SYSTEM_WORKSPACE,
        "workload_subject": docker_delegation_name(
            workload_workspace="default",
            job_id="job-123",
            attempt_id="attempt-1",
            step_id="step-a",
        ),
        "workload_audience": "nemo-helix",
        "workload_workspace": "default",
        "job_id": "job-123",
        "attempt_id": "attempt-1",
        "step_id": "step-a",
        "auth_context": AuthContext.from_principal(Principal(id="creator@example.com")),
        "expires_at": _expires_at(),
    }
    values.update(overrides)
    return WorkloadDelegationEntity.model_validate(values)


def _reference_entity(**overrides) -> WorkloadDelegationEntity:
    values = {
        "name": reference_delegation_name(
            workload_audience="nemo-helix",
            workload_subject="system:serviceaccount:ns:runner",
            bound_reference_name="authentication.kubernetes.io/pod-uid",
            bound_reference_value="pod-uid-123",
        ),
        "workload_subject": "system:serviceaccount:ns:runner",
        "bound_reference_name": "authentication.kubernetes.io/pod-uid",
        "bound_reference_value": "pod-uid-123",
    }
    values.update(overrides)
    return _entity(**values)


def _generic_docker_entity(**overrides) -> WorkloadDelegationEntity:
    scope = WorkloadDelegationScope(
        workload_workspace="default",
        workload_kind="deployment",
        workload_instance_id="deployment-123",
    )
    name = docker_workload_delegation_name(
        scope=scope,
        workload_generation="container-abc",
    )
    values = {
        "name": name,
        "workspace": SYSTEM_WORKSPACE,
        "workload_subject": name,
        "workload_audience": "nemo-helix",
        "workload_workspace": "default",
        "workload_kind": "deployment",
        "workload_id": "deployment-123",
        "workload_generation": "container-abc",
        "auth_context": AuthContext.from_principal(Principal(id="creator@example.com")),
        "opaque_subject_token_hash": "v1:sha256:test",
        "expires_at": _expires_at(),
    }
    values.update(overrides)
    return WorkloadDelegationEntity.model_validate(values)


def _generic_reference_entity(**overrides) -> WorkloadDelegationEntity:
    workload_subject = "system:serviceaccount:deployments:runner"
    pod_uid = "pod-uid-456"
    values = {
        "name": kubernetes_pod_uid_delegation_name(
            workload_audience="nemo-helix",
            workload_subject=workload_subject,
            pod_uid=pod_uid,
        ),
        "workspace": SYSTEM_WORKSPACE,
        "workload_subject": workload_subject,
        "workload_audience": "nemo-helix",
        "workload_workspace": "default",
        "workload_kind": "deployment",
        "workload_id": "deployment-123",
        "workload_generation": pod_uid,
        "auth_context": AuthContext.from_principal(Principal(id="creator@example.com")),
        "bound_reference_name": "authentication.kubernetes.io/pod-uid",
        "bound_reference_value": pod_uid,
        "expires_at": _expires_at(),
    }
    values.update(overrides)
    return WorkloadDelegationEntity.model_validate(values)


def _job_step_entity(**overrides) -> WorkloadDelegationEntity:
    """A job step's row as the Jobs backends write it: kind ``job``, the job's name, and the job's entity ID."""
    values = {
        "workload_kind": "job",
        "workload_id": "job-123",
        "workload_generation": "attempt-1/step-a",
        "job_uid": "platform-job-123",
    }
    values.update(overrides)
    return _entity(**values)


def _list_response(
    data: list[WorkloadDelegationEntity],
    *,
    page: int = 1,
    page_size: int = 100,
    total_pages: int = 1,
) -> ListResponse[WorkloadDelegationEntity]:
    return ListResponse(
        data=data,
        pagination=PaginationInfo(
            page=page,
            page_size=page_size,
            current_page_size=len(data),
            total_pages=total_pages,
            total_results=len(data),
        ),
    )


def _generic_scope(**overrides: str | None) -> WorkloadDelegationScope:
    values = {
        "workload_workspace": "default",
        "workload_kind": "deployment",
        "workload_instance_id": "deployment-123",
        "workload_claim_id": None,
    }
    values.update(overrides)
    return WorkloadDelegationScope.model_validate(values)


def _entity_client() -> AsyncMock:
    return AsyncMock(spec=EntityClient)


def test_docker_delegation_name_is_deterministic() -> None:
    expected_input = '["default","job-123","attempt-1","step-a"]'.encode()
    expected = "job-" + hashlib.sha256(expected_input).hexdigest()[:48]

    name = docker_delegation_name(
        workload_workspace="default",
        job_id="job-123",
        attempt_id="attempt-1",
        step_id="step-a",
    )

    assert name == expected
    assert (
        docker_delegation_name(
            workload_workspace="default",
            job_id="job-123",
            attempt_id="attempt-1",
            step_id="step-a",
        )
        == name
    )
    assert set(inspect.signature(docker_delegation_name).parameters) == {
        "workload_workspace",
        "job_id",
        "attempt_id",
        "step_id",
    }


def test_reference_delegation_name_is_canonical_hash() -> None:
    expected_input = (
        '["nemo-helix","system:serviceaccount:ns:runner","authentication.kubernetes.io/pod-uid","pod-uid-123"]'
    ).encode()
    expected = "ref-" + hashlib.sha256(expected_input).hexdigest()[:48]

    name = reference_delegation_name(
        workload_audience="nemo-helix",
        workload_subject="system:serviceaccount:ns:runner",
        bound_reference_name="authentication.kubernetes.io/pod-uid",
        bound_reference_value="pod-uid-123",
    )

    assert name == expected
    assert (
        reference_delegation_name(
            workload_audience="nemo-helix",
            workload_subject="system:serviceaccount:ns:runner",
            bound_reference_name="authentication.kubernetes.io/pod-uid",
            bound_reference_value="pod-uid-123",
        )
        == name
    )
    assert set(inspect.signature(reference_delegation_name).parameters) == {
        "workload_audience",
        "workload_subject",
        "bound_reference_name",
        "bound_reference_value",
    }


def test_opaque_docker_proof_token_round_trip() -> None:
    token, token_hash = create_opaque_docker_proof_token("job-abc")
    parsed = parse_opaque_docker_proof_token(token)

    assert DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE.endswith("docker-opaque-workload-proof")
    assert token.startswith(f"{OPAQUE_DOCKER_PROOF_PREFIX}.")
    assert len(token.split(".")) == 3
    assert parsed.delegation_name == "job-abc"
    assert len(parsed.secret) == 32
    assert verify_opaque_docker_proof_token_hash(parsed.secret, token_hash)
    assert not verify_opaque_docker_proof_token_hash(parsed.secret, "v1:sha256:bad")


@pytest.mark.parametrize(
    "token",
    [
        "",
        "wrong.job.secret",
        "nhx_obo_v1.only-two-parts",
        "nhx_obo_v1...secret",
        "nhx_obo_v1.am9iOmFiYw.bad-secret",
    ],
)
def test_parse_opaque_docker_proof_token_rejects_malformed_tokens(token: str) -> None:
    with pytest.raises(InvalidWorkloadProofTokenError):
        parse_opaque_docker_proof_token(token)


@pytest.mark.asyncio
async def test_register_creates_system_workspace_entity() -> None:
    entity_client = _entity_client()
    entity = _entity()
    entity_client.add.return_value = entity

    saved = await WorkloadDelegationStore(entity_client).register(entity)

    assert saved == entity
    entity_client.add.assert_awaited_once_with(entity)
    entity_client.get.assert_not_called()
    entity_client.list.assert_not_called()


@pytest.mark.asyncio
async def test_get_uses_direct_system_workspace_lookup() -> None:
    entity_client = _entity_client()
    entity = _entity()
    entity_client.get.return_value = entity

    result = await WorkloadDelegationStore(entity_client).get(entity.name)

    assert result == entity
    entity_client.get.assert_awaited_once_with(WorkloadDelegationEntity, entity.name, workspace=SYSTEM_WORKSPACE)
    entity_client.list.assert_not_called()


@pytest.mark.asyncio
async def test_get_returns_none_when_missing() -> None:
    entity_client = _entity_client()
    entity_client.get.side_effect = EntityNotFoundError("missing")

    result = await WorkloadDelegationStore(entity_client).get("job:missing")

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entity",
    [
        _entity(workload_audience=""),
        _entity(revoked_at=datetime.now(timezone.utc)),
        _entity(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
        _entity(bound_reference_name="authentication.kubernetes.io/pod-uid"),
        _entity(bound_reference_value="pod-uid-123"),
        _entity(
            bound_reference_name="authentication.kubernetes.io/pod-uid",
            bound_reference_value="pod-uid-123",
        ),
        _entity(name="job:wrong"),
        _reference_entity(name="ref:wrong"),
        _entity(name="workload:runner", workload_subject="system:serviceaccount:ns:runner"),
        _entity(workspace="default"),
    ],
)
async def test_register_rejects_invalid_entities(entity: WorkloadDelegationEntity) -> None:
    with pytest.raises(WorkloadDelegationValidationError):
        await WorkloadDelegationStore(_entity_client()).register(entity)


@pytest.mark.asyncio
async def test_register_opaque_docker_requires_stored_hash() -> None:
    with pytest.raises(WorkloadDelegationValidationError):
        await WorkloadDelegationStore(_entity_client()).register(
            _entity(),
            require_opaque_subject_token_hash=True,
        )


@pytest.mark.asyncio
async def test_reference_bound_entity_registers_with_reference_pair() -> None:
    entity_client = _entity_client()
    entity = _reference_entity()
    entity_client.add.return_value = entity

    saved = await WorkloadDelegationStore(entity_client).register(entity)

    assert saved == entity
    entity_client.add.assert_awaited_once_with(entity)


@pytest.mark.asyncio
async def test_renew_preserves_grant_binding() -> None:
    entity_client = _entity_client()
    fetched = _job_step_entity(opaque_subject_token_hash="proof", docker_container_id="container-a")
    fetched._id = "grant"
    fetched._db_version = 7
    entity_client.get_by_id.return_value = fetched
    entity_client.update.side_effect = lambda entity: entity
    expiry = _expires_at() + timedelta(hours=1)
    renewed = await WorkloadDelegationStore(entity_client).renew("grant", expected_db_version=7, expires_at=expiry)
    assert renewed.id == fetched.id and renewed.db_version == 7
    assert renewed.expires_at == expiry
    assert renewed.auth_context == fetched.auth_context
    assert renewed.job_id == fetched.job_id
    assert renewed.job_uid == "platform-job-123"
    assert renewed.docker_container_id == "container-a"
    assert renewed.opaque_subject_token_hash == "proof"
    entity_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_renew_rejects_stale_version() -> None:
    entity_client = _entity_client()
    fetched = _entity()
    fetched._db_version = 7
    entity_client.get_by_id.return_value = fetched
    with pytest.raises(WorkloadDelegationConflictError):
        await WorkloadDelegationStore(entity_client).renew("grant", expected_db_version=6, expires_at=_expires_at())
    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_renew_rejects_existing_revoked_record() -> None:
    entity_client = _entity_client()
    fetched = _entity(revoked_at=datetime.now(timezone.utc))
    fetched._db_version = 8
    entity_client.get_by_id.return_value = fetched
    with pytest.raises(WorkloadDelegationConflictError, match="revoked"):
        await WorkloadDelegationStore(entity_client).renew("grant", expected_db_version=8, expires_at=_expires_at())
    assert not fetched.is_active()
    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_deletes_fetched_id_and_version() -> None:
    entity_client = _entity_client()
    fetched = _entity()
    fetched._id = "grant"
    fetched._db_version = 9
    entity_client.get.return_value = fetched
    await WorkloadDelegationStore(entity_client).revoke(fetched.name)
    entity_client.delete_by_id.assert_awaited_once_with(WorkloadDelegationEntity, "grant", expected_db_version=9)
    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_returns_none_when_missing() -> None:
    entity_client = _entity_client()
    entity_client.get.side_effect = EntityNotFoundError("missing")

    result = await WorkloadDelegationStore(entity_client).revoke("job:missing")

    assert result is None
    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_register_conflict_on_active_row_raises_domain_conflict() -> None:
    entity_client = _entity_client()
    existing = _entity()
    entity_client.add.side_effect = EntityConflictError("already exists")
    entity_client.get.return_value = existing

    with pytest.raises(WorkloadDelegationAlreadyActiveError):
        await WorkloadDelegationStore(entity_client).register(_entity())

    entity_client.update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("inactive", ["expired", "revoked"])
async def test_register_deletes_inactive_grant_and_logs_replacement(
    inactive: str, caplog: pytest.LogCaptureFixture
) -> None:
    entity_client = _entity_client()
    old = _entity(opaque_subject_token_hash="old-proof")
    old._id = "old-grant"
    old._db_version = 12
    if inactive == "expired":
        old.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    else:
        old.revoked_at = datetime.now(timezone.utc)
    replacement = _entity(opaque_subject_token_hash="new-proof")
    saved_replacement = replacement.model_copy()
    saved_replacement._id = "new-grant"
    entity_client.add.side_effect = [EntityConflictError("already exists"), saved_replacement]
    entity_client.get.return_value = old
    with caplog.at_level("INFO", logger="nemo_helix_plugin.auth.workload_delegations"):
        saved = await WorkloadDelegationStore(entity_client).register(replacement)
    entity_client.delete_by_id.assert_awaited_once_with(WorkloadDelegationEntity, old.id, expected_db_version=12)
    entity_client.update.assert_not_called()
    assert saved.id != old.id
    assert saved.name == old.name
    deleted, registered = caplog.records
    assert deleted.__dict__["delegation_id"] == old.id
    assert registered.__dict__["delegation_id"] == saved.id
    assert registered.__dict__["previous_delegation_id"] == old.id
    assert registered.__dict__["reason"] == inactive
    assert "old-proof" not in str([record.__dict__ for record in caplog.records])
    assert "new-proof" not in str([record.__dict__ for record in caplog.records])


@pytest.mark.parametrize("race_at", ["deletion", "creation"])
@pytest.mark.asyncio
async def test_registration_race_fails_closed(race_at: str) -> None:
    entity_client = _entity_client()
    old = _entity(revoked_at=datetime.now(timezone.utc))
    old._id = "old"
    entity_client.get.return_value = old
    entity_client.add.side_effect = EntityConflictError("unique name")
    if race_at == "deletion":
        entity_client.delete_by_id.side_effect = EntityConflictError("stale version")
    with pytest.raises(WorkloadDelegationConflictError, match="changed|raced"):
        await WorkloadDelegationStore(entity_client).register(_entity())
    assert entity_client.add.await_count == (1 if race_at == "deletion" else 2)


def test_sync_registration_deletes_inactive_grant() -> None:
    entity_client = MagicMock()
    old = _entity(revoked_at=datetime.now(timezone.utc))
    old._id = "old"
    old._db_version = 3
    new = _entity()
    saved = new.model_copy()
    saved._id = "new"
    entity_client.get.return_value = old
    entity_client.add.side_effect = [EntityConflictError("unique name"), saved]
    assert SyncWorkloadDelegationStore(entity_client).register(new).id == "new"
    entity_client.delete_by_id.assert_called_once_with(WorkloadDelegationEntity, "old", expected_db_version=3)
    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_deleted_grant_cannot_renew_by_reused_name() -> None:
    entity_client = _entity_client()
    entity_client.get_by_id.side_effect = EntityNotFoundError("old grant deleted")
    with pytest.raises(WorkloadDelegationConflictError, match="no longer exists"):
        await WorkloadDelegationStore(entity_client).renew("old", expected_db_version=1, expires_at=_expires_at())
    entity_client.get.assert_not_called()
    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_lookup_by_id_returns_revoked_grant_without_following_name() -> None:
    entity_client = _entity_client()
    old = _entity(revoked_at=datetime.now(timezone.utc))
    old._id = "old"
    entity_client.get_by_id.return_value = old
    assert await WorkloadDelegationStore(entity_client).get_by_id("old") is old
    entity_client.get.assert_not_called()
    entity_client.get_by_id.assert_awaited_once_with(WorkloadDelegationEntity, "old")


def test_docker_workload_delegation_name_is_deterministic() -> None:
    expected_input = '["default","deployment","deployment-123","container-abc"]'.encode()
    expected = "docker-" + hashlib.sha256(expected_input).hexdigest()[:48]

    name = docker_workload_delegation_name(
        scope=WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="deployment",
            workload_instance_id="deployment-123",
        ),
        workload_generation="container-abc",
    )

    assert name == expected
    assert (
        docker_deployment_delegation_name(
            workload_workspace="default",
            deployment_id="deployment-123",
            container_name="container-abc",
        )
        == name
    )


def test_kubernetes_pod_uid_delegation_name_uses_verified_reference_key() -> None:
    name = kubernetes_pod_uid_delegation_name(
        workload_audience="nemo-helix",
        workload_subject="system:serviceaccount:ns:runner",
        pod_uid="pod-uid-123",
    )

    assert name == reference_delegation_name(
        workload_audience="nemo-helix",
        workload_subject="system:serviceaccount:ns:runner",
        bound_reference_name="authentication.kubernetes.io/pod-uid",
        bound_reference_value="pod-uid-123",
    )


@pytest.mark.asyncio
async def test_register_accepts_generic_docker_workload_delegation() -> None:
    entity_client = _entity_client()
    entity = _generic_docker_entity()
    entity_client.add.return_value = entity

    saved = await WorkloadDelegationStore(entity_client).register(
        entity,
        require_opaque_subject_token_hash=True,
    )

    assert saved == entity
    entity_client.add.assert_awaited_once_with(entity)


@pytest.mark.asyncio
async def test_register_accepts_generic_verified_reference_workload_delegation() -> None:
    entity_client = _entity_client()
    entity = _generic_reference_entity()
    entity_client.add.return_value = entity

    saved = await WorkloadDelegationStore(entity_client).register(entity)

    assert saved == entity
    entity_client.add.assert_awaited_once_with(entity)


@pytest.mark.asyncio
async def test_register_rejects_partial_generic_workload_metadata() -> None:
    with pytest.raises(WorkloadDelegationValidationError):
        await WorkloadDelegationStore(_entity_client()).register(_generic_docker_entity(workload_generation=None))


@pytest.mark.asyncio
async def test_list_by_workload_pages_through_matching_delegations() -> None:
    entity_client = _entity_client()
    first = _generic_docker_entity(workload_generation="container-a")
    second = _generic_reference_entity(workload_generation="pod-uid-b", bound_reference_value="pod-uid-b")
    second.name = kubernetes_pod_uid_delegation_name(
        workload_audience=second.workload_audience,
        workload_subject=second.workload_subject,
        pod_uid="pod-uid-b",
    )
    entity_client.list.side_effect = [
        _list_response([first], page=1, page_size=1, total_pages=2),
        _list_response([second], page=2, page_size=1, total_pages=2),
    ]

    result = await WorkloadDelegationStore(entity_client).list_by_workload(
        _generic_scope(workload_claim_id="logical-deployment"),
        page_size=1,
    )

    assert result == [first, second]
    assert entity_client.list.await_count == 2
    first_call = entity_client.list.await_args_list[0]
    assert first_call.kwargs["workspace"] == SYSTEM_WORKSPACE
    assert first_call.kwargs["page"] == 1
    assert first_call.kwargs["page_size"] == 1
    filter_operation = first_call.kwargs["filter_operation"]
    assert {operation.field: operation.value for operation in filter_operation.operations} == {
        "data.workload_workspace": "default",
        "data.workload_kind": "deployment",
        "data.workload_id": "deployment-123",
    }


def test_workload_delegation_scope_rejects_empty_instance_or_claim() -> None:
    with pytest.raises(ValidationError, match="workload_instance_id"):
        WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="deployment",
            workload_instance_id="",
        )

    with pytest.raises(ValidationError, match="workload_claim_id"):
        WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="deployment",
            workload_instance_id="deployment-123",
            workload_claim_id="",
        )


@pytest.mark.asyncio
async def test_revoke_by_workload_deletes_exact_selected_ids() -> None:
    entity_client = _entity_client()
    first = _generic_docker_entity(workload_generation="container-a")
    first._id = "docker-grant"
    second = _generic_reference_entity()
    second._id = "pod-grant"
    entity_client.list.return_value = _list_response([first, second])
    await WorkloadDelegationStore(entity_client).revoke_by_workload(_generic_scope())
    assert [call.args[1] for call in entity_client.delete_by_id.await_args_list] == [first.id, second.id]
    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_native_pod_renewal_cannot_add_a_docker_proof() -> None:
    entity_client = _entity_client()
    pod = _reference_entity()
    entity_client.get_by_id.return_value = pod
    with pytest.raises(WorkloadDelegationValidationError, match="opaque Docker"):
        await WorkloadDelegationStore(entity_client).renew(
            "grant", expected_db_version=pod.db_version, expires_at=_expires_at(), opaque_subject_token_hash="forged"
        )
    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_by_id_never_resolves_current_lookup_name() -> None:
    entity_client = _entity_client()
    old = _entity()
    old._id = "old-grant"
    entity_client.get_by_id.return_value = old
    await WorkloadDelegationStore(entity_client).revoke_by_id(old.id)
    entity_client.delete_by_id.assert_awaited_once_with(
        WorkloadDelegationEntity, old.id, expected_db_version=old.db_version
    )
    entity_client.get.assert_not_called()
    entity_client.update.assert_not_called()


def test_sync_revoke_by_id_never_resolves_current_lookup_name() -> None:
    entity_client = MagicMock()
    old = _entity()
    old._id = "old-grant"
    entity_client.get_by_id.return_value = old
    SyncWorkloadDelegationStore(entity_client).revoke_by_id(old.id)
    entity_client.delete_by_id.assert_called_once_with(
        WorkloadDelegationEntity, old.id, expected_db_version=old.db_version
    )
    entity_client.get.assert_not_called()
    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_missing_is_idempotent_and_cleans_existing_revoked_record() -> None:
    entity_client = _entity_client()
    entity_client.get_by_id.side_effect = EntityNotFoundError("missing")
    store = WorkloadDelegationStore(entity_client)
    assert await store.revoke_by_id("missing") is None
    entity_client.get_by_id.side_effect = None
    revoked = _entity(revoked_at=datetime.now(timezone.utc))
    revoked._id = "old"
    entity_client.get_by_id.return_value = revoked
    await store.revoke_by_id("old")
    entity_client.delete_by_id.assert_awaited_once_with(
        WorkloadDelegationEntity, "old", expected_db_version=revoked.db_version
    )
    entity_client.update.assert_not_called()
    entity_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_bind_docker_container_is_write_once_and_renewal_preserves_it() -> None:
    entity_client = _entity_client()
    entity = _job_step_entity(opaque_subject_token_hash="proof")
    entity._id = "grant"
    entity_client.get_by_id.return_value = entity
    entity_client.update.side_effect = lambda value: value
    store = WorkloadDelegationStore(entity_client)
    bound = await store.bind_docker_container("grant", "container-a")
    assert bound.docker_container_id == "container-a"
    assert bound.job_uid == "platform-job-123"
    assert entity.docker_container_id is None
    entity_client.get_by_id.return_value = bound
    entity_client.update.reset_mock()
    assert await store.bind_docker_container("grant", "container-a") is bound
    with pytest.raises(WorkloadDelegationConflictError, match="different container"):
        await store.bind_docker_container("grant", "container-b")
    renewed = await store.renew("grant", expected_db_version=bound.db_version, expires_at=_expires_at())
    assert renewed.docker_container_id == "container-a"
    assert renewed.job_uid == "platform-job-123"


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_id", [None, "container-a"])
async def test_copied_label_cannot_revoke_another_container_grant(stored_id: str | None) -> None:
    entity_client = _entity_client()
    grant = _entity(docker_container_id=stored_id, opaque_subject_token_hash="proof")
    grant._id = "grant"
    entity_client.get_by_id.return_value = grant
    with pytest.raises(WorkloadDelegationConflictError, match="does not match"):
        await WorkloadDelegationStore(entity_client).revoke_for_docker_container("grant", "container-b")
    entity_client.delete_by_id.assert_not_awaited()


def test_sync_binding_and_verified_revocation() -> None:
    entity_client = MagicMock()
    entity = _entity(opaque_subject_token_hash="proof")
    entity._id = "grant"
    entity_client.get_by_id.return_value = entity
    entity_client.update.side_effect = lambda value: value
    store = SyncWorkloadDelegationStore(entity_client)
    bound = store.bind_docker_container("grant", "container-a")
    entity_client.get_by_id.return_value = bound
    assert store.bind_docker_container("grant", "container-a") is bound
    with pytest.raises(WorkloadDelegationConflictError):
        store.bind_docker_container("grant", "container-b")
    with pytest.raises(WorkloadDelegationConflictError):
        store.revoke_for_docker_container("grant", "container-b")
    store.revoke_for_docker_container("grant", "container-a")
    entity_client.delete_by_id.assert_called_once_with(
        WorkloadDelegationEntity, "grant", expected_db_version=bound.db_version
    )
    entity_client.get_by_id.side_effect = EntityNotFoundError("deleted")
    with pytest.raises(EntityNotFoundError):
        store.bind_docker_container("grant", "container-a")


def test_sync_renewal_rotates_only_the_existing_docker_proof_and_expiry() -> None:
    entity_client = MagicMock()
    current = _entity(opaque_subject_token_hash="old-proof", docker_container_id="container-a")
    current._id = "grant"
    entity_client.get_by_id.return_value = current
    entity_client.update.side_effect = lambda entity: entity
    expiry = _expires_at()
    renewed = SyncWorkloadDelegationStore(entity_client).renew(
        "grant",
        expected_db_version=current.db_version,
        expires_at=expiry,
        opaque_subject_token_hash="new-proof",
    )
    assert renewed.id == current.id
    assert renewed.docker_container_id == current.docker_container_id
    assert renewed.auth_context == current.auth_context
    assert renewed.opaque_subject_token_hash == "new-proof" and renewed.expires_at == expiry
    entity_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_async_revocation_retries_exact_id_after_renewal() -> None:
    client = _entity_client()
    original = _entity(docker_container_id="container-a")
    original._id = "old-grant"
    original._db_version = 7
    renewed = original.model_copy()
    renewed._db_version = 8
    client.get_by_id.side_effect = [original, renewed]
    client.delete_by_id.side_effect = [EntityConflictError("renewed"), None]

    await WorkloadDelegationStore(client).revoke_by_id(original.id)

    assert [entry.kwargs["expected_db_version"] for entry in client.delete_by_id.await_args_list] == [7, 8]
    assert [entry.args[1] for entry in client.delete_by_id.await_args_list] == [original.id, original.id]
    client.get.assert_not_called()


@pytest.mark.asyncio
async def test_async_revocation_stops_after_three_conflicts() -> None:
    client = _entity_client()
    original = _entity()
    original._id = "grant"
    client.get_by_id.return_value = original
    client.delete_by_id.side_effect = EntityConflictError("renewed")
    with pytest.raises(WorkloadDelegationConflictError):
        await WorkloadDelegationStore(client).revoke_by_id(original.id)
    assert client.delete_by_id.call_count == 3


@pytest.mark.asyncio
async def test_async_docker_revocation_checks_binding_after_conflict() -> None:
    client = _entity_client()
    original = _entity(docker_container_id="container-a")
    original._id = "grant"
    changed = original.model_copy(update={"docker_container_id": "container-b"})
    client.get_by_id.side_effect = [original, changed]
    client.delete_by_id.side_effect = EntityConflictError("changed")
    with pytest.raises(WorkloadDelegationConflictError, match="binding changed"):
        await WorkloadDelegationStore(client).revoke_for_docker_container(original.id, "container-a")
    assert client.delete_by_id.call_count == 1


@pytest.mark.asyncio
async def test_async_batch_revocation_attempts_siblings_after_failure() -> None:
    client = _entity_client()
    first = _generic_docker_entity()
    first._id = "first"
    second = _generic_reference_entity()
    second._id = "second"
    client.list.return_value = _list_response([first, second])
    client.delete_by_id.side_effect = [RuntimeError("entity service unavailable"), None]

    with pytest.raises(WorkloadDelegationError, match="Some workload delegations") as failure:
        await WorkloadDelegationStore(client).revoke_by_workload(_generic_scope())

    assert [entry.args[1] for entry in client.delete_by_id.await_args_list] == [first.id, second.id]
    assert isinstance(failure.value.__cause__, ExceptionGroup)


@pytest.mark.asyncio
async def test_async_revocation_does_not_follow_replacement_after_conflict() -> None:
    client = _entity_client()
    original = _entity()
    original._id = "old-grant"
    client.get_by_id.side_effect = [original, EntityNotFoundError("old grant deleted")]
    client.delete_by_id.side_effect = EntityConflictError("changed")
    await WorkloadDelegationStore(client).revoke_by_id(original.id)
    assert client.delete_by_id.call_count == 1
    client.get.assert_not_called()


def test_sync_revocation_retries_exact_id_after_renewal() -> None:
    client = MagicMock()
    original = _entity(docker_container_id="container-a")
    original._id = "old-grant"
    original._db_version = 7
    renewed = original.model_copy()
    renewed._db_version = 8
    client.get_by_id.side_effect = [original, renewed]
    client.delete_by_id.side_effect = [EntityConflictError("renewed"), None]

    SyncWorkloadDelegationStore(client).revoke_by_id(original.id)

    assert [entry.kwargs["expected_db_version"] for entry in client.delete_by_id.call_args_list] == [7, 8]
    assert [entry.args[1] for entry in client.delete_by_id.call_args_list] == [original.id, original.id]
    client.get.assert_not_called()


def test_sync_revocation_stops_after_three_conflicts() -> None:
    client = MagicMock()
    original = _entity()
    original._id = "grant"
    client.get_by_id.return_value = original
    client.delete_by_id.side_effect = EntityConflictError("renewed")
    with pytest.raises(WorkloadDelegationConflictError):
        SyncWorkloadDelegationStore(client).revoke_by_id(original.id)
    assert client.delete_by_id.call_count == 3


def test_sync_docker_revocation_checks_binding_after_conflict() -> None:
    client = MagicMock()
    original = _entity(docker_container_id="container-a")
    original._id = "grant"
    changed = original.model_copy(update={"docker_container_id": "container-b"})
    client.get_by_id.side_effect = [original, changed]
    client.delete_by_id.side_effect = EntityConflictError("changed")
    with pytest.raises(WorkloadDelegationConflictError, match="binding changed"):
        SyncWorkloadDelegationStore(client).revoke_for_docker_container(original.id, "container-a")
    assert client.delete_by_id.call_count == 1


def test_sync_batch_revocation_attempts_siblings_after_failure() -> None:
    client = MagicMock()
    first = _generic_docker_entity()
    first._id = "first"
    second = _generic_reference_entity()
    second._id = "second"
    client.list.return_value = _list_response([first, second])
    client.delete_by_id.side_effect = [RuntimeError("entity service unavailable"), None]

    with pytest.raises(WorkloadDelegationError, match="Some workload delegations") as failure:
        SyncWorkloadDelegationStore(client).revoke_by_workload(_generic_scope())

    assert [entry.args[1] for entry in client.delete_by_id.call_args_list] == [first.id, second.id]
    assert isinstance(failure.value.__cause__, ExceptionGroup)


def test_sync_revocation_does_not_follow_replacement_after_conflict() -> None:
    client = MagicMock()
    original = _entity()
    original._id = "old-grant"
    client.get_by_id.side_effect = [original, EntityNotFoundError("old grant deleted")]
    client.delete_by_id.side_effect = EntityConflictError("changed")
    SyncWorkloadDelegationStore(client).revoke_by_id(original.id)
    assert client.delete_by_id.call_count == 1
    client.get.assert_not_called()


@pytest.mark.asyncio
async def test_empty_docker_reference_fields_normalize_on_registration() -> None:
    client = _entity_client()
    grant = _entity(opaque_subject_token_hash="proof", bound_reference_name="", bound_reference_value="")
    grant._id = "grant"
    client.add.side_effect = lambda value: value

    registered = await WorkloadDelegationStore(client).register(grant, require_opaque_subject_token_hash=True)

    assert registered.bound_reference_name is None
    assert registered.bound_reference_value is None
    assert client.add.await_args.args[0].bound_reference_name is None


@pytest.mark.asyncio
async def test_existing_empty_docker_references_allow_binding_and_proof_rotation() -> None:
    client = _entity_client()
    grant = _entity(opaque_subject_token_hash="proof")
    grant._id = "grant"
    legacy = grant.model_copy(update={"bound_reference_name": "", "bound_reference_value": ""})
    client.get_by_id.return_value = legacy
    client.update.side_effect = lambda value: value
    store = WorkloadDelegationStore(client)

    bound = await store.bind_docker_container(grant.id, "container")
    assert bound.docker_container_id == "container"
    renewed = await store.renew(
        grant.id, expected_db_version=legacy.db_version, expires_at=_expires_at(), opaque_subject_token_hash="rotated"
    )
    assert renewed.opaque_subject_token_hash == "rotated"


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_name,reference_value", [("reference", "value"), ("", "value")])
async def test_docker_binding_and_rotation_reject_actual_references(reference_name, reference_value) -> None:
    client = _entity_client()
    grant = _entity(
        opaque_subject_token_hash="proof",
        bound_reference_name=reference_name,
        bound_reference_value=reference_value,
    )
    grant._id = "grant"
    client.get_by_id.return_value = grant
    store = WorkloadDelegationStore(client)

    with pytest.raises(WorkloadDelegationConflictError, match="active Docker"):
        await store.bind_docker_container(grant.id, "container")
    with pytest.raises(WorkloadDelegationValidationError, match="opaque Docker"):
        await store.renew(
            grant.id,
            expected_db_version=grant.db_version,
            expires_at=_expires_at(),
            opaque_subject_token_hash="rotated",
        )
    client.update.assert_not_awaited()


def test_job_uid_is_none_when_the_record_has_none_or_an_empty_one() -> None:
    # A record written before the field existed has no such key. An entity ID that was never set reads as "".
    assert _entity().job_uid is None
    assert _entity(job_uid="").job_uid is None
    assert _entity(job_uid="platform-job-123").job_uid == "platform-job-123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entity",
    [
        pytest.param(_job_step_entity(opaque_subject_token_hash="proof"), id="docker-job-step"),
        pytest.param(
            _reference_entity(
                workload_kind="job",
                workload_id="job-123",
                workload_generation="attempt-1/step-a/pod-uid-123",
                job_uid="platform-job-123",
            ),
            id="kubernetes-job-step",
        ),
    ],
)
async def test_register_accepts_job_uid_on_a_job_step_row(entity: WorkloadDelegationEntity) -> None:
    entity_client = _entity_client()
    entity_client.add.side_effect = lambda value: value
    sync_entity_client = MagicMock()
    sync_entity_client.add.side_effect = lambda value: value

    saved = await WorkloadDelegationStore(entity_client).register(entity)
    sync_saved = SyncWorkloadDelegationStore(sync_entity_client).register(entity)

    assert saved.job_uid == "platform-job-123"
    assert sync_saved.job_uid == "platform-job-123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entity",
    [
        pytest.param(_entity(job_uid="platform-job-123"), id="legacy-job-row-without-kind"),
        pytest.param(_generic_docker_entity(job_uid="platform-job-123"), id="deployment-docker-row"),
        pytest.param(
            _generic_reference_entity(workload_kind="job", job_uid="platform-job-123"),
            id="job-kind-row-without-job-fields",
        ),
        # Each job field is required on its own: these rows pass every other register check.
        *[
            pytest.param(
                _reference_entity(
                    workload_kind="job",
                    workload_id="job-123",
                    workload_generation="attempt-1/step-a/pod-uid-123",
                    job_uid="platform-job-123",
                    **{missing: None},
                ),
                id=f"job-kind-row-without-{missing.replace('_', '-')}",
            )
            for missing in ("job_id", "attempt_id", "step_id")
        ],
    ],
)
async def test_register_rejects_job_uid_outside_a_job_step_row(entity: WorkloadDelegationEntity) -> None:
    entity_client = _entity_client()
    sync_entity_client = MagicMock()

    with pytest.raises(WorkloadDelegationValidationError, match="job_uid"):
        await WorkloadDelegationStore(entity_client).register(entity)
    with pytest.raises(WorkloadDelegationValidationError, match="job_uid"):
        SyncWorkloadDelegationStore(sync_entity_client).register(entity)

    entity_client.add.assert_not_called()
    sync_entity_client.add.assert_not_called()


# The argument is left untyped on purpose: a caller that passes None where a job ID is required must get False,
# never a match between two missing IDs.
@pytest.mark.parametrize(
    ("record", "job_uid", "expected"),
    [
        pytest.param(_job_step_entity(), "platform-job-123", True, id="active-job-row-with-the-same-id"),
        pytest.param(_job_step_entity(), "platform-job-999", False, id="another-job"),
        pytest.param(_job_step_entity(), "", False, id="empty-argument"),
        pytest.param(_job_step_entity(), None, False, id="missing-argument"),
        pytest.param(_job_step_entity(job_uid=None), "platform-job-123", False, id="record-without-job-uid"),
        pytest.param(_job_step_entity(job_uid=None), "", False, id="record-without-job-uid-and-empty-argument"),
        pytest.param(_job_step_entity(job_uid=None), None, False, id="record-without-job-uid-and-missing-argument"),
        # An empty ID can only reach the record past validation, as model_copy does.
        pytest.param(
            _job_step_entity().model_copy(update={"job_uid": ""}),
            "",
            False,
            id="record-with-empty-job-uid-and-empty-argument",
        ),
        pytest.param(
            _job_step_entity(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
            "platform-job-123",
            False,
            id="expired-record",
        ),
        pytest.param(
            _job_step_entity(revoked_at=datetime.now(timezone.utc)), "platform-job-123", False, id="revoked-record"
        ),
        pytest.param(
            _job_step_entity(workload_kind="deployment"), "platform-job-123", False, id="deployment-kind-record"
        ),
        pytest.param(_entity(job_uid="platform-job-123"), "platform-job-123", False, id="record-without-a-kind"),
    ],
)
def test_is_step_of_job(record: WorkloadDelegationEntity, job_uid, expected: bool) -> None:
    assert record.is_step_of_job(job_uid) is expected


@pytest.mark.parametrize(
    ("record", "job_uid", "attempt_id", "expected"),
    [
        pytest.param(_job_step_entity(), "platform-job-123", "attempt-1", True, id="same-job-and-attempt"),
        pytest.param(_job_step_entity(), "platform-job-123", "attempt-2", False, id="another-attempt"),
        pytest.param(_job_step_entity(), "platform-job-999", "attempt-1", False, id="another-job"),
        pytest.param(_job_step_entity(), "platform-job-123", "", False, id="empty-attempt-argument"),
        pytest.param(_job_step_entity(), "platform-job-123", None, False, id="missing-attempt-argument"),
        pytest.param(
            _job_step_entity(attempt_id=None), "platform-job-123", "attempt-1", False, id="record-without-attempt"
        ),
        pytest.param(
            _job_step_entity(attempt_id=None),
            "platform-job-123",
            None,
            False,
            id="record-without-attempt-and-missing-argument",
        ),
        pytest.param(
            _job_step_entity(attempt_id=""),
            "platform-job-123",
            "",
            False,
            id="record-with-empty-attempt-and-empty-argument",
        ),
        pytest.param(
            _job_step_entity(job_uid=None), "platform-job-123", "attempt-1", False, id="record-without-job-uid"
        ),
        pytest.param(
            _job_step_entity(revoked_at=datetime.now(timezone.utc)),
            "platform-job-123",
            "attempt-1",
            False,
            id="revoked-record",
        ),
    ],
)
def test_is_step_of_attempt(record: WorkloadDelegationEntity, job_uid, attempt_id, expected: bool) -> None:
    assert record.is_step_of_attempt(job_uid, attempt_id) is expected


def test_is_step_of_job_and_attempt_judge_expiry_at_the_supplied_time() -> None:
    record = _job_step_entity()
    before_expiry = record.expires_at - timedelta(seconds=1)
    after_expiry = record.expires_at + timedelta(seconds=1)

    assert record.is_step_of_job("platform-job-123", now=before_expiry)
    assert record.is_step_of_attempt("platform-job-123", "attempt-1", now=before_expiry)
    assert not record.is_step_of_job("platform-job-123", now=after_expiry)
    assert not record.is_step_of_attempt("platform-job-123", "attempt-1", now=after_expiry)
