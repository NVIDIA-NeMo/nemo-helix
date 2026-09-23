# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from nhx.common.auth import AuthContext, Principal
from nhx.common.auth.workload_delegations import (
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
    OPAQUE_DOCKER_PROOF_PREFIX,
    InvalidWorkloadProofTokenError,
    WorkloadDelegationConflictError,
    WorkloadDelegationEntity,
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
async def test_update_uses_fetched_db_version() -> None:
    entity_client = _entity_client()
    fetched = _entity(workload_audience="old-audience")
    fetched._db_version = 7
    replacement = _entity(workload_audience="new-audience")
    entity_client.get.return_value = fetched
    entity_client.update.side_effect = lambda entity: entity

    updated = await WorkloadDelegationStore(entity_client).update(replacement)

    entity_client.get.assert_awaited_once_with(WorkloadDelegationEntity, replacement.name, workspace=SYSTEM_WORKSPACE)
    entity_client.update.assert_awaited_once()
    update_arg = entity_client.update.call_args.args[0]
    assert update_arg.db_version == 7
    assert update_arg.workload_audience == "new-audience"
    assert updated.db_version == 7


@pytest.mark.asyncio
async def test_update_rejects_stale_expected_db_version() -> None:
    entity_client = _entity_client()
    fetched = _entity(workload_audience="old-audience")
    fetched._db_version = 7
    replacement = _entity(workload_audience="new-audience")
    entity_client.get.return_value = fetched

    with pytest.raises(WorkloadDelegationConflictError):
        await WorkloadDelegationStore(entity_client).update(replacement, expected_db_version=6)

    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_rejects_reactivating_revoked_row() -> None:
    entity_client = _entity_client()
    revoked_at = datetime.now(timezone.utc)
    fetched = _entity(revoked_at=revoked_at)
    fetched._db_version = 8
    stale_refresh = _entity(revoked_at=None)
    stale_refresh._db_version = 7
    entity_client.get.return_value = fetched

    with pytest.raises(WorkloadDelegationConflictError, match="cannot be reactivated"):
        await WorkloadDelegationStore(entity_client).update(stale_refresh)

    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_uses_fetched_db_version() -> None:
    entity_client = _entity_client()
    fetched = _entity()
    fetched._db_version = 9
    entity_client.get.return_value = fetched
    entity_client.update.side_effect = lambda entity: entity
    now = datetime.now(timezone.utc)

    revoked = await WorkloadDelegationStore(entity_client).revoke(fetched.name, now=now)

    assert revoked is not None
    entity_client.get.assert_awaited_once_with(WorkloadDelegationEntity, fetched.name, workspace=SYSTEM_WORKSPACE)
    entity_client.update.assert_awaited_once()
    update_arg = entity_client.update.call_args.args[0]
    assert update_arg.db_version == 9
    assert update_arg.revoked_at == now


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

    with pytest.raises(WorkloadDelegationConflictError):
        await WorkloadDelegationStore(entity_client).register(_entity())

    entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_register_conflict_replaces_expired_row_with_fetched_db_version() -> None:
    entity_client = _entity_client()
    expired = _entity(expires_at=datetime.now(timezone.utc) - timedelta(hours=1))
    expired._db_version = 12
    replacement = _entity()
    entity_client.add.side_effect = EntityConflictError("already exists")
    entity_client.get.return_value = expired
    entity_client.update.side_effect = lambda entity: entity

    saved = await WorkloadDelegationStore(entity_client).register(replacement)

    entity_client.update.assert_awaited_once()
    update_arg = entity_client.update.call_args.args[0]
    assert update_arg.db_version == 12
    assert update_arg.expires_at == replacement.expires_at
    assert saved.db_version == 12


@pytest.mark.asyncio
async def test_register_conflict_replaces_revoked_row_with_fetched_db_version() -> None:
    entity_client = _entity_client()
    revoked = _entity(revoked_at=datetime.now(timezone.utc))
    revoked._db_version = 13
    replacement = _entity()
    entity_client.add.side_effect = EntityConflictError("already exists")
    entity_client.get.return_value = revoked
    entity_client.update.side_effect = lambda entity: entity

    saved = await WorkloadDelegationStore(entity_client).register(replacement)

    entity_client.update.assert_awaited_once()
    update_arg = entity_client.update.call_args.args[0]
    assert update_arg.db_version == 13
    assert update_arg.revoked_at is None
    assert saved.db_version == 13


@pytest.mark.asyncio
async def test_register_conflict_rejects_stale_expected_db_version_for_replacement() -> None:
    entity_client = _entity_client()
    expired = _entity(expires_at=datetime.now(timezone.utc) - timedelta(hours=1))
    expired._db_version = 12
    replacement = _entity()
    entity_client.add.side_effect = EntityConflictError("already exists")
    entity_client.get.return_value = expired

    with pytest.raises(WorkloadDelegationConflictError):
        await WorkloadDelegationStore(entity_client).register(replacement, expected_db_version=11)

    entity_client.update.assert_not_called()


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
async def test_revoke_by_workload_updates_each_matching_delegation_with_one_timestamp() -> None:
    entity_client = _entity_client()
    first = _generic_docker_entity(workload_generation="container-a")
    second = _generic_reference_entity()
    now = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
    entity_client.list.return_value = _list_response([first, second])
    entity_client.update.side_effect = lambda entity: entity

    revoked = await WorkloadDelegationStore(entity_client).revoke_by_workload(
        _generic_scope(),
        now=now,
    )

    assert revoked == [first, second]
    assert [call.args[0].revoked_at for call in entity_client.update.await_args_list] == [now, now]
