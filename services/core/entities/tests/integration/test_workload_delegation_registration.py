# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegation lifecycle against the real entity HTTP API and SQLite repository."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationConflictError,
    WorkloadDelegationEntity,
    WorkloadDelegationScope,
    WorkloadDelegationStore,
    create_opaque_docker_proof_token,
    docker_delegation_name,
)
from nemo_helix_plugin.auth.workload_identity import build_kubernetes_pod_uid_workload_delegation
from nhx.common.auth import Principal
from nhx.common.entities import EntityClient
from nhx.testing import create_test_client


def _delegation() -> WorkloadDelegationEntity:
    name = docker_delegation_name(
        workload_workspace="default", job_id="job-1", attempt_id="attempt-1", step_id="step-1"
    )
    _, proof_hash = create_opaque_docker_proof_token(name)
    return WorkloadDelegationEntity(
        name=name,
        workspace="system",
        workload_subject=name,
        workload_audience="nemo-helix",
        workload_workspace="default",
        workload_generation="attempt-1/step-1",
        workload_kind="job",
        workload_id="job-1",
        job_id="job-1",
        attempt_id="attempt-1",
        step_id="step-1",
        auth_context=AuthContext.from_principal(Principal(id="creator@example.com")),
        opaque_subject_token_hash=proof_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


@pytest.fixture
def entity_client():
    with create_test_client(client_type=EntityClient, workspaces=["system", "default"], projects=[]) as client:
        yield client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_keeps_revoked_id_and_refresh_keeps_current_id(entity_client: EntityClient) -> None:
    store = WorkloadDelegationStore(entity_client)
    old = await store.register(_delegation())
    await store.revoke(old.name)
    current = await store.register(_delegation())
    assert current.id != old.id
    assert current.name == old.name
    listed = await store.list_by_workload(
        WorkloadDelegationScope(workload_workspace="default", workload_kind="job", workload_instance_id="job-1")
    )
    assert [entity.id for entity in listed] == [current.id]
    assert await store.get_by_id(old.id) is None
    resolved = await store.get(old.name)
    assert resolved is not None and resolved.id == current.id
    with pytest.raises(WorkloadDelegationConflictError):
        await store.renew(old.id, expected_db_version=old.db_version, expires_at=old.expires_at)
    _, rotated_hash = create_opaque_docker_proof_token(current.name)
    renewed = await store.renew(
        current.id,
        expected_db_version=current.db_version,
        expires_at=current.expires_at + timedelta(hours=1),
        opaque_subject_token_hash=rotated_hash,
    )
    assert renewed.id == current.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_competing_retries_create_one_active_grant(entity_client: EntityClient) -> None:
    store = WorkloadDelegationStore(entity_client)
    old = await store.register(_delegation())
    await entity_client.update(old.model_copy(update={"revoked_at": datetime.now(timezone.utc)}))
    get = store.get
    both_read = asyncio.Event()
    readers = 0

    async def get_before_race(name: str) -> WorkloadDelegationEntity | None:
        nonlocal readers
        entity = await get(name)
        readers += 1
        if readers == 2:
            both_read.set()
        await asyncio.wait_for(both_read.wait(), timeout=10)
        return entity

    with patch.object(store, "get", side_effect=get_before_race):
        results = await asyncio.gather(
            store.register(_delegation()), store.register(_delegation()), return_exceptions=True
        )
    assert sum(isinstance(result, WorkloadDelegationEntity) for result in results) == 1
    assert sum(isinstance(result, WorkloadDelegationConflictError) for result in results) == 1
    current = await store.get(old.name)
    assert current is not None and current.is_active() and current.id != old.id
    assert await store.get_by_id(old.id) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_registration_recovers_after_interruption_between_delete_and_create(
    entity_client: EntityClient,
) -> None:
    store = WorkloadDelegationStore(entity_client)
    old = await store.register(_delegation())
    await entity_client.update(old.model_copy(update={"revoked_at": datetime.now(timezone.utc)}))
    add = entity_client.add
    calls = 0

    async def interrupted_add(entity: WorkloadDelegationEntity) -> WorkloadDelegationEntity:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controller stopped")
        return await add(entity)

    with patch.object(entity_client, "add", side_effect=interrupted_add):
        with pytest.raises(RuntimeError, match="controller stopped"):
            await store.register(_delegation())
    assert await store.get(old.name) is None
    current = await store.register(_delegation())
    assert current.id != old.id
    assert await store.get_by_id(old.id) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delayed_exact_id_cleanup_preserves_retry_across_store_restart(entity_client: EntityClient) -> None:
    store = WorkloadDelegationStore(entity_client)
    old = await store.register(_delegation())
    await store.revoke_by_id(old.id)
    current = await store.register(_delegation())
    for cleanup_store in (store, WorkloadDelegationStore(entity_client)):
        await cleanup_store.revoke_by_id(old.id)
        resolved = await cleanup_store.get_by_id(current.id)
        assert resolved is not None and resolved.is_active()
    await store.revoke_by_id(current.id)
    revoked = await store.get_by_id(current.id)
    assert revoked is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_persisted_container_binding_survives_restart_and_rejects_copied_label(
    entity_client: EntityClient,
) -> None:
    store = WorkloadDelegationStore(entity_client)
    grant = await store.register(_delegation())
    await store.bind_docker_container(grant.id, "container-a")
    restarted = WorkloadDelegationStore(entity_client)
    with pytest.raises(WorkloadDelegationConflictError):
        await restarted.revoke_for_docker_container(grant.id, "container-b")
    with pytest.raises(WorkloadDelegationConflictError):
        await restarted.bind_docker_container(grant.id, "container-b")
    bound = await restarted.get_by_id(grant.id)
    assert bound is not None and bound.docker_container_id == "container-a" and bound.is_active()
    await restarted.revoke_for_docker_container(grant.id, "container-a")
    revoked = await restarted.get_by_id(grant.id)
    assert revoked is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_renewal_prevents_deleting_the_old_record(entity_client: EntityClient) -> None:
    store = WorkloadDelegationStore(entity_client)
    original = await store.register(_delegation())
    expired = await entity_client.update(
        original.model_copy(
            update={
                "expires_at": datetime.now(timezone.utc) - timedelta(seconds=1),
            }
        )
    )
    delete = entity_client.delete_by_id

    async def renew_before_delete(
        entity_type: type[WorkloadDelegationEntity], entity_id: str, *, expected_db_version: int | None = None
    ) -> None:
        await store.renew(
            expired.id,
            expected_db_version=expired.db_version,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        await delete(entity_type, entity_id, expected_db_version=expected_db_version)
        pytest.fail("stale deletion unexpectedly succeeded")

    with patch.object(entity_client, "delete_by_id", side_effect=renew_before_delete):
        with pytest.raises(WorkloadDelegationConflictError):
            await store.register(_delegation())
    survivor = await store.get_by_id(original.id)
    assert survivor is not None and survivor.is_active()
    current = await store.get(original.name)
    assert current is not None and current.id == original.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_native_pod_deletion_and_renewal_keep_sibling_grants_separate(entity_client: EntityClient) -> None:
    store = WorkloadDelegationStore(entity_client)
    scope = WorkloadDelegationScope(workload_workspace="default", workload_kind="job", workload_instance_id="job-1")
    grants = []
    for uid in ("pod-a", "pod-b"):
        grants.append(
            await store.register(
                build_kubernetes_pod_uid_workload_delegation(
                    scope=scope,
                    workload_audience="nemo-helix",
                    workload_generation=uid,
                    namespace="ns",
                    service_account_name="runner",
                    pod_uid=uid,
                    auth_context=AuthContext.from_principal(Principal(id="creator@example.com")),
                    ttl_seconds_active=3600,
                )
            )
        )
    first, second = grants
    await store.revoke(first.name)
    assert await store.get_by_id(first.id) is None
    renewed = await store.renew(
        second.id, expected_db_version=second.db_version, expires_at=second.expires_at + timedelta(hours=1)
    )
    assert renewed.id == second.id and renewed.is_active()
    assert renewed.bound_reference_value == "pod-b"
    assert renewed.opaque_subject_token_hash is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_revocation_retries_after_concurrent_renewal(entity_client: EntityClient) -> None:
    store = WorkloadDelegationStore(entity_client)
    original = await store.register(_delegation())
    delete = entity_client.delete_by_id
    calls = 0

    async def renew_once_before_delete(
        entity_type: type[WorkloadDelegationEntity], entity_id: str, *, expected_db_version: int | None = None
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await store.renew(
                original.id,
                expected_db_version=original.db_version,
                expires_at=original.expires_at + timedelta(hours=1),
            )
        await delete(entity_type, entity_id, expected_db_version=expected_db_version)

    with patch.object(entity_client, "delete_by_id", side_effect=renew_once_before_delete):
        await store.revoke_by_id(original.id)
    assert calls == 2
    assert await store.get_by_id(original.id) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_empty_references_load_as_absent_and_allow_binding(entity_client: EntityClient) -> None:
    store = WorkloadDelegationStore(entity_client)
    original = await store.register(_delegation())
    await entity_client.update(original.model_copy(update={"bound_reference_name": "", "bound_reference_value": ""}))

    loaded = await store.get_by_id(original.id)
    assert loaded is not None
    assert loaded.bound_reference_name is None and loaded.bound_reference_value is None
    bound = await store.bind_docker_container(original.id, "container")
    _, proof_hash = create_opaque_docker_proof_token(bound.name)
    renewed = await store.renew(
        bound.id,
        expected_db_version=bound.db_version,
        expires_at=bound.expires_at + timedelta(hours=1),
        opaque_subject_token_hash=proof_hash,
    )
    assert renewed.docker_container_id == "container"
    assert renewed.opaque_subject_token_hash == proof_hash
