# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-owned delegation cleanup through the real entity HTTP API and SQLite."""

from unittest.mock import patch

import pytest
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import SyncWorkloadDelegationStore, WorkloadDelegationEntity
from nemo_helix_plugin.auth.workload_identity import (
    WorkloadDelegationScope,
    build_docker_opaque_workload_delegation,
    build_kubernetes_pod_uid_workload_delegation,
)
from nemo_helix_plugin.entities.client import EntitiesClient
from nhx.common.entities import EntityNotFoundError, SyncEntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.controllers.delegation_cleanup import JobDelegationCleanup
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt
from nhx.testing import ClientContext, create_test_client

pytestmark = pytest.mark.integration


def _job(entities: SyncEntityClient, *, workspace: str = "default") -> HelixJob:
    return entities.create(
        HelixJob(
            name="reusable-name",
            workspace=workspace,
            source="test",
            platform_spec=TestConstants.PLATFORM_SPEC,
            fileset="borrowed",
            output_location="borrowed",
            status=HelixJobStatus.ACTIVE,
        )
    )


def _attempt(entities: SyncEntityClient, job: HelixJob, *, name: str, status: HelixJobStatus) -> HelixJobAttempt:
    return entities.create(
        HelixJobAttempt(
            name=name,
            workspace=job.workspace,
            job=job.id,
            platform_spec=job.platform_spec,
            status=status,
        )
    )


def _grant(
    entities: SyncEntityClient, job: HelixJob, attempt: HelixJobAttempt, *, proof: str = "docker", legacy: bool = False
) -> WorkloadDelegationEntity:
    scope = WorkloadDelegationScope(workload_workspace=job.workspace, workload_kind="job", workload_instance_id=job.id)
    if proof == "docker":
        grant, _ = build_docker_opaque_workload_delegation(
            scope=scope,
            workload_audience="nemo-helix",
            workload_generation=f"{attempt.id}/step",
            auth_context=AuthContext(principal_id="creator"),
            ttl_seconds_active=3600,
            job_id=job.id,
            attempt_id=attempt.id,
            step_id="step",
        )
    else:
        grant = build_kubernetes_pod_uid_workload_delegation(
            scope=scope,
            workload_audience="nemo-helix",
            workload_generation=attempt.id,
            namespace="jobs",
            service_account_name="runner",
            pod_uid=attempt.id,
            auth_context=AuthContext(principal_id="creator"),
            ttl_seconds_active=3600,
            job_id=job.id,
            attempt_id=attempt.id,
            step_id="step",
        )
    if legacy:
        grant.workload_kind = None
        grant.workload_id = None
        grant.workload_generation = None
    return SyncWorkloadDelegationStore(entities).register(grant)


@pytest.mark.parametrize("status", HelixJobStatus.terminals())
@pytest.mark.parametrize("proof", ["docker", "kubernetes"])
@pytest.mark.parametrize("legacy", [False, True])
def test_completed_attempt_cleanup_preserves_live_retry(status, proof, legacy):
    with create_test_client(client_type=ClientContext, workspaces=["system", "default"], projects=[]) as ctx:
        entities = SyncEntityClient(EntitiesClient.from_client(ctx.client)).as_service("jobs", internal=True)
        job = _job(entities)
        old_attempt = _attempt(entities, job, name="old", status=status)
        current_attempt = _attempt(entities, job, name="retry", status=HelixJobStatus.ACTIVE)
        old = _grant(entities, job, old_attempt, proof=proof, legacy=legacy)
        current = _grant(entities, job, current_attempt, proof=proof, legacy=legacy)
        job.current_attempt_id = current_attempt.id
        entities.update(job)
        assert old.docker_container_id is None

        JobDelegationCleanup(entities).sweep(on_progress=lambda: None)

        with pytest.raises(EntityNotFoundError):
            entities.get_by_id(WorkloadDelegationEntity, old.id)
        assert entities.get_by_id(WorkloadDelegationEntity, current.id).is_active()
        assert entities.get_by_id(HelixJob, job.id).current_attempt_id == current_attempt.id


def test_job_deletion_waits_for_delegation_cleanup_and_restart_recovers(backend_registry):
    with create_test_client(client_type=ClientContext, workspaces=["system", "default", "other"], projects=[]) as ctx:
        entities = SyncEntityClient(EntitiesClient.from_client(ctx.client)).as_service("jobs", internal=True)
        job = _job(entities)
        attempt = _attempt(entities, job, name="initial", status=HelixJobStatus.ERROR)
        grant = _grant(entities, job, attempt)
        other_job = _job(entities, workspace="other")
        other_attempt = _attempt(entities, other_job, name="initial", status=HelixJobStatus.ACTIVE)
        other_grant = _grant(entities, other_job, other_attempt)
        job.status = HelixJobStatus.DELETION_PENDING
        entities.update(job)
        first = JobReconciler(backend_registry, ctx.client, system_entity_client=entities)
        with patch.object(entities, "delete_by_id", side_effect=RuntimeError("entity service unavailable")):
            first._cleanup_pending_jobs()
        assert entities.get_by_id(HelixJob, job.id).status == HelixJobStatus.DELETION_PENDING
        assert entities.get_by_id(WorkloadDelegationEntity, grant.id).is_active()

        JobReconciler(backend_registry, ctx.client, system_entity_client=entities)._cleanup_pending_jobs()

        for entity_type, entity_id in (
            (HelixJob, job.id),
            (HelixJobAttempt, attempt.id),
            (WorkloadDelegationEntity, grant.id),
        ):
            with pytest.raises(EntityNotFoundError):
                entities.get_by_id(entity_type, entity_id)
        assert entities.get_by_id(WorkloadDelegationEntity, other_grant.id).is_active()


def test_concurrent_renewal_is_retried_from_persisted_grant():
    with create_test_client(client_type=ClientContext, workspaces=["system", "default"], projects=[]) as ctx:
        entities = SyncEntityClient(EntitiesClient.from_client(ctx.client)).as_service("jobs", internal=True)
        job = _job(entities)
        attempt = _attempt(entities, job, name="initial", status=HelixJobStatus.COMPLETED)
        grant = _grant(entities, job, attempt)
        delete = entities.delete_by_id

        def renew_before_delete(
            entity_type: type[WorkloadDelegationEntity], entity_id: str, *, expected_db_version: int | None = None
        ) -> None:
            current = entities.get_by_id(WorkloadDelegationEntity, grant.id)
            SyncWorkloadDelegationStore(entities).renew(
                current.id, expected_db_version=current.db_version, expires_at=current.expires_at
            )
            delete(entity_type, entity_id, expected_db_version=expected_db_version)

        with patch.object(entities, "delete_by_id", side_effect=renew_before_delete):
            JobDelegationCleanup(entities).sweep(on_progress=lambda: None)
        assert entities.get_by_id(WorkloadDelegationEntity, grant.id).is_active()

        JobDelegationCleanup(entities).sweep(on_progress=lambda: None)

        with pytest.raises(EntityNotFoundError):
            entities.get_by_id(WorkloadDelegationEntity, grant.id)


def test_late_registration_after_deletion_does_not_affect_reused_job_name():
    with create_test_client(client_type=ClientContext, workspaces=["system", "default"], projects=[]) as ctx:
        entities = SyncEntityClient(EntitiesClient.from_client(ctx.client)).as_service("jobs", internal=True)
        old_job = _job(entities)
        old_attempt = _attempt(entities, old_job, name="initial", status=HelixJobStatus.COMPLETED)
        entities.delete_by_id(HelixJob, old_job.id)
        old_grant = _grant(entities, old_job, old_attempt)
        current_job = _job(entities)
        current_attempt = _attempt(entities, current_job, name="initial", status=HelixJobStatus.ACTIVE)
        current_grant = _grant(entities, current_job, current_attempt)

        JobDelegationCleanup(entities).sweep(on_progress=lambda: None)

        with pytest.raises(EntityNotFoundError):
            entities.get_by_id(WorkloadDelegationEntity, old_grant.id)
        assert entities.get_by_id(WorkloadDelegationEntity, current_grant.id).is_active()
