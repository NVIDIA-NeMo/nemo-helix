# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from nhx.common.auth import AuthContext, WorkloadDelegationEntity
from nhx.common.entities import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.controllers.delegation_cleanup import JobDelegationCleanup
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt


def _grant(*, grant_id: str = "grant", kind: str | None = "job") -> WorkloadDelegationEntity:
    grant = WorkloadDelegationEntity(
        name=grant_id,
        workspace="system",
        workload_subject=grant_id,
        workload_audience="nemo-helix",
        workload_workspace="default",
        workload_kind=kind,
        workload_id="job-id",
        job_id="job-id",
        attempt_id="old-attempt",
        step_id="step",
        auth_context=AuthContext(principal_id="creator"),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    grant._id = grant_id
    grant._db_version = 7
    return grant


def _attempt(status: HelixJobStatus) -> HelixJobAttempt:
    attempt = HelixJobAttempt(
        name="old",
        workspace="default",
        job="job-id",
        platform_spec=TestConstants.PLATFORM_SPEC,
        status=status,
    )
    attempt._id = "old-attempt"
    return attempt


def _job() -> HelixJob:
    job = HelixJob(
        name="reusable-name",
        workspace="default",
        source="test",
        platform_spec=TestConstants.PLATFORM_SPEC,
        fileset="borrowed",
        output_location="borrowed",
        status=HelixJobStatus.DELETION_PENDING,
    )
    job._id = "job-id"
    return job


@pytest.fixture
def entities() -> MagicMock:
    entities = MagicMock(spec=SyncEntityClient)
    entities.list.return_value = SimpleNamespace(data=[_grant()])
    return entities


@pytest.mark.parametrize("status", HelixJobStatus.terminals())
@pytest.mark.parametrize("kind", ["job", None])
def test_finished_attempt_cleanup_does_not_require_container_binding(entities, status, kind):
    grant = _grant(kind=kind)
    entities.list.return_value = SimpleNamespace(data=[grant])
    entities.get_by_id.side_effect = [_attempt(status), grant]

    JobDelegationCleanup(entities).sweep(on_progress=lambda: None)

    entities.delete_by_id.assert_called_once_with(WorkloadDelegationEntity, grant.id, expected_db_version=7)


@pytest.mark.parametrize("status", HelixJobStatus.non_terminals())
def test_live_or_paused_attempt_keeps_its_grants(entities, status):
    entities.get_by_id.return_value = _attempt(status)

    JobDelegationCleanup(entities).sweep(on_progress=lambda: None)

    entities.delete_by_id.assert_not_called()


@pytest.mark.parametrize("changed", ["job", "workspace"])
def test_attempt_ownership_must_match_grant(entities, changed):
    attempt = _attempt(HelixJobStatus.COMPLETED)
    if changed == "job":
        attempt.job = "another-job"
    else:
        attempt.workspace = "other"
    entities.get_by_id.return_value = attempt

    JobDelegationCleanup(entities).sweep(on_progress=lambda: None)

    entities.delete_by_id.assert_not_called()


def test_deployment_grants_are_not_job_cleanup_candidates(entities):
    entities.list.return_value = SimpleNamespace(data=[_grant(kind="deployment")])

    JobDelegationCleanup(entities).sweep(on_progress=lambda: None)

    entities.get_by_id.assert_not_called()
    entities.delete_by_id.assert_not_called()


def test_missing_attempt_with_existing_job_keeps_grant(entities):
    entities.get_by_id.side_effect = [EntityNotFoundError("attempt missing"), _job()]

    JobDelegationCleanup(entities).sweep(on_progress=lambda: None)

    entities.delete_by_id.assert_not_called()


def test_grant_registered_after_job_deletion_is_swept(entities):
    grant = _grant()
    entities.get_by_id.side_effect = [EntityNotFoundError("attempt missing"), EntityNotFoundError("job missing"), grant]

    JobDelegationCleanup(entities).sweep(on_progress=lambda: None)

    entities.get_by_id.assert_has_calls(
        [call(HelixJobAttempt, "old-attempt"), call(HelixJob, "job-id"), call(WorkloadDelegationEntity, grant.id)]
    )
    entities.delete_by_id.assert_called_once_with(WorkloadDelegationEntity, grant.id, expected_db_version=7)


def test_lookup_failure_keeps_grant_and_restart_retries(entities, caplog):
    entities.get_by_id.side_effect = RuntimeError("entity service unavailable")
    progress = MagicMock()
    JobDelegationCleanup(entities).sweep(on_progress=progress)
    entities.delete_by_id.assert_not_called()
    progress.assert_called_once()
    assert "Could not clean up job workload delegation; will retry" in caplog.text

    entities.get_by_id.side_effect = [_attempt(HelixJobStatus.ERROR), _grant()]
    JobDelegationCleanup(entities).sweep(on_progress=progress)
    entities.delete_by_id.assert_called_once()


def test_version_conflict_leaves_grant_for_later_sweep(entities):
    grant = _grant()
    entities.get_by_id.side_effect = [
        _attempt(HelixJobStatus.ERROR),
        grant,
        grant,
        grant,
        _attempt(HelixJobStatus.ERROR),
        grant,
    ]
    entities.delete_by_id.side_effect = [EntityConflictError("renewed")] * 3 + [None]
    cleanup = JobDelegationCleanup(entities)

    cleanup.sweep(on_progress=lambda: None)
    cleanup.sweep(on_progress=lambda: None)

    assert entities.delete_by_id.call_count == 4


def test_sweep_advances_past_full_page_of_live_grants(entities):
    grant = _grant()
    entities.list.side_effect = [SimpleNamespace(data=[grant] * 100), SimpleNamespace(data=[grant])]
    entities.get_by_id.side_effect = [_attempt(HelixJobStatus.ACTIVE)] * 100 + [
        _attempt(HelixJobStatus.COMPLETED),
        grant,
    ]
    cleanup = JobDelegationCleanup(entities)

    cleanup.sweep(on_progress=lambda: None)
    cleanup.sweep(on_progress=lambda: None)

    assert [entry.kwargs["page"] for entry in entities.list.call_args_list] == [1, 2]
    entities.delete_by_id.assert_called_once()


def test_job_cleanup_collects_all_pages_before_deletion(entities):
    grants = [_grant(grant_id=f"grant-{index}") for index in range(101)]
    entities.list.side_effect = [SimpleNamespace(data=grants[:100]), SimpleNamespace(data=grants[100:])]
    entities.get_by_id.side_effect = grants

    progress = MagicMock()
    JobDelegationCleanup(entities).cleanup_job(_job(), on_progress=progress)

    assert progress.call_count == 103

    assert [entry.kwargs["page"] for entry in entities.list.call_args_list] == [1, 2]
    for entry in entities.list.call_args_list:
        assert entry.kwargs["workspace"] == "system"
        assert entry.kwargs["filter_obj"] == {"workload_workspace": "default", "job_id": "job-id"}
    assert [entry.args[1] for entry in entities.delete_by_id.call_args_list] == [grant.id for grant in grants]
    assert [entry[0] for entry in entities.mock_calls[:2]] == ["list", "list"]


def test_failed_delegation_cleanup_blocks_job_deletion(entities, backend_registry, mock_nemo_client):
    job = _job()
    entities.list.return_value = SimpleNamespace(data=[job])
    controller = JobReconciler(backend_registry, mock_nemo_client, system_entity_client=entities)
    controller._delegation_cleanup = MagicMock(spec=JobDelegationCleanup)
    controller._delegation_cleanup.cleanup_job.side_effect = RuntimeError("entity service unavailable")

    controller._cleanup_pending_jobs()

    entities.delete_by_id.assert_not_called()
