# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry delegation cleanup from persisted job and attempt ownership."""

import logging
from collections.abc import Callable

from nhx.common.auth import SyncWorkloadDelegationStore, WorkloadDelegationEntity
from nhx.common.entities import SYSTEM_WORKSPACE, EntityNotFoundError, SyncEntityClient
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt

logger = logging.getLogger(__name__)
_PAGE_SIZE = 100


class JobDelegationCleanup:
    def __init__(self, entities: SyncEntityClient) -> None:
        self._entities = entities
        self._delegations = SyncWorkloadDelegationStore(entities)
        self._page = 1

    def sweep(self, *, on_progress: Callable[[], None]) -> None:
        """Delete grants for finished attempts, retrying retained records on later passes."""
        try:
            grants = self._entities.list(
                WorkloadDelegationEntity,
                workspace=SYSTEM_WORKSPACE,
                sort="id",
                page=self._page,
                page_size=_PAGE_SIZE,
            ).data
            self._page = self._page + 1 if len(grants) == _PAGE_SIZE else 1
        except Exception:
            logger.exception("Could not list workload delegations for job cleanup")
            return
        for grant in grants:
            try:
                if not self._is_job_grant(grant) or not grant.attempt_id:
                    continue
                if self._attempt_finished(grant):
                    self._delegations.revoke_by_id(grant.id)
            except Exception:
                logger.exception(
                    "Could not clean up job workload delegation; will retry",
                    extra={"delegation_id": grant.id, "job": grant.job_id, "attempt": grant.attempt_id},
                )
            finally:
                on_progress()

    def cleanup_job(self, job: HelixJob, *, on_progress: Callable[[], None]) -> None:
        """Remove this job's grants before deleting its ownership records."""
        page = 1
        selected: list[WorkloadDelegationEntity] = []
        while True:
            grants = self._entities.list(
                WorkloadDelegationEntity,
                workspace=SYSTEM_WORKSPACE,
                filter_obj={"workload_workspace": job.workspace, "job_id": job.id},
                sort="id",
                page=page,
                page_size=_PAGE_SIZE,
            ).data
            on_progress()
            selected.extend(grant for grant in grants if self._is_job_grant(grant))
            if len(grants) < _PAGE_SIZE:
                break
            page += 1
        # Collect before deleting so offset pagination cannot skip a grant.
        for grant in selected:
            try:
                self._delegations.revoke_by_id(grant.id)
            finally:
                on_progress()

    @staticmethod
    def _is_job_grant(grant: WorkloadDelegationEntity) -> bool:
        return bool(grant.job_id) and grant.workload_kind in (None, "job")

    def _attempt_finished(self, grant: WorkloadDelegationEntity) -> bool:
        if not grant.attempt_id or not grant.job_id:
            return False
        try:
            attempt = self._entities.get_by_id(HelixJobAttempt, grant.attempt_id)
        except EntityNotFoundError:
            # A delayed registration can arrive after job deletion cascaded to its attempts.
            # IDs are immutable, so a new job with the same name cannot own this grant.
            try:
                self._entities.get_by_id(HelixJob, grant.job_id)
            except EntityNotFoundError:
                return True
            return False
        return (
            attempt.workspace == grant.workload_workspace
            and attempt.job == grant.job_id
            and attempt.status.is_terminal()
        )
