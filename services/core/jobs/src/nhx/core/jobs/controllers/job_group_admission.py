# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.jobs.types import HelixJobStepWithContext
from nhx.common.entities import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.controllers.backends.exceptions import AdmissionDeferred
from nhx.core.jobs.entities import (
    HelixJob,
    HelixJobAttempt,
)
from nhx.core.jobs.job_groups import HelixJobGroup, JobGroupTransitionError
from pydantic import ValidationError


class JobGroupAdmissionDeferred(AdmissionDeferred):
    """A Step did not pass the JobGroup admission check."""


class JobGroupAdmission:
    def __init__(self, store: SyncEntityClient) -> None:
        self._store = store

    def reserve(self, step: HelixJobStepWithContext, *, job_id: str, current_attempt_id: str | None) -> None:
        try:
            job = self._store.get_by_id(HelixJob, job_id)
            if job.parent is None:
                return
            if (
                job.status in HelixJobStatus.lifecycle_pending()
                or job.workspace != step.workspace
                or step.attempt_id != current_attempt_id
                or step.attempt_id != job.current_attempt_id
            ):
                raise JobGroupAdmissionDeferred("The Attempt is no longer current")
            attempt = self._store.find_one(
                HelixJobAttempt,
                workspace=job.workspace,
                filter_str=json.dumps({"id": step.attempt_id, "parent": job.id}),
            )
            if attempt.status.is_terminal():
                raise JobGroupAdmissionDeferred("The Attempt has settled")
            group = self._store.get_by_id(HelixJobGroup, job.parent)
            if group.workspace != step.workspace:
                raise JobGroupAdmissionDeferred("The group is in a different workspace")
            if group.controller_state.admit_attempt(job.id, attempt.id, group.submission.max_concurrent_jobs):
                self._store.update(group)
        except JobGroupTransitionError as exc:
            raise JobGroupAdmissionDeferred(str(exc)) from exc
        except (EntityConflictError, EntityNotFoundError, NemoClientError, ValidationError) as exc:
            raise JobGroupAdmissionDeferred("Could not confirm JobGroup admission; will retry") from exc
