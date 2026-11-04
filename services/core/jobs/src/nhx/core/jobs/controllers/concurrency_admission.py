# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from typing import cast

from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import HelixJobStepWithContext, ListStepsQueryParams
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.controllers.admission import AdmissionDecision
from nhx.core.jobs.controllers.backends import extract_provider_profile

logger = logging.getLogger(__name__)

# A step in one of these statuses holds a concurrency slot. QUEUED/CREATED/RESUMING are waiting
# for admission and PAUSED holds no running work, so none of those count.
IN_FLIGHT_STATUSES = (
    HelixJobStatus.PENDING,
    HelixJobStatus.ACTIVE,
    HelixJobStatus.CANCELLING,
    HelixJobStatus.PAUSING,
)

ExecutorKey = tuple[str, str]


def principal_of(step: HelixJobStepWithContext) -> str | None:
    """The submitting user's identity, or ``None`` when the step carries no auth context."""
    return step.auth_context.principal_id if step.auth_context is not None else None


def round_robin_by_user(steps: Iterable[HelixJobStepWithContext]) -> list[HelixJobStepWithContext]:
    """Interleave steps one per user per round so no user's backlog starves the others.

    Order within a user is preserved (so FIFO ``created_at`` order holds per user), and users are
    ordered by where their first step appeared in the input. Steps without a principal share one bucket.
    """
    buckets: dict[str | None, deque[HelixJobStepWithContext]] = {}
    for step in steps:
        buckets.setdefault(principal_of(step), deque()).append(step)

    ordered: list[HelixJobStepWithContext] = []
    while buckets:
        for principal in list(buckets):
            bucket = buckets[principal]
            ordered.append(bucket.popleft())
            if not bucket:
                del buckets[principal]
    return ordered


class ConcurrencyAdmission:
    """Admits a step only if it fits under the executor, workspace and per-user limits.

    The check is point-in-time: ``begin_pass`` counts the work currently in flight once per scheduling
    pass, and ``check`` adds each step it admits to those counts so later steps in the same pass see it.
    A step that does not fit is deferred (never failed) and reconsidered on the next pass.
    """

    def __init__(
        self,
        jobs_client: JobsClient,
        *,
        executor_limits: Mapping[ExecutorKey, int],
        max_active_jobs_per_workspace: int,
        max_active_jobs_per_user: int,
    ) -> None:
        self._jobs = jobs_client
        self._executor_limits = dict(executor_limits)
        self._workspace_default = max_active_jobs_per_workspace
        self._user_limit = max_active_jobs_per_user
        self._by_executor: Counter[ExecutorKey] = Counter()
        self._by_workspace: Counter[str] = Counter()
        self._by_user: Counter[str] = Counter()
        self._snapshot_ok = False

    def workspace_limit(self, workspace: str) -> int:
        """Limit for a workspace. A per-workspace value will take precedence here once it exists."""
        return self._workspace_default

    def executor_limit(self, executor: ExecutorKey) -> int | None:
        """Limit for an executor, or ``None`` when the executor is not a configured profile."""
        return self._executor_limits.get(executor)

    def begin_pass(self) -> None:
        """Snapshot current in-flight work. On failure, ``check`` defers until the next pass."""
        by_executor: Counter[ExecutorKey] = Counter()
        by_workspace: Counter[str] = Counter()
        by_user: Counter[str] = Counter()
        query = cast(
            ListStepsQueryParams,
            {"filter[status]": ",".join(s.value for s in IN_FLIGHT_STATUSES)},
        )
        try:
            for step in self._jobs.list_steps(name="-", workspace="-", query_params=query).items():
                if step.step_spec is not None:
                    by_executor[extract_provider_profile(step)] += 1
                by_workspace[step.workspace] += 1
                principal = principal_of(step)
                if principal is not None:
                    by_user[principal] += 1
        except NemoClientError:
            logger.exception("Could not count in-flight job steps; deferring admission this pass")
            self._snapshot_ok = False
            return
        self._by_executor, self._by_workspace, self._by_user = by_executor, by_workspace, by_user
        self._snapshot_ok = True

    def check(self, step: HelixJobStepWithContext) -> AdmissionDecision:
        if not self._snapshot_ok:
            return AdmissionDecision.defer("Waiting for capacity: could not determine current load")

        executor = extract_provider_profile(step)
        principal = principal_of(step)

        # A step must clear every limit; the first saturated one (executor, workspace, user) is reported.
        executor_limit = self.executor_limit(executor)
        if executor_limit is not None and self._by_executor[executor] >= executor_limit:
            return AdmissionDecision.defer(
                f"Waiting for capacity: executor {executor[0]}/{executor[1]} is at its limit "
                f"({self._by_executor[executor]}/{executor_limit} active jobs)"
            )
        workspace_limit = self.workspace_limit(step.workspace)
        if self._by_workspace[step.workspace] >= workspace_limit:
            return AdmissionDecision.defer(
                f"Waiting for capacity: workspace '{step.workspace}' is at its limit "
                f"({self._by_workspace[step.workspace]}/{workspace_limit} active jobs)"
            )
        if principal is not None and self._by_user[principal] >= self._user_limit:
            return AdmissionDecision.defer(
                f"Waiting for capacity: user is at their limit ({self._by_user[principal]}/{self._user_limit} active jobs)"
            )

        self._by_executor[executor] += 1
        self._by_workspace[step.workspace] += 1
        if principal is not None:
            self._by_user[principal] += 1
        return AdmissionDecision.admit()
