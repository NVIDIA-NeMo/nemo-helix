# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved, network-free inputs for the run-record contract."""

from datetime import datetime, timedelta, timezone

import pytest
from nemo_evals.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evals.jobs.agent_spec import AgentEvalSpec, FabricConfigSource, FabricRunnerTarget, ResolvedTask
from nemo_evals.jobs.metric_resolution import to_inline
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.hybrid import HybridMetricBundlePackager
from nemo_evals.taskset_runs.identifiers import derive_run_id
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_evals.taskset_runs.store import RunAcceptance, TasksetRunStore
from nemo_helix_plugin.files.client import AsyncFilesClient
from nhx.testing.files import InMemoryFilesService
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric


@pytest.fixture
def files_service() -> InMemoryFilesService:
    """Return fresh isolated Files storage for each test's resource assertions."""
    return InMemoryFilesService()


@pytest.fixture
def files_client(files_service: InMemoryFilesService) -> AsyncFilesClient:
    """Return a real Files client backed by the test's in-memory Files service."""
    return files_service.client()


@pytest.fixture
def plan(resolved_spec: AgentEvalSpec) -> TasksetRunPlan:
    """Bind the resolved-spec fixture to a fixed keyed run and pinned taskset revision.

    The returned plan has explicit unlimited caps and a fail-on-missing-image
    policy, providing deterministic input for serialization and acceptance tests.
    """
    return TasksetRunPlan(
        run_id=derive_run_id("default", "user:alice", "key"),
        workspace="default",
        taskset_ref="default/suite#3",
        spec=resolved_spec,
        missing_prebuilt_image="fail",
        max_concurrent_task_jobs=None,
        max_failed_tasks=None,
    )


@pytest.fixture
def acceptance(plan: TasksetRunPlan) -> RunAcceptance:
    """Return resolved acceptance inputs for the plan fixture with fixed UTC deadlines.

    Execution and finalization end one and two hours after acceptance, allowing
    retries and races to prove that the winning request's times are preserved.
    """
    accepted_at = datetime(2026, 10, 5, tzinfo=timezone.utc)
    return RunAcceptance(
        plan=plan,
        idempotency_key="key",
        request_digest="request-digest",
        accepted_at=accepted_at,
        execution_deadline_at=accepted_at + timedelta(hours=1),
        finalization_deadline_at=accepted_at + timedelta(hours=2),
    )


@pytest.fixture
def store(entity_store, files_client: AsyncFilesClient) -> TasksetRunStore:
    """Return a store over the entity and Files fixtures without production retries."""
    return TasksetRunStore._with_entity_adapter(entity_store, files_client)


@pytest.fixture
def resolved_spec() -> AgentEvalSpec:
    """Build a fresh fully resolved evaluation spec without external services.

    No fixture inputs are needed. The returned spec contains one evaluator task,
    an inline exact-match metric bundle, and a Fabric target so plan validation
    exercises real nested models without metric or target resolution requests.
    """
    return AgentEvalSpec(
        tasks=[
            ResolvedTask(
                id="task-1",
                spec=ResolvedEvaluatorTaskDefinition(
                    kind="evaluator",
                    intent="Answer exactly.",
                    metrics=[
                        to_inline(
                            bundle_metric(
                                ExactMatchMetric(reference="yes", candidate="yes"), HybridMetricBundlePackager()
                            )
                        )
                    ],
                ),
            )
        ],
        target=FabricRunnerTarget(source=FabricConfigSource(config={}, model="provider/model")),
    )
