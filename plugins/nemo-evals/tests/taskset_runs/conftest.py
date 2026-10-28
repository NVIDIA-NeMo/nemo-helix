# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved, network-free inputs for the run-record contract."""

from datetime import datetime, timedelta, timezone

import pytest
from nemo_evals.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evals.config import HarborTaskJobsConfig
from nemo_evals.harbor.remote.images import PreparedHarborEnvironment, PreparedImage
from nemo_evals.harbor.remote.profile import ResolvedHarborSandboxPolicy
from nemo_evals.jobs.agent_spec import AgentEvalSpec, FabricConfigSource, FabricRunnerTarget, ResolvedTask
from nemo_evals.jobs.metric_resolution import to_inline
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.hybrid import HybridMetricBundlePackager
from nemo_evals.taskset_runs.identifiers import derive_run_id
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_evals.taskset_runs.store import RunAcceptance, TasksetRunStore
from nemo_helix_plugin.files.client import AsyncFilesClient
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric


@pytest.fixture
def plan(
    resolved_spec: AgentEvalSpec, sandbox_policy: ResolvedHarborSandboxPolicy, harbor_config: HarborTaskJobsConfig
) -> TasksetRunPlan:
    """Bind the resolved-spec fixture to a fixed keyed run and pinned taskset revision.

    The returned plan has explicit unlimited caps and a fail-on-missing-image
    policy, providing deterministic input for serialization and acceptance tests.
    """
    assert harbor_config.dispatcher_image is not None
    return TasksetRunPlan(
        run_id=derive_run_id("default", "user:alice", "key"),
        workspace="default",
        taskset_ref="default/suite#3",
        spec=resolved_spec,
        max_concurrent_task_jobs=None,
        sandbox_policy=sandbox_policy,
        dispatcher_image=harbor_config.dispatcher_image,
        dispatcher_profile="local-k8s",
        task_environments=[
            PreparedHarborEnvironment(
                task_revision_digest="r" * 64,
                archive_digest="a" * 64,
                agent=PreparedImage(image_ref=f"registry.invalid/task@sha256:{'b' * 64}", preparation_key="b" * 64),
            )
            for _ in resolved_spec.tasks
        ],
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
        timeout_seconds=3600,
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
