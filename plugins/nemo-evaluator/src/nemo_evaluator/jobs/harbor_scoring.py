# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved scoring for one immutable stored Harbor task revision."""

from nemo_evaluator.harbor.tasks import HarborTaskScoring
from nemo_evaluator.jobs.metric_resolution import to_runtime_bundle, unresolved_model_refs
from nemo_evaluator.shared.metric_bundles.bundles import unbundle_metric
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric


def apply_harbor_scoring(scoring: HarborTaskScoring, task: AgentEvalTask, *, reward_key: str) -> AgentEvalTask:
    metrics = [unbundle_metric(to_runtime_bundle(metric)) for metric in scoring.metrics]
    unresolved = unresolved_model_refs(metrics)
    if unresolved:
        raise ValueError(f"Harbor metric models must be resolved before run: {', '.join(unresolved)}")
    # Reconstruct rather than model_copy: validate duplicate metric types and view signals.
    return AgentEvalTask(
        **task.model_dump(exclude={"metrics", "views"}),
        metrics=[HarborRewardMetric(output_name=reward_key), *metrics],
        views=scoring.views,
    )


def validate_harbor_scoring(scoring: HarborTaskScoring, *, reward_key: str) -> None:
    apply_harbor_scoring(
        scoring,
        AgentEvalTask(id=scoring.task_ref.root, intent="Validate Harbor scoring", inputs={}),
        reward_key=reward_key,
    )
