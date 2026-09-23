# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved scoring for one immutable stored Harbor task revision."""

from nemo_evaluator.api.task_definitions.harbor import ResolvedHarborTaskDefinition
from nemo_evaluator.jobs.metric_resolution import require_resolved_model_refs, to_runtime_metrics
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_archive import normalize_harbor_instruction
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric


def harbor_scoring_task(definition: ResolvedHarborTaskDefinition, *, reward_key: str) -> AgentEvalTask:
    """Build and validate scoring for a resolved Harbor definition.

    Args:
        definition: Snapshot supplying native identity, instruction, metrics, and views.
        reward_key: Primary Harbor reward output name.

    Returns:
        A runtime task with mandatory reward scoring followed by custom metrics.

    Raises:
        ValueError: Metric models are unresolved or task scoring is invalid.
    """
    instruction = normalize_harbor_instruction(definition.instruction, task_id=definition.native_task_id)
    metrics = to_runtime_metrics(definition.metrics)
    require_resolved_model_refs(metrics, subject="Harbor")
    return AgentEvalTask(
        id=definition.native_task_id,
        intent=definition.native_task_id,
        inputs={"instruction": instruction},
        metrics=[HarborRewardMetric(output_name=reward_key), *metrics],
        views=definition.views,
    )
