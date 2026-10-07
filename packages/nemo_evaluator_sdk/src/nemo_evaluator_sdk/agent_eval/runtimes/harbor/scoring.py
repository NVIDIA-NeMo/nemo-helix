# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor reward finalization for generated and imported trials, without native Harbor."""

from collections.abc import Sequence

from nemo_evaluator_sdk.agent_eval.reward_keys import (
    HARBOR_PRIMARY_REWARD_KEY,
    ParsedHarborRewards,
    validate_reward_key,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial
from nemo_evaluator_sdk.enums import MetricType
from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.metrics.utils import metric_type_name


def saved_harbor_reward_key(trials: Sequence[AgentEvalTrial]) -> str:
    """Read the single primary reward key recorded by saved Harbor trials.

    Args:
        trials: Saved trials whose metadata must identify the original reward key.

    Returns:
        The validated reward key shared by every trial.

    Raises:
        ValueError: Metadata is absent, invalid, or inconsistent across trials.
    """
    keys: set[str] = set()
    for trial in trials:
        key = trial.metadata.get(HARBOR_PRIMARY_REWARD_KEY)
        if not isinstance(key, str):
            raise ValueError(f"Harbor trial {trial.id!r} requires {HARBOR_PRIMARY_REWARD_KEY} metadata")
        validate_reward_key(key)
        keys.add(key)
    if len(keys) != 1:
        raise ValueError("Saved Harbor trials must contain one consistent primary reward_key")
    return next(iter(keys))


def harbor_scoring_metrics(task: AgentEvalTask, trials: Sequence[AgentEvalTrial], *, reward_key: str) -> list[Metric]:
    """Expand a task's Harbor reward metric for secondary rewards found in saved trials.

    Algorithm:
        - Collect optional outputs already declared by the Harbor reward metric.
        - Add successful and rejected reward keys observed in trial metadata.
        - Replace only the Harbor reward metric, preserving custom metrics and trial data.

    Args:
        task: Runtime task whose metric list will be reconstructed.
        trials: Saved trials used to discover additional reward names.
        reward_key: Required primary reward output name.

    Returns:
        A new metric list with discovered rewards declared as optional outputs.
    """
    from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric

    keys: set[str] = set()
    for metric in task.metrics:
        if metric_type_name(metric) == MetricType.HARBOR_REWARD:
            keys.update(output.name for output in metric.output_spec() if not output.required)
    for trial in trials:
        rewards = ParsedHarborRewards.from_metadata(trial.metadata)
        keys.update(rewards.values)
        keys.update(rewards.rejected_by_key)
    reward_keys = (reward_key, *sorted(keys - {reward_key}))
    return [
        HarborRewardMetric(output_name=reward_key, reward_keys=reward_keys)
        if metric_type_name(metric) == MetricType.HARBOR_REWARD
        else metric
        for metric in task.metrics
    ]
