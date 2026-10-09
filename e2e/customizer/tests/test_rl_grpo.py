# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU e2e: NeMo RL GRPO on Qwen3-0.6B with the Gym ``math_with_judge`` environment (``wheels-v1``).

GRPO has no separate scoring pass. The job validates on the same holdout prompts before
training (``val_at_start``) and after (``val_at_end``) and records the mean ``math-verify``
reward as ``val_accuracy``; it also records ``train_reward`` every step. The test requires both
to increase.
"""

import logging
from statistics import mean

import pytest
from nemo_helix_plugin.client.client import NemoClient

from e2e.customizer import customizer_jobs as jobs
from e2e.customizer.customization_helpers import unique_name

logger = logging.getLogger(__name__)

TRAINING_TIMEOUT = 3 * 3600

pytestmark = [
    pytest.mark.platform("kubernetes"),
    pytest.mark.feature("gpu", "rl", "grpo", "uplift"),
    pytest.mark.timeout(TRAINING_TIMEOUT + 1800),
]

RlJobInput = pytest.importorskip("nemo_rl_plugin.schema").RlJobInput

MAX_STEPS = 50


def test_rl_grpo_reward_uplift(
    client: NemoClient,
    customizer_workspace: str,
    grpo_math_uplift_fileset: str,
    grpo_math_env_fileset: str,
    rl_base_entity: str,
) -> None:
    jobs.require_kubernetes_backend(client)
    ws = customizer_workspace
    output_name = unique_name("qwen3-grpo")

    # A starting point, not a validated recipe: NeMo-RL's grpo_math_1B learning rate and
    # micro batch, with 8 generations per prompt instead of 16 to halve rollout cost.
    spec = RlJobInput.model_validate(
        {
            "model": f"{ws}/{rl_base_entity}",
            "dataset": f"{ws}/{grpo_math_uplift_fileset}",
            "environment": f"{ws}/{grpo_math_env_fileset}",
            "training": {
                "type": "grpo",
                "finetuning_type": "all_weights",
                "max_steps": MAX_STEPS,
                "num_prompts_per_step": 32,
                "num_generations_per_prompt": 8,
                "batch_size": 256,
                "micro_batch_size": 4,
                "learning_rate": 5e-6,
                "max_seq_length": 2048,
                "max_new_tokens": 1024,
                "val_at_start": True,
                "val_at_end": True,
                "seed": 42,
                "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            },
            "output": {"name": output_name},
        }
    )

    job_name, final = jobs.submit_and_wait_customization_job(client, "rl", spec, ws, timeout=TRAINING_TIMEOUT)
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, ws)

    val = jobs.training_metric_values(client, ws, job_name, "val_accuracy")
    train = jobs.training_metric_values(client, ws, job_name, "train_reward")
    assert len(val) >= 2, f"expected validation before and after training, got val_accuracy={val}"
    quarter = max(1, len(train) // 4)
    early_train, late_train = mean(train[:quarter]), mean(train[-quarter:])
    logger.info(
        "rl grpo: val_accuracy %.4f -> %.4f; train_reward first-quarter mean %.4f -> last-quarter mean %.4f",
        val[0],
        val[-1],
        early_train,
        late_train,
    )
    assert val[-1] > val[0], f"val_accuracy did not increase: start={val[0]} end={val[-1]} (all: {val})"
    assert late_train > early_train, (
        f"train_reward did not increase: first-quarter mean={early_train} last-quarter mean={late_train} "
        f"over {len(train)} steps"
    )
