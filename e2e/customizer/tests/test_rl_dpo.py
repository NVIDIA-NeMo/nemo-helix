# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU e2e: nemo-rl DPO on HelpSteer3, scored on held-out preference accuracy.

DPO trains the policy to rank the chosen response above the rejected one relative to the
reference policy. The job validates on the held-out pairs before and after training and records
``val_accuracy`` (the share of pairs whose chosen response gets the higher implicit reward) and
the mean implicit rewards of the chosen and rejected responses. The test requires the tuned
policy to rank the held-out pairs better than chance and to give chosen responses the higher
mean reward.
"""

import json
import logging
from pathlib import Path

import pytest
from nemo_helix_plugin.client.client import NemoClient

from e2e.customizer import customizer_jobs as jobs
from e2e.customizer.customization_helpers import unique_name

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.platform("kubernetes"), pytest.mark.feature("gpu", "rl", "uplift")]

RlJobInput = pytest.importorskip("nemo_rl_plugin.schema").RlJobInput


def _tie_fraction(validation_path: Path) -> float:
    """Share of HelpSteer3 pairs with ``overall_preference == 0``.

    NeMo-RL trains a tie as chosen == rejected, so its two rewards are always equal and the
    strict ``chosen > rejected`` accuracy can never count it. Accuracy is judged on the rest.
    """
    preferences = [json.loads(line)["overall_preference"] for line in validation_path.read_text().splitlines() if line]
    assert preferences, f"no validation pairs in {validation_path}"
    return sum(1 for preference in preferences if preference == 0) / len(preferences)


def test_rl_dpo_uplift(
    client: NemoClient,
    customizer_workspace: str,
    helpsteer_dpo_fileset: str,
    dpo_cache: Path,
    rl_base_entity: str,
) -> None:
    jobs.require_kubernetes_backend(client)

    ws = customizer_workspace
    output_name = unique_name("qwen3-dpo")

    spec = RlJobInput.model_validate(
        {
            "model": f"{ws}/{rl_base_entity}",
            "dataset": f"{ws}/{helpsteer_dpo_fileset}",
            "training": {
                "type": "dpo",
                "epochs": 1,
                "batch_size": 32,
                "micro_batch_size": 1,
                "max_seq_length": 1024,
                "ref_policy_kl_penalty": 0.05,
                "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            },
            "output": {"name": output_name},
        }
    )

    job_name, final = jobs.submit_and_wait_customization_job(client, "rl", spec, ws)
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, ws)

    metrics = jobs.training_metrics(client, ws, job_name)
    accuracy = jobs.require_metric(metrics, "val_accuracy", job_name)
    chosen = jobs.require_metric(metrics, "val_rewards_chosen_mean", job_name)
    rejected = jobs.require_metric(metrics, "val_rewards_rejected_mean", job_name)
    assert len(accuracy) >= 2, f"expected validation before and after training, got val_accuracy={accuracy}"

    non_tie = 1 - _tie_fraction(dpo_cache / "validation.jsonl")
    pair_accuracy = accuracy[-1] / non_tie
    margin = chosen[-1] - rejected[-1]
    logger.info(
        "rl dpo: val_accuracy %.4f -> %.4f (%.4f on the %.0f%% of pairs with a preference); "
        "reward margin chosen - rejected %.4f -> %.4f",
        accuracy[0],
        accuracy[-1],
        pair_accuracy,
        non_tie * 100,
        chosen[0] - rejected[0],
        margin,
    )
    assert margin > 0, f"held-out chosen reward did not exceed rejected: chosen={chosen[-1]} rejected={rejected[-1]}"
    assert pair_accuracy > 0.5, (
        f"held-out preference accuracy {pair_accuracy:.4f} on pairs with a preference is not above chance "
        f"(val_accuracy={accuracy}, non-tie share={non_tie:.4f})"
    )
