# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU e2e: nemo-rl DPO on HelpSteer3, with a deterministic uplift proxy."""

import logging

import pytest
from nemo_helix_plugin.client.client import NemoClient

from e2e.customizer import customizer_eval as ceval
from e2e.customizer import customizer_jobs as jobs
from e2e.customizer.customization_helpers import unique_name

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.platform("kubernetes"), pytest.mark.feature("gpu", "rl", "uplift")]

RlJobInput = pytest.importorskip("nemo_rl_plugin.schema").RlJobInput


def _deploy_and_score(
    client: NemoClient,
    workspace: str,
    base_url: str,
    entity: str,
    rows: list[dict],
) -> float:
    deployment, config = jobs.deploy_vllm_model(client, workspace, entity, lora_enabled=False)
    try:
        return ceval.score_rows(
            rows,
            base_url,
            workspace,
            deployment,
            ceval.base_model_field(workspace, entity),
            metric="f1",
        )
    finally:
        jobs.delete_deployment(client, workspace, deployment, config)


def test_rl_dpo_uplift(
    client: NemoClient,
    customizer_workspace: str,
    platform_base_url: str,
    helpsteer_dpo_fileset: str,
    dpo_eval_rows: list[dict],
    rl_base_entity: str,
    require_uplift: bool,
) -> None:
    jobs.require_kubernetes_backend(client)

    ws = customizer_workspace
    base_entity = rl_base_entity
    output_name = unique_name("qwen3-dpo")

    spec = RlJobInput.model_validate(
        {
            "model": f"{ws}/{base_entity}",
            "dataset": f"{ws}/{helpsteer_dpo_fileset}",
            "training": {
                "type": "dpo",
                "epochs": 1,
                "batch_size": 32,
                "micro_batch_size": 1,
                "learning_rate": 5e-6,
                "max_seq_length": 1024,
                "ref_policy_kl_penalty": 0.05,
                "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            },
            "output": {"name": output_name},
        }
    )

    job_name, final = jobs.submit_and_wait_customization_job(client, "rl", spec, ws)
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, ws)

    assert dpo_eval_rows, "no non-tie preference rows available for DPO eval"

    base_score = _deploy_and_score(client, ws, platform_base_url, base_entity, dpo_eval_rows)
    tuned_score = _deploy_and_score(client, ws, platform_base_url, output_name, dpo_eval_rows)

    result = ceval.UpliftResult(metric="f1", base_score=base_score, tuned_score=tuned_score, tuned_label="dpo")
    logger.info("rl dpo: base=%.4f tuned=%.4f uplift=%.4f", base_score, tuned_score, result.uplift)
    result.assert_ok(require_uplift=require_uplift)
