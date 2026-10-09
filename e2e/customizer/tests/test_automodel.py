# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU e2e: automodel SFT on SQuAD — LoRA and full-weight, with uplift eval."""

import logging

import pytest
from nemo_automodel_plugin.schema import AutomodelJobInput
from nemo_helix_plugin.client.client import NemoClient

from e2e.customizer import customizer_eval as ceval
from e2e.customizer import customizer_jobs as jobs
from e2e.customizer.customization_helpers import unique_name

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.platform("docker", "kubernetes"), pytest.mark.feature("gpu", "automodel")]


@pytest.mark.parametrize("finetuning_type", ["lora", "all_weights"])
def test_automodel_sft_uplift(
    client: NemoClient,
    customizer_workspace: str,
    platform_base_url: str,
    squad_fileset: str,
    squad_val_rows: list[dict],
    automodel_base_entity: str,
    require_uplift: bool,
    finetuning_type: str,
) -> None:
    ws = customizer_workspace
    base_entity = automodel_base_entity
    output_name = unique_name(f"qwen3-{finetuning_type}")

    spec = AutomodelJobInput.model_validate(
        {
            "model": f"{ws}/{base_entity}",
            "dataset": {"training": f"{ws}/{squad_fileset}", "validation": f"{ws}/{squad_fileset}"},
            "training": {
                "training_type": "sft",
                "finetuning_type": finetuning_type,
                "max_seq_length": 1024,
            },
            "schedule": {"epochs": 1},
            "batch": {"global_batch_size": 8, "micro_batch_size": 1},
            "optimizer": {"learning_rate": 1e-4 if finetuning_type == "lora" else 5e-6},
            "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            "output": {"name": output_name},
        }
    )

    job_name, final = jobs.submit_and_wait_customization_job(client, "automodel", spec, ws)
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, ws)

    if finetuning_type == "lora":
        deployment, config = jobs.deploy_vllm_model(client, ws, base_entity, lora_enabled=True)
        try:
            base_score = ceval.score_rows(
                squad_val_rows,
                platform_base_url,
                ws,
                deployment,
                ceval.base_model_field(ws, base_entity),
            )
            tuned_score = ceval.score_rows(
                squad_val_rows,
                platform_base_url,
                ws,
                deployment,
                ceval.lora_model_field(ws, output_name),
            )
        finally:
            jobs.delete_deployment(client, ws, deployment, config)
    else:
        base_score = _deploy_and_score(client, ws, platform_base_url, base_entity, squad_val_rows)
        tuned_score = _deploy_and_score(client, ws, platform_base_url, output_name, squad_val_rows)

    result = ceval.UpliftResult(
        metric="f1",
        base_score=base_score,
        tuned_score=tuned_score,
        tuned_label=finetuning_type,
    )
    logger.info(
        "automodel %s: base=%.4f tuned=%.4f uplift=%.4f",
        finetuning_type,
        base_score,
        tuned_score,
        result.uplift,
    )
    result.assert_ok(require_uplift=require_uplift)


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
        )
    finally:
        jobs.delete_deployment(client, workspace, deployment, config)
