# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU e2e: unsloth LoRA SFT on SQuAD, with uplift eval."""

import logging

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_unsloth_plugin.schema import UnslothJobInput

from e2e.customizer import customizer_eval as ceval
from e2e.customizer import customizer_jobs as jobs
from e2e.customizer.customization_helpers import unique_name

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.platform("docker", "kubernetes"), pytest.mark.feature("gpu", "unsloth")]


def test_unsloth_lora_uplift(
    client: NemoClient,
    customizer_workspace: str,
    platform_base_url: str,
    unsloth_train_fileset: str,
    unsloth_val_fileset: str,
    squad_val_rows: list[dict],
    unsloth_base_entity: str,
    require_uplift: bool,
) -> None:
    ws = customizer_workspace
    base_entity = unsloth_base_entity
    output_name = unique_name("qwen25-unsloth-lora")

    spec = UnslothJobInput.model_validate(
        {
            "model": {"name": f"{ws}/{base_entity}", "max_seq_length": 1024, "load_in_4bit": False},
            "dataset": {
                "path": f"{ws}/{unsloth_train_fileset}",
                "validation_path": f"{ws}/{unsloth_val_fileset}",
                "apply_chat_template": True,
            },
            "training": {"finetuning_type": "lora"},
            "schedule": {"epochs": 1},
            "optimizer": {"learning_rate": 1e-4},
            "output": {"name": output_name, "save_method": "lora"},
        }
    )

    job_name, final = jobs.submit_and_wait_customization_job(client, "unsloth", spec, ws)
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, ws)

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

    result = ceval.UpliftResult(metric="f1", base_score=base_score, tuned_score=tuned_score, tuned_label="lora")
    logger.info("unsloth lora: base=%.4f tuned=%.4f uplift=%.4f", base_score, tuned_score, result.uplift)
    result.assert_ok(require_uplift=require_uplift)
