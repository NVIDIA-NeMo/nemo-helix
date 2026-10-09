# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU e2e: automodel LoRA SFT of Nemotron 3.5 Lightning 30B-A3B on SQuAD, with uplift eval.

Weights are pulled from Hugging Face at runtime rather than staged from S3. The base
model and the LoRA adapter are served from one LoRA-enabled vLLM deployment.
Needs 4x 80GB H100s.
"""

import logging

import pytest
from nemo_automodel_plugin.schema import AutomodelJobInput
from nemo_helix_plugin.client.client import NemoClient

from e2e.customizer import customizer_eval as ceval
from e2e.customizer import customizer_jobs as jobs
from e2e.customizer.customization_helpers import assert_output_registered, unique_name

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.platform("docker", "kubernetes"),
    pytest.mark.feature("gpu", "automodel", "h100", "uplift"),
    pytest.mark.skip(reason="Disabled until an H100 runner (4x 80GB) is available"),
]

NUM_TRAINING_GPUS = 4
NUM_SERVING_GPUS = 4


@pytest.mark.timeout(5 * 3600)
def test_automodel_nemotron_lightning_lora_uplift(
    client: NemoClient,
    customizer_workspace: str,
    platform_base_url: str,
    squad_fileset: str,
    squad_val_rows: list[dict],
    nemotron_lightning_base_entity: str,
    require_uplift: bool,
) -> None:
    ws = customizer_workspace
    base_entity = nemotron_lightning_base_entity
    output_name = unique_name("nemotron-lightning-lora")

    spec = AutomodelJobInput.model_validate(
        {
            "model": f"{ws}/{base_entity}",
            "dataset": {"training": f"{ws}/{squad_fileset}", "validation": f"{ws}/{squad_fileset}"},
            "training": {
                "training_type": "sft",
                "finetuning_type": "lora",
                "lora": {"rank": 8, "alpha": 32, "dropout": 0.0, "use_triton": True},
                "max_seq_length": 2048,
                "precision": "bf16",
                "attn_implementation": "sdpa",
            },
            "schedule": {"epochs": 1, "seed": 1111},
            "batch": {"global_batch_size": 8, "micro_batch_size": 1, "sequence_packing": False},
            "optimizer": {
                "optimizer": "AdamW",
                "learning_rate": 1e-4,
                "min_learning_rate": 1e-5,
                "weight_decay": 0.1,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_eps": 1e-8,
                "lr_decay_style": "cosine",
                "warmup_steps": 100,
            },
            # MoE training distributes experts with expert parallelism; tensor parallel must stay 1.
            "parallelism": {
                "num_nodes": 1,
                "num_gpus_per_node": NUM_TRAINING_GPUS,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "context_parallel_size": 1,
                "expert_parallel_size": NUM_TRAINING_GPUS,
                "sequence_parallel": False,
            },
            "output": {"name": output_name},
        }
    )

    # First run downloads ~60GB of weights from Hugging Face inside the training job.
    job_name, final = jobs.submit_and_wait_customization_job(
        client, "automodel", spec, ws, timeout=3 * 3600, image_pull_timeout=3600
    )
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, ws)
    assert_output_registered(client, ws, base_model_name=base_entity, output_name=output_name)

    deployment, config = jobs.deploy_vllm_model(
        client,
        ws,
        base_entity,
        lora_enabled=True,
        gpu=NUM_SERVING_GPUS,
        disk_size="200Gi",
        # The model spec for this hybrid Mamba MoE may not carry the fields the
        # service uses to size tensor parallelism, so set it explicitly.
        additional_args=["--tensor-parallel-size", str(NUM_SERVING_GPUS)],
    )
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
    logger.info("automodel nemotron lora: base=%.4f tuned=%.4f uplift=%.4f", base_score, tuned_score, result.uplift)
    result.assert_ok(require_uplift=require_uplift)
