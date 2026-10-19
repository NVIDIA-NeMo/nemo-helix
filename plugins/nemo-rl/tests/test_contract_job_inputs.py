# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract fixtures for submit-time RlJobInput JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nemo_rl_plugin.schema import OutputRequest, RlJobInput

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize(
    "fixture_name",
    ["minimal_dpo.json", "integrations_wandb_mlflow.json"],
)
def test_contract_job_input_validates(fixture_name: str) -> None:
    path = FIXTURES_DIR / fixture_name
    spec = RlJobInput.model_validate(json.loads(path.read_text()))
    assert spec.training.type == "dpo"

    if spec.integrations is None:
        return

    assert spec.integrations.wandb is not None
    assert spec.integrations.wandb.project == "my-project"
    assert spec.integrations.wandb.name == "run-001"
    assert spec.integrations.wandb.api_key_secret is not None
    assert spec.integrations.wandb.api_key_secret.root == "default/wandb-api-key"
    assert spec.integrations.mlflow is not None
    assert spec.integrations.mlflow.tracking_uri == "http://mlflow:5000"
    assert spec.integrations.mlflow.name == "run-001"


def test_minimal_grpo_fixture_validates() -> None:
    """Full-weight GRPO, taken from a configuration that has been run on 8 GPUs."""
    path = FIXTURES_DIR / "minimal_grpo.json"
    spec = RlJobInput.model_validate(json.loads(path.read_text()))
    assert spec.training.type == "grpo"
    assert spec.environment == "default/math-with-judge-env"
    assert spec.training.finetuning_type == "all_weights"
    assert spec.training.lora is None
    assert spec.training.num_generations_per_prompt == 16
    assert spec.training.parallelism.num_gpus_per_node == 8


def test_minimal_grpo_lora_fixture_validates() -> None:
    """GRPO + LoRA, same 8-GPU configuration with an adapter instead of full weights."""
    path = FIXTURES_DIR / "minimal_grpo_lora.json"
    spec = RlJobInput.model_validate(json.loads(path.read_text()))
    assert spec.training.type == "grpo"
    assert spec.environment == "default/math-with-judge-env"
    assert spec.training.finetuning_type == "lora"
    assert spec.training.lora is not None
    assert spec.training.lora.rank == 128
    assert spec.training.lora.alpha == 256


def test_minimal_moe_grpo_lora_fixture_validates() -> None:
    """MoE GRPO + LoRA on 4 GPUs, exercising expert parallelism and reward shaping.

    ``RlSchema`` sets ``extra="forbid"``, so every key here is one the schema declares;
    a per-architecture setting it does not know is rejected at submit regardless of what
    the compiler would do with it.
    """
    path = FIXTURES_DIR / "minimal_moe_grpo_lora.json"
    spec = RlJobInput.model_validate(json.loads(path.read_text()))

    assert spec.training.type == "grpo"
    assert spec.training.parallelism.expert_parallel_size == 4
    assert spec.training.lora is not None
    assert spec.training.lora.exclude_modules == ["*out_proj*"]
    # Triton LoRA kernels are declined explicitly here; only an explicit true above
    # tensor_parallel_size 1 is rejected.
    assert spec.training.lora.use_triton is False
    assert spec.training.vllm_tensor_parallel_size == 4
    assert spec.training.overlong_filtering is True
    assert spec.training.reward_shaping is not None
    assert spec.training.reward_scaling is not None


@pytest.mark.parametrize(
    "fixture_name",
    ["minimal_grpo.json", "minimal_grpo_lora.json", "minimal_moe_grpo_lora.json"],
)
def test_grpo_fixtures_satisfy_training_divisibility(fixture_name: str) -> None:
    """Both rules the compiler enforces at submit, checked on every shipped GRPO fixture.

    A fixture that violates one is rejected before any GPU is claimed, which would make it
    useless as an acceptance input.
    """
    spec = RlJobInput.model_validate(json.loads((FIXTURES_DIR / fixture_name).read_text()))
    t, p = spec.training, spec.training.parallelism
    model_parallel = (
        p.tensor_parallel_size * p.pipeline_parallel_size * p.context_parallel_size * p.expert_parallel_size
    )
    total_gpus = p.num_nodes * p.num_gpus_per_node
    assert total_gpus % model_parallel == 0
    data_parallel = total_gpus // model_parallel
    assert t.batch_size % (t.micro_batch_size * data_parallel) == 0

    prompts = t.num_prompts_per_step or max(t.batch_size // max(t.num_generations_per_prompt, 1), 1)
    assert (prompts * t.num_generations_per_prompt) % t.batch_size == 0

    if t.vllm_tensor_parallel_size is not None:
        assert t.vllm_tensor_parallel_size <= p.num_gpus_per_node
        assert total_gpus % t.vllm_tensor_parallel_size == 0


def test_output_name_cannot_exceed_response_limit() -> None:
    assert len(OutputRequest(name="x" * 255).name or "") == 255
    with pytest.raises(ValueError):
        OutputRequest(name="x" * 256)
