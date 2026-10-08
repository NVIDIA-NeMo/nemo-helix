# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract fixtures for submit-time AutomodelJobInput JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nemo_automodel_plugin.schema import AutomodelJobInput

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize(
    "fixture_name",
    ["integrations_wandb_mlflow.json"],
)
def test_contract_job_input_validates(fixture_name: str) -> None:
    path = FIXTURES_DIR / fixture_name
    spec = AutomodelJobInput.model_validate(json.loads(path.read_text()))
    assert spec.integrations is not None
    assert spec.integrations.wandb is not None
    assert spec.integrations.wandb.project == "my-project"
    assert spec.integrations.wandb.name == "run-001"
    assert spec.integrations.wandb.api_key_secret is not None
    assert spec.integrations.wandb.api_key_secret.root == "default/wandb-api-key"
    assert spec.integrations.mlflow is not None
    assert spec.integrations.mlflow.tracking_uri == "http://mlflow:5000"
    assert spec.integrations.mlflow.name == "run-001"


def test_nemotron_moe_lora_job_input_validates() -> None:
    """The Nemotron MoE LoRA layout the customizer skill recommends is accepted at submit."""
    path = FIXTURES_DIR / "nemotron_moe_sft_lora.json"
    spec = AutomodelJobInput.model_validate(json.loads(path.read_text()))
    assert spec.training.lora is not None
    assert spec.training.lora.exclude_modules == ["*.out_proj"]
    assert spec.parallelism.num_gpus_per_node == 4
    assert spec.parallelism.expert_parallel_size == 4
    assert spec.parallelism.tensor_parallel_size == 1
    assert spec.schedule.max_steps is None
