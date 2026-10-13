# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit-time validation for ``RlJobInput`` fields owned by the plugin."""

from __future__ import annotations

from typing import Any

import pytest
from nemo_rl_plugin.schema import DeploymentParams, RlJobInput


def _job(**overrides: Any) -> dict[str, Any]:
    return {
        "model": "default/base-model",
        "dataset": "default/prefs",
        "training": {"type": "dpo"},
        **overrides,
    }


def _grpo_lora_job(**overrides: Any) -> dict[str, Any]:
    return {
        "model": "default/base-model",
        "dataset": "default/gym",
        "environment": "default/my-env",
        "training": {"type": "grpo", "finetuning_type": "lora"},
        **overrides,
    }


def test_deployment_config_defaults_to_none() -> None:
    assert RlJobInput.model_validate(_job()).deployment_config is None


def test_deployment_config_accepts_a_string_ref() -> None:
    spec = RlJobInput.model_validate(_job(deployment_config="shared/my-config"))

    assert spec.deployment_config == "shared/my-config"


def test_deployment_config_accepts_inline_params() -> None:
    spec = RlJobInput.model_validate(_job(deployment_config={"gpu": 2}))

    assert isinstance(spec.deployment_config, DeploymentParams)
    assert spec.deployment_config.gpu == 2
    assert spec.deployment_config.lora_enabled is True


def test_grpo_lora_job_trains_an_adapter() -> None:
    assert RlJobInput.model_validate(_grpo_lora_job()).trains_lora_adapter is True


def test_dpo_job_does_not_train_an_adapter() -> None:
    assert RlJobInput.model_validate(_job()).trains_lora_adapter is False


def test_grpo_lora_rejects_lora_enabled_false() -> None:
    with pytest.raises(ValueError, match="lora_enabled must be true"):
        RlJobInput.model_validate(_grpo_lora_job(deployment_config={"lora_enabled": False}))


def test_full_weight_job_allows_lora_enabled_false() -> None:
    """A full-weight run produces its own model, so a non-LoRA deployment is fine."""
    spec = RlJobInput.model_validate(_job(deployment_config={"lora_enabled": False}))

    assert isinstance(spec.deployment_config, DeploymentParams)
    assert spec.deployment_config.lora_enabled is False


def test_a_string_ref_is_not_rejected_at_submit_time_for_lora() -> None:
    """String refs are resolved and checked by the compiler, not the schema."""
    spec = RlJobInput.model_validate(_grpo_lora_job(deployment_config="shared/base-cfg"))

    assert spec.deployment_config == "shared/base-cfg"


@pytest.mark.parametrize("gpu", [0, -1])
def test_deployment_config_rejects_non_positive_gpu(gpu: int) -> None:
    """Caught at submit, not at compile time where the task-side schema would reject it."""
    with pytest.raises(ValueError, match="greater than 0"):
        RlJobInput.model_validate(_job(deployment_config={"gpu": gpu}))
