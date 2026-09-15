# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_automodel_plugin.schema import AutomodelJobInput, DeploymentParams


def test_reject_output_model() -> None:
    with pytest.raises(ValueError, match="output_model"):
        AutomodelJobInput.model_validate(
            {
                "model": "llama",
                "dataset": {"training": "default/train"},
                "training": {"training_type": "sft"},
                "output_model": "old-field",
            },
        )


def test_distillation_requires_teacher() -> None:
    with pytest.raises(ValueError, match="teacher_model"):
        AutomodelJobInput.model_validate(
            {
                "model": "llama",
                "dataset": {"training": "default/train"},
                "training": {"training_type": "distillation"},
            },
        )


def test_training_recipe_defaults_to_auto() -> None:
    spec = AutomodelJobInput.model_validate(
        {
            "model": "llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "all_weights"},
        }
    )

    assert spec.training.recipe == "auto"
    assert spec.optimizer.optimizer == "auto"


@pytest.mark.parametrize("optimizer", ["auto", "Adam", "AdamW", "FusedAdam"])
def test_optimizer_is_selectable(optimizer: str) -> None:
    spec = AutomodelJobInput.model_validate(
        {
            "model": "llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "all_weights"},
            "optimizer": {"optimizer": optimizer},
        }
    )

    assert spec.optimizer.optimizer == optimizer


def test_distillation_rejects_encoder_recipe() -> None:
    with pytest.raises(ValueError, match="only supports the sft recipe"):
        AutomodelJobInput.model_validate(
            {
                "model": "llama",
                "dataset": {"training": "default/train"},
                "training": {
                    "training_type": "distillation",
                    "teacher_model": "default/teacher",
                    "recipe": "cross_encoder",
                },
            }
        )


def test_distillation_auto_rejects_resolved_encoder_recipe() -> None:
    spec = AutomodelJobInput.model_validate(
        {
            "model": "embed",
            "dataset": {"training": "default/train"},
            "training": {
                "training_type": "distillation",
                "teacher_model": "default/teacher",
                "recipe": "auto",
            },
        }
    )
    with pytest.raises(ValueError, match="only supports the sft recipe"):
        spec.with_resolved_recipe("embedding")
    with pytest.raises(ValueError, match="only supports the sft recipe"):
        spec.with_resolved_recipe("cross_encoder")
    resolved = spec.with_resolved_recipe("causal_lm")
    assert resolved.training.recipe == "auto"


def test_encoder_recipes_apply_nemotron_job_defaults() -> None:
    embed = AutomodelJobInput.model_validate(
        {
            "model": "embed",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "recipe": "bi_encoder", "finetuning_type": "all_weights"},
        }
    ).with_resolved_recipe("embedding")
    rerank = AutomodelJobInput.model_validate(
        {
            "model": "rerank",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "recipe": "cross_encoder", "finetuning_type": "all_weights"},
        }
    ).with_resolved_recipe("cross_encoder")

    assert embed.batch.global_batch_size == 128
    assert embed.batch.micro_batch_size == 4
    assert embed.optimizer.learning_rate == 1e-5
    assert embed.optimizer.warmup_steps == 5
    assert rerank.optimizer.learning_rate == 3e-6
    assert rerank.optimizer.warmup_steps == 100


def test_encoder_recipe_defaults_do_not_override_explicit_hparams() -> None:
    spec = AutomodelJobInput.model_validate(
        {
            "model": "embed",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "recipe": "bi_encoder", "finetuning_type": "all_weights"},
            "batch": {"global_batch_size": 16, "micro_batch_size": 2},
            "optimizer": {"learning_rate": 2e-5, "warmup_steps": 1},
        }
    ).with_resolved_recipe("embedding")
    assert spec.batch.global_batch_size == 16
    assert spec.batch.micro_batch_size == 2
    assert spec.optimizer.learning_rate == 2e-5
    assert spec.optimizer.warmup_steps == 1


def test_auto_recipe_does_not_apply_retrieval_defaults_until_resolved() -> None:
    spec = AutomodelJobInput.model_validate(
        {
            "model": "embed",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "recipe": "auto", "finetuning_type": "all_weights"},
        }
    )
    assert spec.batch.global_batch_size == 8
    assert spec.optimizer.learning_rate == 5e-6

    resolved = spec.with_resolved_recipe("embedding")
    assert spec.training.recipe == "auto"
    assert spec.batch.global_batch_size == 8
    assert resolved.training.recipe == "bi_encoder"
    assert resolved.batch.global_batch_size == 128
    assert resolved.optimizer.learning_rate == 1e-5
    assert resolved.optimizer.warmup_steps == 5


def test_deployment_config_defaults_to_none() -> None:
    spec = AutomodelJobInput.model_validate(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "lora"},
        }
    )
    assert spec.deployment_config is None


def test_deployment_config_accepts_string_ref_and_inline_params() -> None:
    base = {
        "model": "meta/llama",
        "dataset": {"training": "default/train"},
        "training": {"training_type": "sft", "finetuning_type": "lora"},
    }
    by_ref = AutomodelJobInput.model_validate({**base, "deployment_config": "shared/my-config"})
    assert by_ref.deployment_config == "shared/my-config"

    inline = AutomodelJobInput.model_validate({**base, "deployment_config": {"gpu": 2}})
    assert isinstance(inline.deployment_config, DeploymentParams)
    assert inline.deployment_config.gpu == 2
    assert inline.deployment_config.lora_enabled is True


def test_lora_adapter_rejects_lora_enabled_false() -> None:
    with pytest.raises(ValueError, match="lora_enabled must be true"):
        AutomodelJobInput.model_validate(
            {
                "model": "meta/llama",
                "dataset": {"training": "default/train"},
                "training": {"training_type": "sft", "finetuning_type": "lora"},
                "deployment_config": {"lora_enabled": False},
            }
        )


def test_merged_lora_allows_lora_enabled_false() -> None:
    """merge=True produces a full-weight model, so a non-LoRA deployment is fine."""
    spec = AutomodelJobInput.model_validate(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "lora", "lora": {"merge": True}},
            "deployment_config": {"lora_enabled": False},
        }
    )
    assert spec.trains_standalone_lora_adapter() is False


def test_all_weights_allows_lora_enabled_false() -> None:
    spec = AutomodelJobInput.model_validate(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "all_weights"},
            "deployment_config": {"lora_enabled": False},
        }
    )
    assert spec.trains_standalone_lora_adapter() is False


@pytest.mark.parametrize("gpu", [0, -1])
def test_deployment_config_rejects_non_positive_gpu(gpu: int) -> None:
    """Caught at submit, not at compile time where the task-side schema would reject it."""
    with pytest.raises(ValueError, match="greater than 0"):
        AutomodelJobInput.model_validate(
            {
                "model": "meta/llama",
                "dataset": {"training": "default/train"},
                "training": {"training_type": "sft", "finetuning_type": "lora"},
                "deployment_config": {"gpu": gpu},
            }
        )


def test_deployment_config_accepts_a_positive_gpu_count() -> None:
    spec = AutomodelJobInput.model_validate(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "lora"},
            "deployment_config": {"gpu": 1},
        }
    )

    assert isinstance(spec.deployment_config, DeploymentParams)
    assert spec.deployment_config.gpu == 1
