# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from nemo_automodel_plugin.cli.inputs import load_job_json
from nemo_automodel_plugin.schema import AutomodelJobInput, ExportSpec


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


def test_export_precision_rejects_unverified_bf16() -> None:
    with pytest.raises(ValueError, match="precision"):
        ExportSpec.model_validate({"precision": "bf16"})


def test_legacy_embedding_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="embedding"):
        AutomodelJobInput.model_validate(
            {
                "model": "llama",
                "dataset": {"training": "default/train"},
                "training": {
                    "training_type": "sft",
                    "finetuning_type": "all_weights",
                    "embedding": {"train_n_passages": 7},
                },
            }
        )


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

    assert embed.batch.global_batch_size == 256
    assert embed.batch.micro_batch_size == 8
    assert embed.optimizer.learning_rate == 1e-5
    assert embed.optimizer.warmup_steps == 5
    assert rerank.batch.global_batch_size == 128
    assert rerank.batch.micro_batch_size == 8
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
    assert resolved.batch.global_batch_size == 256
    assert resolved.optimizer.learning_rate == 1e-5
    assert resolved.optimizer.warmup_steps == 5


def test_cli_serialization_preserves_unset_fields_for_recipe_defaults(tmp_path) -> None:
    """A full dump reports every field as set, which suppressed the encoder defaults."""
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "model": "embed",
                "dataset": {"training": "default/train"},
                "training": {"training_type": "sft", "recipe": "bi_encoder", "finetuning_type": "all_weights"},
            }
        )
    )

    resolved = AutomodelJobInput.model_validate(json.loads(load_job_json(job))).with_resolved_recipe("embedding")

    assert resolved.batch.global_batch_size == 256
    assert resolved.batch.micro_batch_size == 8
    assert resolved.optimizer.learning_rate == 1e-5
    assert resolved.optimizer.warmup_steps == 5


def test_cli_serialization_still_carries_explicit_fields(tmp_path) -> None:
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "model": "embed",
                "dataset": {"training": "default/train"},
                "training": {"training_type": "sft", "recipe": "bi_encoder", "finetuning_type": "all_weights"},
                "batch": {"micro_batch_size": 2},
            }
        )
    )

    resolved = AutomodelJobInput.model_validate(json.loads(load_job_json(job))).with_resolved_recipe("embedding")

    assert resolved.batch.micro_batch_size == 2
    assert resolved.batch.global_batch_size == 256
