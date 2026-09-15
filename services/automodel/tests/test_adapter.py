# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_automodel_plugin.schema import AutomodelJobOutput
from nmp.automodel.adapter import automodel_spec_to_compiler_output
from nmp.automodel.api.v2.jobs.schemas import DistillationTraining, SFTTraining
from nmp.customization_common.integrations import collect_integration_secret_envs


def test_adapter_sft() -> None:
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "lora"},
            "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
        },
    )
    assert isinstance(spec.training, SFTTraining)
    assert spec.dataset == "default/train"


def test_adapter_plumbs_previously_dropped_fields() -> None:
    """CLI/SDK fields the adapter used to drop must now reach the v2 SFTTraining."""
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {
                "training_type": "sft",
                "recipe": "cross_encoder",
                "finetuning_type": "lora",
                "precision": "bf16",
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.1},
            },
            "optimizer": {
                "learning_rate": 2e-4,
                "min_learning_rate": 1e-5,
                "adam_beta1": 0.8,
                "adam_beta2": 0.95,
            },
            "parallelism": {"num_gpus_per_node": 2, "tensor_parallel_size": 2, "sequence_parallel": True},
            "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
        },
    )
    assert isinstance(spec.training, SFTTraining)
    assert spec.training.recipe == "cross_encoder"
    assert spec.training.precision is not None and spec.training.precision.value == "bf16"
    assert spec.training.min_learning_rate == 1e-5
    assert spec.training.adam_beta1 == 0.8
    assert spec.training.adam_beta2 == 0.95
    assert spec.training.parallelism.sequence_parallel is True
    assert spec.training.peft is not None
    assert spec.training.peft.dropout == 0.1
    assert spec.training.retrieval is None


def test_adapter_plumbs_retrieval_spec() -> None:
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {
                "training_type": "sft",
                "recipe": "bi_encoder",
                "finetuning_type": "all_weights",
                "retrieval": {
                    "train_n_passages": 7,
                    "query_prefix": "query:",
                    "passage_prefix": "passage:",
                    "export": {"dimensions": True, "opset": 18},
                },
            },
            "output": {"name": "out", "type": "model", "fileset": "out-fs"},
        },
    )
    assert isinstance(spec.training, SFTTraining)
    assert spec.training.retrieval is not None
    assert spec.training.retrieval.train_n_passages == 7
    assert spec.training.retrieval.query_prefix == "query:"
    assert spec.training.retrieval.passage_prefix == "passage:"
    assert spec.training.retrieval.export is not None
    assert spec.training.retrieval.export.dimensions is True
    assert spec.training.retrieval.export.opset == 18


def test_adapter_accepts_legacy_embedding_block() -> None:
    """Accept the pre-rename ``training.embedding`` key."""
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {
                "training_type": "sft",
                "recipe": "bi_encoder",
                "finetuning_type": "all_weights",
                "embedding": {"train_n_passages": 7, "query_prefix": "query:"},
            },
            "output": {"name": "out", "type": "model", "fileset": "out-fs"},
        },
    )
    assert isinstance(spec.training, SFTTraining)
    assert spec.training.retrieval is not None
    assert spec.training.retrieval.train_n_passages == 7
    assert spec.training.retrieval.query_prefix == "query:"


def test_adapter_new_fields_default_when_omitted() -> None:
    """Omitting the new fields keeps the v2 SFTTraining defaults."""
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "lora"},
            "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
        },
    )
    assert isinstance(spec.training, SFTTraining)
    assert spec.training.precision is None
    assert spec.training.min_learning_rate is None
    assert spec.training.adam_beta1 == 0.9
    assert spec.training.adam_beta2 == 0.999
    assert spec.training.parallelism.sequence_parallel is False
    assert spec.training.peft is not None
    assert spec.training.peft.dropout == 0.0


def test_adapter_plumbs_pass2_hyperparameters() -> None:
    """Pass-2 hyperparameters set on the plugin spec must reach the v2 SFTTraining."""
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {
                "training_type": "sft",
                "finetuning_type": "lora",
                "attn_implementation": "flash_attention_2",
                "lora": {"rank": 8, "exclude_modules": ["*.out_proj"], "use_triton": False},
            },
            "optimizer": {"adam_eps": 1e-6, "optimizer": "AdamW", "lr_decay_style": "linear"},
            "batch": {"sequence_packing": True, "sequence_packing_max_samples": 256},
            "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
        },
    )
    assert isinstance(spec.training, SFTTraining)
    assert spec.training.adam_eps == 1e-6
    assert spec.training.optimizer == "AdamW"
    assert spec.training.lr_decay_style == "linear"
    assert spec.training.attn_implementation == "flash_attention_2"
    assert spec.training.sequence_packing_max_samples == 256
    assert spec.training.peft is not None
    assert spec.training.peft.exclude_modules == ["*.out_proj"]
    assert spec.training.peft.use_triton is False


def test_adapter_pass2_defaults_when_omitted() -> None:
    """Omitting pass-2 fields preserves effective defaults through ``auto``."""
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "lora"},
            "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
        },
    )
    assert isinstance(spec.training, SFTTraining)
    assert spec.training.adam_eps == 1e-8
    assert spec.training.optimizer == "auto"
    assert spec.training.lr_decay_style == "cosine"
    assert spec.training.attn_implementation == "sdpa"
    assert spec.training.sequence_packing_max_samples == 1000
    assert spec.training.peft is not None
    assert spec.training.peft.exclude_modules is None
    assert spec.training.peft.use_triton is True


def test_adapter_distillation() -> None:
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {
                "training_type": "distillation",
                "finetuning_type": "all_weights",
                "teacher_model": "meta/teacher",
            },
            "output": {"name": "out", "type": "model", "fileset": "out-fs"},
        },
    )
    assert isinstance(spec.training, DistillationTraining)
    assert spec.training.teacher_model == "meta/teacher"


def test_adapter_no_integrations() -> None:
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "lora"},
            "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
        },
    )
    assert spec.integrations is None
    assert collect_integration_secret_envs(spec.integrations) == []


def test_adapter_integrations_full_round_trip() -> None:
    spec = automodel_spec_to_compiler_output(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "lora"},
            "output": {"name": "my-output", "type": "adapter", "fileset": "out-fs"},
            "integrations": {
                "wandb": {
                    "project": "my-project",
                    "name": "run-001",
                    "entity": "my-team",
                    "tags": ["sft", "llama"],
                    "notes": "experiment notes",
                    "base_url": "https://wandb.internal",
                    "api_key_secret": "default/wandb-key",
                },
                "mlflow": {
                    "experiment_name": "exp-1",
                    "name": "run-001",
                    "tracking_uri": "http://mlflow:5000",
                    "tags": {"team": "nlp"},
                    "description": "SFT run",
                },
            },
        },
    )

    assert spec.integrations is not None
    wandb = spec.integrations.wandb
    assert wandb is not None
    assert wandb.project == "my-project"
    assert wandb.name == "run-001"
    assert wandb.entity == "my-team"
    assert wandb.tags == ["sft", "llama"]
    assert wandb.notes == "experiment notes"
    assert wandb.base_url == "https://wandb.internal"
    assert wandb.api_key_secret is not None
    assert wandb.api_key_secret.root == "default/wandb-key"

    mlflow = spec.integrations.mlflow
    assert mlflow is not None
    assert mlflow.experiment_name == "exp-1"
    assert mlflow.name == "run-001"
    assert mlflow.tracking_uri == "http://mlflow:5000"
    assert mlflow.tags == {"team": "nlp"}
    assert mlflow.description == "SFT run"

    secret_envs = collect_integration_secret_envs(spec.integrations)
    assert secret_envs == [
        {"name": "WANDB_API_KEY", "from_secret": {"name": "default/wandb-key"}},
    ]


def test_adapter_integrations_from_automodel_job_output() -> None:
    job_output = AutomodelJobOutput.model_validate(
        {
            "model": "meta/llama",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft", "finetuning_type": "lora"},
            "schedule": {"epochs": 1},
            "batch": {"global_batch_size": 8, "micro_batch_size": 1},
            "optimizer": {"learning_rate": 1e-4},
            "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
            "integrations": {"wandb": {"project": "plugin-project"}},
        },
    )

    spec = automodel_spec_to_compiler_output(job_output)

    assert spec.integrations is not None
    assert spec.integrations.wandb is not None
    assert spec.integrations.wandb.project == "plugin-project"
