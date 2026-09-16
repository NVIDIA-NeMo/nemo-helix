<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-unsloth-plugin

Unsloth GPU fine-tuning **customization contributor** for NeMo Helix.

Registered under `nemo.customization.contributors` (key `unsloth`) — the `nemo-customizer-plugin` hub composes it under `/apis/customization/v2/workspaces/{workspace}/unsloth/` (HTTP) and `client.customization.unsloth.*` (SDK), and mounts the CLI at `nemo customization unsloth ...`.

Unsloth is **submit-only**: training executes remotely on the platform's GPU cluster as a 4-step container job (download → train → upload → model-entity), mirroring `nemo-automodel-plugin`. The plugin itself stays lightweight — heavy ML deps (`unsloth`, `trl`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `torch`) live only inside the `nhx-unsloth-training` container image.

## Install

The plugin is part of `enabled-plugins` once `uv sync` runs. No GPU / ML deps are installed locally; only the container image needs them.

Container image build / push instructions live in [`docker/unsloth/README.md`](../../docker/unsloth/README.md).

## Submit a training job

```bash
nemo customization unsloth submit /path/to/job.json -w default
```

Job JSON uses the `UnslothJobInput` schema (see `nemo_unsloth_plugin/schema.py`). Minimal example:

```json
{
  "name": "qwen-tutorial-smoke",
  "model": {"name": "unsloth/Qwen2.5-0.5B-Instruct", "max_seq_length": 2048},
  "dataset": {"path": "default/my-dataset", "text_field": "text"},
  "schedule": {"max_steps": 60, "warmup_ratio": 0.1}
}
```

What happens after submit:

1. The plugin's `to_spec` validates the model entity + dataset fileset against the live platform.
2. `UnslothJob.compile` produces a 4-step `PlatformJobSpec` (see `nhx.unsloth.images` for the exact container commands):

   | Step | Image | Container command |
   |------|-------|-------------------|
   | **`model-and-dataset-download`** | `nhx-customizer-tasks` | `python -m nhx.customization_common.tasks.file_io --service-source unsloth --service-name unsloth` |
   | **`training`** | `nhx-unsloth-training` | `python -m nhx.unsloth.tasks.training` |
   | **`model-upload`** | `nhx-customizer-tasks` | `python -m nhx.customization_common.tasks.file_io --service-source unsloth --service-name unsloth` |
   | **`model-entity-creation`** | `nhx-customizer-tasks` | `python -m nhx.customization_common.tasks.model_entity --service-name unsloth` |

   Shared CPU tasks require explicit identity flags on the module entrypoints (`nhx.customization_common.tasks.file_io.__main__` requires both `--service-source` and `--service-name`; `model_entity.__main__` requires `--service-name`). The compiler sets these via `FILE_IO_TASK_COMMAND` and `MODEL_ENTITY_TASK_COMMAND` in `services/unsloth/src/nhx/unsloth/images.py`. `--service-source` is stamped on upload-created filesets; `--service-name` drives SDK auth/telemetry.
3. The platform Jobs runner schedules each step; tail logs with the standard jobs API.

## CLI surface

```bash
nemo customization unsloth --help
nemo customization unsloth submit JOB_JSON -w WORKSPACE [--profile P] [--cluster C] [-o k=v]
nemo customization unsloth explain      # prints schemas
```

`submit`'s positional `JOB_JSON` replaces the `--spec` / `--spec-file` shape used by some other backends.

## GPU selection

Set `hardware.gpus = "0"` (or `"0,1"`) in the job JSON. The training container picks the value up via `CUDA_VISIBLE_DEVICES` *before* importing `unsloth` / `torch` so the var is observed at torch-init time. Selection, not reservation — Unsloth picks one GPU per process.

The container image targets the same compute capabilities NVIDIA's stock `pytorch` base supports (Ampere+). Pre-Ampere users should set `hardware.precision = "fp16"` in the job JSON.

## Schema reference

- `model: ModelLoadSpec` — `name`, `max_seq_length`, `load_in_4bit`, `load_in_8bit`, `dtype`, `trust_remote_code`.
- `dataset: DatasetSpec` — `path` (required), `text_field`, `apply_chat_template`, `validation_path`, `packing`.
- `training: TrainingSpec` — `training_type`, `finetuning_type` (`lora` or `all_weights`), `lora: LoRAParams`, `use_gradient_checkpointing`.
- `schedule: ScheduleSpec` — `epochs` xor `max_steps`, `warmup_steps` xor `warmup_ratio`, `lr_scheduler_type`, `logging_steps`, `save_steps`, `eval_steps`, `seed`.
- `batch: BatchSpec` — `per_device_train_batch_size`, `gradient_accumulation_steps`.
- `optimizer: OptimizerSpec` — `learning_rate`, `weight_decay`, `optim`.
- `hardware: HardwareSpec` — `gpus`, `precision` (`bf16` / `fp16`).
- `integrations: IntegrationsSpec | None` — optional W&B / MLflow (`nemo_helix_plugin.integrations`). Request by presence; `api_key_secret` carries a secret *reference* that the jobs launcher resolves into `WANDB_API_KEY` in the training container **at runtime** (compile only records the reference). Example: `plugins/nemo-unsloth/tests/fixtures/integrations_wandb_mlflow.json`.
- `output: OutputRequest | None` — `name`, `description`, `save_method` (`lora` / `merged_16bit` / `merged_4bit`).

`UnslothJobOutput` is the canonical post-`to_spec` form: same as the input plus a resolved `output: OutputResponse` carrying the auto-generated name, inferred type (adapter vs model), and the destination fileset name.

## Architecture (plugin ↔ service split)

This plugin is the **thin contributor wrapper**. The heavy code lives in `services/unsloth/` (`nhx-unsloth`):

- **Plugin** (`plugins/nemo-unsloth/`, `nemo_unsloth_plugin`) — `UnslothContributor`, `UnslothJob` (lifecycle + `compile()`), submitter-facing schema (`UnslothJobInput`), CLI overrides, SDK shapes, contributor wiring.
- **Service** (`services/unsloth/`, `nhx.unsloth`) — canonical schemas (`UnslothJobOutput` and shared sub-shapes), the `train_sft` training driver, CPU task commands wired through `nhx.customization_common.tasks.*` (`nhx-customizer-tasks` image), the `nhx.unsloth.tasks.training` GPU entrypoint, and the `platform_job_config_compiler`.

The plugin imports two things from the service:

- `nhx.unsloth.compile.platform_job_config_compiler` — invoked from `UnslothJob.compile()` to build the 4-step `PlatformJobSpec`.
- `nhx.unsloth.config.config` — for the default execution profile.

## See also

- `plugins/nemo-customizer/` — the customization router hub. Owns `/apis/customization`, `nemo customization`, `client.customization`.
- `services/unsloth/` — the heavy code this plugin delegates to.
- `docker/unsloth/` — Dockerfile + build instructions for the `nhx-unsloth-training` image.
- `plugins/nemo-automodel/` — sibling plugin with the same submit shape.
