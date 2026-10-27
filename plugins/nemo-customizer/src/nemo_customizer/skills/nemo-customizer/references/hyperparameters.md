<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hyperparameters

Three backend job schemas live in this skill. Each backend has its own field reference file — **pick by plugin**:

| Plugin | Schema class | Schema dump | Field reference |
|--------|--------------|-------------|-----------------|
| `automodel` | `AutomodelJobInput` | `nemo customization automodel explain` | **`hyperparameters-automodel.md`** |
| `unsloth` | `UnslothJobInput` | `nemo customization unsloth explain` | **`hyperparameters-unsloth.md`** |
| `rl` (DPO / GRPO) | `RlJobInput` | `nemo customization rl explain` | **`hyperparameters-rl.md`** |

All three schemas use `extra="forbid"` — unknown keys raise validation errors. Field names are **not** interchangeable across backends (e.g. automodel uses `micro_batch_size` / `global_batch_size` / `parallelism`; unsloth uses `per_device_train_batch_size` / `gradient_accumulation_steps` / `hardware`; rl uses `batch_size` / `micro_batch_size` under `training` and takes `model` / `dataset` as plain strings). Use the right schema for the chosen plugin.

**Batch sizing, 48 GB VRAM tables, multi-GPU (data parallel vs tensor parallel), and throughput tuning** live in **`batch-sizing.md`** (automodel + unsloth). These per-backend files are the **field glossary**, full JSON template per backend, distillation/KD (automodel), and DPO knobs (rl) — not the place to pick batch sizes for production runs.

## Table of contents

| Read this file | For |
|----------------|-----|
| **`hyperparameters-automodel.md`** | Automodel job JSON layout, full template, `training` / `schedule` / `batch` / `optimizer` / `parallelism` field reference, LR & LoRA-rank tuning, presets, distillation/KD |
| **`hyperparameters-unsloth.md`** | Unsloth job JSON layout, full template, `model` / `dataset` / `training` / `schedule` / `batch` / `optimizer` / `hardware` / `output` field reference, LR & LoRA-rank tuning, save-method picker |
| **`hyperparameters-rl.md`** | NeMo-RL (DPO + GRPO) job JSON layout, shared knobs, DPO-specific (`ref_policy_kl_penalty` = β), GRPO-specific (`num_generations_per_prompt`, environment FileSet), convert CLI |
| **`recipes.md`** | Benchmarked Automodel / NeMo-RL recipes per model (Nemotron first) and the recipe → job JSON field mapping |
| **`batch-sizing.md`** | ≥48 GB VRAM batch tables, multi-GPU (data vs tensor parallel), OOM / throughput tuning (automodel + unsloth) |
| **Integrations** (below) | W&B / MLflow `integrations` object — all three backends (automodel, unsloth, rl) |

---

## Integrations (all backends)

**All three backends** (automodel, unsloth, rl) accept the same `integrations` object on job JSON (`IntegrationsSpec` in `nemo_helix_plugin.integrations`) — **W&B** and **MLflow**. A non-null `wandb` / `mlflow` block **requests** that integration; the training runtime **activates** it only when credentials/URIs are available (W&B needs `WANDB_API_KEY`, MLflow needs a tracking URI). Omit the field or set a block to `null` to disable. There is no `enabled` flag and no `report_to` on input — `report_to` is derived at runtime from activated integrations. The compiler logs a warning when W&B is requested without `api_key_secret` or MLflow without `tracking_uri`.

```json
"integrations": {
  "wandb": {
    "project": "my-project",
    "name": "run-001",
    "entity": "my-team",
    "tags": ["sft", "llama"],
    "notes": "Experiment notes",
    "base_url": "https://wandb.internal",
    "api_key_secret": "default/wandb-api-key"
  },
  "mlflow": {
    "experiment_name": "llama-finetuning",
    "name": "run-001",
    "tracking_uri": "http://mlflow:5000",
    "tags": { "team": "nlp" },
    "description": "SFT experiment"
  }
}
```

| Field | Notes |
|-------|-------|
| `wandb` | Non-null requests W&B (requires `WANDB_API_KEY` at runtime). |
| `wandb.project` | W&B project; defaults to `output.name` at runtime if unset. |
| `wandb.name` | W&B run name; defaults to job ID. Legacy `run_name` is accepted with a deprecation warning. |
| `wandb.entity` | W&B team or username. |
| `wandb.tags` / `wandb.notes` | Optional run metadata. |
| `wandb.base_url` | Self-hosted W&B server URL. Without `api_key_secret`, W&B may still activate when `base_url` is set **and** the server allows access without a cloud API key — a compile-time warning is logged. |
| `wandb.api_key_secret` | Platform secret ref (`secret_name` or `workspace/secret_name`). The compiler injects `WANDB_API_KEY` into the training step environment. |
| `mlflow` | Non-null requests MLflow (requires tracking URI at runtime). |
| `mlflow.tracking_uri` | MLflow tracking server; can also come from `MLFLOW_TRACKING_URI` in the container. |
| `mlflow.experiment_name` | Defaults to `output.name` if unset. |
| `mlflow.name` | MLflow run name; defaults to job ID. Legacy `run_name` is accepted with a deprecation warning. |
| `mlflow.tags` / `mlflow.description` | Optional run metadata. |

Set `"integrations": null` or omit the field when tracking is not needed.

**Local setup (MLflow server, `docker0` tracking URI, jobs-launcher, W&B secret) — Docker-runtime (automodel / unsloth):** `references/integrations-setup.md`.

**rl (DPO) note:** rl supports **W&B and MLflow** through this object exactly like automodel and unsloth. Two rl specifics: the run name defaults to the **job id** (stable across pause/resume) and NeMo-RL auto-adds tags (`service:rl`, `framework:…`, plus workspace / job / task / model); and because rl runs on **Kubernetes / Ray** (not the Docker executor), point `tracking_uri` and any self-hosted W&B `base_url` at an endpoint **reachable from the cluster** — the `docker0` local-MLflow recipe above is Docker-runtime only. (NeMo-RL's TensorBoard / SwanLab logger slots aren't exposed via `integrations`, same as the other backends — `IntegrationsSpec` carries only `wandb` + `mlflow`.)

**Unsloth note:** HuggingFace `TrainingArguments.run_name` is shared by W&B and MLflow. When both backends are active, `wandb.name` wins if set; otherwise `mlflow.name` is used. If both names are set to different values, a runtime warning is logged and W&B's name is used.
