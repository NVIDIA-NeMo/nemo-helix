<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hyperparameters

Three backend job schemas live in this skill. Each backend has its own field reference file — **pick by plugin**:

| Plugin | Schema class | Schema dump | Field reference |
|--------|--------------|-------------|-----------------|
| `automodel` | `AutomodelJobInput` (`plugins/nemo-automodel/src/nemo_automodel_plugin/schema.py`) | `nemo customization automodel explain` | **`hyperparameters-automodel.md`** |
| `unsloth` | `UnslothJobInput` (`plugins/nemo-unsloth/src/nemo_unsloth_plugin/schema.py`) | `nemo customization unsloth explain` | **`hyperparameters-unsloth.md`** |
| `rl` (DPO / GRPO) | `RlJobInput` (`plugins/nemo-rl/src/nemo_rl_plugin/schema.py`) | `nemo customization rl explain` | **`hyperparameters-rl.md`** |

All three schemas use `extra="forbid"` — unknown keys raise validation errors. Field names are **not** interchangeable across backends (e.g. automodel uses `micro_batch_size` / `global_batch_size` / `parallelism`; unsloth uses `per_device_train_batch_size` / `gradient_accumulation_steps` / `hardware`; rl uses `batch_size` / `micro_batch_size` under `training` and takes `model` / `dataset` as plain strings). Use the right schema for the chosen plugin.

**Batch sizing, 48 GB VRAM tables, multi-GPU (data parallel vs tensor parallel), and throughput tuning** live in **`batch-sizing.md`** (automodel + unsloth). These per-backend files are the **field glossary**, full JSON template per backend, distillation/KD (automodel), and DPO knobs (rl) — not the place to pick batch sizes for production runs.

## Table of contents

| Read this file | For |
|----------------|-----|
| **`hyperparameters-automodel.md`** | Automodel job JSON layout, full template, `training` / `schedule` / `batch` / `optimizer` / `parallelism` field reference, LR & LoRA-rank tuning, presets, distillation/KD |
| **`hyperparameters-unsloth.md`** | Unsloth job JSON layout, full template, `model` / `dataset` / `training` / `schedule` / `batch` / `optimizer` / `hardware` / `output` field reference, LR & LoRA-rank tuning, save-method picker |
| **`hyperparameters-rl.md`** | NeMo-RL (DPO + GRPO) job JSON layout, shared knobs, DPO-specific (`ref_policy_kl_penalty` = β), GRPO-specific (`num_generations_per_prompt`, environment FileSet), convert CLI |
| **`batch-sizing.md`** | ≥48 GB VRAM batch tables, multi-GPU (data vs tensor parallel), OOM / throughput tuning (automodel + unsloth) |
| **Integrations** (below) | W&B / MLflow `integrations` object — all three backends (automodel, unsloth, rl) |
| **Source of truth** (below) | What is authoritative for a field, and what is only an example |

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

Set `"integrations": null` or omit the field when tracking is not needed. For the payload shape only — the field set is `IntegrationsSpec`, above — automodel → `plugins/nemo-automodel/tests/fixtures/integrations_wandb_mlflow.json`; unsloth → `plugins/nemo-unsloth/tests/fixtures/integrations_wandb_mlflow.json`; rl → `plugins/nemo-rl/tests/fixtures/integrations_wandb_mlflow.json`.

**Local setup (MLflow server, `docker0` tracking URI, jobs-launcher, W&B secret) — Docker-runtime (automodel / unsloth):** `references/integrations-setup.md`.

**rl (DPO) note:** rl supports **W&B and MLflow** through this object exactly like automodel and unsloth. Two rl specifics: the run name defaults to the **job id** (stable across pause/resume) and NeMo-RL auto-adds tags (`service:rl`, `framework:…`, plus workspace / job / task / model); and because rl runs on **Kubernetes / Ray** (not the Docker executor), point `tracking_uri` and any self-hosted W&B `base_url` at an endpoint **reachable from the cluster** — the `docker0` local-MLflow recipe above is Docker-runtime only. (NeMo-RL's TensorBoard / SwanLab logger slots aren't exposed via `integrations`, same as the other backends — `IntegrationsSpec` carries only `wandb` + `mlflow`.)

**Unsloth note:** HuggingFace `TrainingArguments.run_name` is shared by W&B and MLflow. When both backends are active, `wandb.name` wins if set; otherwise `mlflow.name` is used. If both names are set to different values, a runtime warning is logged and W&B's name is used.

---

# Source of truth

**A test fixture is not the schema.** Fixtures are inputs to smoke tests: they carry
whatever made a test run fast and cheap, they exercise one path rather than the field
set, and nothing fails when the schema gains a field they never set. Reading one to
answer "what fields exist" or "what is the default" gives an answer that is wrong in
the direction of *too small* — `max_steps: 20` and a 0.6B model are test scaffolding,
not recommendations.

Order to trust, highest first:

| Rank | Ask | Why it wins |
|------|-----|-------------|
| 1 | `nemo customization <plugin> explain` | The live schema of the installed build. Cannot drift from what the server accepts. |
| 2 | The schema source files below | The same contract, with field descriptions and validators. Use when the CLI is unavailable or you need the *why*. |
| 3 | This reference and its per-backend files | Curated, but written by hand and can lag a schema change. |
| 4 | Fixtures, READMEs, design docs | Illustrative only. Copy the *shape*, never the values, and never infer absence from them. |

If a fixture and the live schema disagree, the schema is right and the fixture is
stale. Say so rather than following the fixture.

## Authoritative

| Resource | Path | Use for |
|----------|------|---------|
| **Batch / multi-GPU / 48 GB LoRA (automodel)** | `batch-sizing.md` § Batch sizing — automodel, § Multi-GPU | Choosing `micro`, GBS, LR, TP vs data parallel |
| **Batch (unsloth, single GPU)** | `batch-sizing.md` § Batch sizing — unsloth | `per_device_train_batch_size` × `gradient_accumulation_steps` starting points |
| Submit schema (automodel) | `plugins/nemo-automodel/src/nemo_automodel_plugin/schema.py` | Allowed JSON fields |
| Schema → compiler mapping (automodel) | `services/automodel/src/nhx/automodel/adapter.py` | `dataset.training` → compiler `dataset` string |
| API field descriptions (automodel) | `services/automodel/src/nhx/automodel/api/v2/jobs/schemas.py` | Compiler-internal shape (not submit JSON) |
| Submit schema (unsloth) | `plugins/nemo-unsloth/src/nemo_unsloth_plugin/schema.py` | Allowed JSON fields (`UnslothJobInput`) |
| Canonical schema (unsloth) | `services/unsloth/src/nhx/unsloth/schemas.py` | Post-`to_spec` shape; what `train_sft` consumes |
| Training driver (unsloth) | `services/unsloth/src/nhx/unsloth/tasks/training/backends/unsloth_sft.py` | Field → call-site mapping (FastLanguageModel.from_pretrained, SFTTrainer, save_pretrained{,_merged}) |
| Submit schema (rl / DPO) | `plugins/nemo-rl/src/nemo_rl_plugin/schema.py` | Allowed JSON fields (`RlJobInput` / `DPOTraining`) |
| Canonical schema (rl / DPO) | `services/rl/src/nhx/rl/schemas.py` | Post-transform shape (`RlJobOutput`); divisibility validator |
| DPO config builder (rl) | `services/rl/src/nhx/rl/tasks/training/backends/nemo_rl/dpo_config.py` | Field → NeMo-RL YAML mapping |

## Illustrative — shape only, never the contract

Nothing here is authoritative. Read it for orientation, then confirm every field
against `explain` or a schema file above.

| Resource | Path | Read it for | Do not read it for |
|----------|------|-------------|--------------------|
| JSON fixtures (automodel) | `plugins/nemo-automodel/tests/fixtures/*.json` | Where a block sits in the payload | Field set, defaults, sensible values |
| JSON fixture (unsloth) | `plugins/nemo-unsloth/tests/fixtures/minimal_unsloth_sft.json` | Same | Same |
| JSON fixture (rl / DPO) | `plugins/nemo-rl/tests/fixtures/minimal_dpo.json` | Same | Same |
| Full spec doc (automodel) | `plugins/nemo-automodel/SCOPE.md` | Design intent | Current field set |
| Plugin README (unsloth) | `plugins/nemo-unsloth/README.md` | Submit-only CLI, 4-step container job, GPU selection | Field reference |
| Plugin README (rl) | `plugins/nemo-rl/README.md` | Submit-only CLI, Kubernetes/Ray runtime, constraints | Field reference |
| Plugin design doc (rl) | `docs/customizer/nemo-rl-dpo-plugin-design.md` | Architecture, 4-step job, image split | Field reference |

Every fixture above sets `max_steps` so a smoke test finishes in a minute. That is the
clearest case of the rule: it is the single most-copied wrong value in this repo, and
it caps a real run mid-epoch.
