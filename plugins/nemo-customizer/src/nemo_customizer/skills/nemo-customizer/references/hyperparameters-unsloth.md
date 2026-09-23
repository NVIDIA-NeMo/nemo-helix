<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Unsloth job JSON reference. Index + integrations + source-of-truth: `hyperparameters.md`. Batch sizing (single GPU): `batch-sizing.md`. -->

# Unsloth job JSON

Job JSON for `nemo customization unsloth submit` uses **`UnslothJobInput`** (`plugins/nemo-unsloth/src/nemo_unsloth_plugin/schema.py`). Only fields in that schema are accepted (`extra="forbid"`). The canonical post-transform shape lives in `services/unsloth/src/nhx/unsloth/schemas.py` (`UnslothJobOutput`) and is what the training driver consumes in the GPU container.

**Schema dump:**

```bash
nemo customization unsloth explain
```

Unsloth is **submit-only, single-GPU inside the training container**. There is no `parallelism` block and no `training.execution_profile` in job JSON — pass `--profile` on `nemo customization unsloth submit` instead (default `gpu`). `hardware.gpus` sets `CUDA_VISIBLE_DEVICES` in the container before `import torch`. Multi-GPU sharding → use automodel.

## Job JSON layout (unsloth)

| Section | Purpose |
|---------|---------|
| `name` | Optional job name (auto-generated if omitted) |
| `model` | **Object** — base model entity ref + how to load it (4-bit, dtype, max_seq_length) |
| `dataset` | Single fileset ref (`path`) + optional `validation_path`; row shape selector (`text_field`, `apply_chat_template`, `packing`) |
| `training` | Method (`sft`), adapter shape (`lora`/`full`), LoRA hyperparams, gradient checkpointing |
| `schedule` | `epochs` xor `max_steps`; `warmup_steps` xor `warmup_ratio`; logging / save / eval cadence; LR scheduler |
| `batch` | `per_device_train_batch_size` × `gradient_accumulation_steps` = effective batch |
| `optimizer` | LR, weight decay, optimizer choice (`adamw_8bit` default) |
| `hardware` | GPU selection (`CUDA_VISIBLE_DEVICES`) + mixed precision (`bf16` / `fp16`) |
| `integrations` | Optional W&B / MLflow (same shape as automodel) |
| `output` | Output entity name, optional description, **`save_method`** (controls what's persisted) |

Full template (every section, defaults inline):

```json
{
  "name": "<job-name>",
  "model": {
    "name": "default/<model-entity>",
    "max_seq_length": 2048,
    "load_in_4bit": true,
    "load_in_8bit": false,
    "dtype": "auto",
    "trust_remote_code": false,
    "device_map": null,
    "rope_scaling": null
  },
  "dataset": {
    "path": "default/<dataset-fileset>",
    "validation_path": null,
    "text_field": "text",
    "apply_chat_template": true,
    "packing": false
  },
  "training": {
    "training_type": "sft",
    "finetuning_type": "lora",
    "lora": {
      "rank": 16,
      "alpha": 16,
      "dropout": 0.0,
      "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
      "bias": "none",
      "use_rslora": false,
      "random_state": 3407,
      "use_dora": false,
      "loftq_config": null,
      "modules_to_save": null,
      "layers_to_transform": null,
      "layer_replication": null,
      "init_lora_weights": true
    },
    "use_gradient_checkpointing": "unsloth"
  },
  "schedule": {
    "epochs": 1,
    "max_steps": null,
    "warmup_steps": 0,
    "warmup_ratio": null,
    "lr_scheduler_type": "linear",
    "lr_scheduler_kwargs": null,
    "logging_steps": 1,
    "save_steps": null,
    "eval_steps": null,
    "seed": 3407
  },
  "batch": {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4
  },
  "optimizer": {
    "learning_rate": 5e-5,
    "weight_decay": 0.0,
    "optim": "adamw_8bit",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "label_smoothing_factor": 0.0,
    "neftune_noise_alpha": null
  },
  "hardware": {
    "gpus": "0",
    "precision": "bf16"
  },
  "integrations": null,
  "output": {
    "name": "<output-name>",
    "description": null,
    "save_method": "lora"
  }
}
```

## Field reference (unsloth)

### `model`

`model` is an **object** (not a string). `name` is the platform model entity ref.

| Field | Default | Notes |
|-------|---------|-------|
| `name` | — | Model entity ref: `"name"` (uses job workspace) or `"workspace/name"`. Plugin resolves to a local path before training. |
| `max_seq_length` | `2048` | Truncate / pack to this length; lower if VRAM tight. |
| `load_in_4bit` | `true` | bitsandbytes 4-bit. Mutex with `load_in_8bit`. Default for Unsloth's headline path; required to fit larger models on small GPUs. |
| `load_in_8bit` | `false` | bitsandbytes 8-bit. Mutex with `load_in_4bit`. |
| `dtype` | `"auto"` | One of `"auto"`, `"bfloat16"`, `"float16"`, `"float32"`. |
| `trust_remote_code` | `false` | HF `trust_remote_code` flag for custom model code (required by some hybrid Mamba/MoE models, e.g. Nemotron-H). |
| `device_map` | `null` | Placement for `FastLanguageModel.from_pretrained`. `null` pins the whole model to the single visible GPU (`{"": 0}`) — the right default for this single-GPU backend. Leave unset unless experimenting; `"auto"`/`"balanced"`/`"sequential"` can spill layers to CPU on unified-memory hosts (GB10 / DGX Spark) and abort 4-bit loads. |
| `rope_scaling` | `null` | RoPE scaling for long-context extension, e.g. `{"type": "linear", "factor": 2.0}`. `null` uses the model's native context length. |

**Mutex:** `load_in_4bit` xor `load_in_8bit`. Both quantization flags are also **incompatible with `training.finetuning_type: "all_weights"`** — full SFT must use a non-quantized base.

> **Hybrid Mamba/MoE models (e.g. NVIDIA Nemotron-H `*-A3B`):** load in **16-bit** (`load_in_4bit: false`, `load_in_8bit: false`) — Unsloth's supported path for these. The 4-bit (bitsandbytes) path can hit a dtype mismatch inside the model's MoE expert accumulation. Keep `device_map` unset (single-GPU default) and set `trust_remote_code: true`.

### `dataset`

See `references/dataset-formats.md` § Unsloth for row-shape rules.

| Field | Default | Notes |
|-------|---------|-------|
| `path` | — | Training fileset ref (`"name"` or `"workspace/name"`). |
| `validation_path` | `null` | Optional validation fileset ref. |
| `text_field` | `"text"` | Column SFTTrainer reads. In `apply_chat_template: true` mode, the rendered template string is written into this column. |
| `apply_chat_template` | `false` | Set `true` for rows with a `messages` array (preferred when the tokenizer has a chat template). |
| `packing` | `false` | trl.SFTTrainer packing for throughput on short rows. |

### Unsloth `training`

| Field | Default | Notes |
|-------|---------|-------|
| `training_type` | `"sft"` | Only `"sft"` is implemented today. |
| `finetuning_type` | `"lora"` | `"lora"` (adapter; default) or `"all_weights"` (full SFT — heavy, no quantization). |
| `lora` | auto-filled when `finetuning_type` is `lora` | See LoRA subsection below. |
| `use_gradient_checkpointing` | `"unsloth"` | `"unsloth"` (recommended), `"true"`, or `"false"`. Unsloth's variant is faster than HF's. |

**LoRA block (`training.lora`):**

| Field | Default | Notes |
|-------|---------|-------|
| `rank` | `16` | Higher → more capacity, more VRAM. Cap at 32 if the adapter will deploy via default NIM / vLLM. |
| `alpha` | `16` | LoRA scaling; common rule of thumb `alpha ≈ rank` or `2× rank`. |
| `dropout` | `0.0` | LoRA dropout (0.0–<1.0). |
| `target_modules` | Unsloth 7-module set: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Full attention + MLP. Override with a subset like `["q_proj","v_proj"]` for a lighter touch. |
| `bias` | `"none"` | `"none"` / `"all"` / `"lora_only"`. |
| `use_rslora` | `false` | Rank-stabilized LoRA. |
| `random_state` | `3407` | Reproducibility seed for the LoRA init. |
| `use_dora` | `false` | DoRA (weight-decomposed LoRA). Better quality at low ranks; adds overhead. |
| `loftq_config` | `null` | LoftQ init config for quantized bases. `null` disables. |
| `modules_to_save` | `null` | Extra non-LoRA modules trained & saved in full, e.g. `["embed_tokens","lm_head"]` (vocab changes / continued pretraining). |
| `layers_to_transform` | `null` | Restrict LoRA to specific layer index(es). `null` = all layers. |
| `layer_replication` | `null` | Layer-replication ranges for stacking, e.g. `[[0,16],[8,24]]`. |
| `init_lora_weights` | `true` | Init scheme. `true` = PEFT default; `"gaussian"`/`"pissa"`/`"olora"`/`"loftq"` for advanced inits. |

`lora` is auto-filled with these defaults when `finetuning_type: "lora"` and the user omits the block. Must be `null` / omitted when `finetuning_type: "all_weights"`.

### Unsloth `schedule`

| Field | Default | Notes |
|-------|---------|-------|
| `epochs` | `null` | Full passes. **`epochs` xor `max_steps`** — exactly one is required. |
| `max_steps` | `null` | Global step cap. Use alone for smoke tests; do not combine with `epochs`. |
| `warmup_steps` | `0` | Linear warmup. Mutex with `warmup_ratio`. |
| `warmup_ratio` | `null` | Fractional warmup over total steps. Mutex with `warmup_steps`. |
| `lr_scheduler_type` | `"linear"` | `"linear"`, `"cosine"`, `"constant"`, `"constant_with_warmup"`, `"cosine_with_restarts"`. |
| `lr_scheduler_kwargs` | `null` | Extra scheduler kwargs, e.g. `{"num_cycles": 3}` for `cosine_with_restarts`. `null` uses defaults. |
| `logging_steps` | `1` | Loss-log cadence. |
| `save_steps` | `null` | If set, save checkpoint every N steps. |
| `eval_steps` | `null` | If set with `validation_path`, eval every N steps. When `null` and `validation_path` is set, the training driver defaults to **one validation pass per effective epoch** at `max(1, effective_steps - 1)` (same effective-step cap as automodel's default `val_check_interval`). |
| `seed` | `3407` | Trainer seed (`TrainingArguments.seed`). |

**Hard mutex enforced by the schema:** `epochs` xor `max_steps`; `warmup_steps` xor `warmup_ratio`. Validation errors surface at submit time.

### Unsloth `batch`

| Field | Default | Notes |
|-------|---------|-------|
| `per_device_train_batch_size` | `1` | Forwarded verbatim to `TrainingArguments`. Drives peak VRAM. |
| `gradient_accumulation_steps` | `1` | Multiplies effective batch without raising VRAM. |

`effective_batch = per_device_train_batch_size × gradient_accumulation_steps`. No GBS divisibility math (single GPU). Starting points by model size are in `batch-sizing.md` § Batch sizing — unsloth.

### Unsloth `optimizer`

| Field | Default | Notes |
|-------|---------|-------|
| `learning_rate` | `2e-4` (schema default; skill uses `5e-5` for LoRA SFT) | See LR table below. |
| `weight_decay` | `0.0` | L2-style regularization. |
| `optim` | `"adamw_8bit"` | `"adamw_torch"`, `"adamw_torch_fused"` (Hopper+), `"adamw_8bit"`, `"paged_adamw_8bit"`, `"sgd"`. `adamw_8bit` has the smallest optimizer state and is Unsloth's notebook default. |
| `adam_beta1` | `0.9` | Adam/AdamW beta1. |
| `adam_beta2` | `0.999` | Adam/AdamW beta2. |
| `adam_epsilon` | `1e-8` | Adam/AdamW epsilon. |
| `max_grad_norm` | `1.0` | Gradient-clipping max norm (TRL default). |
| `label_smoothing_factor` | `0.0` | Label smoothing for the CE loss. `0.0` disables. |
| `neftune_noise_alpha` | `null` | NEFTune embedding-noise alpha (quality boost). `null` disables. |

`warmup_steps` is on `schedule`, not on `optimizer` (different from the automodel schema).

### `hardware`

| Field | Default | Notes |
|-------|---------|-------|
| `gpus` | `null` | Comma-separated CUDA indices inside the training container: `"0"` (typical). Sets `CUDA_VISIBLE_DEVICES` **before** `import torch`. **Selection, not reservation.** Unsloth uses one GPU per training process. |
| `precision` | `"bf16"` | `"bf16"` (Ampere+) or `"fp16"`. |

### Unsloth `integrations`

See **Integrations (all backends)** in `hyperparameters.md`.

### `output`

| Field | Default | Notes |
|-------|---------|-------|
| `name` | auto-derived from `<model-entity>-<dataset>-<hex12>` | The output model entity / fileset name. |
| `description` | `null` | Free-form description carried onto the entity and fileset. |
| `save_method` | `"lora"` | `"lora"` (adapter — hot-reloads on base LoRA deployment; no new inference deploy), `"merged_16bit"` (merged checkpoint — **deploy** `output.name` as model entity), `"merged_4bit"` (lossy, storage-tight; deploy like merged). `merged_*` requires `training.finetuning_type: "lora"`. |

After `to_spec`, the canonical `OutputResponse` also carries `type` (`"adapter"` for `save_method: "lora"`, `"model"` otherwise) and `fileset` (defaults to `name`); both are derived — submitter doesn't set them.

## Tuning guide (unsloth)

VRAM / batch tuning is in **`batch-sizing.md` § Batch sizing — unsloth**. Below covers non-batch fields.

### Unsloth learning rate (LoRA SFT, starting points)

Same scale as automodel (the underlying optimizer math is the same):

| Model scale | Suggested `learning_rate` |
|-------------|---------------------------|
| ≤ 3B | `5e-5` – `1e-4` |
| 3B – 8B | `2e-5` – `5e-5` |
| > 8B | `1e-5` – `2e-5` |

Schema default is `2e-4` (Unsloth notebook default — works for small adapters with `adamw_8bit`). Skill defaults are conservative `5e-5`.

### Unsloth LoRA rank / alpha

| Use case | `rank` | `alpha` |
|----------|--------|---------|
| Default / balanced | 16 | 16 |
| Lighter touch | 8 | 16 |
| More capacity (inference-safe max on default NIM/vLLM) | 32 | 32 or 64 |

Drop `rank` before lowering batch when OOM. Higher `alpha/rank` ratios amplify adapter influence; Unsloth's defaults keep `alpha == rank`.

### Save-method picker

| User wants | `save_method` | Inference after training |
|------------|---------------|--------------------------|
| Smallest artefact; hot-reload on base LoRA deployment | `lora` | No new deploy — adapter loads on existing `lora_enabled` deployment |
| Full-weight checkpoint as standalone model | `merged_16bit` | **Deploy** `output.name` as new model entity |
| Disk-tight merged checkpoint (lossy) | `merged_4bit` | **Deploy** `output.name` as new model entity |
| Full SFT (no LoRA) | `lora` is invalid; output is always a full model | **Deploy** `output.name` as new model entity |

`merged_*` require `training.finetuning_type: "lora"`. The schema validator surfaces a clear error if violated.

### Smoke test (unsloth)

```json
"schedule": { "max_steps": 50 }
```

(omit `epochs`).

### Distillation

Not supported by unsloth today (`training_type` is `Literal["sft"]`). Use automodel for distillation.

