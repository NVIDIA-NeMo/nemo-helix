<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Automodel job JSON reference. Index + integrations + source-of-truth: `hyperparameters.md`. Batch sizing / multi-GPU / 48 GB VRAM tables: `batch-sizing.md`. -->

# Automodel job JSON

Job JSON for `nemo customization automodel submit` uses **`AutomodelJobInput`** (`plugins/nemo-automodel/src/nemo_automodel_plugin/schema.py`). Only fields in that schema are accepted (`extra="forbid"`).

**Schema dump:**

```bash
nemo customization automodel explain
```

**Contract examples:** `services/automodel/tests/contract/input_configs/` (legacy shape; map `batch_size` → `global_batch_size` in submit JSON).

## Job JSON layout

| Section | Purpose |
|---------|---------|
| `model` | **Base model entity** ref (`default/<model-entity>`) — weights to fine-tune |
| `dataset` | **Dataset filesets** (`default/<dataset-fileset>`); optional `prompt_template` for CUSTOM schema |
| `training` | Method, LoRA, `max_seq_length`, distillation/KD fields |
| `schedule` | Epochs, optional step cap, validation cadence, seed |
| `batch` | Global/micro batch, sequence packing |
| `optimizer` | LR, weight decay, warmup |
| `parallelism` | Nodes, GPUs, TP/PP/CP/EP |
| `output` | Output adapter/model fileset name |
| `integrations` | Optional W&B / MLflow |

### `model` field (base model entity)

`model` must name a **Models API entity** for the checkpoint being trained — not a dataset fileset, not an output adapter from a prior job, and not a raw Hugging Face repo id.

| Valid | Invalid |
|-------|---------|
| `default/qwen3-1.7b` (entity from `nemo models create`) | `Qwen/Qwen3-1.7B` (HF id) |
| `default/llama-3.2-1b-instruct` | `default/commonsense_qa` (dataset fileset) |
| `other-ws/my-model` (qualified ref) | `qwen3-1.7b-commonsense-qa-lora` (output fileset only, unless registered as entity) |

Register before submit (same as skill fast path): HF **model** fileset → `nemo models create <model-entity> …` with `"fileset":"default/<weights-fileset>"`. List: `nemo models list --workspace default`.

Full template:

```json
{
  "model": "default/<model-entity>",
  "dataset": {
    "training": "default/<dataset-fileset>",
    "validation": "default/<dataset-fileset>",
    "prompt_template": null
  },
  "training": {
    "training_type": "sft",
    "recipe": "auto",
    "finetuning_type": "lora",
    "lora": {
      "rank": 16,
      "alpha": 32,
      "dropout": 0.0,
      "merge": false,
      "target_modules": null,
      "exclude_modules": null,
      "use_triton": true
    },
    "max_seq_length": 2048,
    "precision": null,
    "attn_implementation": "sdpa",
    "execution_profile": null
  },
  "schedule": {
    "epochs": 1,
    "max_steps": null,
    "val_check_interval": null,
    "seed": null
  },
  "batch": {
    "global_batch_size": 4,
    "micro_batch_size": 1,
    "sequence_packing": false,
    "sequence_packing_max_samples": 1000
  },
  "optimizer": {
    "learning_rate": 5e-5,
    "min_learning_rate": null,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "optimizer": "auto",
    "lr_decay_style": "cosine",
    "warmup_steps": 0
  },
  "parallelism": {
    "num_nodes": 1,
    "num_gpus_per_node": 1,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "context_parallel_size": 1,
    "expert_parallel_size": null,
    "sequence_parallel": false
  },
  "output": { "name": "<output-name>", "description": null },
  "integrations": null
}
```

---

## Field reference

### Automodel `training`

| Field | Default | Notes |
|-------|---------|-------|
| `training_type` | `sft` | `distillation` requires `teacher_model` (entity ref) |
| `recipe` | `auto` | `auto` uses the checkpoint head; `sft`, `bi_encoder`, or `cross_encoder` explicitly selects the recipe. Explicit encoder recipes can wrap a causal-LM checkpoint. Unset batch/LR/warmup fields are rewritten for encoder recipes — see **Retrieval recipe defaults** below. |
| `finetuning_type` | `lora` | `all_weights` (full fine-tune), `lora_merged` (merge adapter into base) |
| `lora.rank` | `16` | Higher → more capacity, more VRAM. Typical training range 8–32; **cap at 32** if the adapter will be served with default NIM / vLLM (rank > 32 may not load) |
| `lora.alpha` | `32` | Scaling; common rule of thumb **alpha ≈ 2× rank** |
| `lora.dropout` | `0.0` | LoRA dropout (0.0–1.0) for regularization |
| `lora.merge` | `false` | If true with `lora_merged`, output is full weights not adapter |
| `lora.target_modules` | `null` | e.g. `["q_proj","v_proj"]`; null = platform default targets |
| `lora.exclude_modules` | `null` | Patterns to exclude from LoRA, e.g. `["*.out_proj"]` |
| `lora.use_triton` | `true` | Use the optimized Triton LoRA kernel |
| `max_seq_length` | `2048` | Truncate/pack to this length; lower if OOM |
| `precision` | `null` | `bf16` \| `fp16` \| `fp32` \| `fp8`; null auto-detects from the checkpoint |
| `attn_implementation` | `sdpa` | `sdpa` (PyTorch native) \| `flash_attention_2` \| `eager` |
| `teacher_model` | — | **Model entity ref** (not HF id). Required for distillation; see below |
| `distillation_ratio` | `0.5` | KD blend (0–1) |
| `distillation_temperature` | `1.0` | KD temperature |
| `teacher_precision` | `bf16` | `bf16` \| `fp16` \| `fp32` |
| `offload_teacher` | `false` | Offload teacher weights to CPU |

LoRA block is auto-created when `finetuning_type` is `lora` or `lora_merged`.

### Automodel `schedule`

| Field | Default | Notes |
|-------|---------|-------|
| `epochs` | `1` | Must be **≥ 1**. Full passes over training set |
| `max_steps` | `null` | **Global step cap.** Omit for epoch-based runs |
| `val_check_interval` | `null` | `≤ 1.0` = fraction of epoch; `> 1` = every N steps |
| `seed` | `null` | Reproducibility |

**Gotcha:** Do **not** set `max_steps` with `epochs` for normal training. `max_steps` stops early (e.g. `epochs: 1` + `max_steps: 100` ends at step 100). Use `max_steps` **alone** only for smoke tests.

### Automodel `batch`

| Field | Default | Notes |
|-------|---------|-------|
| `global_batch_size` | `8` (schema) | Encoder recipes rewrite unset values (**256** bi-encoder, **128** cross-encoder). Effective batch across all GPUs; **≥48 GB LoRA tables → `batch-sizing.md`** |
| `micro_batch_size` | `1` (schema) | Encoder recipes rewrite unset values (**8** bi-encoder, **8** cross-encoder). **Per GPU**; same SKILL tables for single- and multi-GPU (TP=1) |
| `sequence_packing` | `false` | Pack short sequences for throughput (needs compatible data) |
| `sequence_packing_max_samples` | `1000` | Samples analyzed to estimate the optimal pack size (only when packing) |

**Validation:** `global_batch_size` must be divisible by `micro_batch_size × data_parallel_size`, where:

`data_parallel_size = (num_nodes × num_gpus_per_node) / (tensor_parallel_size × pipeline_parallel_size × context_parallel_size)`

Example: 1 node, 2 GPUs, TP=1 → DP=2 → GBS must be a multiple of `2 × micro_batch_size`. See **`batch-sizing.md` § Multi-GPU** for data parallel vs tensor parallel.

### Automodel `optimizer`

| Field | Default | Notes |
|-------|---------|-------|
| `learning_rate` | `5e-6` (schema) | Encoder recipes rewrite unset values (`1e-5` bi-encoder, `3e-6` cross-encoder). Skill uses **5e-5** for small LoRA SFT; see tuning below |
| `min_learning_rate` | `null` | Floor for the cosine LR decay; null lets it decay toward 0 |
| `weight_decay` | `0.01` | L2-style regularization |
| `adam_beta1` | `0.9` | Adam optimizer beta1 |
| `adam_beta2` | `0.999` | Adam optimizer beta2 |
| `adam_eps` | `1e-8` | Adam/AdamW epsilon for numerical stability |
| `optimizer` | `auto` | `auto` \| `Adam` \| `AdamW` \| `FusedAdam`; `auto` picks FusedAdam for retrieval recipes, Adam for SFT |
| `lr_decay_style` | `cosine` | `cosine` \| `linear` \| `constant` |
| `warmup_steps` | `0` | Encoder recipes rewrite unset values (`5` bi-encoder, `100` cross-encoder). Linear warmup; try ~10% of total steps for long SFT runs |

### Retrieval recipe defaults

When the resolved recipe is `bi_encoder` or `cross_encoder` (explicit, or `auto` after the checkpoint head is known), unset batch and optimizer fields are filled with Nemotron retrieval values. Fields you set in the job JSON are left alone. `sft` keeps the schema defaults above.

| Field | Schema default | `bi_encoder` | `cross_encoder` |
|-------|----------------|--------------|-----------------|
| `batch.global_batch_size` | `8` | `256` | `128` |
| `batch.micro_batch_size` | `1` | `8` | `8` |
| `optimizer.learning_rate` | `5e-6` | `1e-5` | `3e-6` |
| `optimizer.warmup_steps` | `0` | `5` | `100` |

`optimizer.optimizer: auto` still selects FusedAdam for these recipes and Adam for SFT.

For `bi_encoder`, `micro_batch_size` also sets the in-batch negative pool, so
lowering it costs retrieval quality rather than just speed — see
**`batch-sizing.md` § Retrieval recipes** before overriding it. On 48 GB, `micro`
12 / GBS 384 is the confirmed ceiling; prefer more data-parallel GPUs over a
higher `micro` to widen negatives.

### Retrieval data from Stage 1

Previous: the Data Designer `retrieval-prepare` / `retrieval-run` `artifacts`
fileset, holding `training.jsonl` (`query`, `pos_doc`, `neg_doc`) and `eval_beir/`
at the root. Wrapped `train.json`, corpus parquet, and mining caches live under
`additional/`. Pass that fileset as `dataset.training`.

**Pre-submit:** convert-only Stage 1 leaves `neg_doc: []`. The retrieval collator
samples `train_n_passages - 1` negatives (default 4) and raises
`neg_doc must contain at least 1 document to sample N negatives`. Count
`len(neg_doc)` on a sample of `training.jsonl` (see `nemo-retrieval-recipes`
`references/sdg.md`). Every checked row must contain a non-empty list: a single
empty `neg_doc` can be selected by the collator and fail the run. If any row is
empty, run `retrieval-prepare` with `enable_mining: true` before `automodel
submit`. Do not paper over this by lowering `train_n_passages`.

The base model must be a model entity with a non-null `fileset`. An Inference
Gateway auto-discovered entity with only `api_endpoint` cannot be downloaded for
training.

Next after a successful job: `nemo-retrieval-recipes` or `nemo evaluator
retrieve-eval submit` on the **frozen** `eval_beir` — not CHAT `evaluate`.

```json
{
  "model": "default/nemotron-3-embed-1b",
  "dataset": {"training": "default/retrieval-stage1-artifacts"},
  "training": {
    "recipe": "bi_encoder",
    "training_type": "sft",
    "finetuning_type": "lora_merged"
  },
  "output": {"name": "nemotron-3-embed-1b-tuned"}
}
```

Use `"recipe": "cross_encoder"` and the rerank model entity for ranking. Leave batch/LR unset to take the table above. `finetuning_type` must be `all_weights` or `lora_merged` (unmerged LoRA cannot be exported to ONNX for ranking NIM).

### `training.retrieval`

Dataset, collator, and export knobs for `bi_encoder` / `cross_encoder`. Use `training.retrieval` only; `training.embedding` is rejected.

`retrieval.export` writes ONNX plus the HF checkpoint. `primary` (`onnx` by default, or `hf`) selects the fileset root; the other artifact goes under `alternates/`. `precision` defaults to `fp16` to match typical Hugging Face checkpoints. Embeddings emit pooled `embeddings`; cross-encoders emit `logits`. Set `dimensions: true` for Matryoshka truncation. Unmerged LoRA is not exported.

### `parallelism`

| Field | Default | Notes |
|-------|---------|-------|
| `num_nodes` | `1` | Multi-node distributed jobs |
| `num_gpus_per_node` | `1` | GPUs per node |
| `tensor_parallel_size` | `1` | **> 1** when the model does not fit on one ≥48 GB GPU — see **`batch-sizing.md` § Multi-GPU** |
| `pipeline_parallel_size` | `1` | Pipeline stages |
| `context_parallel_size` | `1` | Long-context sharding |
| `expert_parallel_size` | `null` | MoE only; must divide `data_parallel_size × context_parallel_size` |
| `sequence_parallel` | `false` | Shard activations along the sequence dim (pairs with tensor parallelism) |

**MoE:** If `expert_parallel_size > 1` and multiple GPUs, `tensor_parallel_size` must be **1**.

### Automodel `integrations` (optional)

See **Integrations (all backends)** in `hyperparameters.md`.

---

## Tuning guide (when the user asks)

Apply user overrides to `/tmp/job.json` before submit. For **batch / GPU count / parallelism**, follow **`batch-sizing.md`** (defaults table + § Batch sizing + § Multi-GPU). Below covers **non-batch** fields and defers VRAM/batch symptoms to the skill.

| Symptom / goal | Try first |
|----------------|-----------|
| CUDA OOM | **`batch-sizing.md` tuning loop:** halve `micro_batch_size`, then `global_batch_size`, then `max_seq_length`; use TP > 1 only if the model does not fit one ≥48 GB GPU |
| Slow / low GPU use | **`batch-sizing.md`:** step toward high-util column or double `micro`+GBS until ~35–40 GiB; multi-GPU data parallel if model fits one GPU |
| Underfitting | More `epochs`, slightly higher `learning_rate`, higher LoRA `rank` (≤ 32 for NIM/vLLM deploy) |
| Overfitting | Fewer `epochs`, lower `learning_rate`, higher `weight_decay`, smaller `rank` |
| Quick smoke test | `max_steps` only (e.g. 10–50), **omit or ignore epoch goal**; or `epochs: 1` on tiny slice |
| Reproducibility | Set `schedule.seed` |

### Automodel learning rate (LoRA SFT, starting points)

| Model scale | Suggested `learning_rate` |
|-------------|---------------------------|
| ≤ 3B | `5e-5` – `1e-4` |
| 3B – 8B | `2e-5` – `5e-5` |
| > 8B | `1e-5` – `2e-5` |

Schema default is `5e-6` (conservative). Fixtures: `qwen3_0.6b_sft_lora.json` uses `5e-5`; `minimal_sft_lora.json` uses `5e-6`.

### Automodel LoRA rank / alpha

**Deployment cap:** Default **NIM** and **vLLM** LoRA serving paths support rank **≤ 32**. Use `rank` 32 (not higher) when the fine-tuned adapter will be deployed for inference on those stacks unless the user confirms a higher rank is supported.

| Use case | `rank` | `alpha` |
|----------|--------|---------|
| Default / balanced | 16 | 32 |
| Low VRAM / light touch | 8 | 16 |
| More capacity (inference-safe max) | 32 | 64 |

### Epochs vs dataset size

One epoch = one full pass over `train.jsonl`. Steps per epoch ≈ `train_samples / global_batch_size` (e.g. ~10k samples, GBS 64 → ~153 steps). Plan poll time from the **GBS you chose in `batch-sizing.md`**, not the unknown-VRAM default (GBS 4).

---

## Presets (non-batch fields)

Use **`batch-sizing.md` § Batch sizing** and **§ Multi-GPU** for `batch` and `parallelism` on ≥48 GB GPUs. Presets below only override schedule / training / optimizer.

**Smoke test (step-capped)**

```json
"schedule": { "epochs": 1, "max_steps": 50 }
```

**Higher-quality LoRA (more VRAM/time)**

```json
"training": { "lora": { "rank": 32, "alpha": 64 }, "max_seq_length": 2048 },
"schedule": { "epochs": 3 },
"optimizer": { "learning_rate": 2e-5, "warmup_steps": 100 }
```

Pair with batch rows from **`batch-sizing.md`** (e.g. ≤4B default `micro` 32 / GBS 128, not `micro` 1 / GBS 4).

---

## Distillation (`training_type: "distillation"`)

Use only when the user requests KD/distillation. **`model`** is the **student** entity; **`teacher_model`** is a separate **teacher** entity in the same workspace (unless qualified as `other-ws/name`).

### Teacher model entity

`teacher_model` must be a registered **model entity ref**, same shape as `model`:

| Form | Example |
|------|---------|
| Same workspace | `default/llama-3.2-3b-instruct` |
| Explicit workspace | `default/<teacher-entity>` |

It is **not** a Hugging Face repo id. Register the teacher like the student before submit:

```bash
TEACHER_WEIGHTS=llama-3.2-3b-instruct   # fileset name
TEACHER_ENTITY=llama-3.2-3b-instruct    # entity name
TEACHER_HF=meta-llama/Llama-3.2-3B-Instruct

nemo files filesets create "$TEACHER_WEIGHTS" --workspace default --purpose model --exist-ok \
  --storage '{"type":"huggingface","repo_id":"'"$TEACHER_HF"'","repo_type":"model","revision":"main"}'

nemo models create "$TEACHER_ENTITY" --workspace default --exist-ok \
  --input-data '{"name":"'"$TEACHER_ENTITY"'","fileset":"default/'"$TEACHER_WEIGHTS"'","custom_fields":{"hf_model_id":"'"$TEACHER_HF"'"}}'
```

Verify: `nemo models get <teacher-entity> --workspace default`. Reuse an existing entity with `nemo models list` when present.

**Compatibility:** Student and teacher must share the **same vocabulary / tokenizer family** (compiler loads both for KD). Mismatched tokenizers fail at runtime. Prefer a larger instruct model as teacher and a smaller base/chat model as student in the same family when possible.

**VRAM:** Set `offload_teacher: true` if the job OOMs loading student + teacher; `teacher_precision: "bf16"` is the default.

### Job JSON

```json
{
  "model": "default/<student-entity>",
  "dataset": { "training": "default/<dataset-fileset>" },
  "training": {
    "training_type": "distillation",
    "finetuning_type": "lora",
    "teacher_model": "default/<teacher-entity>",
    "distillation_ratio": 0.5,
    "distillation_temperature": 1.0,
    "teacher_precision": "bf16",
    "offload_teacher": false,
    "max_seq_length": 2048
  },
  "schedule": { "epochs": 1 },
  "batch": { "global_batch_size": 64, "micro_batch_size": 16 },
  "optimizer": { "learning_rate": 8e-5 },
  "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1, "tensor_parallel_size": 1 },
  "output": { "name": "<output-name>" }
}
```

(`batch` / `parallelism` example uses an 8B-scale row from **`batch-sizing.md`**; adjust for student size.)

| Field | Meaning |
|-------|---------|
| `distillation_ratio` | Blend of KD vs CE loss (`0` = CE only, `1` = KD only) |
| `distillation_temperature` | Softmax temperature for teacher logits |
| `offload_teacher` | CPU-offload frozen teacher weights to save GPU memory |

