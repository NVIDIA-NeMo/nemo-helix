<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Batch sizing

VRAM tables, multi-GPU rules, and throughput tuning for **automodel** and **unsloth** on ≥48 GB GPUs. Field glossary and full JSON templates live in `hyperparameters.md`; the skill workflow lives in `SKILL.md`.

## Batch sizing — automodel (≥48 GB VRAM)

Tables, multi-GPU rules, and the tuning loop below are **automodel-specific** (fields `global_batch_size` / `micro_batch_size` / `tensor_parallel_size` / `num_gpus_per_node`). For unsloth see **Batch sizing — unsloth** further down.

Assume **one GPU with at least 48 GB** (e.g. RTX 5880 / A6000 / L40), `parallelism` = 1 node × 1 GPU, `tensor_parallel_size` 1, bf16, `training_type` `sft`, LoRA **rank 16** unless the user asks otherwise.

**How to size**

1. Read **model size** from the entity (`nemo models get`) or HF card (parameter count).
2. Pick **`finetuning_type`**: `lora` (adapter only, default) vs `all_weights` (full SFT — much heavier).
3. Set **`max_seq_length`** (2048 is the skill default; shorter seq → more batch headroom).
4. Set **`micro_batch_size`** first (drives peak VRAM), then **`global_batch_size`** as a multiple of `micro_batch_size` (gradient accumulation when GBS > micro).

**Constraint:** `global_batch_size` must be divisible by `micro_batch_size × data_parallel_size`, where `data_parallel_size = (num_nodes × num_gpus_per_node) / (tensor_parallel_size × pipeline_parallel_size × context_parallel_size)` (1 for a single-GPU job).

### LoRA (`finetuning_type: lora`) — `max_seq_length` 2048

**VRAM does not scale linearly with `micro_batch_size`.** LoRA loads the full base weights once; activation memory grows slowly. On 48 GB, **`micro_batch_size` must decrease as model size grows** (smaller models always ≥ larger models in the table). Use **`global_batch_size` ≈ 4 × `micro_batch_size`**.

**Default batch** — start here for a reliable full epoch. **High utilization** — optional; double from default (or ramp in steps) to reach **~35–40 GiB**. Halve both if OOM (exit **137**) or training crashes (exit **1**).

| Model params | Default `micro` | Default GBS | `learning_rate` | High-util `micro` | High-util GBS |
|--------------|------------------:|------------:|----------------:|------------------:|--------------:|
| ≤4B | 32 | 128 | `1e-4` | 64 | 256 |
| 4B–8B | 24 | 96 | `8e-5` | 48 | 192 |
| 8B–14B | 16 | 64 | `8e-5` | 24 | 96 |
| >14B | 8 | 32 | `5e-5` | 16 | 64 |

Validated (`commonsense_qa` @ 2048, 48 GB, one job per GPU): **Qwen3-1.7B** — `micro` 16 / GBS 64 ~8 min; defaults above leave headroom to ramp. **Qwen3-8B** — `micro` 2–4 ≈16–18.5 GiB (under-filled); **`micro` 16 / GBS 64** stable default (~153 steps/epoch); high-util **`micro` 24 / GBS 96** (32 / 128 hit ~40 GiB but failed mid-epoch with exit 1).

### Multi-GPU (same node)

Pick the path by whether the **base model fits in ~48 GB on one GPU** (LoRA or full SFT):

| Situation | `tensor_parallel_size` | Goal |
|-----------|------------------------:|------|
| Model **fits** on one ≥48 GB GPU | **1** | **Data parallel** — more GPUs = faster training; keep `micro` per GPU, scale `global_batch_size` |
| Model **does not fit** on one ≥48 GB GPU | **> 1** (e.g. 2 on a 2-GPU node) | **Tensor parallel** — shard layers across GPUs so the model fits; lower `micro` / GBS vs single-GPU tables |

**Data parallel (TP = 1)** — default for Qwen3-8B LoRA and similar on 48 GB cards:

| Rule | Detail |
|------|--------|
| `micro_batch_size` | **Per GPU** — same as a stable single-GPU run |
| `global_batch_size` | ≈ **single-GPU GBS × `num_gpus_per_node`**; step count ≈ `samples / GBS` |
| Divisibility | `global_batch_size` ÷ **`micro_batch_size × num_gpus_per_node`** must be an integer |
| Scheduling | **One job** owns all GPUs; no overlapping 1-GPU and multi-GPU jobs |

```json
"parallelism": { "num_nodes": 1, "num_gpus_per_node": 2, "tensor_parallel_size": 1 },
"batch": { "global_batch_size": 128, "micro_batch_size": 16 }
```

**Tensor parallel (TP > 1)** — when weights + activations OOM on a single ≥48 GB GPU (large full SFT, very long `max_seq_length`, or models above the LoRA sizing table without fitting):

- Set **`num_gpus_per_node`** and **`tensor_parallel_size`** so **`num_gpus_per_node` is divisible by `tensor_parallel_size`** (e.g. 2 GPUs → `tensor_parallel_size: 2`, or 4 GPUs → TP 2 or 4).
- **`data_parallel_size`** = `(num_nodes × num_gpus_per_node) / (tensor_parallel_size × pipeline_parallel_size × context_parallel_size)` — use this in the GBS divisibility rule instead of raw GPU count.
- Start with **lower `micro_batch_size`** than the single-GPU table; increase only if VRAM allows. MoE models: if `expert_parallel_size > 1`, **`tensor_parallel_size` must be 1**.

```json
"parallelism": { "num_nodes": 1, "num_gpus_per_node": 2, "tensor_parallel_size": 2 },
"batch": { "global_batch_size": 8, "micro_batch_size": 1 }
```

`execution_profile` is usually still **`"gpu"`** — confirm with `nemo jobs list-execution-profiles -f json`.

**Example — Qwen3-8B LoRA, 2× 48 GB (fits one GPU):** single-GPU **micro 16 / GBS 64** → 2-GPU data parallel **micro 16 / GBS 128**, `learning_rate` `8e-5`.

### Full-weight SFT (`finetuning_type: all_weights`) — `max_seq_length` 2048

| Model params | `micro_batch_size` | `global_batch_size` | `learning_rate` |
|--------------|-------------------:|--------------------:|----------------:|
| ≤2B | 2 | 8 | `2e-5` |
| 2B–4B | 1 | 4 | `1e-5` |
| 4B–8B | 1 | 2 | `5e-6` |
| >8B | 1 | 1 | lower LR or use TP / shorter seq |

Output type is **model** (full checkpoint), not adapter. Expect much longer runs than LoRA at the same batch. **Inference:** deploy `default/<output.name>` as a new model entity — full SFT does not hot-reload onto the base model's LoRA deployment.

### Retrieval recipes (`bi_encoder` / `cross_encoder`)

The tables above assume one sequence per sample. Retrieval samples cost more:
`bi_encoder` encodes the query plus `retrieval.train_n_passages` passages (default
5, so 6 sequences per sample) at `retrieval.query_max_length` /
`retrieval.passage_max_length` (512 each) rather than `max_seq_length`.

| Recipe | Default `micro` | Default GBS | `learning_rate` | Sequences per sample |
|--------|----------------:|------------:|----------------:|---------------------:|
| `bi_encoder` | 8 | 256 | `1e-5` | `1 + train_n_passages` |
| `cross_encoder` | 8 | 128 | `3e-6` | `train_n_passages` |

These are applied automatically for unset fields — do not copy them into the job
JSON unless you intend to override.

Measured on a 1B embedding model, LoRA rank 16, `train_n_passages` 5, 512/512
query/passage, one 48 GB (47.37 GiB usable) GPU:

| `micro` | Peak | Verdict |
|--------:|-----:|---------|
| 1 | 4.55 GiB | Wastes the card, and no in-batch negatives |
| 8 | 28.3 GiB (range 14.8–28.3 over 20 steps) | **Default.** Real headroom for long batches |
| 12 | 41.5 GiB (~87%) | **Maximum confirmed on 48 GB.** Allocator hits cache-flush retries |
| 16, 32 | OOM | — |

`micro` 12 pairs with **GBS 384**, not 256: `global_batch_size` must be divisible
by `micro_batch_size × data_parallel_size`. It runs, but at 87% the caching
allocator logs `memory allocation failed with OOM` warnings and frees cached
blocks to continue — survivable, slower, and one long batch away from a hard OOM.
Use it only when you want the wider in-batch negative pool and accept that risk.

On 80 GB cards, `micro` 24 / GBS 384 is the analogous step up (untested here).

Retrieval activations are ~6× a plain SFT sample at the same sequence length, so
do **not** carry the LoRA tables above over to these recipes.

Peak is also data-dependent: the collator pads to the longest sequence in the
batch, so a batch of long passages spikes well above the average step. Leave
real headroom rather than tuning to the first step that fits, and reach for
`retrieval.do_gradient_checkpointing: true` (default `false`) before giving up
`micro` — it trades step time for a large cut in activation memory, which is the
right trade when `micro` is carrying in-batch-negative quality.

**`micro_batch_size` is a quality knob for `bi_encoder`, not just a speed one.**
The contrastive loss draws in-batch negatives from the local batch and gathers
across data-parallel ranks, but **not** across gradient-accumulation steps. A
`micro_batch_size` of 1 therefore trains against the mined negatives alone with
no in-batch negatives at all, which is a much weaker signal — raising GBS does not
compensate. When you hit OOM on a bi-encoder, prefer lowering
`retrieval.train_n_passages` or the passage length before dropping `micro` below
about 8. `cross_encoder` has no such coupling.

The gather is across **data-parallel ranks**, so `micro` 8 on two GPUs gives the
same 16-query negative pool as `micro` 16 on one — without either card paying the
`micro` 16 activation bill. Scaling GPUs is the cheap way to widen negatives;
`micro` is the expensive one.

### `max_seq_length` scaling

Scale **`micro_batch_size`** from the 2048 tables (round down, minimum 1):

| `max_seq_length` | Multiply `micro_batch_size` by |
|------------------|-------------------------------:|
| 512 | 4× |
| 1024 | 2× |
| 2048 | 1× (tables above) |
| 4096 | 0.5× |

Then set `global_batch_size` to a multiple of the new `micro_batch_size` (often keep the same ratio as the table, e.g. GBS = 4 × micro for LoRA).

### LoRA rank

Higher rank uses more VRAM. If OOM at rank 16, drop to rank 8 before lowering batch; if headroom remains, rank 32 is fine for training (deploy rank ≤32 on default NIM/vLLM).

### Tuning loop

| Symptom | Action |
|---------|--------|
| CUDA OOM | Halve `micro_batch_size`, then `global_batch_size`, then `max_seq_length` |
| Slow / low GPU memory use | Step up toward the **high-util** column (or double default `micro`+GBS); stop at ~35–40 GiB or when training fails, then use **default** for the retry |
| User wants max throughput | Raise `micro_batch_size` first; keep GBS ≈ 4× micro — avoid `micro_batch_size` 1 with huge GBS |

Field glossary, distillation/KD, and schema pointers: `references/hyperparameters.md` (batch/multi-GPU → **this file**, not hyperparameters).

## Batch sizing — rl / GRPO

GRPO has an extra axis the SFT backends do not: every prompt is rolled out
`num_generations_per_prompt` times *before* any training happens, so the generation cost —
not the gradient step — usually sets the ceiling.

### Derive the numbers, do not copy them

Work in this order. Each step depends only on facts you can read off the model, the dataset
and the cluster.

| # | Decide | From | Rule |
|---|---|---|---|
| 1 | `parallelism` | GPUs available; model size | Start 1 node × N GPUs, `tensor_parallel_size` 1. Raise TP only when weights do not fit — TP is a memory tool, not a speed one |
| 2 | `policy_backend` | `finetuning_type` | Leave unset to set the default `lora` → `automodel` , `all_weights` → `dtensor`. |
| 3 | `max_seq_length` | prompt + expected answer length in the dataset | Measure it: tokenize a sample and take ~p99, round up. Over-provisioning here costs generation time on every rollout |
| 4 | `num_generations_per_prompt` | how noisy the reward is | `8` is the standard start. Below `4` the group advantage is too noisy to learn from; above `16` you pay linearly in rollouts for diminishing signal |
| 5 | `batch_size` | steps 1 and 4 | Must satisfy **both** divisibility rules below |
| 6 | `micro_batch_size` | VRAM | Start `1`. This is the OOM knob, and it does not change the math of step 5 |

### The two rules the compiler enforces

Both are checked at submit (`validate_for_training`), so getting them wrong costs a
round-trip, not a job:

```text
data_parallel_size = (num_nodes × num_gpus_per_node)
                     ÷ (tensor_parallel × pipeline_parallel × context_parallel × expert_parallel)

1.  batch_size % (micro_batch_size × data_parallel_size) == 0
2.  (num_prompts_per_step × num_generations_per_prompt) % batch_size == 0
```

`num_prompts_per_step` defaults to `batch_size // num_generations_per_prompt`, and that
**floor division is the trap**: `batch_size 32` with `num_generations_per_prompt 5` derives
`6`, giving a rollout batch of `30` against a train batch of `32`. Rule 2 catches it, but the
fix is to pick a `num_generations_per_prompt` that *divides* `batch_size`.

Safe pairings on 1 node:

| GPUs | `batch_size` | `num_generations_per_prompt` | Rollouts per step |
|---|---|---|---|
| 1 | 8 | 4 | 32 |
| 1 | 32 | 8 | 256 |
| 2 | 32 | 8 | 256 |
| 4 | 32 | 8 | 256 |
| 8 | 64 | 8 | 512 |

### Memory, in the order to reach for it

GRPO holds the policy *and* a vLLM engine on the same GPUs, so the split matters as much as
the batch:

1. `activation_checkpointing: true` — first, always.
2. `vllm_gpu_memory_utilization` (default `0.5`) — the fraction vLLM reserves for weights plus
   KV cache. Raise toward `0.7` for a large model that cannot load; lower it when training
   OOMs but generation is fine.
3. `micro_batch_size` down to `1` if it is not already.
4. `max_seq_length` down — cheapest real win, since it bounds every rollout.
5. `tensor_parallel_size` up — last, and remember `lora.use_triton` must then be unset or `false`.

`batching_strategy` defaults to `dynamic`, which already packs short rollouts together; leave
it unless the model is VLM/multimodal or context-parallel, where `sequence_packing` is rejected.

### Dataset size

Rollouts per epoch = `rows × num_generations_per_prompt`. A 17k-row dataset at `8` generations
is 136k rollouts per epoch — hours on one GPU. For a smoke run, cap rows at prep time rather
than setting `max_steps`, so the epoch means something.

Validation runs **one** rollout per `validation.jsonl` row (there is no validation counterpart
to `num_generations_per_prompt`), so validation cost is just the row count.

## Batch sizing — unsloth (single GPU)

Unsloth is single-GPU by design. The effective batch is the **product** of two fields, not a global/micro split:

```text
effective_batch = batch.per_device_train_batch_size × batch.gradient_accumulation_steps
```

There is no `parallelism` block, no TP / PP / DP, no GBS divisibility math. Multi-GPU sharding → switch to automodel.

**Field mapping from the automodel tables above:**

| Automodel field | Unsloth analogue | Notes |
|-----------------|------------------|-------|
| `micro_batch_size` | `batch.per_device_train_batch_size` | Drives peak VRAM. |
| `global_batch_size` | `batch.per_device_train_batch_size × batch.gradient_accumulation_steps` | Set `gradient_accumulation_steps` so the product matches the GBS you'd pick on automodel. |
| `parallelism.num_gpus_per_node` | n/a — single GPU | Use `hardware.gpus: "0"` to pin to one GPU. |
| `tensor_parallel_size` | n/a | If the model doesn't fit on one GPU → use automodel. |

**Starting points (LoRA, `max_seq_length` 2048, one ≥48 GB GPU):**

| Model params | `per_device_train_batch_size` | `gradient_accumulation_steps` | Effective batch | `learning_rate` |
|--------------|------------------------------:|------------------------------:|----------------:|----------------:|
| ≤4B | 8 | 16 | 128 | `1e-4` |
| 4B–8B | 4 | 24 | 96 | `8e-5` |
| 8B–14B | 2 | 32 | 64 | `8e-5` |
| >14B | 1 | 32 | 32 | `5e-5` |

`load_in_4bit: true` (default) keeps base weights in 4-bit, which is what makes the "smaller per-device batch on bigger models" rule milder than vanilla HF. If you raise `per_device_train_batch_size` and hit OOM (exit 137) or training crashes (exit 1), halve `per_device_train_batch_size` first and double `gradient_accumulation_steps` to keep the effective batch the same.

**Save method.** Default `output.save_method: "lora"` (adapter only — small, fast, hot-reloads on LoRA-enabled deployments). Use `"merged_16bit"` if the user wants a full-weight checkpoint to deploy as a standalone model entity; `"merged_4bit"` only when storage is tight (lossy). Merged methods require `training.finetuning_type: "lora"`. Merged and full SFT outputs must be **deployed for inference** — they do not hot-reload onto the base adapter deployment.

**Tuning loop (unsloth):**

| Symptom | Action |
|---------|--------|
| CUDA OOM | Halve `per_device_train_batch_size` (keep effective batch via `gradient_accumulation_steps`); then lower `model.max_seq_length`; then drop `lora.rank` to 8 |
| Missing `nmp-unsloth-training` image | Build/pull the Unsloth container image — see `references/troubleshooting.md` and `docker/unsloth/README.md` |
| `Unsloth training requires platform.runtime: docker` (platform not using the Docker executor) | Start platform with `platform.runtime: docker` and a GPU execution profile; training runs in containers on that host's Docker daemon |
| Loss not moving | Raise `learning_rate` one step (e.g. `5e-5` → `1e-4`); confirm `apply_chat_template` matches the data shape; check the LoRA `target_modules` covers the right layers (defaults are Unsloth's 7-module set) |
