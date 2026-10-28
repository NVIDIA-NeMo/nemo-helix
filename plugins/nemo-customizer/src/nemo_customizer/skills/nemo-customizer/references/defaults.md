<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Defaults per backend

Default values per backend when the user gives no specifics. The shared defaults are in `SKILL.md` § **Defaults**.

Automodel-specific:

| Field | Value |
|-------|-------|
| Parallelism | 1 node, 1 GPU, TP=1 |
| Batch, optimizer | `SKILL.md` § **Defaults** |

Unsloth-specific:

| Field | Value |
|-------|-------|
| Hardware | `hardware.gpus` `"0"`, `hardware.precision` `bf16` (selection only, single GPU) |
| Model load | `load_in_4bit: true`, `dtype: "auto"` |
| Batch | `batch.per_device_train_batch_size` 2, `batch.gradient_accumulation_steps` 4 (effective batch 8; see `references/batch-sizing.md` for ≥48 GB ramp) |
| Optimizer | `optim` `adamw_8bit`; `learning_rate` unset |
| Output | `save_method: "lora"` (adapter-only) unless user asks for merged checkpoint |
| Gradient checkpointing | `training.use_gradient_checkpointing: "unsloth"` |

rl-specific (DPO):

| Field | Value |
|-------|-------|
| Training | DPO, full-weight (`type: "dpo"`; no LoRA) |
| Model (if user gives none) | `Qwen/Qwen3-0.6B` |
| Dataset (if user gives none) | `nvidia/HelpSteer3` (preference subset; uploaded raw — see `references/dataset-formats.md` § NeMo-RL) |
| Schedule (if user gives none) | small demo run: `max_steps` 20, which finishes fast and proves the pipeline. A real run sets `epochs`. |
| Parallelism | 1 node, 1 GPU (`parallelism.num_nodes`/`num_gpus_per_node`) |
| Batch | `batch_size` 32, `micro_batch_size` 1 |
| Optimizer | AdamW + cosine; `learning_rate` unset |
| DPO | `ref_policy_kl_penalty` (β) 0.05, `sft_loss_weight` 0.0 |
| Max sequence length | 1024 |
| Output | full-weight model entity (`output.name`); no adapter |

When the user asks for a DPO job **without specifics**, default to the above: a
20-step run of `Qwen/Qwen3-0.6B` on `nvidia/HelpSteer3` — small enough to finish
quickly and confirm the pipeline end-to-end.

rl-specific (GRPO) — values below are a **real 8×H100 LoRA configuration**, not a smoke test:

| Field | Value |
|-------|-------|
| Training | GRPO (`type: "grpo"`), **`finetuning_type: "lora"`** (`references/fast-path-rl.md` says when to use `all_weights`). `lora_merged` is rejected. |
| Model (if user gives none) | `Qwen/Qwen3-8B-Base` → entity `qwen3-8b-base` |
| Environment (if user gives none) | **Stop and ask**: `SKILL.md` § **GRPO** |
| LoRA | `rank` 128, `alpha` 256, `dropout` 0.0, all linear layers (omit `target_modules`/`exclude_modules`). A rank-128 adapter loads only on a deployment with `--max-lora-rank` ≥ 128; default deployments stop at 16 (`post-training-eval.md` § **LoRA rank limit**). Omit `use_triton`: the compiler picks `true` at TP 1, `false` above |
| Parallelism | 1 node × 8 GPUs, `tensor_parallel_size` 1, `execution_profile` `"h100"` |
| Batch | `batch_size` 512, `micro_batch_size` 1, `num_prompts_per_step` 32, `num_generations_per_prompt` 16 |
| Dynamic sampling | `use_dynamic_sampling` true, `batch_multiplier` 2.0, `dynamic_sampling_max_gen_batches` 10 |
| Sequence | `max_seq_length` 3072, `max_new_tokens` 2048, `temperature` 1.0, `activation_checkpointing` true |
| Optimizer | `learning_rate` unset, `min_learning_rate` 9e-6, `adam_eps` 1e-8, `warmup_steps` 10, `weight_decay` 0.01 |
| Schedule | `epochs` 1 (a full pass). `max_steps` 200 only if the user explicitly wants a bounded first run |
| GRPO loss | `ref_policy_kl_penalty` 0.0, `ratio_clip_min`/`max`/`c` 0.2 / 0.28 / 10.0, `normalize_rewards` true, `use_leave_one_out_baseline` true, `max_rollout_turns` 1 |
| Importance sampling | `truncated_importance_sampling_type` `"tis"`, ratio 2.0, `use_importance_sampling_correction` true |
| Rollout engine | `vllm_tensor_parallel_size` 8 (≤ `num_gpus_per_node`), `vllm_gpu_memory_utilization` 0.6 |
| Validation | `val_at_start` + `val_at_end` true, `val_check_interval` 25, `keep_top_k` 1, `seed` 42 |
| Output | `output.name` only — the type is **inferred** (`lora` → HF PEFT adapter entity; `all_weights` → model entity) |

**These target 8 GPUs.** On different hardware, rescale rather than copy: keep `batch_size % (micro_batch_size × data_parallel_size) == 0` and `(num_prompts_per_step × num_generations_per_prompt) % batch_size == 0`, cap `vllm_tensor_parallel_size` at `num_gpus_per_node`, and drop `execution_profile` unless the platform defines it. Derivation: `references/batch-sizing.md` § **Batch sizing — rl / GRPO**.
