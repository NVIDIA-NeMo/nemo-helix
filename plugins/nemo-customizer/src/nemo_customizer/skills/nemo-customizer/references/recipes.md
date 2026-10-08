<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Reference recipes

Before you write job JSON for a named model, start from a benchmarked recipe in the training library and translate it into the job schema. Recipes carry tested values for learning rate, LoRA rank/alpha, batch size, sequence length, and parallel layout for each model. Use the skill defaults only when no recipe matches.

| Backend | Library | Ref (matches the training image) | Applicable recipes |
|---|---|---|---|
| `automodel` | [Automodel](https://github.com/NVIDIA-NeMo/Automodel/tree/r0.6.0/examples) | `r0.6.0` | `examples/llm_finetune/<family>/`, `examples/llm_kd/` |
| `rl` | [NeMo-RL](https://github.com/NVIDIA-NeMo/RL/tree/r0.8.0/examples/configs) | `r0.8.0` | `examples/configs/recipes/llm/*-fsdp2*`, `*-automodel*`, and the base configs they extend. Customizer trains on the DTensor/Automodel policy, so `*-megatron*` recipes do not apply. |

## Workflow

1. Find a recipe for the model family, method, and GPU count. File names encode the layout: `2n8g` = 2 nodes × 8 GPUs, `ep4` = expert parallel 4, `peft`/`lora` = LoRA.
2. Open the YAML. NeMo-RL recipes set only overrides: follow `defaults:` up to the base config (`dpo.yaml`, `grpo_math_1B.yaml`, …) to get the full value set.
3. Map the fields with the tables below. Confirm every field with `nemo customization <plugin> explain`.
4. Rescale for the user's GPUs. Keep the per-GPU micro batch and the parallel layout, then pick a global batch that satisfies the divisibility rules in `batch-sizing.md`.
5. Do **not** copy `max_steps`, `ci:`, `val_every_steps`/`ckpt_every_steps`, or dataset paths. These are CI smoke settings. Train on `epochs`.

## Nemotron

Nemotron 3.5 Lightning 30B-A3B and Nemotron 3 Nano 30B-A3B share the `NemotronH` hybrid Mamba-2/MoE architecture (128 experts, 6 active), so Nano v3 recipes apply to Lightning. Lightning also ships an MTP head (`training.mtp`).

| Model | Method | Recipe |
|---|---|---|
| Nemotron 3.5 Lightning / 3 Nano 30B-A3B | LoRA, 4 GPUs, EP 4 | Automodel [`customizer_nemotron_nano_peft.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/customizer_nemotron_nano_peft.yaml), [`nemotron_nano_v3_hellaswag_peft.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml) |
| | LoRA + sequence packing | Automodel [`customizer_nemotron_nano_peft_packing.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/customizer_nemotron_nano_peft_packing.yaml) |
| | LoRA, 1 GPU | Automodel [`nemotron_nano_v3_singlegpu_lora.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml) |
| | Full SFT, 8 GPUs, EP 8 | Automodel [`customizer_nemotron_nano_full_sft.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/customizer_nemotron_nano_full_sft.yaml), [`customizer_nemotron_nano_full_sft_chat.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/customizer_nemotron_nano_full_sft_chat.yaml) |
| | GRPO (DAPO), Lightning | NeMo-RL [`dapo-nanov3.5-30BA3B-4n8g-automodel.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/r0.8.0/examples/configs/recipes/llm/dapo-nanov3.5-30BA3B-4n8g-automodel.yaml) |
| | GRPO LoRA / full weight | NeMo-RL [`grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/r0.8.0/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml), [`grpo-nanov3-30BA3B-1n8g-fsdp2.v2.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/r0.8.0/examples/configs/recipes/llm/grpo-nanov3-30BA3B-1n8g-fsdp2.v2.yaml) |
| | DPO | NeMo-RL [`dpo-nanov3-30B3AB-1n8g-fsdp8ep8-automodel.v2.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/r0.8.0/examples/configs/recipes/llm/dpo-nanov3-30B3AB-1n8g-fsdp8ep8-automodel.v2.yaml). Only LR, β, batch, and length transfer, because customizer DPO has no `expert_parallel_size`. |
| Nemotron 3 Super 120B-A12B | LoRA / full SFT | Automodel [`nemotron_super_v3_hellaswag_peft.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/nemotron_super_v3_hellaswag_peft.yaml), [`nemotron_super_v3_hellaswag.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/nemotron_super_v3_hellaswag.yaml) |
| | GRPO | NeMo-RL [`grpo-nemotron3-super-120BA12B-16n8g-automodel-ep8.v2.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/r0.8.0/examples/configs/recipes/llm/grpo-nemotron3-super-120BA12B-16n8g-automodel-ep8.v2.yaml) |
| Nemotron 3 Ultra | LoRA | Automodel [`nemotron_ultra_v3_hellaswag_peft.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/nemotron_ultra_v3_hellaswag_peft.yaml) |
| Nemotron Nano 9B v2 | LoRA / full SFT | Automodel [`nemotron_nano_9b_squad_peft.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/nemotron_nano_9b_squad_peft.yaml), [`nemotron_nano_9b_squad.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron/nemotron_nano_9b_squad.yaml) |
| Nemotron Nano 12B v2 | GRPO | NeMo-RL [`grpo-nano-v2-12b-2n8g-fsdp2tp1.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/r0.8.0/examples/configs/recipes/llm/grpo-nano-v2-12b-2n8g-fsdp2tp1.yaml) |
| Nemotron Flash 1B | LoRA / full SFT | Automodel [`nemotron_flash_1b_squad_peft.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad_peft.yaml), [`nemotron_flash_1b_squad.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad.yaml) |

All of [`examples/llm_finetune/nemotron/`](https://github.com/NVIDIA-NeMo/Automodel/tree/r0.6.0/examples/llm_finetune/nemotron) also covers Llama-Nemotron 8B / 49B.

**Nemotron gotchas**

- **Hopper or Blackwell only.** Nemotron customization is supported on NVIDIA Hopper or Blackwell era GPUs (H100, H200, B100, B200). Check `nemo jobs list-execution-profiles -f json` for a Hopper or Blackwell profile before you submit, and set `training.execution_profile` if the default GPU profile is neither. If the platform has no such GPUs, tell the user rather than submitting.
- Train the **BF16** checkpoint. Automodel `r0.6.0` dequantizes FP8 and GPT-OSS MXFP4 base checkpoints, but not ModelOpt NVFP4 (`…-NVFP4`). Use NVFP4 for inference only.
- Set `lora.exclude_modules: ["*.out_proj"]`. Mamba `out_proj` runs through a fused kernel that LoRA cannot wrap.
- MoE layout: `parallelism.expert_parallel_size` equal to the GPU count, `tensor_parallel_size: 1` (`hyperparameters-automodel.md` § `parallelism`).
- Lightning and Nano v3 default to reasoning mode. For short-answer eval, send `"chat_template_kwargs": {"enable_thinking": false}`.

## Other families

| Need | Where |
|---|---|
| Any family on automodel | [`examples/llm_finetune/<family>/`](https://github.com/NVIDIA-NeMo/Automodel/tree/r0.6.0/examples/llm_finetune), e.g. [`qwen/qwen3_0p6b_hellaswag.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml), [`qwen/qwen3_0p6b_hellaswag_peft.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/r0.6.0/examples/llm_finetune/qwen/qwen3_0p6b_hellaswag_peft.yaml) |
| Knowledge distillation | Automodel [`examples/llm_kd/`](https://github.com/NVIDIA-NeMo/Automodel/tree/r0.6.0/examples/llm_kd). `kd_ratio` → `distillation_ratio`, `kd_loss_fn.temperature` → `distillation_temperature` |
| DPO, small dense models | NeMo-RL base [`dpo.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/r0.8.0/examples/configs/dpo.yaml) |
| GRPO, small dense models | NeMo-RL [`grpo-qwen3-8B-base-1n8g-fsdp2-lora.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/r0.8.0/examples/configs/recipes/llm/grpo-qwen3-8B-base-1n8g-fsdp2-lora.yaml) (the skill's GRPO default), base [`grpo_math_1B.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/r0.8.0/examples/configs/grpo_math_1B.yaml) |
| Embedding / rerank | Automodel [`examples/retrieval/`](https://github.com/NVIDIA-NeMo/Automodel/tree/r0.6.0/examples/retrieval) — route through `nemo-retrieval-recipes` |

## Field mapping — Automodel YAML → `automodel` job JSON

| Recipe YAML | Job JSON |
|---|---|
| `model.pretrained_model_name_or_path` | HF repo for the model fileset; `model` is the entity ref |
| `peft.dim` / `alpha` / `dropout` | `training.lora.rank` / `alpha` / `dropout` |
| `peft.exclude_modules` / `target_modules` / `use_triton` / `use_memory_efficient_lora` | `training.lora.*` (same names) |
| no `peft:` block | `training.finetuning_type: "all_weights"` |
| `step_scheduler.global_batch_size` / `local_batch_size` | `batch.global_batch_size` / `micro_batch_size` |
| `step_scheduler.num_epochs` | `schedule.epochs` |
| `dataset.seq_length` | `training.max_seq_length` |
| `packed_sequence.packed_sequence_size` (> 0) | `batch.sequence_packing: true` + `batch.packed_sequence_size` |
| `optimizer.lr` / `weight_decay` / `eps` | `optimizer.learning_rate` / `weight_decay` / `adam_eps` |
| `optimizer._target_` (`torch.optim.Adam`, …) | `optimizer.optimizer` (`Adam` / `AdamW` / `FusedAdam`) |
| `lr_scheduler.lr_decay_style` / `min_lr` / `lr_warmup_steps` | `optimizer.lr_decay_style` / `min_learning_rate` / `warmup_steps` |
| `distributed.tp_size` / `pp_size` / `cp_size` / `ep_size` / `sequence_parallel` | `parallelism.tensor_parallel_size` / `pipeline_parallel_size` / `context_parallel_size` / `expert_parallel_size` / `sequence_parallel` |
| `ci.nproc_per_node` (or GPUs implied by the layout) | `parallelism.num_gpus_per_node` |
| `distributed.activation_checkpointing` or `parallelizer.activation_checkpointing` | `training.activation_checkpointing` |
| `model.backend.{attn,linear,rms_norm,experts,dispatcher,…}` | `training.backend.*` (same names) |
| `model.mtp_loss_scaling_factor` / `mtp_use_repeated_layer` / `num_nextn_predict_layers` | `training.mtp.loss_scaling_factor` / `use_repeated_layer` / `num_nextn_predict_layers` |
| `teacher_model.pretrained_model_name_or_path` | `training.teacher_model` (a registered **entity**) |
| `checkpoint.*`, `dist_env`, `rng`, `dataloader`, `loss_fn` | set by the platform — leave out |

## Field mapping — NeMo-RL YAML → `rl` job JSON

All fields go under `training` unless noted.

| Recipe YAML | Job JSON |
|---|---|
| `policy.model_name` | HF repo for the model fileset; top-level `model` is the entity ref |
| `dpo.reference_policy_kl_penalty` / `grpo`'s `loss_fn.reference_policy_kl_penalty` | `ref_policy_kl_penalty` |
| `dpo.preference_loss_weight` / `sft_loss_weight` / `*_average_log_probs` | same names |
| `dpo.max_num_epochs` / `grpo.max_num_epochs` | `epochs` |
| `grpo.num_prompts_per_step` / `num_generations_per_prompt` | same names |
| `grpo.use_dynamic_sampling` / `batch_multiplier` / `dynamic_sampling_max_gen_batches` / `overlong_filtering` | same names |
| `loss_fn.ratio_clip_min` / `ratio_clip_max` / `ratio_clip_c` | same names |
| `policy.train_global_batch_size` / `train_micro_batch_size` | `batch_size` / `micro_batch_size` |
| `policy.max_total_sequence_length` | `max_seq_length` |
| `policy.optimizer.kwargs.lr` / `weight_decay` / `eps` | `learning_rate` / `weight_decay` / `adam_eps` |
| `policy.scheduler` min LR / warmup | `min_learning_rate` / `warmup_steps` |
| `policy.dtensor_cfg.{tensor,context,expert}_parallel_size` | `parallelism.*` (`expert_parallel_size` is GRPO-only and needs `policy_backend: "automodel"`) |
| `policy.dtensor_cfg._v2: true` | `policy_backend: "automodel"` (GRPO) |
| `policy.dtensor_cfg.activation_checkpointing` | `activation_checkpointing` |
| `policy.dtensor_cfg.lora_cfg.{dim,alpha,dropout,exclude_modules}` | `finetuning_type: "lora"` + `lora.{rank,alpha,dropout,exclude_modules}` (GRPO-only) |
| `policy.generation.temperature` / `max_new_tokens` / `top_k` | `temperature` / `max_new_tokens` / `top_k` |
| `policy.generation.vllm_cfg.tensor_parallel_size` / `gpu_memory_utilization` | `vllm_tensor_parallel_size` / `vllm_gpu_memory_utilization` |
| `cluster.num_nodes` / `gpus_per_node` | `parallelism.num_nodes` / `num_gpus_per_node` |
| `checkpointing.keep_top_k`, `grpo.val_period` | `keep_top_k`, `val_check_interval` |
| `policy.megatron_cfg.*`, `logger.*`, `data.*`, `env.*` | not mapped — Megatron is disabled; data and environment come from FileSets |

If a recipe sets something the job schema has no field for, leave it out and tell the user which setting was dropped. Do not invent field names: `extra="forbid"` rejects unknown keys at submit.
