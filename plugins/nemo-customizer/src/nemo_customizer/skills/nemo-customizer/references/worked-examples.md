<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Worked examples

End-to-end examples for automodel, unsloth and rl (DPO).

**Automodel, the default (1 GPU):** `Qwen/Qwen3-1.7B` + `tau/commonsense_qa` → CHAT JSONL, fileset `commonsense_qa`, entity `qwen3-1.7b`, output `qwen3-1-7b-commonsense-qa-lora`. `learning_rate` is left unset, so the schema picks the LoRA default:

```json
{
  "model": "default/qwen3-1.7b",
  "dataset": { "training": "default/commonsense_qa", "validation": "default/commonsense_qa" },
  "training": {
    "training_type": "sft",
    "finetuning_type": "lora",
    "lora": { "rank": 16, "alpha": 32 },
    "max_seq_length": 2048
  },
  "schedule": { "epochs": 1 },
  "batch": { "global_batch_size": 4, "micro_batch_size": 1 },
  "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1, "tensor_parallel_size": 1 },
  "output": { "name": "qwen3-1-7b-commonsense-qa-lora" }
}
```

On a ≥48 GB GPU, take batch and learning rate from the LoRA ≤4B **default** row of `references/batch-sizing.md`: `micro` 32, GBS 128, `learning_rate` `1e-4` (high-util: 64 / 256). For **full SFT**, set `"finetuning_type": "all_weights"`, drop `lora`, and take the ≤2B row of § **Full-weight SFT** in the same file.

**Automodel, 4 or more Hopper/Blackwell GPUs (Nemotron LoRA):** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` + `tau/commonsense_qa` → CHAT JSONL, fileset `commonsense_qa`, entity `nemotron-3-5-lightning-30b-a3b`, output `nemotron-3-5-lightning-commonsense-qa-lora`, `epochs: 1`. Values come from the Nemotron 3 Nano LoRA recipes in `references/recipes.md` (same architecture):

```json
{
  "model": "default/nemotron-3-5-lightning-30b-a3b",
  "dataset": { "training": "default/commonsense_qa", "validation": "default/commonsense_qa" },
  "training": {
    "training_type": "sft",
    "finetuning_type": "lora",
    "lora": { "rank": 16, "alpha": 32, "exclude_modules": ["*.out_proj"] },
    "max_seq_length": 2048
  },
  "schedule": { "epochs": 1 },
  "batch": { "global_batch_size": 8, "micro_batch_size": 1 },
  "optimizer": { "learning_rate": 1e-5, "min_learning_rate": 1e-6 },
  "parallelism": { "num_nodes": 1, "num_gpus_per_node": 4, "tensor_parallel_size": 1, "expert_parallel_size": 4 },
  "output": { "name": "nemotron-3-5-lightning-commonsense-qa-lora" }
}
```

**Unsloth:** same small-dense model + dataset + entity + fileset, but `nemo customization unsloth --job-json /tmp/job.json -w default`. Job JSON ≤4B row: `batch.per_device_train_batch_size` 8, `batch.gradient_accumulation_steps` 16 (effective 128), `learning_rate` `1e-4`, `hardware.gpus` `"0"`, `output.save_method` `"lora"`. Poll to completion.

**rl (DPO):** the no-details default — `Qwen/Qwen3-0.6B` + `nvidia/HelpSteer3` (preference subset, uploaded raw), output `qwen3-0.6b-dpo`. Upload `training.jsonl` + `validation.jsonl` to one fileset, register the model entity, then submit a **small 20-step demo** job:

```json
{
  "model": "default/qwen3-0.6b",
  "dataset": "default/helpsteer3-dpo",
  "training": { "type": "dpo", "max_steps": 20, "batch_size": 32, "micro_batch_size": 1,
                "max_seq_length": 1024, "ref_policy_kl_penalty": 0.05,
                "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1 } },
  "output": { "name": "qwen3-0.6b-dpo" }
}
```

`nemo customization rl --job-json /tmp/job.json -w default`, then poll to completion. This is the 20-step demo from `references/defaults.md`; a real run sets `epochs` instead of `max_steps`.

