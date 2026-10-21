<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Worked examples

End-to-end examples for automodel, unsloth and rl (DPO).

**Automodel (Nemotron LoRA):** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` + `tau/commonsense_qa` → CHAT JSONL, fileset `commonsense_qa`, entity `nemotron-3-5-lightning-30b-a3b`, output `nemotron-3-5-lightning-commonsense-qa-lora`, `epochs: 1` (no `max_steps`). Values come from the Nemotron 3 Nano LoRA recipes in `references/recipes.md` (same architecture):

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

**Automodel (small dense, 1 GPU):** `Qwen/Qwen3-1.7B` + the same dataset, entity `qwen3-1.7b`. On a ≥48 GB GPU, use the LoRA ≤4B **default** row: `micro` 32, GBS 128, `learning_rate` `1e-4` (high-util: 64 / 256).

**Unsloth:** same small-dense model + dataset + entity + fileset, but `nemo customization unsloth --job-json /tmp/job.json -w default`. Job JSON ≤4B row: `batch.per_device_train_batch_size` 8, `batch.gradient_accumulation_steps` 16 (effective 128), `learning_rate` `1e-4`, `hardware.gpus` `"0"`, `output.save_method` `"lora"`. Poll the submit `"name"` (already `unsloth-…`) verbatim to completion. For payload shape only (not the field set or values): `plugins/nemo-unsloth/tests/fixtures/minimal_unsloth_sft.json` — a smoke-test input, so confirm fields against `unsloth explain`.

**rl (DPO):** the no-details default — `Qwen/Qwen3-0.6B` + `nvidia/HelpSteer3` (preference subset, uploaded raw), output `qwen3-0.6b-dpo`. First confirm `kubernetes_job` backend (see **Plugin pick** → rl runtime gate). Upload `training.jsonl` + `validation.jsonl` to one fileset, register the model entity, then submit a **small 20-step demo** job:

```json
{
  "model": "default/qwen3-0.6b",
  "dataset": "default/helpsteer3-dpo",
  "training": { "type": "dpo", "max_steps": 20, "batch_size": 32, "micro_batch_size": 1,
                "learning_rate": 5e-6, "max_seq_length": 1024, "ref_policy_kl_penalty": 0.05,
                "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1 } },
  "output": { "name": "qwen3-0.6b-dpo" }
}
```

`nemo customization rl --job-json /tmp/job.json -w default`, derive the `rl-<hex>` id (submit has no `--name`), poll to completion. For payload shape only: `plugins/nemo-rl/tests/fixtures/minimal_dpo.json` — a smoke-test input, so confirm fields against `rl explain`. For a real run, replace `max_steps: 20` with `epochs`.

