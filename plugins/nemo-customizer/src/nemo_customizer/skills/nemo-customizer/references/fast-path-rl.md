<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fast path — rl (GRPO and DPO)

Commands and job JSON for the `rl` backend, after `SKILL.md` § **Pre-flight** and the **platform check** in **Plugin pick**. The rules each step must follow are in `SKILL.md` § **Workflow**.

## Fast path — rl (GRPO)

**Default to LoRA** (`finetuning_type: "lora"`). It trains an adapter instead of every weight, so it is markedly cheaper and faster than a full-weight run and fits on far less GPU memory — the right starting point unless the user asks for full weights or needs a standalone model entity to deploy. Full-weight remains the schema default (`all_weights`), so LoRA is something you set.

**1. Environment and 2. Dataset** — the **nemo-gym-environments** skill builds, validates and uploads both FileSets. Come back with their two refs.

**3. Model** — same as step 2 of the automodel fast path in `references/fast-path-sft.md` (HF weights fileset + model entity; gated repos need `token_secret`).

**4. Job JSON** — `model`, `dataset`, and `environment` are all **strings**; the method is `training.type: "grpo"`. Full field reference: `references/hyperparameters-rl.md`. The block below is the **LoRA default**.

```json
{
  "model": "default/qwen3-8b-base",
  "dataset": "default/<gym-dataset-fileset>",
  "environment": "default/<environment-fileset>",
  "training": {
    "type": "grpo",
    "finetuning_type": "lora",
    "lora": { "rank": 128, "alpha": 256, "dropout": 0.0 },

    "max_seq_length": 3072,
    "max_new_tokens": 2048,
    "temperature": 1.0,
    "activation_checkpointing": true,

    "min_learning_rate": 9e-6,
    "adam_eps": 1e-8,
    "warmup_steps": 10,
    "weight_decay": 0.01,

    "epochs": 1,

    "batch_size": 512,
    "micro_batch_size": 1,
    "num_prompts_per_step": 32,
    "num_generations_per_prompt": 16,

    "use_dynamic_sampling": true,
    "batch_multiplier": 2.0,
    "dynamic_sampling_max_gen_batches": 10,

    "ref_policy_kl_penalty": 0.0,
    "ratio_clip_min": 0.2,
    "ratio_clip_max": 0.28,
    "ratio_clip_c": 10.0,
    "normalize_rewards": true,
    "use_leave_one_out_baseline": true,
    "max_rollout_turns": 1,

    "truncated_importance_sampling_type": "tis",
    "truncated_importance_sampling_ratio": 2.0,
    "use_importance_sampling_correction": true,

    "vllm_tensor_parallel_size": 8,
    "vllm_gpu_memory_utilization": 0.6,

    "val_at_start": true,
    "val_at_end": true,
    "val_check_interval": 25,
    "keep_top_k": 1,
    "seed": 42,

    "parallelism": { "num_nodes": 1, "num_gpus_per_node": 8, "tensor_parallel_size": 1 },
    "execution_profile": "h100"
  },
  "output": { "name": "<output-name>" }
}
```

It runs one full epoch; `references/defaults.md` has the bounded-run alternative.

**Scale it to the hardware you actually have** — this targets 8 GPUs. On fewer, keep both divisibility rules satisfied (`references/batch-sizing.md` § **Batch sizing — rl / GRPO**): `batch_size % (micro_batch_size × data_parallel_size) == 0` and `(num_prompts_per_step × num_generations_per_prompt) % batch_size == 0`. Drop `vllm_tensor_parallel_size` to at most `num_gpus_per_node`, and `execution_profile` unless the platform defines that profile.

`use_triton` is deliberately absent from the `lora` block: unset, the compiler picks `true` at `tensor_parallel_size` 1 and `false` above, and an explicit `true` above TP 1 is rejected at submit.

**Full-weight variant.** Drop `finetuning_type` and the `lora` block — `all_weights` is the schema default. It registers a model entity that needs its own deployment, where LoRA registers an adapter that hot-reloads onto the base. Do not set `lora_merged` (rejected) or pass a `lora` block alongside `all_weights` (rejected).

**5. Submit and poll**

```bash
nemo customization rl --job-json /tmp/job.json --workspace default > /tmp/rl-submit.json
JOB=$(python3 -c "import json;print(json.load(open('/tmp/rl-submit.json'))['name'])")
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"
```

Report progress with `references/reporting.md` § **RL (GRPO) example**.

## Fast path — rl (DPO)

DPO is full-weight. Model-entity setup (step 2) is identical to automodel; the dataset is **preference data** and the job JSON is the `RlJobInput` shape.

**1. Preference dataset** — rows are `{prompt, chosen, rejected}` (see `references/dataset-formats.md` § NeMo-RL). Upload **both** files to **one** fileset:

```bash
DATASET=<preference-fileset>   # e.g. dpo-data
nemo files filesets create "$DATASET" --workspace default --purpose dataset
nemo files upload /tmp/dpo-train.jsonl "$DATASET" --workspace default --remote-path training.jsonl
nemo files upload /tmp/dpo-val.jsonl   "$DATASET" --workspace default --remote-path validation.jsonl
nemo files list "$DATASET" --workspace default
```

**2. Model** — same as step 2 of the automodel fast path in `references/fast-path-sft.md` (HF weights fileset + model entity; gated repos need `token_secret`).

**3. Job JSON** — write `/tmp/job.json`. `model` and `dataset` are **strings**; the method is under `training` with `type: "dpo"`. Full field reference: `references/hyperparameters-rl.md`.

```json
{
  "model": "default/<model-entity>",
  "dataset": "default/<preference-fileset>",
  "training": {
    "type": "dpo",
    "epochs": 1,
    "max_seq_length": 1024,
    "batch_size": 32,
    "micro_batch_size": 1,
    "ref_policy_kl_penalty": 0.05,
    "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1 }
  },
  "output": { "name": "<output-name>" }
}
```

**4. Submit and poll**

```bash
nemo customization rl --job-json /tmp/job.json --workspace default > /tmp/rl-submit.json   # add --profile <name> if the default gpu profile is wrong
JOB=$(python3 -c "import json;print(json.load(open('/tmp/rl-submit.json'))['name'])")
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"
```

If submit fails on an unknown profile, re-list execution profiles and pass `--profile <name>`. `nemo customization rl explain` prints the live schema.
