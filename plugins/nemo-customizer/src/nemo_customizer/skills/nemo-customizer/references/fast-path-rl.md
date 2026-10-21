<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fast path — rl (GRPO and DPO)

Kubernetes-only. Confirm `backend: kubernetes_job` before anything else (see `references/gotchas.md` § rl runtime gate).

GRPO on a Ray cluster — **Kubernetes runtime only**. Same runtime gate as DPO: confirm `nemo jobs list-execution-profiles -f json` shows `cpu`/`gpu` at `backend: kubernetes_job` before anything else.

**Default to LoRA** (`finetuning_type: "lora"`). It trains an adapter instead of every weight, so it is markedly cheaper and faster than a full-weight run and fits on far less GPU memory — the right starting point unless the user asks for full weights or needs a standalone model entity to deploy. Full-weight remains the schema default (`all_weights`), so LoRA is something you set.

GRPO differs from every other backend in one structural way: it needs **two FileSets**, an **environment** (code that runs a rollout and returns a reward) and a **dataset** of prompt rows. There are no labelled completions — the reward comes from the environment.

**1. Environment** — the bulk of the work, and it has its own reference: **`references/gym-environments.md`**. Pick the format first. All three run on the same Gym runtime; the format decides only where dependencies come from and where `config_paths` may live:

| The user has… | Format | How |
|---|---|---|
| A Prime Intellect hub env, or any `verifiers` env | **`adapter-wheels-v1`** | `pi-to-gym-conversion` (below) — the only format with a converter |
| A Gym server tree **and** the cluster can reach a package index at spin-up | **`native-v1`** | Package the server dir + add a manifest; **strip any `.jsonl`** |
| Anything else — own code, **or a Gym server tree on a deny-default cluster** | **`wheels-v1`** | Hand-build: manifest + configs + server dirs + vendored wheel closure |

Do not pick `native-v1` just because the environment came from Gym: it ships no wheels, so its per-server venv resolves from an index and the job needs egress. Ask the operator (`NHX_RL_SANDBOX_ALLOW_INTERNET`) before committing to it; otherwise the same tree ships as `wheels-v1`.

For a hub env, run the converter on an **internet-capable host** — training clusters have no hub egress:

```bash
# Not the project .venv — see references/gym-environments.md § Path B.
UV_PROJECT_ENVIRONMENT=.venv-conversion uv sync --frozen --package nhx-rl --extra conversion
.venv-conversion/bin/pi-to-gym-conversion \
  --hub-id primeintellect/ascii-tree --hub-version 0.1.5 \
  --out-dir ./ascii-tree-pkg --dataset-dir ./ascii-tree-data \
  --validation-fraction 0.1 --upload --workspace default
```

`--upload` creates both FileSets and uploads them; the converter runs the same layout check as `--validate-only` on the package it writes, before anything is uploaded. Without it, upload by hand (`--purpose environment` for the package, `--purpose dataset` for the JSONL) — see `references/gym-environments.md` § **Upload**. Before a hand upload, and after any edit to the package, **always** validate first: `pi-to-gym-conversion --validate-only ./ascii-tree-pkg`.

**2. Dataset** — the converter writes it for a hub env. For the user's own prompts, rows are Gym rollout rows with the prompt under `responses_create_params.input` and an `agent_ref` object — **not** `messages[]`, **not** prompt/completion, **not** preference triples. Schema and a conversion snippet: `references/dataset-formats.md` § **NeMo-RL (GRPO)**.

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

    "learning_rate": 1e-5,
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

It runs one full epoch (`epochs: 1`, no `max_steps`). Add `"max_steps": 200` only if the user explicitly wants a bounded first run (`references/defaults.md`).

**Scale it to the hardware you actually have** — this targets 8 GPUs. On fewer, keep both divisibility rules satisfied (`references/batch-sizing.md` § **Batch sizing — rl / GRPO**): `batch_size % (micro_batch_size × data_parallel_size) == 0` and `(num_prompts_per_step × num_generations_per_prompt) % batch_size == 0`. Drop `vllm_tensor_parallel_size` to at most `num_gpus_per_node`, and `execution_profile` unless the platform defines that profile.

`use_triton` is deliberately absent from the `lora` block: unset, the compiler picks `true` at `tensor_parallel_size` 1 and `false` above, and an explicit `true` above TP 1 is rejected at submit.

**Full-weight variant.** Drop `finetuning_type` and the `lora` block — `all_weights` is the schema default. It registers a model entity that needs its own deployment, where LoRA registers an adapter that hot-reloads onto the base. Do not set `lora_merged` (rejected) or pass a `lora` block alongside `all_weights` (rejected).

**5. Submit and poll** — identical to DPO (no `--name`; read the `rl-<hex>` id from submit stdout):

```bash
nemo customization rl --job-json /tmp/job.json --workspace default > /tmp/rl-submit.json
JOB=$(python3 -c "import json;print(json.load(open('/tmp/rl-submit.json'))['name'])")
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"
```

**Read GRPO progress on reward, not loss.** The GRPO surrogate loss oscillates near zero and says nothing about run quality. `train_reward` and `val_accuracy` are the curves that matter — `val_accuracy` is the validation pass's mean reward under NeMo-RL's name for it. `train_truncation_rate` rising or `train_baseline_reward/pct_mixed` falling toward zero explain a reward curve that stopped moving. See `references/reporting.md` § **RL (GRPO) example**.

## Fast path — rl (DPO)

DPO on a Ray cluster — **Kubernetes runtime only**, full-weight. **Before anything else**, confirm the platform dispatches jobs to Kubernetes: `nemo jobs list-execution-profiles -f json` must show `cpu`/`gpu` at `backend: kubernetes_job` (not `docker`/`subprocess`). If it doesn't, stop — do not start/use a docker-runtime platform; configure a Kubernetes-runtime one per `references/rl-kubernetes-runtime.md`. Model-entity setup (step 2) is identical to automodel; the dataset is **preference data** and the job JSON is the `RlJobInput` shape.

**1. Preference dataset** — rows are `{prompt, chosen, rejected}` (see `references/dataset-formats.md` § NeMo-RL). Upload **both** files to **one** fileset:

```bash
DATASET=<preference-fileset>   # e.g. dpo-data; new name when the data changes (no --exist-ok — see references/gotchas.md)
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
    "learning_rate": 5e-6,
    "max_seq_length": 1024,
    "batch_size": 32,
    "micro_batch_size": 1,
    "ref_policy_kl_penalty": 0.05,
    "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1 }
  },
  "output": { "name": "<output-name>" }
}
```

**4. Submit and poll** — `rl --job-json` has **no `--name` flag** (the platform auto-generates the `rl-<hex>` job id), so read it from submit stdout:

```bash
nemo customization rl --job-json /tmp/job.json --workspace default > /tmp/rl-submit.json   # add --profile <name> if the default gpu profile is wrong
JOB=$(python3 -c "import json;print(json.load(open('/tmp/rl-submit.json'))['name'])")
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"
```

**Do not** derive the job id by picking the newest `rl-*` from `nemo jobs list` — a concurrent job or an earlier failed submit selects the wrong one. If `/tmp/rl-submit.json` does not parse, the submit result is unknown: stop and inspect it (re-submitting risks a duplicate job).

**Do not use `2>&1`** before `json.load` — warnings on stderr break parsing; see `references/gotchas.md`. Or poll manually: `nemo jobs get-status "$JOB"` every 30–60s. If submit fails on an unknown profile, re-list execution profiles and pass `--profile <name>`. `nemo customization rl explain` prints the live schema.

