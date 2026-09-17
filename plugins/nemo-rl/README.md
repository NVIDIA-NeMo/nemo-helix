<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-rl-plugin

NeMo-RL customization contributor for the NeMo Platform. Adds **DPO** and **GRPO**
training on a Ray cluster (via [NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
v0.6.0) as the `rl` backend under `/apis/customization`.

Thin contributor layer only — the heavy compile glue and container tasks live in
[`services/rl`](../../services/rl) (`nmp-rl`).

## Surfaces

- **CLI:** `nemo customization rl submit <job.json> -w <workspace>` (submit-only;
  there is no local `run` verb or local execution path).
- **REST:** `POST /apis/customization/v2/workspaces/{workspace}/rl/jobs`
- **SDK:** `client.customization.rl.jobs.create(...).data()`

## Constraints

- **Remote Kubernetes only** — gated via `require_distributed_runtime`. There is
  no local Docker fallback (unlike automodel/unsloth).
- **Single-node multi-GPU and multi-node** both supported (`parallelism.num_nodes`).
  Multi-node requires `NMP_RL_MULTINODE_SHARED_STORAGE_PATH`.
- **DPO is full-weight** (no PEFT).
- **GRPO** trains against a NeMo Gym environment supplied as an environment FileSet
  using any environment you author, in any of the three packaging formats described
  in Environments section below.
- **GRPO sandboxed mode** defaults from platform config (`sandboxed_gym_default=true`).
  Compile fails closed when OpenSandbox is unavailable (set
  `NMP_PLATFORM_SANDBOX_CLUSTER_CAPABLE=true` once installed) or when
  `NMP_RL_JOB_STORAGE_PVC_CLAIM` is unset — the Gym sandbox re-mounts that claim to
  read the downloaded environment and dataset. Set `NMP_RL_SANDBOXED_GYM_DEFAULT=false`
  for trusted dev smoke tests only.
- **`training.type` is required** in submitted JSON (union discriminator); it does not
  default to `dpo`.

## Job spec

### DPO

`model` and `dataset` are string refs; the method lives under `training` with
`type: "dpo"`. The `dataset` fileset holds **both** `training.jsonl` and
`validation.jsonl` as `{prompt, chosen, rejected}` preference rows.

```json
{
  "model": "default/qwen3-0.6b",
  "dataset": "default/dpo-data",
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
  "output": { "name": "qwen3-0.6b-dpo" }
}
```

Configurable `training` knobs (full reference: the skill's
`references/hyperparameters.md` § NeMo-RL (DPO)): the optimizer/schedule/batch
fields, `parallelism`, `optimizer_type`, `adam_eps`, `activation_checkpointing`,
`keep_top_k`, `val_at_end`, and the DPO-specific `ref_policy_kl_penalty`,
`preference_loss_weight`, `sft_loss_weight`, `preference_average_log_probs`,
`sft_average_log_probs`, `max_grad_norm`. `RlJobInput` (`schema.py`) is the
authoritative input shape; `nemo customization rl explain` prints it live.

### GRPO (NeMo Gym)

GRPO needs two FileSets: an **environment** (the code that serves prompts and scores
rollouts) and a **dataset** (the Gym JSONL rows fed through it). See
[Environments](#environments) for what goes in each and how to package one.

1. **Package** your environment into one of the supported formats and upload both
   FileSets (`purpose=environment` and `purpose=dataset`).

   If you are starting from a Prime Intellect hub environment, `pi-to-gym-conversion`
   is a worked example of producing an `adapter-wheels-v1` package plus its Gym JSONL.
   It is a convenience for that one source, not a required step — environments authored
   any other way are packaged the same way and treated identically:

   The `conversion` extra (verifiers + pip) is not in the training image, and it goes in
   its own environment: the converter installs the untrusted hub wheel into whatever
   interpreter runs it, and a `uv sync` of the repo `.venv` prunes both back out.

   ```bash
   UV_PROJECT_ENVIRONMENT=.venv-conversion uv sync --package nmp-rl --extra conversion
   .venv-conversion/bin/pi-to-gym-conversion --hub-id primeintellect/ascii-tree --out-dir ./ascii-tree-pkg
   ```

2. **Submit** GRPO with `environment` + Gym JSONL `dataset`:

   ```json
   {
     "model": "default/qwen3-0.6b",
     "dataset": "default/ascii-tree-gym-data",
     "environment": "default/ascii-tree-env",
     "training": {
       "type": "grpo",
       "epochs": 1,
       "batch_size": 32,
       "micro_batch_size": 1,
       "num_generations_per_prompt": 8,
       "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1 }
     },
     "output": { "name": "qwen3-0.6b-grpo" }
   }
   ```

   The compiler downloads model + dataset + environment, emits path-only NeMo-RL
   YAML with `env.nemo_gym.sandboxed` from platform config (default `true`), and
   injects vLLM/broker egress env vars for sandboxed rollouts.

## Environments

An **environment** is the code that runs on the other side of a rollout: it turns a
dataset row into a prompt, receives the policy's response, and returns a reward. NeMo
Gym starts it as a local HTTP server for the duration of the job; NeMo-RL calls it once
per rollout. Anything you can express as a Gym resources server or agent works — there
is no fixed catalogue, and no dependency on any particular environment hub.

An **environment FileSet** (`purpose=environment`) holds that code plus a
`nemo-environment.yaml` manifest at its root. The manifest declares the packaging
`format`, the `config_paths` Gym should load, and provenance `metadata`. Prompt rows do
**not** live here — they go in the separate dataset FileSet, so one environment can be
reused across datasets.

| `format` | Layout | Use when |
|---|---|---|
| `native-v1` | Gym source trees under `responses_api_agents/`, `resources_servers/`, `responses_api_models/`. No wheels. | Your environment is plain source with no third-party dependencies beyond what the training image ships. |
| `wheels-v1` | Any `config_paths` layout, plus a `wheels/` directory of pre-built `.whl` files. | Your environment needs Python dependencies. The wheels are installed offline, so the job needs no cluster egress. |
| `adapter-wheels-v1` | `configs/` YAMLs plus `wheels/`, and an `adapter.agent` naming an agent harness the training image already ships. | Your environment is driven by a shipped agent harness (e.g. `verifiers_agent`) rather than its own server code. |

Folder trees, Gym YAML for resources / Responses API servers, and worked examples:
[GRPO Environment Packages](../../docs/customizer/tutorials/grpo-environment-packages.mdx).

`wheels-v1` and `adapter-wheels-v1` are installed from the vendored `wheels/` directory
only, which is what lets a sandboxed rollout run with no internet access. Package one
resolved closure per environment; vendoring several versions of the same distribution
leaves the resolver to pick, and the job warns about it.

`adapter.agent` is checked against an allowlist of harnesses built into the training
image (`services/rl/src/nmp/rl/tasks/environment/allowlist.py`), since the manifest
selects code that already exists in the image rather than shipping it.

## Compiled job (4 steps)

`submit` → `RlJobInput` → transform → `RlJobOutput` → compiled `PlatformJobSpec`:

1. **download** — model fileset + dataset (+ environment for GRPO) → PVC (CPU, `nmp-customizer-tasks`)
2. **dpo-training** / **grpo-training** — Ray step (GPU, `nmp-rl-training`); single-node `gpu` or
   multi-node `gpu_distributed` executor, selected by `parallelism.num_nodes`
3. **upload** — trained checkpoint → output fileset (CPU)
4. **model-entity** — register the full-weight output `ModelEntity`

## Related

- **Skill:** the `nemo-customizer` skill documents the end-to-end DPO workflow
  (`plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/`).
- **Design:** [`docs/customizer/nemo-rl-dpo-plugin-design.md`](../../docs/customizer/nemo-rl-dpo-plugin-design.md).
- **GRPO environment packages:** [`docs/customizer/tutorials/grpo-environment-packages.mdx`](../../docs/customizer/tutorials/grpo-environment-packages.mdx) (`native-v1` / `wheels-v1` / `adapter-wheels-v1`).
- **GRPO job + cluster:** [`docs/customizer/grpo-training.mdx`](../../docs/customizer/grpo-training.mdx).
- **GPU e2e smoke test:** [`scripts/gpu-dpo-smoke/`](../../scripts/gpu-dpo-smoke).
- **Images:** [`docker/rl/Dockerfile.nmp-rl-base`](../../docker/rl/Dockerfile.nmp-rl-base),
  [`docker/rl/Dockerfile.nmp-rl-training`](../../docker/rl/Dockerfile.nmp-rl-training),
  `docker/Dockerfile.nmp-customizer-tasks`.
