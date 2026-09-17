<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GRPO environment examples

**These are examples.** Nothing in the platform calls them, and they define no behaviour: the
supported contract is the environment FileSet layout that
`nmp.rl.tasks.environment.validate` defines. Use them to get a working environment and dataset
quickly, or as a starting point for your own.

A GRPO job needs three things that no deployment creates for you — a model entity, an
**environment** FileSet, and a **dataset** FileSet of Gym rollout rows. The job JSON fixtures in
`plugins/nemo-rl/tests/fixtures/` reference them by name; these scripts build two of them.

## Prerequisites

| Requirement | Check | If missing |
|---|---|---|
| Platform on `platform.runtime: kubernetes` | `nemo jobs list-execution-profiles -f json` reports `backend: kubernetes_job` | GRPO cannot run; refer to the skill's `rl-kubernetes-runtime.md` |
| Sandboxed Gym enabled | operator has set `NMP_SANDBOX_CLUSTER_CAPABLE` and `NMP_RL_JOB_STORAGE_PVC_CLAIM` | Submit fails before any GPU is claimed. Operator-only |
| A NeMo Gym checkout | `ls $GYM_ROOT/resources_servers` | Gym is **not** vendored here: `git clone https://github.com/NVIDIA-NeMo/Gym ~/workspace/Gym` |
| Internet on **this** host | — | `wheels-v1` resolves a wheel closure; the dataset script pulls from HuggingFace |

## 1. Build the environment package

`gym_to_env_package.py` packages **any** server from a Gym checkout:

```bash
uv run scripts/grpo-examples/gym_to_env_package.py \
  --gym-root ~/workspace/Gym \
  --server resources_servers/math_with_judge \
  --format wheels-v1 --arch x86_64 \
  --expect-nemo-gym-version <v> --ray-version <v> --openai-version <v> \
  --out-dir /tmp/mwj-env
```

`wheels-v1` requires those three versions, because Gym pins every per-server virtualenv to the
training image's `nemo-gym`, `ray` and `openai`. A closure built against different ones is
ignored and resolved from an index instead. All three come from a NeMo-RL checkout at the commit
`NEMO_RL_REF` pins in `docker-bake.hcl` -- the training image is built from it, so no access to
the image is needed: `nemo-gym` from the Gym submodule, `ray`/`openai` from NeMo-RL's `uv.lock`.
Not from Gym's own lock, which pins different versions.

The same script emits `native-v1` — one flag, not a second script:

```bash
uv run scripts/grpo-examples/gym_to_env_package.py \
  --gym-root ~/workspace/Gym \
  --server resources_servers/math_with_judge \
  --format native-v1 \
  --out-dir /tmp/mwj-native
```

The two differ in exactly two ways, both handled for you:

| | `wheels-v1` | `native-v1` |
|---|---|---|
| `wheels/` | full closure vendored | absent — resolved from a package index at job start |
| `policy_model.yaml` | `configs/` | `responses_api_models/vllm_model/configs/` (the format requires a Gym server prefix) |
| Cluster egress at job start | not needed | **required** (`NMP_RL_SANDBOX_ALLOW_INTERNET`) |

`--arch` is ignored for `native-v1`, since it vendors nothing.

| Flag | Notes |
|---|---|
| `--gym-root` | Required. Fails with clone instructions if omitted |
| `--server` | `<server_type>/<implementation>`, e.g. `resources_servers/math_with_judge` |
| `--format` | `wheels-v1` vendors the closure and needs no egress at job start; `native-v1` ships no wheels and resolves from an index, so the cluster must allow internet |
| `--arch` | `x86_64` or `aarch64` — the training images ship for both, so match the nodes: `kubectl get nodes -o jsonpath='{.items[*].status.nodeInfo.architecture}'` |
| `--config` | Repeatable. Defaults to `<implementation>.yaml`; other configs in the directory usually pair the server with an agent this package does not carry |
| `--expect-nemo-gym-version` | Fail unless the checkout builds this exact version. Gym pins each per-server venv to the image's `nemo-gym` version, so a mismatch is silently resolved from PyPI instead |

It copies the server tree (dropping `data/`, `tests/` and any `.jsonl`), strips inline
`datasets:` blocks pointing at in-tree files, writes `policy_model.yaml` where the format
requires it, and emits the manifest.

For a `verifiers` / Prime Intellect environment use the converter instead, which needs no Gym
checkout — note it vendors `x86_64` wheels today:

```bash
uv run --package nmp-rl pi-to-gym-conversion \
  --hub-id primeintellect/ascii-tree --hub-version 0.1.5 \
  --out-dir ./ascii-tree-pkg --dataset-dir ./ascii-tree-data --validation-fraction 0.1
```

## 2. Build the dataset

`prepare_math_with_judge.py` builds Gym rollout rows from DAPO-Math-17k (train) plus a holdout
or AIME24 (validation), adding the `agent_ref` every platform row needs and the
`expected_answer` field `math_with_judge` scores against:

```bash
uv run --with datasets scripts/grpo-examples/prepare_math_with_judge.py \
  --out-dir /tmp/mwj-data --train-size 512
```

## 3. Validate and upload

```bash
uv run --package nmp-rl pi-to-gym-conversion --validate-only /tmp/mwj-env

nemo files filesets create math-with-judge-env -w default --purpose environment --exist-ok
nemo files upload /tmp/mwj-env/ math-with-judge-env -w default

nemo files filesets create math-with-judge-gym-data -w default --purpose dataset --exist-ok
nemo files upload /tmp/mwj-data/ math-with-judge-gym-data -w default
```

The trailing slash matters: it uploads the directory's *contents*. `--purpose
environment` is enforced at submit. Confirm with `nemo files list` before submitting.

## 4. Submit

The names above match what the shipped fixtures reference:

```bash
nemo customization rl submit plugins/nemo-rl/tests/fixtures/minimal_grpo_lora.json -w default
```

| Fixture | Shape |
|---|---|
| `minimal_grpo.json` | Full-weight GRPO, 8 GPUs |
| `minimal_grpo_lora.json` | GRPO + LoRA, 8 GPUs — cheaper and faster; start here |
| `minimal_moe_grpo_lora.json` | MoE + LoRA on 4 GPUs, with expert parallelism and reward shaping |

Each also needs the model entity it names. Create it the same way as for any other backend:

```bash
nemo files filesets create qwen3-8b-base -w default --purpose model --exist-ok \
  --storage '{"type":"huggingface","repo_id":"Qwen/Qwen3-8B-Base","repo_type":"model","revision":"main"}'
nemo models create qwen3-8b-base -w default --exist-ok \
  --input-data '{"name":"qwen3-8b-base","fileset":"default/qwen3-8b-base","custom_fields":{"hf_model_id":"Qwen/Qwen3-8B-Base"}}'
```

## Files

| File | Role |
|---|---|
| `gym_to_env_package.py` | Packages any Gym server directory as `wheels-v1` or `native-v1` from a local Gym checkout |
| `prepare_math_with_judge.py` | Builds Gym rollout rows from DAPO-Math-17k and a holdout or AIME24 |
