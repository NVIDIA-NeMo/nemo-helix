---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-gym-environments
description: >-
  Build, convert, validate and upload a NeMo Gym reward environment for GRPO
  training: package your own reward code or a NeMo Gym server, convert a Prime
  Intellect or verifiers environment, and write the rollout-row dataset. Hand
  off to nemo-customizer to submit the GRPO job.
triggers:
  - nemo gym
  - nemo-gym
  - gym environment
  - reward environment
  - environment package
  - environment fileset
  - verifiers env
  - prime intellect
  - pi-to-gym-conversion
  - native-v1
  - wheels-v1
  - adapter-wheels-v1
  - rollout rows
  - package an environment
  - convert an environment
  - vendor wheels
  - environment spin-up
not-for:
  - nemo-customizer (submitting and monitoring the GRPO, DPO, SFT or LoRA job itself)
  - nemo-evaluator-plugin (scoring a model or agent, not training against a reward)
compatibility: >-
  Requires the `rl` customization backend on a Kubernetes-runtime platform with sandboxed
  Gym enabled. Converting a hub environment needs internet access on the host that runs
  the converter.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read, Write, Edit, Grep
permissions:
  - env
  - network
  - file_write
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Gym environments for GRPO

GRPO scores sampled responses with an **environment**: code that runs a rollout and returns a reward. A GRPO job takes two FileSets, built here; the **nemo-customizer** skill submits the job that uses them.

```text
environment FileSet (purpose=environment)   dataset FileSet (purpose=dataset)
  nemo-environment.yaml   ← manifest          training.jsonl    ← required (any name with `train`)
  configs/ or server dirs                     validation.jsonl  ← optional (any name with `val`)
  wheels/                 ← 2 of 3 formats
```

The job JSON references both as strings: `"environment": "default/<env-fileset>"`, `"dataset": "default/<dataset-fileset>"`.

## Before you start

Each of these invalidates the work if it turns out false:

| Check | How | If it fails |
|---|---|---|
| The platform can run GRPO at all | The **nemo-customizer** skill's platform check: a GPU execution profile with `backend: kubernetes_job` or `volcano_job` | Stop; GRPO needs a Kubernetes-runtime platform |
| The cluster runs sandboxed Gym | Operator confirms `sandbox_cluster_capable` and the job-storage PVC claim (`../nemo-customizer/references/rl-kubernetes-runtime.md` § **Sandboxed Gym (GRPO)**) | Stop; submit fails and no package change helps |
| Whether the sandbox has egress | Operator confirms `NHX_RL_SANDBOX_ALLOW_INTERNET` | Decides `native-v1` vs `wheels-v1`; settle it **before** the layout |
| Which Gym agent runs the environment | A `verifiers` env runs `verifiers_agent`; a resources server usually runs `simple_agent` | Decides the format and every row's `agent_ref.name` |
| The training image's `nemo-gym`, `ray` and `openai` versions | `references/dependencies.md` § **Vendoring a wheel closure** | Needed only for a `wheels-v1` closure |

**Get the converter.** `pi-to-gym-conversion` is an example script in the nemo-helix repository, not part of the installed `nemo` CLI or the `rl` plugin. Run it from a nemo-helix checkout, in a dedicated environment rather than the project `.venv`: the converter installs the hub wheel into the interpreter that runs it.

```bash
UV_PROJECT_ENVIRONMENT=.venv-conversion uv sync --frozen --package nhx-rl --extra conversion
export PATH="$PWD/.venv-conversion/bin:$PATH"
pi-to-gym-conversion --help
```

Converting a hub environment needs that `conversion` extra; `--validate-only` does not.

## 1. Pick a format

| The user has… | Format | Ships wheels? | `config_paths` must live under |
|---|---|---|---|
| A Prime Intellect hub env, or any `verifiers` env installable as a wheel | **`adapter-wheels-v1`** | Yes | `configs/` |
| A Gym server tree, and the sandbox can reach a package index at spin-up | **`native-v1`** | No | `responses_api_agents/`, `resources_servers/`, `responses_api_models/` |
| Anything else: own code, or a Gym server tree on a deny-default cluster | **`wheels-v1`** | Yes | anywhere in the package |

All three run on the same Gym runtime. The format decides only where dependencies come from and where `config_paths` may live, so do not rank them by support. `native-v1` ships no wheels, so it needs egress; on a deny-default cluster, ship the same tree as `wheels-v1`. Why: `references/dependencies.md`.

## 2. Build the package

| Format | How | Read |
|---|---|---|
| `adapter-wheels-v1` | Run the converter on a host with internet (training clusters have no hub egress) | `references/formats.md` § **Path B** |
| `native-v1` | Slice the server out of a NeMo Gym checkout and add a manifest | `references/formats.md` § **Path A** |
| `wheels-v1` | Hand-build the manifest, configs and server dirs, and vendor a wheel closure | `references/formats.md` § **Path C**, `references/dependencies.md` |

```bash
pi-to-gym-conversion \
  --hub-id primeintellect/ascii-tree --hub-version 0.1.5 \
  --nemo-rl-root ~/workspace/RL \
  --out-dir ./ascii-tree-pkg --dataset-dir ./ascii-tree-data \
  --validation-fraction 0.1
```

**Pass `--nemo-rl-root`**: a NeMo-RL checkout, submodules included, at the commit `NEMO_RL_REF` pins in `docker-bake.hcl`. The converter reads the training image's `nemo-gym`, `ray` and `openai` versions from it; without it the package needs sandbox egress to start.

**Pin `--hub-version`.** Unset, the converter takes whatever the index offers now, and a later release can narrow `Requires-Python` or ship code the training image's Python cannot run.

Five rules a package must meet. Each one fails at spin-up or, worse, trains silently on the wrong thing; `references/gym-config.md` explains the mechanism behind each:

1. **A `policy_model` config, listed first in `config_paths`.** Every format needs it, not just the adapter one; without it spin-up fails with `ServerRefNotFoundError`.
2. **Refs name instances, not implementations.** `{type, name}` refs and every row's `agent_ref.name` use the top-level YAML key, not the server directory name.
3. **`domain` on every `resources_servers` block, from the closed set.** An empty or unknown value aborts spin-up with `AlmostServerError`. Agent and model blocks carry no `domain`.
4. **A custom server directory ships `requirements.txt` (or `pyproject.toml`), and the package root has no `pyproject.toml`.** Without the marker, a same-named Gym built-in runs instead, with no error. A root `pyproject.toml` drops the `nemo-gym` pin. Keep the Gym directory structure; never repackage the server as a setuptools distribution.
5. **No `.jsonl` anywhere in the package.** Drop Gym's `data/` and `tests/` and the `datasets` block; rows go in the dataset FileSet.

## 3. Write the dataset rows

The converter writes `training.jsonl` (and `validation.jsonl` with `--validation-fraction`) for a hub env; use those. For anything else, rows put the prompt under `responses_create_params.input` and route with an `agent_ref` **object**. They are not `messages[]`, prompt/completion or preference triples.

Before converting, read the environment's `*VerifyRequest` model: the fields it scores are never validated at submit, so a row missing one starts a GPU job that trains on a meaningless reward. Schema, conversion snippet and a row check: `references/rollout-rows.md`.

## 4. Validate the package

```bash
pi-to-gym-conversion --validate-only ./ascii-tree-pkg
# {"valid": true, "format": "adapter-wheels-v1", "name": "ascii-tree"}
```

It exits 1 with the exact violation, and runs the same checks submit runs against the FileSet listing. It checks **layout only**: not wheel platform tags, not closure completeness, not whether `config_paths` define the servers your rows route to, and nothing about rows. Those surface at spin-up.

## 5. Upload both FileSets

The converter can create and upload both: add `--upload --workspace default`. It names them `<slug>-env` and `<slug>-env-dataset`; read the names back from the `uploaded` block of its JSON output.

By hand (the only option for `native-v1` and `wheels-v1`):

```bash
ENV=ascii-tree-env
nemo files filesets create "$ENV" --workspace default --purpose environment --exist-ok
nemo files upload ./ascii-tree-pkg/ "$ENV" --workspace default
nemo files list "$ENV" --workspace default

DATA=ascii-tree-env-dataset   # a new name whenever the rows change
nemo files filesets create "$DATA" --workspace default --purpose dataset
nemo files upload ./ascii-tree-data/ "$DATA" --workspace default
nemo files list "$DATA" --workspace default
```

- **`--purpose environment` is enforced at submit.** A package in a `dataset`-purpose fileset is rejected.
- **Keep the trailing slash** on the local directory: it uploads the directory's contents. Without it everything nests one level deeper, and submit fails with `Missing nemo-environment.yaml at environment root` or `must contain training.jsonl`. `nemo files list` must show the manifest and `training.jsonl` at the root.
- **The dataset gets a new fileset name whenever the rows change.** Re-running the create with `--exist-ok` keeps the files uploaded the first time.

## 6. Hand off to nemo-customizer

Give the **nemo-customizer** skill the two refs (`default/<env-fileset>`, `default/<dataset-fileset>`). It writes the GRPO job JSON, submits it and reports on reward. A spin-up or rollout failure after submit usually comes back here: `references/troubleshooting.md`.

**No environment yet?** A dataset alone cannot make a GRPO job. Follow nemo-customizer's "no environment supplied" rule. If the user picks the `math_with_judge` reference run, `references/example-scripts.md` builds both FileSets from a nemo-helix checkout.

## Reference files

| When | Read |
|---|---|
| Config YAML, `domain`, server dirs, `policy_model`, manifest | `references/gym-config.md` |
| Building each format, including a worked tool-calling example | `references/formats.md` |
| Where server packages come from; vendoring a wheel closure | `references/dependencies.md` |
| Row schema, finding the fields an environment scores, converting a dataset | `references/rollout-rows.md` |
| Error messages and fixes | `references/troubleshooting.md` |
| `math_with_judge` example scripts (nemo-helix checkout) | `references/example-scripts.md` |
