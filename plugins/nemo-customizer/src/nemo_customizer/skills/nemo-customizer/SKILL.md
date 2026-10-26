---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-customizer
description: >-
  Fine-tune or train an open model (Nemotron, Llama, Qwen and others) on your own
  data with NeMo Helix: SFT, LoRA, DPO preference tuning and GRPO reinforcement
  learning. Picks the backend, prepares the dataset, submits and monitors the job,
  and reports results. For embedding or reranking retrieval models, use
  nemo-retrieval-recipes.
triggers:
  - nemo-customizer
  - nemo customizer
  - fine-tune
  - fine tune
  - finetune
  - train a model
  - train on my data
  - customize a model
  - sft
  - lora
  - dpo
  - grpo
  - run grpo
  - grpo job
  - rlhf
  - reinforcement learning
  - direct preference optimization
  - preference optimization
  - preference tuning
  - automodel
  - unsloth
  - nemo-rl
  - nemo rl
  - nemo customization
  - nemo-customization
  - customizer
  - customization training
  - automodel --job-json
  - unsloth --job-json
  - rl --job-json
not-for:
  - nemo-gym-environments (building, converting or validating a GRPO reward environment and its rollout rows)
  - nemo-build-agent (agent scaffold/deploy, not weight training)
  - nemo-setup (platform install; route here when CLI resolution fails)
  - safe-synthesizer (tabular synthetic data training)
  - nemo-retrieval-recipes (domain embed/rerank SDG → train → retrieve-eval conductor)
compatibility: >-
  Requires nemo-customizer-plugin and a customization contributor (`nemo.customization.contributors`).
  Platform must expose jobs, files, and models APIs.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read, Grep
permissions:
  - env
  - network
  - file_write
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Customizer

End-to-end **SFT + LoRA** (automodel/unsloth), **DPO**, and **GRPO** (rl) on NeMo Helix. Three backend plugins, all submitted with `nemo customization <plugin> --job-json …`. None has a local `run` verb, and none takes `--venv`:

| Backend | Invocation | Trains | Where it runs | Pick when |
|---------|------------|--------|---------------|-----------|
| **`automodel`** (default) | `automodel --job-json` | SFT / LoRA | **Kubernetes GPU pods or the platform's Docker GPU executor**. Multi-node (`num_nodes > 1`) needs Kubernetes | General SFT/LoRA; multi-GPU; distillation; full-weight SFT |
| **`unsloth`** | `unsloth --job-json` | SFT / LoRA | Same as automodel; single GPU inside the container | User asks for Unsloth, or its 4-bit LoRA path on one GPU |
| **`rl`** | `rl --job-json` | **DPO** (preference) or **GRPO** (Gym env) | **Kubernetes only**: provisions a **Ray** cluster, no Docker fallback | DPO (full-weight), or GRPO with an environment FileSet + Gym rollout rows (full-weight or LoRA) |

The backend command posts to the platform API; training runs in container steps on the platform, **not** in the CLI shell. Batch shell work, and skip CLI `--help` unless a command fails.

## Pre-flight

**CLI.** Run from the **nemo-helix** git root (top-level `pyproject.toml`), not a plugin subfolder. Resolve the invocation **once** before any other step:

```bash
cd /path/to/nemo-helix
if command -v nemo >/dev/null 2>&1; then
  echo "nemo"
elif command -v uv >/dev/null 2>&1 && uv run nemo --help >/dev/null 2>&1; then
  echo "uv run nemo"
else
  echo "CLI_NOT_FOUND"
fi
```

`nemo` → use `nemo …`. `uv run nemo` → prefix every command with `uv run`. `CLI_NOT_FOUND` → stop and route to **nemo-setup** (`make bootstrap` then `nemo setup`).

**Platform URL.** Export `NHX_BASE_URL` only when the user gives a platform URL, and remember whether they did. If a platform call fails with a connection error:

- **The user gave a URL** (or you exported a non-default one): stop and report the platform unreachable at that address. Do not offer to start services.
- **Default URL only:** **ask** whether to start the platform locally; never start it without a yes. The command is in `references/troubleshooting.md` § **Platform unreachable**. It starts a **Docker-runtime** platform, which runs automodel/unsloth only; never start or reuse one for `rl`.

**Authentication.** Check `nemo auth status`. If authentication is disabled, skip login. Otherwise: unsigned JWT allowed (local dev) → `nemo auth login --unsigned-token --email <user email or admin@example.com>`; OIDC → `nemo auth login` (or `--username` / `--password` for non-interactive). On 401/403 run the matching login and retry. Use `admin@example.com` unless the user specifies another email. Run `nemo auth status` after login to confirm.

**Gated HF models.** Gated or private HF repos need a platform secret (convention: **`hf-token`**) set as **`token_secret`** on the **model fileset**, not in job JSON; the Files service does not read local `~/.cache/huggingface` or `HF_TOKEN`. Before creating the fileset, run `nemo secrets list --workspace default` and confirm `hf-token` exists. If it is missing, **ask the user** for the token and **stop**. Commands: `references/troubleshooting.md` § **Gated HuggingFace models**.

## Plugin pick

1. Run `nemo jobs list-execution-profiles -f json` against the user's platform. Do **not** use local `docker info`: the platform runs the jobs, often on another host.
2. **DPO / preference optimization** (`{prompt, chosen, rejected}` data, "align", "RLHF-style"), **GRPO / Gym / Prime Intellect env**, or an explicit NeMo-RL request → **`rl`** (needs a GPU profile **and** `platform.runtime: kubernetes`).
3. Else explicit Unsloth request → **`unsloth`**.
4. Else explicit Automodel request → **`automodel`**.
5. Else any profile with `provider: gpu` or `gpu_distributed` → **`automodel`** (default).
6. Else stop and tell the user GPU customization is unavailable.

**`rl` runtime gate:** before submitting rl, confirm the `cpu`/`gpu` profiles report `backend: kubernetes_job` (or `volcano_job`). If they report `docker`/`subprocess`, **stop**: tell the user DPO and GRPO need a Kubernetes-runtime platform. Do **not** start or reuse a docker-runtime platform and do **not** fall back to automodel/unsloth, which train neither. To configure one: `references/rl-kubernetes-runtime.md`.

## Workflow

Commands and job JSON for each backend: `references/fast-path-sft.md` (automodel, unsloth) and `references/fast-path-rl.md` (DPO, GRPO).

1. **Convert the dataset.** An HF dataset id from the user → convert it locally (`references/hf-conversion.md`); do not ask for local paths. Write CHAT `messages` rows when the model has a chat template: `references/dataset-formats.md` says why and gives each backend's row shape. GRPO rows come from the environment instead (see **GRPO** below).
2. **Upload it to a dataset fileset.** Name the fileset after the HF dataset only (`tau/commonsense_qa` → `commonsense_qa`), not the model. **When the data changes, upload to a new fileset name** (`sft-data-v2/`): `--exist-ok` reuses an existing fileset **without re-uploading**, so edited files silently train on the old contents. Do not update a fileset in place unless the user confirms nothing else uses it, and never delete one without their explicit confirmation of the exact `<workspace>/<name>`. Run `nemo files list` to verify.
3. **Register the model.** HF weights fileset + model entity, if missing (gated repos: see Pre-flight). Submit without polling `nemo models get`; the model spec fills in asynchronously. To create the model and dataset during submit instead: `references/fast-path-sft.md` § **Create resources at submit**. **LoRA job:** if the base model is already deployed without LoRA, ask the user before submitting (`references/post-training-eval.md` § **Base already deployed without LoRA**).
4. **Write the job JSON.** For a named model, start from its recipe in `references/recipes.md`; otherwise use **Defaults** below. **Do not add `max_steps` to a real run**: it overrides `epochs` and can stop mid-epoch. Test fixtures set it only so smoke tests finish fast. Every schema accepts `epochs` and `max_steps` together, so when the user sets both, keep both. The live schema, `nemo customization <plugin> explain`, outranks this skill and its references.
5. **Submit and read the job id.** Parse **stdout only**: `submit`, `explain` and `-f json` write JSON on stdout and warnings on stderr, so `2>&1` into `json.load` fails even when the job was created, and re-submitting after that parse error is the usual cause of duplicate jobs. If submit output does not parse, check `nemo jobs list` before submitting again. The job id is the `"name"` field, already prefixed (`automodel-…`, `unsloth-…`, `rl-…`); pass it verbatim. `rl` has no `--name` flag, and the newest `rl-*` in `nemo jobs list` may be someone else's job.

   ```bash
   nemo customization automodel --job-json /tmp/job.json --workspace default > /tmp/submit.json
   JOB=$(python3 -c "import json;print(json.load(open('/tmp/submit.json'))['name'])")
   ```

6. **Poll to a terminal status.** The job is done only when its **top-level** `status` is `completed`, `error` or `cancelled`. Every step can show `completed` while the job is still `active`, and `status_details.phase` can sit at `training` with `progress_pct: 100` for a while. `scripts/poll_customization_job.sh "$JOB"` polls any backend's job every 15s (append seconds to change it) and exits 1 on `error`/`cancelled`.
7. **Report** with `references/reporting.md`. Then, for inference or eval, `references/post-training-eval.md`: a LoRA adapter hot-reloads onto a READY `lora_enabled: true` deployment of its base, while full-weight and merged outputs are new model entities to deploy (vLLM by default).

A job that fails on a missing training image, a gated download or a platform error: `references/troubleshooting.md`.

## GRPO

A GRPO job takes two FileSets beyond the model: an **environment** package and a **dataset** of Gym rollout rows. The **nemo-gym-environments** skill builds, converts, validates and uploads both; this skill writes the GRPO job JSON (`references/fast-path-rl.md`), submits it and reports on reward.

**No environment supplied.** A dataset alone cannot produce a GRPO job: the reward comes from an environment written for that dataset's rows, and a mismatched pair does not error, it trains on a meaningless reward. If the user names a model and dataset but no environment, **do not guess, and do not pair their dataset with a demo environment.** Tell them, in order:

1. They have not supplied an environment, and GRPO requires one: unlike SFT or DPO, the labels do not come from the dataset.
2. Their dataset is probably not compatible with `math_with_judge`, the one worked example. It scores a short final answer against an `expected_answer` field per row, so it fits math and short-answer datasets only.
3. Offer exactly two ways forward. **A:** they supply an environment (or a hub / `verifiers` env) written to score their dataset; build it with **nemo-gym-environments**. **B:** run the `math_with_judge` reference with **DAPO-Math-17k**, a known-good pair. Say plainly that B **replaces** their dataset; do not offer a third path that keeps it.

## Defaults

| Field | Value |
|-------|-------|
| Workspace | `default` |
| Plugin | `automodel` (override per **Plugin pick**) |
| Training | SFT + LoRA, `max_seq_length` 2048 |
| Schedule | `epochs` ≥ 1 |
| Automodel batch / optimizer | `global_batch_size` 4, `micro_batch_size` 1 (unknown VRAM; ≥48 GB → `references/batch-sizing.md`); `learning_rate` 5e-5 |

Unsloth, rl DPO and rl GRPO defaults: `references/defaults.md`. Worked examples (Nemotron LoRA, small dense, unsloth, DPO): `references/worked-examples.md`.

## Reference files

| When | Read |
|------|------|
| Commands + job JSON: automodel, unsloth | `references/fast-path-sft.md` |
| Commands + job JSON: rl DPO, GRPO | `references/fast-path-rl.md` |
| Defaults per backend | `references/defaults.md` |
| Worked examples | `references/worked-examples.md` |
| Hyperparameters for a named model (benchmarked recipes) | `references/recipes.md` |
| HF dataset conversion / MCQA shaping | `references/hf-conversion.md` |
| Dataset formats (CHAT/SFT/CUSTOM, unsloth text, DPO preference data) | `references/dataset-formats.md` |
| Field glossary, JSON templates, distillation, W&B/MLflow fields | `references/hyperparameters.md` → `references/hyperparameters-automodel.md`, `references/hyperparameters-unsloth.md`, `references/hyperparameters-rl.md` |
| Batch sizing, multi-GPU, OOM / throughput | `references/batch-sizing.md` |
| Reporting template | `references/reporting.md` |
| Errors: platform unreachable, profiles, images, gated HF | `references/troubleshooting.md` |
| rl Kubernetes runtime, sandboxed Gym | `references/rl-kubernetes-runtime.md` |
| W&B secret, MLflow, jobs-launcher | `references/integrations-setup.md` |
| Post-training eval (base vs LoRA) | `references/post-training-eval.md`, `references/eval_helpers.py` |
| Job polling script | `scripts/poll_customization_job.sh` |
| GRPO environment package and rollout rows | the **nemo-gym-environments** skill |

Related skills: **`nemo-files`**, **`nemo-status`**, **`nemo-secrets`**.
