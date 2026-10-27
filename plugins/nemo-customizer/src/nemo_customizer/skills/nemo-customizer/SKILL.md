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

## Plan the job

Settle the plan before any data work, and confirm it with the user in one message: model, method, fine-tuning type, dataset and GPUs.

- **Method.** SFT, unless the data is preference pairs (DPO) or the reward comes from an environment (GRPO).
- **Fine-tuning type.** LoRA by default, since it is much cheaper. Use full weights when the user asks for them or needs a standalone model. DPO trains full weights only. GRPO LoRA needs Hopper or newer GPUs, so plan full-weight GRPO on older ones (`references/hyperparameters-rl.md` § **`policy_backend`**).
- **Model, when the user names none.** SFT: `Qwen/Qwen3-1.7B`. It is ungated, Apache-2.0 and has a chat template, and it fits LoRA and full SFT on one GPU. For LoRA on **4 or more Hopper/Blackwell GPUs**, use `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` instead (`references/recipes.md`); full SFT stays on Qwen3-1.7B. Both are thinking models, so short-answer eval turns thinking off (`references/post-training-eval.md`). DPO and GRPO: `references/defaults.md`.
- **Dataset, when the user names none.** SFT: `tau/commonsense_qa`, converted to CHAT rows (`references/hf-conversion.md`). DPO: `nvidia/HelpSteer3` preference pairs (`references/defaults.md`). GRPO: the environment supplies the rows (**GRPO** below).
- **User-supplied values win.** When the user gives hyperparameters or a job JSON, keep their values. Only validate them (`explain`, batch divisibility, the `rl` runtime gate), and tell the user every field you changed and why.

## Workflow

Commands and job JSON for each backend: `references/fast-path-sft.md` (automodel, unsloth) and `references/fast-path-rl.md` (DPO, GRPO).

1. **Convert the dataset.** An HF dataset id from the user → convert it locally (`references/hf-conversion.md`); do not ask for local paths. Write CHAT `messages` rows when the model has a chat template: `references/dataset-formats.md` says why and gives each backend's row shape. DPO keeps its preference rows instead (`references/dataset-formats.md` § **NeMo-RL (DPO)**), and GRPO rows come from the environment (see **GRPO** below). For SFT and LoRA, then **freeze a test split** before anything is uploaded (`references/post-training-eval.md` § **1. Freeze a test split**); DPO and GRPO skip it (step 8).
2. **Upload it to a dataset fileset.** Name the fileset after the HF dataset only (`tau/commonsense_qa` → `commonsense_qa`), not the model. **When the data changes, upload to a new fileset name** (`sft-data-v2/`): `--exist-ok` reuses an existing fileset **without re-uploading**, so edited files silently train on the old contents. Do not update a fileset in place unless the user confirms nothing else uses it, and never delete one without their explicit confirmation of the exact `<workspace>/<name>`. Run `nemo files list` to verify. The test split goes to its own fileset, `<dataset-fileset>-test`.
3. **Register the model.** HF weights fileset + model entity, if missing (gated repos: see Pre-flight). Submit without polling `nemo models get`; the model spec fills in asynchronously. To create the model and dataset during submit instead: `references/fast-path-sft.md` § **Create resources at submit**.
4. **Score the base model** on the test split, before training: serve it (LoRA: with `lora_enabled: true`) and hand off to **nemo-evals** (`references/post-training-eval.md` § **2. Serve the base model and score it**). SFT and LoRA only; GRPO compares validation reward inside the job. **LoRA:** if the base is already deployed without LoRA, ask the user before deploying it again (same file, § **Base already deployed without LoRA**).
5. **Write the job JSON.** For a named model, start from its recipe in `references/recipes.md`; otherwise use **Defaults** below. Set `deployment_config` so the job serves its output when training finishes. **Do not add `max_steps` to a real run**: it overrides `epochs` and can stop mid-epoch. Test fixtures set it only so smoke tests finish fast. The live schema, `nemo customization <plugin> explain`, outranks this skill and its references.
6. **Submit and read the job id.** Parse **stdout only**: `submit`, `explain` and `-f json` write JSON on stdout and warnings on stderr, so `2>&1` into `json.load` fails even when the job was created, and re-submitting after that parse error is the usual cause of duplicate jobs. If submit output does not parse, check `nemo jobs list` before submitting again. The job id is the `"name"` field, already prefixed (`automodel-…`, `unsloth-…`, `rl-…`); pass it verbatim. `rl` has no `--name` flag, and the newest `rl-*` in `nemo jobs list` may be someone else's job.

   ```bash
   nemo customization automodel --job-json /tmp/job.json --workspace default > /tmp/submit.json
   JOB=$(python3 -c "import json;print(json.load(open('/tmp/submit.json'))['name'])")
   ```

7. **Poll to a terminal status, watching training health.** The job is done only when its **top-level** `status` is `completed`, `error` or `cancelled`. Every step can show `completed` while the job is still `active`, and `status_details.phase` can sit at `training` with `progress_pct: 100` for a while. `bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"` polls any backend's job every 15s (append seconds to change it) and prints progress with the latest loss, or reward and health for GRPO. It exits 0 on `completed` and 1 on `error`/`cancelled`. Report every `WARN:` line it prints. On an `ALERT:` it exits 3 while the job keeps running: show the user the alert and ask whether to cancel (`nemo jobs cancel "$JOB"`) or keep going (re-run with `"$JOB" 15 --continue-on-alert`). Causes and fixes: `references/troubleshooting.md` § **Runtime failures**.
8. **Score the tuned model** on the same test split, with the same settings (`references/post-training-eval.md` § **3. Score the tuned model**). An eval failure never fails the run: report it separately. DPO and GRPO have no test split: report the validation metrics the job logs (`references/reporting.md` § **RL (DPO) example**, § **RL (GRPO) example**).
9. **Report** with `references/reporting.md`, leading with base vs tuned for SFT and LoRA.

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
| Model, dataset | `Qwen/Qwen3-1.7B` on `tau/commonsense_qa` (**Plan the job**) |
| Training | SFT + LoRA, `max_seq_length` 2048 |
| Schedule | `epochs` ≥ 1 |
| Automodel batch | `global_batch_size` 4, `micro_batch_size` 1 (unknown VRAM; ≥48 GB → `references/batch-sizing.md`) |
| Learning rate | Leave `learning_rate` unset: each backend's schema picks it per method and fine-tuning type, and `explain` lists the values |

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
| Evaluation: frozen test split, base vs tuned, model IDs | `references/post-training-eval.md` |
| Job polling and training health | `scripts/poll_customization_job.sh`, `scripts/job_health.py` |
| GRPO environment package and rollout rows | the **nemo-gym-environments** skill |

Related skills: **`nemo-files`**, **`nemo-status`**, **`nemo-secrets`**.
