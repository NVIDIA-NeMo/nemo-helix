---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-customizer
description: >-
  Fine-tune models on NeMo Helix with `automodel`, `unsloth`, or `rl` (all
  Jobs-service backends using `--job-json`): HF dataset conversion, filesets, model entities, and job JSON
  (hyperparameters, batch, schedule, optimizer) + job polling. `automodel`/`unsloth`
  run SFT/LoRA as Docker GPU jobs; `rl` runs DPO (preference) or GRPO (NeMo Gym
  environment + reward) on a Ray cluster (Kubernetes). Covers building and converting
  NeMo Gym environment packages (native-v1, wheels-v1, adapter-wheels-v1) and Gym
  rollout-row datasets. Use for train, fine-tune, customize, SFT, LoRA, DPO, GRPO,
  RLHF, reinforcement learning, reward environment, NeMo Gym, verifiers, Prime
  Intellect, preference optimization, learning rate, epochs, or nemo customization.
  For domain embedding/rerank recipes (retrieval SDG, bi_encoder, retrieve-eval), use
  `nemo-retrieval-recipes` instead of this skill as the conductor.
triggers:
  - nemo-customizer
  - nemo customizer
  - fine-tune
  - fine tune
  - finetune
  - train a model
  - customize a model
  - sft
  - lora
  - dpo
  - grpo
  - rlhf
  - reinforcement learning
  - reward environment
  - nemo gym
  - nemo-gym
  - gym environment
  - verifiers env
  - prime intellect
  - environment fileset
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

End-to-end **SFT + LoRA** (automodel/unsloth), **DPO**, and **GRPO** (rl) on NeMo Helix. Three backend plugins ship in this repo. All submit through `nemo customization <plugin> --job-json ...` and expose no local `run` verb:

| Backend | Invocation | Trains | Where it runs | Pick when |
|---------|------------|--------|---------------|-----------|
| **`automodel`** (default) | `automodel --job-json` | SFT / LoRA | **Kubernetes GPU pods or the platform's Docker GPU executor**. Multi-node (`num_nodes > 1`) needs Kubernetes | General SFT/LoRA; multi-GPU; distillation; full-weight SFT |
| **`unsloth`** | `unsloth --job-json` | SFT / LoRA | Same as automodel; single GPU inside the container | User asks for Unsloth, or its 4-bit LoRA path on one GPU |
| **`rl`** | `rl --job-json` | **DPO** (preference) or **GRPO** (Gym env) | **Kubernetes only**: provisions a **Ray** cluster, no Docker fallback | Preference DPO (full-weight), or GRPO with an environment FileSet + Gym rollout rows (GRPO also supports LoRA) |

The backend command posts to the platform API; training runs in container steps on the platform, **not** in the CLI shell. A docker-runtime platform runs SFT/LoRA fine and cannot run DPO or GRPO at all. Batch shell work; reuse resources with `--exist-ok` (but see the `--exist-ok` rule below); skip CLI `--help` unless a command fails.

## Pre-flight — CLI resolution

Run from the **nemo-helix** git root (top-level `pyproject.toml`), not a plugin subfolder. Resolve the invocation **once** before any other step:

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

## Authentication (optional)

Check `nemo auth status`. If authentication is disabled, skip login. Otherwise: unsigned JWT allowed (local dev) → `nemo auth login --unsigned-token --email <user email or admin@example.com>`; OIDC → `nemo auth login` (or `--username` / `--password` for non-interactive). On 401/403 run the matching login and retry. Use `admin@example.com` unless the user specifies another email. Run `nemo auth status` after login to confirm.

## HuggingFace token (gated models)

Gated HF repos need a platform secret (convention: **`hf-token`**) set as **`token_secret`** on the **model fileset**, not in job JSON. The Files service does not read local `~/.cache/huggingface` or `HF_TOKEN`. For a gated/private repo, before creating the fileset or submitting: `nemo secrets list --workspace default` and confirm `hf-token` exists. If missing, **ask the user** for the token and **stop**. Public repos: omit `token_secret`. Details: `references/troubleshooting.md` § **Gated HuggingFace models**.

## Plugin pick

1. Run `nemo jobs list-execution-profiles -f json` (login first only if auth is enabled; parsing tips in `references/troubleshooting.md`).
2. **DPO / preference optimization** (`{prompt, chosen, rejected}` data, "align", "RLHF-style"), **GRPO / Gym / Prime Intellect env**, or an explicit NeMo-RL request → **`rl`** (needs a GPU profile **and** `platform.runtime: kubernetes`).
3. Else explicit Unsloth request → **`unsloth`**.
4. Else explicit Automodel request → **`automodel`**.
5. Else any profile with `provider: gpu` or `gpu_distributed` → **`automodel`** (default).
6. Else stop and tell the user GPU customization is unavailable.

**`rl` runtime gate:** before submitting rl, confirm the `cpu`/`gpu` profiles report `backend: kubernetes_job` (or `volcano_job`). If they report `docker`/`subprocess`, **stop**: tell the user DPO and GRPO need a Kubernetes-runtime platform. Do **not** start or reuse a docker-runtime platform and do **not** fall back to automodel/unsloth. To configure one: `references/rl-kubernetes-runtime.md`.

## Critical rules

The full list (about 40 bullets, including all rl/GRPO environment pitfalls) is in **`references/gotchas.md`**. Read it before the first submit. Never miss these:

- **Platform unreachable:** if the user gave a custom URL (or you exported a non-default `NHX_BASE_URL`), stop and report it unreachable. On the default `http://127.0.0.1:8080` only, **ask** before starting services locally; never start without asking, and never use that docker-runtime start for `rl`.
- **Parse stdout only.** `submit`, `explain` and `-f json` write JSON on stdout; never pipe `2>&1` into `json.load`. A parse error after a *successful* submit is the usual cause of duplicate jobs. If submit output can't be parsed, stop and inspect before re-submitting.
- **`--exist-ok` reuses a fileset WITHOUT re-uploading files.** Changed data trains on the old contents with no error. When data changed, upload to a **new** fileset name (e.g. `sft-data-v2/`). Never delete a fileset without explicit user confirmation of the exact `<workspace>/<name>`.
- **A `model`/`dataset` ref belongs in one place:** submit fails if the job JSON sets it and you also pass the matching `--upload-*` flag.
- **Don't add `max_steps` to a real run.** It takes precedence over `epochs` and can stop mid-epoch; fixtures set it only so smoke tests finish fast, so do not copy it. Every backend's schema accepts `epochs` and `max_steps` together: keep both when the user sets both, and don't reject the job for it.
- **Fixtures are not the schema.** Authoritative order: `nemo customization <plugin> explain`, then `references/hyperparameters.md` § **Source of truth**, then this skill.
- **Job done = top-level `status`** in `completed` | `error` | `cancelled`. Steps can all be `completed` while the job is still `active`. Poll with `scripts/poll_customization_job.sh <job-id>` (works for `automodel-…`, `unsloth-…`, `rl-…`; exits 1 on `error`/`cancelled`).
- **Job id comes from submit stdout** (`"name"` field). `rl` has no `--name` flag; never pick the newest `rl-*` from `nemo jobs list`.
- **Use `nemo jobs list-execution-profiles -f json`, not local `docker info`,** to pick a backend.
- **Remote platform, missing training image:** do **not** `docker build`/`pull` on the agent machine; report using `references/reporting.md` and `references/troubleshooting.md` § **Missing training images**.
- **Prefer CHAT JSONL** when the model has a chat template (`references/dataset-formats.md`). HF dataset id from the user → convert locally (`references/hf-conversion.md`); dataset fileset name = HF dataset **name** only.
- **Named model → start from its recipe** in `references/recipes.md`. Nemotron needs Hopper/Blackwell GPUs; train the **BF16** checkpoint for Nemotron MoE.
- **Unsloth is single-GPU per job** (no `parallelism` block); multi-GPU sharding → automodel.
- **LoRA adapters hot-reload** onto a READY `lora_enabled: true` deployment; full-weight/merged outputs register a new model entity you must deploy. Eval routing: `references/post-training-eval.md`.
- **Deploy with vLLM by default.** For NIM, prefer the model-free NIM over the multi-LLM NIM (`llm-nim`), which cannot serve `lora_merged` outputs. See `references/post-training-eval.md` § **Pick the engine**.
- **GRPO:** needs TWO filesets (environment + dataset) and **no environment supplied ⇒ stop and ask** (below); the environment package must contain no `.jsonl`; validate with `pi-to-gym-conversion --validate-only <dir>` before upload; report progress on **reward, not loss**. DPO: one `dataset` string ref holding both `training.jsonl` and `validation.jsonl`.

### No environment supplied (GRPO)

A dataset alone cannot produce a GRPO job: the reward comes from an environment written for that dataset's rows. If the user names a model and dataset but no environment, **do not guess and do not pair their dataset with a demo environment** (a mismatch trains on a meaningless reward). Tell them they have not supplied an environment, that their dataset is probably not compatible with the shipped `math_with_judge` example, and offer exactly two options: **A** supply a matching environment (or hub/`verifiers` env), or **B** run the `math_with_judge` reference with **DAPO-Math-17k** (this *replaces* their dataset). Details: `references/defaults.md`.

## Workflow

Common steps, then branch by plugin pick (full per-backend checklists: `references/workflow-checklists.md`):

```text
- [ ] Resolve CLI; cd nemo-helix; export NHX_BASE_URL only if the user gave one
- [ ] nemo auth status (login only if enabled)
- [ ] nemo jobs list-execution-profiles -f json → Plugin pick (rl: confirm kubernetes_job or volcano_job, per the rl runtime gate)
- [ ] Convert HF dataset → /tmp/train-data/*.jsonl (references/hf-conversion.md)
- [ ] Create dataset fileset (new name when the data changes; no --exist-ok), upload JSONL, `nemo files list` to verify
      (automodel/unsloth: train.jsonl [+ validation.jsonl]; rl: training.jsonl + validation.jsonl)
- [ ] Gated HF base model? confirm `hf-token`
- [ ] LoRA job: base already deployed without LoRA? ask the user before submit (references/post-training-eval.md § **Base already deployed without LoRA**)
- [ ] Create HF weights fileset + model entity if missing
- [ ] Look up the model's recipe in references/recipes.md
- [ ] Write /tmp/job.json, submit `nemo customization <plugin> --job-json /tmp/job.json --workspace default`
- [ ] Poll to a terminal status, then report (references/reporting.md)
```

## Fast paths

Copy-paste commands and job JSON per backend. Read the one for the chosen backend:

- **automodel / unsloth (SFT, LoRA):** `references/fast-path-sft.md`
- **rl (GRPO and DPO):** `references/fast-path-rl.md`. Its model-entity step reuses step 2 of `references/fast-path-sft.md`. GRPO environment formats and building one: `references/gym-environments.md`.

Minimal automodel job JSON (`model` is the registered model **entity** `default/<model-entity>`, not an HF id):

```json
{
  "model": "default/<model-entity>",
  "dataset": { "training": "default/<dataset-fileset>", "validation": "default/<dataset-fileset>" },
  "training": { "training_type": "sft", "finetuning_type": "lora", "lora": { "rank": 16, "alpha": 32 }, "max_seq_length": 2048 },
  "schedule": { "epochs": 1 },
  "batch": { "global_batch_size": 4, "micro_batch_size": 1 },
  "optimizer": { "learning_rate": 5e-5, "weight_decay": 0.01, "warmup_steps": 0 },
  "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1, "tensor_parallel_size": 1 },
  "output": { "name": "<output-name>" }
}
```

Submit and poll (`JOB` is the `"name"` field of submit stdout; it already carries the `automodel-` prefix, so pass it verbatim):

```bash
nemo customization automodel --job-json /tmp/job.json --workspace default > /tmp/automodel-submit.json
JOB=$(python3 -c "import json;print(json.load(open('/tmp/automodel-submit.json'))['name'])")
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"
```

## Defaults

| Field | Value |
|-------|-------|
| Workspace | `default` |
| Plugin | `automodel` (override per **Plugin pick**) |
| Training | SFT + LoRA, `max_seq_length` 2048 |
| Schedule | `epochs` ≥ 1; omit `max_steps` |
| Automodel batch / optimizer | `global_batch_size` 4, `micro_batch_size` 1 (unknown VRAM; ≥48 GB → `references/batch-sizing.md`); `learning_rate` 5e-5 |

Unsloth, rl DPO and rl GRPO defaults: `references/defaults.md`. Batch sizing and OOM/throughput tuning: `references/batch-sizing.md`.

## Worked examples

Automodel (Nemotron LoRA), automodel small dense, unsloth and rl DPO: `references/worked-examples.md`.

## Report to user

After polling reaches a **terminal** status (`completed`, `error`, or `cancelled`), report using the template in **`references/reporting.md`** (one format for all backends: result header, training configuration table, using the adapter vs the fine-tuned model, metrics, notes by status, error follow-ups).

## Reference files

| When | Read |
|------|------|
| Gotchas, full list (all backends + rl) | `references/gotchas.md` |
| Per-backend workflow checklists | `references/workflow-checklists.md` |
| Fast path commands + job JSON | `references/fast-path-sft.md`, `references/fast-path-rl.md` |
| Defaults per backend; GRPO "no environment" rule | `references/defaults.md` |
| Worked examples | `references/worked-examples.md` |
| Hyperparameters for a named model (benchmarked recipes) | `references/recipes.md` |
| HF conversion / MCQA shaping | `references/hf-conversion.md` |
| Dataset formats (CHAT/SFT/CUSTOM, unsloth text, DPO triples, Gym rollout rows) | `references/dataset-formats.md` |
| GRPO environments (`native-v1`, `wheels-v1`, `adapter-wheels-v1`), example scripts | `references/gym-environments.md`, `scripts/grpo-examples/README.md` |
| Field glossary, JSON templates, distillation, W&B/MLflow fields | `references/hyperparameters.md` → `-automodel.md` / `-unsloth.md` / `-rl.md` |
| Batch sizing, multi-GPU, OOM / throughput | `references/batch-sizing.md` |
| Reporting template | `references/reporting.md` |
| Errors: backend choice, profiles, images, gated HF, CLI JSON parsing, connection | `references/troubleshooting.md` |
| rl Kubernetes runtime, sandboxed Gym | `references/rl-kubernetes-runtime.md` |
| W&B secret, MLflow, jobs-launcher | `references/integrations-setup.md` |
| Post-training eval (base vs LoRA) | `references/post-training-eval.md`, `references/eval_helpers.py` |
| **Live JSON schema (authoritative)** | `nemo customization automodel\|unsloth\|rl explain` |
| Payload **shape** only (never field set or values) | `plugins/nemo-{automodel,unsloth,rl}/tests/fixtures/*.json`; MoE LoRA layout: `plugins/nemo-automodel/tests/fixtures/nemotron_moe_sft_lora.json` |
| Environment manifest + validation source of truth | `services/rl/src/nhx/rl/schemas/environment.py`, `services/rl/src/nhx/rl/tasks/environment/validate.py` |

Related: `plugins/nemo-{automodel,unsloth,rl}/README.md`, `plugins/nemo-customizer/docs/CUSTOMIZATION.md`, skills **`nemo-files`**, **`nemo-status`**, **`nemo-secrets`**.
