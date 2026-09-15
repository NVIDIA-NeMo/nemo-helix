---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-customizer
description: >-
  Fine-tune models on NeMo Platform with `automodel`, `unsloth`, or `rl` (all
  `submit`-only): HF dataset conversion, filesets, model entities, and job JSON
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
  - automodel submit
  - unsloth submit
  - rl submit
not-for:
  - nemo-build-agent (agent scaffold/deploy, not weight training)
  - nemo-explore (agent design only)
  - nemo-setup (platform install; route here when CLI resolution fails)
  - safe-synthesizer (tabular synthetic data training)
  - nemo-retrieval-recipes (domain embed/rerank SDG → train → retrieve-eval conductor)
compatibility: >-
  Requires nemo-customizer-plugin and a customization contributor (`nemo.customization.contributors`).
  Platform must expose jobs, files, and models APIs.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Grep]
---

# NeMo Customizer

End-to-end **SFT + LoRA** (automodel/unsloth), **DPO**, and **GRPO** (rl) on NeMo Platform. Three backend plugins ship in this repo — all are **`submit`-only** and expose no local `run` verb:

| Backend | Verb | Trains | Where it runs | Pick when |
|---------|------|--------|---------------|-----------|
| **`automodel`** (default) | `submit` | SFT / LoRA | **Kubernetes GPU pods or the platform's Docker GPU executor** — whichever the platform provides. Multi-node (`num_nodes > 1`) needs Kubernetes | General SFT/LoRA; multi-GPU (data/tensor parallel); distillation; full-weight SFT |
| **`unsloth`** | `submit` | SFT / LoRA | Same as automodel — either runtime, 4 steps (download → train → upload → model-entity). Single-GPU inside the container | User asks for Unsloth, or wants Unsloth's 4-bit LoRA path / optimizer defaults on a single GPU |
| **`rl`** | `submit` | **DPO** (preference) or **GRPO** (Gym env) | **Kubernetes only** — provisions a **Ray** cluster, no Docker fallback; 4 steps (download → train → upload → model-entity) | Preference DPO (full-weight), or GRPO with an environment FileSet (`native-v1` / `wheels-v1` / `adapter-wheels-v1`) + Gym rollout rows; GRPO also supports LoRA |

`nemo-customizer` is the router (`nemo customization …`); training backends are separate plugins (`nemo-automodel`, `nemo-unsloth`, `nemo-rl`). `submit` posts to the platform API; the platform runs training in container steps — **not** in the CLI shell. Heavy ML deps live in container images only.

**Runtime split:** `automodel`/`unsloth` run on **either** `platform.runtime` — Kubernetes GPU pods or the local Docker GPU executor (`require_container_runtime`), whichever the platform offers. Their one exception is **multi-node** (`parallelism.num_nodes > 1`), which compiles to a `gpu_distributed` executor only Volcano/Kubernetes can place. `rl` is Kubernetes-only in every case (`require_distributed_runtime`) — it provisions a Ray cluster and has no Docker fallback. So a docker-runtime platform runs SFT/LoRA fine and cannot run DPO or GRPO at all.

Decision rule below in **Plugin pick**. Batch shell work; reuse resources with `--exist-ok`; skip CLI `--help` unless a command fails.

## Pre-flight — CLI resolution

Run from the **nemo-platform** git root (top-level `pyproject.toml`), not a plugin subfolder. Example commands below use `nemo …` — resolve the invocation **once** before any other step:

```bash
cd /path/to/nemo-platform
if command -v nemo >/dev/null 2>&1; then
  echo "nemo"
elif command -v uv >/dev/null 2>&1 && uv run nemo --help >/dev/null 2>&1; then
  echo "uv run nemo"
else
  echo "CLI_NOT_FOUND"
fi
```

| Result | Action |
|--------|--------|
| `nemo` | Use `nemo …` for all commands in this workflow |
| `uv run nemo` | Prefix every command with `uv run` (repo dev checkout without `nemo` on `PATH`) |
| `CLI_NOT_FOUND` | Stop. Route to **nemo-setup** (`make bootstrap` then `nemo setup` from the nemo-platform repo root). Do not continue. |

## Authentication (optional)

Platform auth is **not required** to run customization when the cluster has authentication disabled. Check with `nemo auth status` — if it reports authentication is disabled, skip login and proceed.

When auth **is** enabled on the connected platform, API calls need credentials:

| Situation | Action |
|-----------|--------|
| Auth disabled | Skip login |
| Auth enabled, unsigned JWT allowed (typical local dev: `auth.allow_unsigned_jwt: true`) | `nemo auth login --unsigned-token --email <user email or admin@example.com>` |
| Auth enabled, OIDC configured | `nemo auth login` (or `--username` / `--password` for non-interactive) |
| 401/403 on any platform call | Run the matching login above, then retry |

Use `admin@example.com` unless the user specifies another email. Run `nemo auth status` after login to confirm.

## HuggingFace token (gated models)

Gated HF repos (Llama, Gemma, Mistral instruct, …) need a platform secret (convention: **`hf-token`**) referenced as **`token_secret`** on the **model fileset** — not in job JSON (unlike W&B's `api_key_secret`). The Files service does **not** read your local `~/.cache/huggingface` or shell `HF_TOKEN`.

| Model access | Action |
|--------------|--------|
| Public (e.g. `Qwen/Qwen3-1.7B`) | Skip; omit `token_secret` on the fileset |
| Gated / private HF repo | Before model fileset creation or job submit: `nemo secrets list --workspace default` and confirm `hf-token` exists. If missing, **ask the user** for their HF token and **stop** — do not create the fileset or submit until wired up. |

Full create/update commands, fileset `token_secret`, license acceptance, and download-phase errors: `references/troubleshooting.md` § **Gated HuggingFace models**.

## Plugin pick

1. Run `nemo jobs list-execution-profiles -f json` (login first only if auth is enabled — see **Authentication**; see `references/troubleshooting.md` for parsing).
2. If the task is **DPO / preference optimization** (a `{prompt, chosen, rejected}` dataset, "align", "preference", "RLHF-style"), **GRPO / Gym / Prime Intellect env**, **or** the user explicitly asked for NeMo-RL → **`rl`** (requires a GPU profile **and** `platform.runtime: kubernetes`).
3. Else if the user explicitly asked for Unsloth → **`unsloth`**.
4. Else if the user explicitly asked for Automodel → **`automodel`**.
5. Else if any profile has `provider: gpu` or `gpu_distributed` → **`automodel`** (default, SFT/LoRA).
6. Else stop and tell the user GPU customization is unavailable (all backends need a GPU execution profile; `automodel`/`unsloth` accept either runtime — except multi-node, which needs Kubernetes — while `rl` needs `platform.runtime: kubernetes`).

**`rl` runtime gate:** `rl submit` fails fast unless the platform runs `platform.runtime: kubernetes` (`require_distributed_runtime`). rl job steps execute as **Kubernetes pods via the `kubernetes_job` execution backend** — the **`docker` job backend cannot run rl**. Before submitting rl, confirm with `nemo jobs list-execution-profiles -f json` that the `cpu`/`gpu` profiles report `backend: kubernetes_job` (or `volcano_job`). If they report `backend: docker`/`subprocess`, the platform is **not** configured for rl: stop and tell the user DPO and GRPO need a Kubernetes-runtime platform — do **not** start/reuse a docker-runtime platform, and do **not** fall back to automodel/unsloth (those are SFT/LoRA, neither DPO nor GRPO). To stand up or configure one, see `references/rl-kubernetes-runtime.md`.

For **`automodel`/`unsloth`**, training never runs inside the `nemo` CLI process. After `submit`, the platform launches GPU container steps — as Kubernetes pods, or on the Docker daemon attached to that platform host when the platform is Docker-backed. On a Docker-backed platform that daemon is often the same machine as `http://127.0.0.1:8080`, but always query the platform for its executors — not the agent's shell GPU or a separate `docker info` on another box. **`rl` does not use the Docker executor** — its steps run on the Kubernetes cluster the platform is configured against.

## Gotchas

- Resolve the CLI per **Pre-flight — CLI resolution** before any `nemo …` command; run from the **nemo-platform** git root, not a plugin subfolder.
- Set `NMP_BASE_URL` only when the user gives a platform URL; default `http://127.0.0.1:8080` (same as `http://localhost:8080`). The `nemo` CLI reads this env var (see SDK `NMP_BASE_URL`). Track whether the user **overrode** the base URL — see **Platform unreachable** below.
- **Platform unreachable** — if any platform API call fails with a connection error (`Connection error`, timeout, refused):
  - **User gave a custom URL** or you exported a non-default `NMP_BASE_URL`: stop and tell the user the platform is not reachable at that address. Do **not** offer to start local services.
  - **Default URL only** (no user override): **ask** whether to start the platform locally. If they agree, from the **nemo-platform** git root run in the **background**:

    ```bash
    nemo services run \
      --host 0.0.0.0 \
      --port 8080 \
      --controllers jobs,entities,models \
      --service-group all
    ```

    Poll until healthy (`curl -sf http://127.0.0.1:8080/health/ready` or retry `nemo jobs list-execution-profiles -f json`), then continue the workflow. Do not start services without asking.
    - ⚠️ **This default start is a DOCKER-runtime platform — valid for single-node `automodel`/`unsloth` only.** It is **NOT** valid for **`rl`**: rl needs `platform.runtime: kubernetes` with a `kubernetes_job` execution backend. Starting this default and submitting rl will fail the runtime gate. For rl, configure/point at a Kubernetes-runtime platform instead — see `references/rl-kubernetes-runtime.md`. Never start or reuse a docker-runtime platform for rl.
- **All backends are `submit` only** — use `nemo customization <plugin> submit …`; automodel, unsloth, and rl expose no local `run` verb. Do not improvise verbs or pass `--venv`.
- **Test fixtures are not the schema.** `tests/fixtures/*.json` are smoke-test inputs: they carry whatever made a test cheap, exercise one path rather than the field set, and nothing fails when the schema gains a field they never set. Read one for where a block sits in the payload — never for which fields exist, what a default is, or what a sensible value looks like, and never conclude a field is unsupported because a fixture omits it. Authoritative, in order: `nemo customization <plugin> explain` (the installed build's live schema), then the schema source files in `references/hyperparameters.md` § **Source of truth**, then this skill. When a fixture and `explain` disagree, the fixture is stale — say so rather than following it.
- **Never set `max_steps` together with `epochs`** (automodel + unsloth; rl has the same caveat — see **rl (DPO / GRPO) gotchas**). `max_steps` is a global cap and stops mid-epoch. Every fixture in this repo sets it so a smoke test finishes in a minute — the most-copied wrong value here. Unsloth's schema enforces this as a hard mutex; automodel allows both but the result is surprising.
- **Job done (all backends) = top-level `status`** in `completed` | `error` | `cancelled`. Steps can all be `completed` while the job is still `active` (upload, entity registration). `status_details.phase` may stay `training` with `progress_pct: 100` for a long time — keep polling. `poll_customization_job.sh` works for any job id (`automodel-…`, `unsloth-…`, or `rl-…`); it exits **1** on `error` or `cancelled`.
- Model spec fills async: **submit without polling** `nemo models get` unless submit fails.
- HF dataset id from the user → convert locally; do not ask for local paths first.
- Dataset fileset name = HF dataset **name** only (`tau/commonsense_qa` → `commonsense_qa`), not the model name.
- Prefer **CHAT** JSONL when the model has a chat template; details in `references/dataset-formats.md` (automodel auto-detects schema; unsloth needs `dataset.apply_chat_template: true` to consume `messages`).
- User asks to tune **batch or parallelism** (automodel) → `references/batch-sizing.md`. Other fields (LR, epochs, LoRA rank, distillation) → `references/hyperparameters-automodel.md`. For unsloth batch sizing see `references/batch-sizing.md`; for unsloth fields see `references/hyperparameters-unsloth.md`. Run `nemo customization <plugin> explain` for the live schema.
- Skill **defaults** (`micro_batch_size` 1, `global_batch_size` 4) are safe on unknown VRAM. When the user has **≥48 GB** on one GPU, use `references/batch-sizing.md` instead of defaults. Unsloth's analogues are `batch.per_device_train_batch_size` and `batch.gradient_accumulation_steps` (effective batch = product).
- **Unsloth training is single-GPU per job** (inside the container). `hardware.gpus` sets `CUDA_VISIBLE_DEVICES` before `import torch` — **selection, not reservation**. No `parallelism`/TP/PP block in job JSON. Multi-GPU sharding → use automodel. Pass `--profile <name>` on `unsloth submit` when the default `gpu` profile is wrong (automodel sets `training.execution_profile` in JSON instead).
- **Unsloth validation defaults** — when `dataset.validation_path` is set and `schedule.eval_steps` is omitted, the trainer runs validation once per effective epoch automatically. Report final `metrics.val_loss` from job status (see `references/reporting.md`). Set `eval_steps` explicitly to override cadence.
- **Do not use local `docker info`** to pick automodel vs unsloth. Run `nemo jobs list-execution-profiles -f json` against the user's platform (login first only if auth is enabled — see **Authentication**; see `references/troubleshooting.md`). Default output is a table — **`-f json` is required** for scripting; parse **stdout only** (do not pipe `2>&1` into `json.load`).
- **Do not merge stderr into stdout when parsing JSON** — `submit`, `explain`, and `-f json` commands write **JSON on stdout**; harmless warnings like `Configuration file not found, using defaults` go to **stderr**. Piping with **`2>&1`** before `json.load` raises `JSONDecodeError` even when submit **succeeded** — a common cause of **duplicate jobs** when the agent re-submits after a parse error. Parse stdout only; redirect stderr if needed (`2>/dev/null`). See `references/troubleshooting.md` § **Parsing CLI JSON**.
- For submit/image/plugin errors (all backends), read `references/troubleshooting.md`. Unsloth needs the `nmp-unsloth-training` container image on the **platform host's** Docker daemon (see `docker/unsloth/README.md`); rl needs the `nmp-customizer-tasks` / `nmp-rl-training` images on the Kubernetes cluster (see **rl (DPO) gotchas** and `references/rl-kubernetes-runtime.md`).
- **Missing training image on a remote platform** — if the user gave a non-localhost `NMP_BASE_URL` and the job errors with `Failed to pull image`, `manifest unknown`, or missing `nmp-unsloth-training` / automodel training image: **do not** run `docker build`, `docker pull`, or `docker buildx bake` on the agent machine. Report with the template in `references/reporting.md` (use **Output adapter fileset (planned):** on error), then append on-target build steps from `references/troubleshooting.md` § **Missing training images**.
- **Gated HuggingFace models** (Llama, Gemma, …) — confirm `hf-token` + fileset `token_secret` before submit; download fails with `Failed to access upstream storage` / 502 when missing. See **HuggingFace token (gated models)** and `references/troubleshooting.md` § **Gated HuggingFace models**.
- **Post-training eval format** — CHAT SFT: use the same CHAT `messages` JSONL as training. **Do not** flatten rows to `prompt`/`expected`. Send `messages[:-1]` at inference; score against `messages[-1].content`. See `references/post-training-eval.md`. **Embedding / rerank jobs** (`recipe: bi_encoder` / `cross_encoder`): do **not** use CHAT `evaluate`. Hand off to `nemo-retrieval-recipes` or submit `nemo evaluator retrieve-eval` on the frozen `eval_beir` fileset from Stage 1.
- **LoRA adapters load automatically for eval** — when a LoRA job completes (automodel/unsloth `save_method: lora`, or **rl GRPO with `finetuning_type: "lora"`**), the adapter is registered on the base model entity and hot-reloaded on any **READY** deployment with `lora_enabled: true`. **Do not** create or update deployments before LoRA eval. **Full SFT** (`finetuning_type: all_weights`) and **merged checkpoints** (`merged_16bit` / `merged_4bit`) register a new **model** entity at `output.name` — **deploy that entity for inference** before chat or eval; full weights are not hot-reloaded onto the base deployment. For LoRA eval, route through the **provider** gateway (`/provider/<name>/-/v1` with `model: default--<adapter>`); the model-entity path (`/model/<entity>/-/v1`) always hits the base model. See `references/post-training-eval.md` § **Request routing (base vs LoRA)**.

### rl (DPO / GRPO) gotchas

- **rl is DPO or GRPO, not SFT** — DPO trains on **preference pairs** `{prompt, chosen, rejected}`; GRPO needs an **environment** FileSet + a Gym rollout-row dataset. Don't route SFT/LoRA work here, and don't route DPO/GRPO to automodel/unsloth.
- **DPO is full-weight only; GRPO does LoRA too** — set `finetuning_type: "lora"` on the GRPO `training` block (plus an optional `lora` block). The output type is **inferred**, so `output` still carries only `name`. Three things the schema enforces: `lora` must be omitted when `finetuning_type` is `all_weights`; `lora_merged` is rejected outright (no merge at export — train full-weight if merged weights are the goal); and `lora.use_triton` must be left **unset** — the compiler picks (`true` at TP 1, `false` above), and an explicit `true` with `tensor_parallel_size > 1` is *rejected at submit*, not downgraded. Fields and module-selection rules: `references/hyperparameters-rl.md` § **LoRA (GRPO only)**.
- **There is no `grpo` subcommand** — GRPO submits through **`nemo customization rl submit`** like DPO, selected by `training.type: "grpo"` in the job JSON. `training.type` is the union discriminator and is **required**: omitting it fails with `union_tag_not_found` rather than defaulting.
- **GRPO needs TWO FileSets** — an `environment` (code + config, `purpose=environment`) and a `dataset` (prompt rows, `purpose=dataset`). Both are plain string refs in the job JSON. Three environment formats are supported — `native-v1`, `wheels-v1`, `adapter-wheels-v1` — and picking one is the first question to settle. Full guide: `references/gym-environments.md`.
- **Never put `.jsonl` in the environment package** — validation rejects it outright. This bites the `native-v1` path especially: Gym's own configs point `datasets[].jsonl_fpath` at an in-tree file, so an environment copied straight from the Gym source tree fails until the data dir is stripped and the prompts move to the dataset FileSet.
- **Gym YAML: instance ≠ implementation.** The top-level key is the **instance** (unique at runtime); the key under the server type is the **implementation directory** Gym runs (`{server_type}/{implementation}/`). Both `{type, name}` refs and a dataset row's `agent_ref.name` name the **instance**. They're often equal in Gym's own configs, which is why this gets missed. Every package also needs a `policy_model` `responses_api_models` config, listed **first** in `config_paths`, or spin-up dies with `ServerRefNotFoundError: ... Available responses_api_models: (none)`.
- **Two things silently break a hand-built environment.** (1) A custom `{server_type}/{implementation}/` directory with **no `requirements.txt` or `pyproject.toml`** is not recognised as a server — if a Gym built-in shares the name, Gym runs *that* instead and trains the wrong environment with no error. (2) A `pyproject.toml` at the **package root** makes Gym think it is inside a Gym checkout and switch to its editable-install branch, which **drops the automatic `nemo-gym==<image version>` pin**. What breaks then depends on the server's requirements: keep Gym's `-e nemo-gym[dev] @ ../../` line and pip installs your *package root* as `nemo-gym`; omit it and the venv gets no `nemo-gym` at all. Keep the server's original directory structure from the Gym checkout; never repackage as a setuptools distribution.
- **`domain` is required on every `resources_servers` block** and validated against a closed set (`math`, `coding`, `agent`, `knowledge`, `instruction_following`, `long_context`, `safety`, `games`, `translation`, `e2e`, `rlhf`, `other`). Empty or invalid makes it an "almost-server" and **aborts spin-up** with `AlmostServerError` — it is not skipped. Agent and model blocks must not carry `domain`.
- **Offline is about wheelhouse completeness, not the format name.** There are two installs: Gym builds each server's venv with an **index still enabled** (`wheels/` is only a `UV_FIND_LINKS` candidate pool), then NeMo-RL installs the closure with `--no-index`. A `wheels-v1` job is offline-clean only when `wheels/` also covers that first step — the server's requirements **plus** `nemo-gym` at the training image's exact version, `ray[default]` and `openai` at Gym's pins. `references/gym-environments.md` § **Dependency installation**.
- **Validate the environment package before uploading** — `uv run --package nmp-rl pi-to-gym-conversion --validate-only <dir>` prints `{"valid": true, ...}` or exits 1 with the exact violation. The same checks run at submit against the FileSet listing, so this catches the failure minutes earlier and for free.
- **GRPO convert is CLI-first** — run `pi-to-gym-conversion` on a host with internet; training clusters consume uploaded FileSets only (no hub egress). **Pin `--hub-version`**: unset takes whatever the index offers now, and a later release can narrow `Requires-Python` or ship code the training image's Python cannot run. `sandboxed` is platform config (`NMP_RL_SANDBOXED_GYM_DEFAULT`, default true), not a job JSON field.
- **GRPO progress is read on reward, not loss** — the GRPO surrogate loss oscillates around zero and carries no signal about run quality. Report `train_reward` and `val_accuracy` (NeMo-RL's name for the validation pass's **mean reward**, not an accuracy in the classifier sense — there is no `val_reward`). When reward stalls, look at `train_truncation_rate` (rising) and `train_baseline_reward/pct_mixed` (falling toward zero means every prompt group agrees with itself, so there is no gradient left). See `references/reporting.md`.
- **One preference fileset, two files (DPO)** — `dataset` is a **single string** ref to a fileset that holds **both** `training.jsonl` and `validation.jsonl` (uploaded with `--remote-path`). Unlike automodel (`dataset.training`/`dataset.validation`) and unsloth (`dataset.path`/`validation_path`), there is no separate validation ref. See `references/dataset-formats.md` § NeMo-RL.
- **String refs** — `model`, `dataset`, and (for GRPO) `environment` are plain strings (`"workspace/name"`), not objects. The training method goes under `training` with `type: "dpo"` or `"grpo"`.
- **Kubernetes job backend, not Docker** — rl steps run as Kubernetes pods via the `kubernetes_job` backend; the docker job backend cannot run rl. `rl submit` fails fast on a docker-runtime platform. The target cluster must have the **job-step images** (`nmp-customizer-tasks`, `nmp-rl-training`), the **jobs-launcher** image (the per-step init container), and a **job-storage PVC**. Verify the platform with `nemo jobs list-execution-profiles -f json` (expect `backend: kubernetes_job`); to configure one, see `references/rl-kubernetes-runtime.md`. Multi-node (`parallelism.num_nodes > 1`) also needs the platform-side `NMP_RL_MULTINODE_SHARED_STORAGE_PATH` (shared FS for Ray coordination) or compile fails fast.
- **Job id prefix is `rl-<hex>`** and the platform auto-generates it — `rl submit` has **no `--name` flag** (the job JSON `name` is the *output* name, not the job id). Read the job id from the `"name"` field in **submit stdout** (JSON), same as automodel/unsloth; `poll_customization_job.sh rl-<id>` works. **Do not** pick the newest `rl-*` from `nemo jobs list` — a concurrent job or an earlier failed submit selects the wrong one. If submit stdout could not be parsed, stop and re-check rather than guessing a job id.
- **DPO main knob is `ref_policy_kl_penalty`** (β). For OOM, enable `activation_checkpointing: true` first. Full field reference: `references/hyperparameters-rl.md`.
- **GRPO main knobs are `num_generations_per_prompt`** (group size — the spread of rewards inside a group is the whole learning signal) **and `temperature`** (must stay > 0; greedy sampling makes every rollout in a group identical and the run a no-op). For OOM, enable `activation_checkpointing: true`, then lower `num_generations_per_prompt` keeping `batch_size` divisible.
- **`max_steps` + `epochs`** — same caveat as the other backends: `max_steps` caps mid-epoch; it's in the smoke fixture (`plugins/nemo-rl/tests/fixtures/minimal_dpo.json`) — omit for real runs.

## Workflow

Common steps then **branch by plugin pick**:

```text
- [ ] Resolve CLI (Pre-flight — CLI resolution); cd nemo-platform
- [ ] export NMP_BASE_URL (if user provided endpoint); note whether base URL is user-overridden
- [ ] nemo auth status — skip login if auth disabled; if auth enabled and unsigned JWT allowed, `nemo auth login --unsigned-token --email <…>`; if OIDC, `nemo auth login`
- [ ] nemo jobs list-execution-profiles -f json — apply Plugin pick rules above (retry login on 401/403)
- [ ] On connection error: default URL → ask to start platform (see Platform unreachable); custom URL → report unreachable and stop
- [ ] Convert HF dataset → /tmp/train-data/*.jsonl (see references/hf-conversion.md)
- [ ] Create dataset fileset (--exist-ok), upload the JSONL files, nemo files list to verify — automodel/unsloth: train.jsonl (+ validation.jsonl); rl: training.jsonl + validation.jsonl (see rl branch)
- [ ] Gated HF base model? → confirm `hf-token` exists; ask user and stop if missing (see HuggingFace token + troubleshooting § Gated HuggingFace models)
- [ ] Create HF weights fileset + model entity if missing (--exist-ok; gated repos need `token_secret` on fileset — see troubleshooting)

# automodel branch (submit → Docker GPU job)
- [ ] Write /tmp/job.json (batch sizing for ≥48 GB GPU; else Defaults table)
- [ ] nemo customization automodel submit /tmp/job.json --workspace default
- [ ] Poll until top-level terminal (`poll_customization_job.sh`; default 15s interval, or 30–60s manual polls)
- [ ] Report using the template in `references/reporting.md`
- [ ] Optional: compare base vs adapter on validation — `references/eval_helpers.py …` (LoRA only; CHAT format; adapters hot-reload automatically; see `references/post-training-eval.md`)

# unsloth branch (submit → Docker GPU job)
- [ ] Write /tmp/job.json using the UnslothJobInput shape (see Fast path — unsloth)
- [ ] nemo customization unsloth submit /tmp/job.json --workspace default [--profile <gpu-profile>]
- [ ] Poll until top-level terminal (`poll_customization_job.sh unsloth-<job-id>`; default 15s interval)
- [ ] Report using the template in `references/reporting.md`
- [ ] Optional: compare base vs adapter on validation — `references/eval_helpers.py …` (LoRA only; CHAT format; adapters hot-reload automatically; see `references/post-training-eval.md`)

# rl branch (DPO; submit → Kubernetes/Ray job) — requires platform.runtime: kubernetes
- [ ] Verify execution backend: `nemo jobs list-execution-profiles -f json` shows cpu/gpu at `backend: kubernetes_job` (NOT docker/subprocess). If not → stop; do not start a docker platform; configure per references/rl-kubernetes-runtime.md
- [ ] Dataset is PREFERENCE data: upload training.jsonl + validation.jsonl ({prompt,chosen,rejected}) to ONE fileset
- [ ] Write /tmp/job.json using the RlJobInput shape (see Fast path — rl (DPO))
- [ ] nemo customization rl submit /tmp/job.json --workspace default [--profile <gpu-profile>]
- [ ] Read job id from the "name" field in submit stdout (JSON) — submit has no --name flag; do NOT pick the newest rl-* from `nemo jobs list`
- [ ] Poll until top-level terminal (`poll_customization_job.sh rl-<job-id>`; default 15s interval)
- [ ] Report using the template in `references/reporting.md`

# rl branch (GRPO; submit → Kubernetes/Ray job) — requires platform.runtime: kubernetes
- [ ] Verify execution backend (same gate as DPO above)
- [ ] Confirm the cluster runs sandboxed Gym: sandbox_cluster_capable + job-storage PVC claim are operator config and fail AT SUBMIT (references/rl-kubernetes-runtime.md § Sandboxed Gym (GRPO)); ask the operator about egress before choosing native-v1
- [ ] Pick the ENVIRONMENT format — verifiers/hub env → adapter-wheels-v1; Gym tree WITH egress → native-v1; everything else (incl. Gym tree, no egress) → wheels-v1 (see references/gym-environments.md)
- [ ] Build the ENV package. Gym server → `scripts/grpo-examples/gym_to_env_package.py --gym-root <Gym checkout> --server resources_servers/<name> --format wheels-v1|native-v1 --arch x86_64|aarch64` (NeMo Gym is NOT vendored here: `git clone https://github.com/NVIDIA-NeMo/Gym`). Hub/verifiers env → `pi-to-gym-conversion --hub-id … --hub-version … --out-dir …` on an internet-capable host. Both are EXAMPLES; hand-building the manifest + configs/policy_model.yaml + server dirs works too
- [ ] Build the DATASET rows. math_with_judge → `scripts/grpo-examples/prepare_math_with_judge.py --out-dir … --train-size …` (adds agent_ref + expected_answer). Otherwise convert per references/dataset-formats.md, reading the env's *VerifyRequest first
- [ ] Create the MODEL entity (HF weights fileset + `nemo models create`) — same as automodel; see scripts/grpo-examples/README.md § 4
- [ ] Validate BEFORE upload: `pi-to-gym-conversion --validate-only <pkg-dir>` — cheapest possible failure. It checks LAYOUT ONLY: not wheel tags, not closure completeness, not rows
- [ ] Upload environment (--purpose environment — enforced at submit) and dataset (--purpose dataset) as TWO filesets; trailing slash on the local dir; no .jsonl inside the env package; `nemo files list` to confirm nothing nested
- [ ] Dataset rows are GYM ROLLOUT ROWS (prompt under responses_create_params.input + agent_ref object) — see references/dataset-formats.md § NeMo-RL (GRPO)
- [ ] Write /tmp/job.json with training.type "grpo" + the `environment` string ref (see Fast path — rl (GRPO))
- [ ] nemo customization rl submit /tmp/job.json --workspace default [--profile <gpu-profile>]
- [ ] Read job id from the "name" field in submit stdout (JSON) — submit has no --name flag
- [ ] Poll until top-level terminal (`poll_customization_job.sh rl-<job-id>`)
- [ ] Report on REWARD, not loss (references/reporting.md)
```

## Fast path — automodel

Substitute `<hf-repo>`, `<hf-dataset>`, `<model-entity>`, `<weights-fileset>`, `<dataset-fileset>`, `<output-name>`.

**Setup**

```bash
export NMP_BASE_URL=http://127.0.0.1:8080   # user override only
cd /path/to/nemo-platform
nemo auth status   # skip login if auth disabled; if enabled + unsigned JWT allowed → login --unsigned-token --email admin@example.com
nemo jobs list-execution-profiles -f json   # platform GPU profiles → automodel; set training.execution_profile if needed
```

**1. Dataset** — convert per `references/hf-conversion.md`, then:

```bash
DATASET=<dataset-fileset>   # e.g. commonsense_qa
nemo files filesets create "$DATASET" --workspace default --purpose dataset --exist-ok
nemo files upload /tmp/train-data/train.jsonl "$DATASET" --workspace default --remote-path train.jsonl
# validation.jsonl if present
nemo files list "$DATASET" --workspace default
```

**2. Model** — skip if entity exists (`nemo models list --workspace default`). For **gated** HF repos, complete **HuggingFace token (gated models)** first — see `references/troubleshooting.md` § **Gated HuggingFace models** for `token_secret` on the fileset.

```bash
WEIGHTS=<weights-fileset>   # e.g. qwen3-1.7b
MODEL_ENTITY=<model-entity>   # Models API entity (not dataset fileset, not HF id)
HF_REPO=<hf-repo>           # e.g. Qwen/Qwen3-1.7B

nemo files filesets create "$WEIGHTS" --workspace default --purpose model --exist-ok \
  --storage '{"type":"huggingface","repo_id":"'"$HF_REPO"'","repo_type":"model","revision":"main"}'

nemo models create "$MODEL_ENTITY" --workspace default --exist-ok \
  --input-data '{"name":"'"$MODEL_ENTITY"'","fileset":"default/'"$WEIGHTS"'","custom_fields":{"hf_model_id":"'"$HF_REPO"'"}}'
```

For gated repos, add `"token_secret":"hf-token"` to the `--storage` JSON (after creating the secret). See troubleshooting § **Gated HuggingFace models**.

**3. Job JSON** — write `/tmp/job.json`. `model` is the **registered model entity** (`default/<model-entity>`), not an HF repo id or dataset fileset. Full hyperparameter reference: `references/hyperparameters-automodel.md`.

```json
{
  "model": "default/<model-entity>",
  "dataset": {
    "training": "default/<dataset-fileset>",
    "validation": "default/<dataset-fileset>"
  },
  "training": {
    "training_type": "sft",
    "finetuning_type": "lora",
    "lora": { "rank": 16, "alpha": 32 },
    "max_seq_length": 2048
  },
  "schedule": { "epochs": 1 },
  "batch": { "global_batch_size": 4, "micro_batch_size": 1 },
  "optimizer": { "learning_rate": 5e-5, "weight_decay": 0.01, "warmup_steps": 0 },
  "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1, "tensor_parallel_size": 1 },
  "output": { "name": "<output-name>" }
}
```

**4. Submit and poll**

```bash
nemo customization automodel submit /tmp/job.json --workspace default
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh automodel-<job-id>
```

Read `<job-id>` from the `"name"` field in submit stdout (JSON). **Do not use `2>&1`** before `json.load` — warnings on stderr break parsing; see Gotchas. Optional interval override: append seconds (e.g. `… 30`). Or poll manually: `nemo jobs get-status automodel-<job-id>` every 30–60s.

## Fast path — unsloth

Same substitutions as automodel. Steps 1 (dataset) and 2 (model entity) are identical — the differences are the job JSON shape (`UnslothJobInput`) and the `unsloth submit` command.

**1. Dataset** — same as automodel Fast path step 1.

**2. Model** — same as automodel Fast path step 2.

**3. Job JSON** — write `/tmp/job.json` using the **`UnslothJobInput`** shape (see `references/hyperparameters-unsloth.md`). `model` is an **object** (not a string), `dataset.path` is a single fileset ref, `hardware.gpus` replaces the `parallelism` block (single GPU in the training container). `nemo customization unsloth explain` prints the live schema.

```json
{
  "name": "<job-name>",
  "model": {
    "name": "default/<model-entity>",
    "max_seq_length": 2048,
    "load_in_4bit": true,
    "dtype": "auto"
  },
  "dataset": {
    "path": "default/<dataset-fileset>",
    "text_field": "text",
    "apply_chat_template": true
  },
  "training": {
    "training_type": "sft",
    "finetuning_type": "lora",
    "lora": { "rank": 16, "alpha": 32 }
  },
  "schedule": { "epochs": 1, "warmup_ratio": 0.1 },
  "batch": { "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4 },
  "optimizer": { "learning_rate": 5e-5, "optim": "adamw_8bit" },
  "hardware": { "gpus": "0", "precision": "bf16" },
  "output": { "name": "<output-name>", "save_method": "lora" }
}
```

If the model uses `messages` chat format (preferred when the tokenizer has a chat template), keep `dataset.apply_chat_template: true`. Otherwise emit a single `text` column from your converter and set `apply_chat_template: false`.

**4. Submit and poll**

```bash
nemo customization unsloth submit /tmp/job.json --workspace default
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh unsloth-<job-id>
```

Read `<job-id>` from the `"name"` field in submit stdout (JSON). **Do not use `2>&1`** before `json.load` — warnings on stderr break parsing; see Gotchas. Optional interval override: append seconds (e.g. `… 30`). Or poll manually: `nemo jobs get-status unsloth-<job-id>` every 30–60s. If submit fails on an unknown profile, re-list execution profiles and pass `--profile <name>` on submit (default is `gpu`).

Use `submit` for Unsloth jobs; the contributor CLI exposes no local `run` verb.

## Fast path — rl (GRPO)

GRPO on a Ray cluster — **Kubernetes runtime only**. Same runtime gate as DPO: confirm `nemo jobs list-execution-profiles -f json` shows `cpu`/`gpu` at `backend: kubernetes_job` before anything else.

**Default to LoRA** (`finetuning_type: "lora"`). It trains an adapter instead of every weight, so it is markedly cheaper and faster than a full-weight run and fits on far less GPU memory — the right starting point unless the user asks for full weights or needs a standalone model entity to deploy. Full-weight remains the schema default (`all_weights`), so LoRA is something you set.

GRPO differs from every other backend in one structural way: it needs **two FileSets**, an **environment** (code that runs a rollout and returns a reward) and a **dataset** of prompt rows. There are no labelled completions — the reward comes from the environment.

**1. Environment** — the bulk of the work, and it has its own reference: **`references/gym-environments.md`**. Pick the format first. All three run on the same Gym runtime; the format decides only where dependencies come from and where `config_paths` may live:

| The user has… | Format | How |
|---|---|---|
| A Prime Intellect hub env, or any `verifiers` env | **`adapter-wheels-v1`** | `pi-to-gym-conversion` (below) — the only format with a converter |
| A Gym server tree **and** the cluster can reach a package index at spin-up | **`native-v1`** | Package the server dir + add a manifest; **strip any `.jsonl`** |
| Anything else — own code, **or a Gym server tree on a deny-default cluster** | **`wheels-v1`** | Hand-build: manifest + configs + server dirs + vendored wheel closure |

Do not pick `native-v1` just because the environment came from Gym: it ships no wheels, so its per-server venv resolves from an index and the job needs egress. Ask the operator (`NMP_RL_SANDBOX_ALLOW_INTERNET`) before committing to it; otherwise the same tree ships as `wheels-v1`.

For a hub env, run the converter on an **internet-capable host** — training clusters have no hub egress:

```bash
uv run --package nmp-rl pi-to-gym-conversion \
  --hub-id primeintellect/ascii-tree --hub-version 0.1.5 \
  --out-dir ./ascii-tree-pkg --dataset-dir ./ascii-tree-data \
  --validation-fraction 0.1 --upload --workspace default
```

`--upload` creates both FileSets and uploads them. Without it, upload by hand (`--purpose environment` for the package, `--purpose dataset` for the JSONL) — see `references/gym-environments.md` § **Upload**. **Always** validate first: `pi-to-gym-conversion --validate-only ./ascii-tree-pkg`.

**2. Dataset** — the converter writes it for a hub env. For the user's own prompts, rows are Gym rollout rows with the prompt under `responses_create_params.input` and an `agent_ref` object — **not** `messages[]`, **not** prompt/completion, **not** preference triples. Schema and a conversion snippet: `references/dataset-formats.md` § **NeMo-RL (GRPO)**.

**3. Model** — same as automodel Fast path step 2 (HF weights fileset + model entity; gated repos need `token_secret`).

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
    "max_steps": 200,

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

**Scale it to the hardware you actually have** — this targets 8 GPUs. On fewer, keep both divisibility rules satisfied (`references/batch-sizing.md` § **Batch sizing — rl / GRPO**): `batch_size % (micro_batch_size × data_parallel_size) == 0` and `(num_prompts_per_step × num_generations_per_prompt) % batch_size == 0`. Drop `vllm_tensor_parallel_size` to at most `num_gpus_per_node`, and `execution_profile` unless the platform defines that profile.

`use_triton` is deliberately absent from the `lora` block: unset, the compiler picks `true` at `tensor_parallel_size` 1 and `false` above, and an explicit `true` above TP 1 is rejected at submit.

**Full-weight variant.** Drop `finetuning_type` and the `lora` block — `all_weights` is the schema default. It registers a model entity that needs its own deployment, where LoRA registers an adapter that hot-reloads onto the base. Do not set `lora_merged` (rejected) or pass a `lora` block alongside `all_weights` (rejected).

**5. Submit and poll** — identical to DPO (no `--name`; read the `rl-<hex>` id from submit stdout):

```bash
nemo customization rl submit /tmp/job.json --workspace default > /tmp/rl-submit.json
JOB=$(python3 -c "import json;print(json.load(open('/tmp/rl-submit.json'))['name'])")
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"
```

**Read GRPO progress on reward, not loss.** The GRPO surrogate loss oscillates near zero and says nothing about run quality. `train_reward` and `val_accuracy` are the curves that matter — `val_accuracy` is the validation pass's mean reward under NeMo-RL's name for it. `train_truncation_rate` rising or `train_baseline_reward/pct_mixed` falling toward zero explain a reward curve that stopped moving. See `references/reporting.md` § **RL (GRPO) example**.

## Fast path — rl (DPO)

DPO on a Ray cluster — **Kubernetes runtime only**, full-weight. **Before anything else**, confirm the platform dispatches jobs to Kubernetes: `nemo jobs list-execution-profiles -f json` must show `cpu`/`gpu` at `backend: kubernetes_job` (not `docker`/`subprocess`). If it doesn't, stop — do not start/use a docker-runtime platform; configure a Kubernetes-runtime one per `references/rl-kubernetes-runtime.md`. Model-entity setup (step 2) is identical to automodel; the dataset is **preference data** and the job JSON is the `RlJobInput` shape.

**1. Preference dataset** — rows are `{prompt, chosen, rejected}` (see `references/dataset-formats.md` § NeMo-RL). Upload **both** files to **one** fileset:

```bash
DATASET=<preference-fileset>   # e.g. dpo-data
nemo files filesets create "$DATASET" --workspace default --purpose dataset --exist-ok
nemo files upload /tmp/dpo-train.jsonl "$DATASET" --workspace default --remote-path training.jsonl
nemo files upload /tmp/dpo-val.jsonl   "$DATASET" --workspace default --remote-path validation.jsonl
nemo files list "$DATASET" --workspace default
```

**2. Model** — same as automodel Fast path step 2 (HF weights fileset + model entity; gated repos need `token_secret`).

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

**4. Submit and poll** — `rl submit` has **no `--name` flag** (the platform auto-generates the `rl-<hex>` job id), so read it from submit stdout:

```bash
nemo customization rl submit /tmp/job.json --workspace default > /tmp/rl-submit.json   # add --profile <name> if the default gpu profile is wrong
JOB=$(python3 -c "import json;print(json.load(open('/tmp/rl-submit.json'))['name'])")
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"
```

**Do not** derive the job id by picking the newest `rl-*` from `nemo jobs list` — a concurrent job or an earlier failed submit selects the wrong one. If `/tmp/rl-submit.json` does not parse, the submit result is unknown: stop and inspect it (re-submitting risks a duplicate job).

**Do not use `2>&1`** before `json.load` — warnings on stderr break parsing; see Gotchas. Or poll manually: `nemo jobs get-status rl-<job-id>` every 30–60s. If submit fails on an unknown profile, re-list execution profiles and pass `--profile <name>`. `nemo customization rl explain` prints the live schema.

## Defaults

Shared:

| Field | Value |
|-------|-------|
| Workspace | `default` |
| Plugin | `automodel` (override per **Plugin pick**) |
| Training | SFT + LoRA, `max_seq_length` 2048 |
| Schedule | `epochs` ≥ 1; omit `max_steps` |
| Auth email (when login required) | `admin@example.com` unless user specifies |

Automodel-specific:

| Field | Value |
|-------|-------|
| Parallelism | 1 node, 1 GPU, TP=1 |
| Batch | `global_batch_size` 4, `micro_batch_size` 1 (unknown VRAM; see `references/batch-sizing.md` for ≥48 GB) |
| Optimizer | `learning_rate` 5e-5 |

Unsloth-specific:

| Field | Value |
|-------|-------|
| Hardware | `hardware.gpus` `"0"`, `hardware.precision` `bf16` (selection only, single GPU) |
| Model load | `load_in_4bit: true`, `dtype: "auto"` |
| Batch | `batch.per_device_train_batch_size` 2, `batch.gradient_accumulation_steps` 4 (effective batch 8; see `references/batch-sizing.md` for ≥48 GB ramp) |
| Optimizer | `learning_rate` 5e-5, `optim` `adamw_8bit` |
| Output | `save_method: "lora"` (adapter-only) unless user asks for merged checkpoint |
| Gradient checkpointing | `training.use_gradient_checkpointing: "unsloth"` |

rl-specific (DPO):

| Field | Value |
|-------|-------|
| Training | DPO, full-weight (`type: "dpo"`; no LoRA) |
| Model (if user gives none) | `Qwen/Qwen3-0.6B` |
| Dataset (if user gives none) | `nvidia/HelpSteer3` (preference subset; uploaded raw — see `references/dataset-formats.md` § NeMo-RL) |
| Schedule (if user gives none) | small demo run: `max_steps` 20 (completes fast, proves the pipeline). For a real run, set `epochs` and **omit** `max_steps`. |
| Parallelism | 1 node, 1 GPU (`parallelism.num_nodes`/`num_gpus_per_node`) |
| Batch | `batch_size` 32, `micro_batch_size` 1 |
| Optimizer | `learning_rate` 5e-6 (DPO uses a low LR), AdamW + cosine |
| DPO | `ref_policy_kl_penalty` (β) 0.05, `sft_loss_weight` 0.0 |
| Max sequence length | 1024 |
| Output | full-weight model entity (`output.name`); no adapter |

When the user asks for a DPO job **without specifics**, default to the above: a
20-step run of `Qwen/Qwen3-0.6B` on `nvidia/HelpSteer3` — small enough to finish
quickly and confirm the pipeline end-to-end.

rl-specific (GRPO) — values below are a **real 8×H100 LoRA configuration**, not a smoke test:

| Field | Value |
|-------|-------|
| Training | GRPO (`type: "grpo"`), **`finetuning_type: "lora"`** — cheaper and faster than full-weight, and fits far less GPU memory. Use `all_weights` only when the user asks, or needs a standalone deployable model entity. `lora_merged` is rejected. |
| Model (if user gives none) | `Qwen/Qwen3-8B-Base` → entity `qwen3-8b-base` |
| Environment (if user gives none) | **Stop and ask** — see **No environment supplied** below |
| LoRA | `rank` 128, `alpha` 256, `dropout` 0.0, all linear layers (omit `target_modules`/`exclude_modules`). Omit `use_triton`: the compiler picks `true` at TP 1, `false` above |
| Parallelism | 1 node × 8 GPUs, `tensor_parallel_size` 1, `execution_profile` `"h100"` |
| Batch | `batch_size` 512, `micro_batch_size` 1, `num_prompts_per_step` 32, `num_generations_per_prompt` 16 |
| Dynamic sampling | `use_dynamic_sampling` true, `batch_multiplier` 2.0, `dynamic_sampling_max_gen_batches` 10 |
| Sequence | `max_seq_length` 3072, `max_new_tokens` 2048, `temperature` 1.0, `activation_checkpointing` true |
| Optimizer | `learning_rate` 1e-5, `min_learning_rate` 9e-6, `adam_eps` 1e-8, `warmup_steps` 10, `weight_decay` 0.01 |
| Schedule | `epochs` 1 with `max_steps` 200 — a bounded first run. Drop `max_steps` for a full pass |
| GRPO loss | `ref_policy_kl_penalty` 0.0, `ratio_clip_min`/`max`/`c` 0.2 / 0.28 / 10.0, `normalize_rewards` true, `use_leave_one_out_baseline` true, `max_rollout_turns` 1 |
| Importance sampling | `truncated_importance_sampling_type` `"tis"`, ratio 2.0, `use_importance_sampling_correction` true |
| Rollout engine | `vllm_tensor_parallel_size` 8 (≤ `num_gpus_per_node`), `vllm_gpu_memory_utilization` 0.6 |
| Validation | `val_at_start` + `val_at_end` true, `val_check_interval` 25, `keep_top_k` 1, `seed` 42 |
| Output | `output.name` only — the type is **inferred** (`lora` → HF PEFT adapter entity; `all_weights` → model entity) |

**These target 8 GPUs.** On different hardware, rescale rather than copy: keep `batch_size % (micro_batch_size × data_parallel_size) == 0` and `(num_prompts_per_step × num_generations_per_prompt) % batch_size == 0`, cap `vllm_tensor_parallel_size` at `num_gpus_per_node`, and drop `execution_profile` unless the platform defines it. Derivation: `references/batch-sizing.md` § **Batch sizing — rl / GRPO**.

### No environment supplied

A dataset alone cannot produce a GRPO job: the reward comes from an **environment**, and
an environment only scores the rows it was written for. So when the user names a model and
a dataset but no environment — *"run a GRPO job with X on Y"* — **do not guess, and do not
pair their dataset with a demo environment.** A mismatched pair does not error; it trains on
a meaningless reward signal and wastes the GPU hours.

Stop and tell them, in this order:

1. **They have not supplied an environment, and GRPO requires one.** Unlike SFT or DPO, the
   labels do not come from the dataset.
2. **Their dataset is probably not compatible with `math_with_judge`**, the one worked
   example this skill ships. It scores a short final answer against an `expected_answer`
   field per row, so it fits math and short-answer datasets and nothing else.
3. **Offer exactly two ways forward:**

| Option | What it means | When |
|---|---|---|
| **A — supply a matching environment** | The user provides an environment package (or a hub / `verifiers` env) written to score *their* dataset. Then follow the normal path. | Their dataset is what matters |
| **B — run the `math_with_judge` reference** | Swap in **DAPO-Math-17k** as the dataset and `math_with_judge` as the environment — a known-good pair that trains end to end. | They want to see GRPO work, and the dataset is negotiable |

Say plainly that option B **replaces** their dataset. Do not offer a third path that keeps
their dataset and bolts on `math_with_judge`.

If they pick B, the example scripts under `scripts/grpo-examples/` build both FileSets — see
`references/gym-environments.md` § **Example scripts**.

## Batch sizing

`micro_batch_size` / `global_batch_size` (automodel) and `per_device_train_batch_size` × `gradient_accumulation_steps` (unsloth) on **≥48 GB GPUs**, multi-GPU (data vs tensor parallel), and OOM / throughput tuning live in **`references/batch-sizing.md`**. On unknown VRAM the **Defaults** above are safe — read batch-sizing before raising batch on a known ≥48 GB card. rl (DPO) batch knobs (`batch_size` / `micro_batch_size`) are in `references/hyperparameters-rl.md`. **GRPO** batch sizing is its own problem — rollouts, not gradients, set the ceiling, and two divisibility rules are enforced at submit: `references/batch-sizing.md` § **Batch sizing — rl / GRPO**.

## Worked example

**Automodel:** `Qwen/Qwen3-1.7B` + `tau/commonsense_qa` → CHAT JSONL, fileset `commonsense_qa`, entity `qwen3-1.7b`, output `qwen3-1.7b-commonsense-qa-lora`, `epochs: 1` (no `max_steps`). On ≥48 GB GPU use LoRA ≤4B **default**: `micro` 32, GBS 128, `learning_rate` `1e-4` (high-util: 64 / 256).

**Unsloth:** same model + dataset + entity + fileset, but `nemo customization unsloth submit /tmp/job.json -w default`. Job JSON ≤4B row: `batch.per_device_train_batch_size` 8, `batch.gradient_accumulation_steps` 16 (effective 128), `learning_rate` `1e-4`, `hardware.gpus` `"0"`, `output.save_method` `"lora"`. Poll `unsloth-<job-id>` to completion. For payload shape only (not the field set or values): `plugins/nemo-unsloth/tests/fixtures/minimal_unsloth_sft.json` — a smoke-test input, so confirm fields against `unsloth explain`.

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

`nemo customization rl submit /tmp/job.json -w default`, derive the `rl-<hex>` id (submit has no `--name`), poll to completion. For payload shape only: `plugins/nemo-rl/tests/fixtures/minimal_dpo.json` — a smoke-test input, so confirm fields against `rl explain`. For a real run, replace `max_steps: 20` with `epochs`.

## Report to user

After polling reaches a **terminal** status (`completed`, `error`, or `cancelled`), report using the template in **`references/reporting.md`** — one format for all backends. It covers the **Fine-tune result** header, the **Training configuration** table (with per-backend examples: automodel, unsloth, rl/DPO), and **Using the adapter** (automodel/unsloth LoRA) vs **Using the fine-tuned model** (full SFT / merged / rl DPO), plus metrics extraction, notes by status, `/tmp` report saving, and error follow-ups.

## Reference files

| When | Read |
|------|------|
| HF conversion or MCQA shaping | `references/hf-conversion.md` |
| CHAT vs SFT vs CUSTOM (automodel); text vs messages (unsloth); preference triples (rl/DPO); Gym rollout rows (rl/GRPO) | `references/dataset-formats.md` |
| **GRPO: build an example env + dataset, and the job JSON fixtures that reference them** | `scripts/grpo-examples/README.md`, `plugins/nemo-rl/tests/fixtures/minimal_grpo*.json` |
| **GRPO environments** — the three formats (`native-v1`, `wheels-v1`, `adapter-wheels-v1`), converting a verifiers / Prime Intellect env, packaging a Gym source-tree env, bringing your own code, validation, upload, packaging errors | `references/gym-environments.md` |
| Field glossary, full JSON template, distillation/KD, live-schema pointers (index routes per backend) | `references/hyperparameters.md` → `hyperparameters-automodel.md` / `hyperparameters-unsloth.md` / `hyperparameters-rl.md` |
| Batch sizing (≥48 GB), OOM / throughput (automodel + unsloth) | `references/batch-sizing.md` |
| Multi-GPU same node | `references/batch-sizing.md` § **Multi-GPU (same node)** (unsloth is single-GPU) |
| Reporting: result template, Training configuration, Using the adapter / fine-tuned model | `references/reporting.md` |
| Backend choice, execution profiles, submit failure, container images, missing image on remote platform, gated HF auth / download 502, CLI, connection errors | `references/troubleshooting.md` (§ **Parsing CLI JSON** for `2>&1` / `json.load`; § **Gated HuggingFace models** for `hf-token`) |
| rl (DPO **and GRPO**) needs Kubernetes job execution — verifying / configuring `runtime: kubernetes` + `kubernetes_job` executors (local platform → remote cluster, launcher image, PVC, loopback), plus **sandboxed Gym**: OpenSandbox install, `sandbox_cluster_capable`, job-storage PVC, egress, rollout transport | `references/rl-kubernetes-runtime.md` |
| **Live JSON schema — authoritative, check here first** | `uv run nemo customization automodel explain` / `uv run nemo customization unsloth explain` / `uv run nemo customization rl explain` |
| Payload **shape** only — not the field set, defaults, or sensible values (automodel) | `plugins/nemo-automodel/tests/fixtures/qwen3_0.6b_sft_lora.json` |
| Payload **shape** only (unsloth) | `plugins/nemo-unsloth/tests/fixtures/minimal_unsloth_sft.json` |
| Payload **shape** only (rl / DPO) | `plugins/nemo-rl/tests/fixtures/minimal_dpo.json` |
| Environment manifest + validation rules (source of truth) | `services/rl/src/nmp/rl/schemas/environment.py`, `services/rl/src/nmp/rl/tasks/environment/validate.py` |
| Payload **shape** only — integrations (W&B / MLflow) | automodel: `plugins/nemo-automodel/tests/fixtures/integrations_wandb_mlflow.json` · unsloth: `plugins/nemo-unsloth/tests/fixtures/integrations_wandb_mlflow.json` · rl: `plugins/nemo-rl/tests/fixtures/integrations_wandb_mlflow.json` |
| Automodel compile-path contract configs | `services/automodel/tests/contract/input_configs/` → YAML in `output_configs/` (legacy `TrainingStepConfig` shape, not submit JSON) |
| W&B / MLflow field reference (all backends) | `references/hyperparameters.md` § **Integrations (all backends)** |
| W&B secret + MLflow local server + jobs-launcher | `references/integrations-setup.md` |
| Gated HF model auth (`hf-token`, fileset `token_secret`) | `references/troubleshooting.md` § **Gated HuggingFace models** |
| Post-training eval (base vs LoRA, CHAT format parity) | `references/post-training-eval.md`, `references/eval_helpers.py` |

Related: `plugins/nemo-automodel/README.md`, `plugins/nemo-unsloth/README.md`, `plugins/nemo-rl/README.md`, `plugins/nemo-customizer/docs/CUSTOMIZATION.md`, skills **`nemo-files`**, **`nemo-status`**, **`nemo-secrets`**.
