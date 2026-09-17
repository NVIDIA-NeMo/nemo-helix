<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Read this file when submit fails, jobs fail on images, the platform is unreachable, W&B/MLflow integrations fail, gated HuggingFace model download fails, or the user asks for Unsloth.

Resolve the CLI first per **Pre-flight — CLI resolution** in `SKILL.md` (`nemo` on `PATH`, else `uv run nemo`, else route to **nemo-setup**). Example commands below use `nemo …`.

## Prerequisites

Before working through any section below, confirm:

- **NeMo CLI available** — `nemo` on `PATH`, otherwise `uv run nemo` from the nemo-platform repo root (see **Pre-flight — CLI resolution** in `SKILL.md`).
- **Platform base URL configured** — via the CLI context, `--base-url`, or `$NMP_BASE_URL`; defaults to `http://localhost:8080`.
- **Workspace access** — authenticated (`nemo auth login`) with access to the target workspace.

## Platform unreachable (connection error)

Any `nemo …` call may fail with `Connection error`, timeout, or connection refused — typically on the first `nemo jobs list-execution-profiles`. Auth is not required when the cluster has authentication disabled (`nemo auth status`); on 401/403 see **Authentication** in `SKILL.md`.

**Did the user override the base URL?**

| Situation | Action |
|-----------|--------|
| User gave a platform host/URL (e.g. `<remote-host>:8080`) or you set `NMP_BASE_URL` to something other than `http://127.0.0.1:8080` or `http://localhost:8080` | Report that the platform is not reachable at that address. Ask them to confirm the host is up and the URL is correct. **Do not** start local services. |
| Default URL only — no user override | **Ask** whether to start the platform locally. If they agree, from the **nemo-platform** git root run in the **background**, then poll until healthy and retry the failed command: |

```bash
nemo services run \
  --host 0.0.0.0 \
  --port 8080 \
  --controllers jobs,entities,models \
  --service-group all
```

Health check (repeat until success or ~2 min):

```bash
curl -sf http://127.0.0.1:8080/health/ready
# or
nemo jobs list-execution-profiles -f json
```

Do **not** auto-start services without asking. Customization needs **jobs**, **entities** (filesets), and **models** controllers — the command above is the minimal local set for this skill.

If the user already has a listener on `:8080` but health fails, see **nemo-status** (stale lock / wedged platform) before starting a second instance.

## Backend choice (automodel vs unsloth)

**Do not** run `docker info` on the agent machine. The platform often runs elsewhere (`NMP_BASE_URL`). Ask the **connected platform** what executors it exposes.

List profiles (login first only if auth is enabled — see **Authentication** in `SKILL.md`):

```bash
nemo jobs list-execution-profiles -f json
```

REST equivalent (same payload): `GET /apis/jobs/v2/execution-profiles` on the platform base URL with the saved auth token.

Each entry has `provider`, `profile` (name), and `backend` (e.g. `docker`, `kubernetes_job`, `volcano_job`, `subprocess`).

| Condition | Plugin |
|-----------|--------|
| User explicitly asks for Unsloth | `unsloth` |
| User explicitly asks for Automodel | `automodel` |
| Response includes **`provider`: `gpu` or `gpu_distributed`** | **`automodel`** (default) |
| No GPU profiles (only `subprocess` and/or CPU `provider`) | Report that GPU customization is unavailable |

Automodel and unsloth are **`submit`-only**. After submit, the platform's **Docker executor** runs GPU container steps on the daemon attached to the connected platform host (`platform.runtime: docker`). (rl is also submit-only but runs on Kubernetes/Ray — see `rl-kubernetes-runtime.md`.) Training does not run in the CLI shell — query execution profiles on the platform (`NMP_BASE_URL`), not GPU availability in the agent's terminal.

### Pick execution profile

**Automodel** — set `training.execution_profile` in job JSON to the **`profile`** string of a GPU row from the list (e.g. `default`, `docker_gpu`). If omitted, the plugin default is usually `gpu` — submit errors mentioning an unknown profile mean you should re-list and set an exact name from the API.

**Unsloth** — pass `--profile <name>` on `nemo customization unsloth submit …` when the default `gpu` profile is wrong. There is no `execution_profile` field in `UnslothJobInput` today.

Quick filter (stdout only — do not use `2>&1` or `json.load` breaks on stderr warnings):

```bash
nemo jobs list-execution-profiles -f json 2>/dev/null | python3 -c "
import sys, json
for p in json.load(sys.stdin):
    if p.get('provider') in ('gpu', 'gpu_distributed'):
        print(p['profile'], p.get('backend'), p.get('provider'))
"
```

Do not run `nemo customization --help` unless submit returns unknown plugin.

## Parsing CLI JSON

`submit`, `explain`, and `-f json` commands write **JSON on stdout**. Harmless config warnings (e.g. `Configuration file not found, using defaults`) go to **stderr**, not stdout.

**Do not** merge stderr into stdout with **`2>&1`** before `json.load` or jq — the warnings prefix the stream and cause `JSONDecodeError` even when submit **succeeded** and the job is already queued. That parse failure often leads to **duplicate jobs** when the agent re-runs submit.

| Do | Don't |
|----|-------|
| Pipe **stdout only**: `… submit /tmp/job.json \| python3 -c "import sys,json; print(json.load(sys.stdin)['name'])"` | `… submit /tmp/job.json 2>&1 \| python3 -c "import sys,json; json.load(sys.stdin)"` |
| Suppress stderr if noisy: append `2>/dev/null` to the `nemo` command (not `2>&1`) | Merge stderr into the JSON pipe with `2>&1` |
| On `json.load` failure, check `nemo jobs list` before re-submitting | Assume submit failed and submit again |

Same rule for `nemo jobs list-execution-profiles -f json`: parse stdout only; use `2>/dev/null` if needed, never `2>&1` into `json.load`.

## Verb is backend-specific (both submit-only)

- **Automodel**, **Unsloth**, and **RL** use **`submit` only**. Use `nemo customization <plugin> submit …`; these backends expose no local `run` verb.
- Dataset refs in job JSON: `default/<fileset>` (automodel: `dataset.training` / `dataset.validation`; unsloth: `dataset.path` / optional `dataset.validation_path`).

## Gated HuggingFace models

Gated or private HuggingFace repos (e.g. Llama, Gemma) require a **platform secret** and **`token_secret`** on the model fileset. The Files service does **not** use your local `~/.cache/huggingface` or shell `HF_TOKEN`. Unlike W&B, the HF token is **not** set in job JSON — it is wired on the **model fileset** storage config.

| Symptom / log excerpt | Likely cause | Fix |
|-----------------------|--------------|-----|
| Job fails in **download** step; `Failed to access upstream storage`; `InternalServerError` 502; `Verify that the referenced credentials are valid` | Missing/stale `hf-token` secret, or fileset created without `token_secret` | Steps below — then re-submit |
| Secret exists but download still fails | User has not **accepted the model license** on huggingface.co for that repo | Accept license with the same HF account as the token, then re-submit |
| Public model (e.g. `Qwen/Qwen3-1.7B`) | No secret needed | Omit `token_secret` on the fileset |

**Convention:** secret name `hf-token` in workspace `default`. Any valid secret name works if referenced consistently in `token_secret`.

### 1. Check whether the secret exists

```bash
nemo secrets list --workspace default
```

If `hf-token` is missing, **ask the user** for a HuggingFace token with **Read** access (https://huggingface.co/settings/tokens). They must also accept the model license on the model's HF page.

### 2. Create or update the secret

```bash
HF_SECRET=hf-token
printf '%s' "$HF_TOKEN" | nemo secrets create "$HF_SECRET" \
  --workspace default \
  --from-file -

# Or update if it already exists but is stale:
printf '%s' "$HF_TOKEN" | nemo secrets update "$HF_SECRET" \
  --workspace default \
  --from-file -
```

### 3. Create or update the model fileset with `token_secret`

**New fileset (gated repo):**

```bash
WEIGHTS=<weights-fileset>
HF_REPO=<hf-repo>          # e.g. google/gemma-2-2b-it
HF_SECRET=hf-token

nemo files filesets create "$WEIGHTS" --workspace default --purpose model --exist-ok \
  --storage '{"type":"huggingface","repo_id":"'"$HF_REPO"'","repo_type":"model","revision":"main","token_secret":"'"$HF_SECRET"'"}'
```

**Existing fileset missing `token_secret`:**

```bash
nemo files filesets update "$WEIGHTS" --workspace default \
  --input-data '{"storage":{"type":"huggingface","repo_id":"'"$HF_REPO"'","repo_type":"model","revision":"main","token_secret":"'"$HF_SECRET"'"}}'
```

Then create or reuse the model entity pointing at `default/<weights-fileset>`.

**Public repos** — omit `token_secret`:

```bash
nemo files filesets create "$WEIGHTS" --workspace default --purpose model --exist-ok \
  --storage '{"type":"huggingface","repo_id":"'"$HF_REPO"'","repo_type":"model","revision":"main"}'
```

### 4. Re-submit

After secret + fileset are wired, re-submit the same job JSON (use a fresh `output.name` if a prior partial run already registered an adapter).

**Note:** `files-hf-token` in platform config is **internal** service-to-service auth between Models and Files — it is **not** your HuggingFace Hub token. Do not confuse the two.

## Missing training images

Job errors like `Failed to pull image … nmp-unsloth-training:… Not Found`, `manifest unknown`, or a missing automodel training image mean the **connected platform's Docker daemon** (the one that runs GPU job steps) does not have the image. With the default `NMP_BASE_URL` (`127.0.0.1:8080` / `localhost:8080`), that daemon is usually on the same machine as the agent; with a user-overridden URL (e.g. `<remote-host>:8080`), it is on the remote target host instead.

**Did the user override the base URL?** (same rule as **Platform unreachable** — track this from the start of the workflow.)

| Situation | Action |
|-----------|--------|
| **Remote platform** — user gave a host/URL (e.g. `<remote-host>:8080`) or you set `NMP_BASE_URL` to something other than `http://127.0.0.1:8080` or `http://localhost:8080` | **Do not** run `docker build`, `docker pull`, or `docker buildx bake` on the agent machine — that only affects the agent's local daemon, not the remote platform. Tell the user they must build or load the image **on the target host** (the machine whose Docker daemon runs the GPU job steps). Report with the template in `references/reporting.md`, then append **Report follow-up — missing image (remote platform)** below. Stop; do not retry submit until the user confirms the image is available on the target. |
| **Local platform** — default URL only (`127.0.0.1:8080` / `localhost:8080`) | Build or pull on **that same host** where `nemo services run` and Docker share a daemon. See build commands below and `docker/unsloth/README.md` (unsloth) or automodel docker docs. Set env vars **before** starting/restarting the platform. |

Image env vars are read when the platform starts (not per job):

```bash
export NMP_IMAGE_REGISTRY=<registry>
export NMP_IMAGE_TAG=<tag>
```

**Automodel** — also set `NMP_AUTOMODEL_IMAGE_REGISTRY=$NMP_IMAGE_REGISTRY`.

**Unsloth** — set `NMP_UNSLOTH_TRAINING_IMAGE` (and optionally `NMP_UNSLOTH_TASKS_IMAGE`) to the full built ref, then restart platform services so the env var takes effect.

### Build on the target host (unsloth)

Run on the **platform host** (SSH, console, or CI on that box — not from the agent when the platform is remote):

```bash
cd /path/to/nemo-platform

# Local build (platform and Docker on the same machine)
docker buildx bake \
  -f docker-bake.hcl \
  nmp-unsloth-training \
  --load \
  --set "*.platform=linux/amd64"

export NMP_UNSLOTH_TRAINING_IMAGE="${IMAGE_REGISTRY:-my-registry/nemo-platform-dev}/nmp-unsloth-training:${BAKE_TAG:-local}"
# Restart platform so the env var is picked up
nemo services restart
```

Or push to a registry the target can pull from — see **Option B** in `docker/unsloth/README.md` — then set `NMP_UNSLOTH_TRAINING_IMAGE` to that full ref before restart.

After the image is on the target, re-submit the same job JSON (use a fresh `output.name` if a prior partial run already registered an adapter).

### Report follow-up — missing image (remote platform)

When submit or poll returns a missing-image error and the base URL is **user-overridden**, start with the **Report to user** template in `references/reporting.md` (status `error`, **Output adapter fileset (planned):**, Notes quoting the pull error and naming the target host). Then append these sections:

**What you need to do on the target host** — build or load the training image on the machine running the NeMo platform (where `docker info` works for the platform's daemon), set `NMP_UNSLOTH_TRAINING_IMAGE` or automodel image env vars, and restart platform services. Full steps: `docker/unsloth/README.md` (unsloth) or automodel docker docs.

**Re-submit after the image is available:**

```bash
export NMP_BASE_URL=<user's platform URL>
cd /path/to/nemo-platform
nemo customization <plugin> submit /tmp/job.json --workspace default [--profile <gpu-profile>]
```

Then poll until terminal status. Offer to re-submit once the user confirms the image is on the target — do not attempt a local Docker build from the agent for a remote platform.

## W&B / integrations not working

Job JSON has `integrations.wandb` (and/or `integrations.mlflow`) but tracking fails or never starts. Full setup: `references/integrations-setup.md`.

| Symptom / log excerpt | Likely cause | Fix |
|-----------------------|--------------|-----|
| Training logs **omit** `[launcher]` lines; entrypoint is the training module directly (e.g. `Running main process: /opt/venv/bin/python [-m nmp.unsloth.tasks.training]` with **no** preceding `Fetching secret wandb-api-key`) | **jobs-launcher** binary missing or `launcher_tool_path` wrong — secrets are never injected | Build launcher on the **platform host**, set absolute `launcher_tool_path`, restart services. See § **jobs-launcher missing** below and `integrations-setup.md` § **jobs-launcher**. |
| `wandb: ERROR` / HTTP 401 / `permission denied` after launcher lines present | Platform secret `wandb-api-key` has wrong or placeholder value; local `wandb login` cache is **not** used | `uv run nemo secrets update wandb-api-key --value "$WANDB_API_KEY" --workspace default` (or `--from-file -`). Re-submit. |
| `RuntimeError: WandbCallback requires wandb to be installed` (unsloth) | `nmp-unsloth-training` image lacks `wandb` | Rebuild image with `nmp-unsloth[integrations]` extra; set `NMP_UNSLOTH_TRAINING_IMAGE`; restart platform. See **Missing training images**. |
| Compile/submit warning: `integrations.wandb is configured but api_key_secret is missing` | Job JSON has `wandb` block without `api_key_secret` | Add `"api_key_secret": "default/wandb-api-key"` (or your secret ref). |
| MLflow run never appears; W&B works | `tracking_uri` unreachable from container (`localhost`, wrong port) | Use `docker0` host IP + published port (e.g. `http://${DOCKER_HOST_IP}:5001`). See `integrations-setup.md` § **`tracking_uri`**. |

### jobs-launcher missing (W&B secret not injected)

The Docker executor wraps the training entrypoint with **jobs-launcher** only when `launcher_tool_path` points to a built binary on the platform host. If the path is missing or still the default stub, the container runs training **without** secret injection — `WANDB_API_KEY` never reaches the process even when `integrations.wandb.api_key_secret` is set in job JSON.

**Working** — launcher fetched the secret before training:

```text
[launcher] 2026/06/11 22:03:03 Fetching secret wandb-api-key from workspace default...
[launcher] 2026/06/11 22:03:03 Successfully fetched secret wandb-api-key and mapped to WANDB_API_KEY
[launcher] 2026/06/11 22:03:03 Injected 1 secret(s) as environment variables
[launcher] 2026/06/11 22:03:03 Running main process: /opt/venv/bin/python [-m nmp.unsloth.tasks.training]
...
wandb: Syncing run my-run
```

**Broken** — no launcher wrapper (typical when binary was never built or path is wrong):

```text
2026-06-11 21:47:42,848 - __main__ - INFO - Container: CUDA_VISIBLE_DEVICES=0
...
# No wandb: Syncing run … — W&B never initialized because WANDB_API_KEY was not injected
```

On the platform host:

```bash
cd /path/to/nemo-platform/services/core/jobs/jobs-launcher
./build-manual.sh linux amd64
```

Set `jobs.executors.docker.launcher_tool_path` in `~/.nemo/config.yaml` to the **absolute** path of the built `jobs-launcher` binary, then `uv run nemo services restart`. Re-submit with a fresh `output.name`.

## Unsloth submit errors

| Error / symptom | Cause | Fix |
|-----------------|-------|-----|
| `Unsloth training requires platform.runtime: docker` | Platform not configured for Docker GPU jobs | Start platform with Docker runtime and a GPU execution profile |
| Unknown execution profile | Default `gpu` profile missing or wrong | Re-list profiles; pass `--profile <exact-name>` on submit |
| Missing `nmp-unsloth-training` image / `Failed to pull image` / `manifest unknown` | Image not on the **platform host's** Docker daemon | **Remote platform** (`NMP_BASE_URL` not localhost): tell user to build on the target — **do not** `docker build` locally. **Local platform**: build on same host; see **Missing training images** above and `docker/unsloth/README.md` |
| `torch.cuda.is_available()` False in training step logs | GPU not exposed to the container step | Confirm the execution profile is GPU-backed; check platform Docker GPU setup |
| Job stuck in `active` after training step completes | Upload / model-entity steps still running | Keep polling top-level status (same as automodel) |

See `plugins/nemo-unsloth/README.md` for the 4-step job flow (download → train → upload → model-entity).

## Empty `neg_doc` on retrieval jobs

Symptom: `bi_encoder` / `cross_encoder` dies in the first batch with
`neg_doc must contain at least 1 document to sample N negatives`.

Cause: Data Designer convert-only Stage 1 (`enable_mining: false`) writes
`training.jsonl` with `neg_doc: []`. Dataset validation still accepts empty
lists. The collator then samples `train_n_passages - 1` negatives (default 4).

Fix: sample `training.jsonl` and require `neg_doc` to be a non-empty list on
every checked row. A single empty list can be selected by the collator and fail
the run. If any row is empty, run `nemo data-designer retrieval-prepare` with
`enable_mining: true` (or `train_input_file` on the existing convert-only
fileset so frozen `eval_beir` is not regenerated). Do not lower
`train_n_passages` to hide an unmined fileset. Conductor:
`nemo-retrieval-recipes` `references/sdg.md`.

## CLI quick reference

Shared:

| Action | Command |
|--------|---------|
| Execution profiles | `nemo jobs list-execution-profiles -f json` |
| Create dataset fileset | `nemo files filesets create <name> --workspace default --purpose dataset --exist-ok` |
| Create HF weights fileset (public) | `nemo files filesets create <name> --workspace default --purpose model --exist-ok --storage '{"type":"huggingface","repo_id":"<repo>","repo_type":"model","revision":"main"}'` |
| Create HF weights fileset (gated) | Same as above plus `"token_secret":"hf-token"` — see § **Gated HuggingFace models** |
| Upload | `nemo files upload <local> <fileset> --workspace default --remote-path train.jsonl` |
| List files | `nemo files list <fileset> --workspace default` |
| Create model | `nemo models create <name> --workspace default --exist-ok --input-data '<json>'` |
| Poll job | `nemo jobs get-status <automodel\|unsloth\|rl>-<job-id>` |

Automodel:

| Action | Command |
|--------|---------|
| Submit | `nemo customization automodel submit <job.json> --workspace default` |
| Status | `nemo jobs get-status automodel-<job-id>` |
| Live schema | `nemo customization automodel explain` |

Unsloth:

| Action | Command |
|--------|---------|
| Submit | `nemo customization unsloth submit <job.json> --workspace default [--profile P] [--cluster C]` |
| Status | `nemo jobs get-status unsloth-<job-id>` |
| Live schema | `nemo customization unsloth explain` |

All backends return a job id from `submit` — poll until top-level status is terminal (`completed`, `error`, or `cancelled`).
