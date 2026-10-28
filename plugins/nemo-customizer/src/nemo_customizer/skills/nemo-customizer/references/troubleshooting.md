<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Read this file when submit fails, jobs fail on images, the platform is unreachable, W&B/MLflow integrations fail, gated HuggingFace model download fails, or the user asks for Unsloth.

Resolve the CLI and the platform URL first: `SKILL.md` § **Pre-flight**.

## Platform unreachable (connection error)

`SKILL.md` § **Pre-flight** decides whether to start the platform. When the user agrees, run this from the **nemo-helix** git root. It starts the platform in the background and logs to `nemo-services.log`; poll until healthy, then retry the failed command:

```bash
nohup nemo services run \
  --host 0.0.0.0 \
  --port 8080 \
  --controllers jobs,entities,models \
  --service-group all > nemo-services.log 2>&1 &
```

Health check (repeat until success or ~2 min):

```bash
curl -sf http://127.0.0.1:8080/health/ready
# or
nemo jobs list-execution-profiles -f json
```

Customization needs **jobs**, **entities** (filesets), and **models** controllers — the command above is the minimal local set for this skill.

If the user already has a listener on `:8080` but health fails, see **nemo-status** (stale lock / wedged platform) before starting a second instance.

## Execution profiles

`nemo jobs list-execution-profiles -f json` (REST: `GET /apis/jobs/v2/execution-profiles`) returns one entry per profile, with `provider`, `profile` (name) and `backend` (`docker`, `kubernetes_job`, `volcano_job`, `subprocess`). `SKILL.md` § **Plugin pick** maps them to a backend.

### Pick execution profile

**Automodel** — set `training.execution_profile` in job JSON to the **`profile`** string of a GPU row from the list (e.g. `default`, `docker_gpu`). If omitted, the plugin default is usually `gpu` — submit errors mentioning an unknown profile mean you should re-list and set an exact name from the API.

**Unsloth** — pass `--profile <name>` on `nemo customization unsloth --job-json …` when the default `gpu` profile is wrong. There is no `execution_profile` field in `UnslothJobInput` today.

Quick filter:

```bash
nemo jobs list-execution-profiles -f json 2>/dev/null | python3 -c "
import sys, json
for p in json.load(sys.stdin):
    if p.get('provider') in ('gpu', 'gpu_distributed'):
        print(p['profile'], p.get('backend'), p.get('provider'))
"
```

Do not run `nemo customization --help` unless submit returns unknown plugin.

## Gated HuggingFace models

Commands for the gated-model rule in `SKILL.md` § **Pre-flight**.

| Symptom / log excerpt | Likely cause | Fix |
|-----------------------|--------------|-----|
| Job fails in **download** step; `Failed to access upstream storage`; `InternalServerError` 502; `Verify that the referenced credentials are valid` | Missing/stale `hf-token` secret, or fileset created without `token_secret` | Steps below — then re-submit |
| Secret exists but download still fails | User has not **accepted the model license** on huggingface.co for that repo | Accept license with the same HF account as the token, then re-submit |
| Public model (e.g. `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`) | No secret needed | Omit `token_secret` on the fileset |

**Convention:** secret name `hf-token` in workspace `default`. Any valid secret name works if referenced consistently in `token_secret`.

### 1. Check whether the secret exists

```bash
nemo secrets list --workspace default
```

The token needs **Read** access (https://huggingface.co/settings/tokens), and the user must accept the model license on the model's Hugging Face page with the same account.

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
HF_REPO=<hf-repo>          # gated repo, <org>/<model>
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

Job errors like `Failed to pull image … nhx-unsloth-training:… Not Found`, `manifest unknown`, or a missing automodel training image mean the **connected platform's Docker daemon** (the one that runs GPU job steps) does not have the image. With the default `NHX_BASE_URL` (`127.0.0.1:8080` / `localhost:8080`), that daemon is usually on the same machine as the agent; with a user-overridden URL (e.g. `<remote-host>:8080`), it is on the remote target host instead.

**Did the user override the base URL?** (same rule as **Platform unreachable** — track this from the start of the workflow.)

| Situation | Action |
|-----------|--------|
| **Remote platform** — user gave a host/URL (e.g. `<remote-host>:8080`) or you set `NHX_BASE_URL` to something other than `http://127.0.0.1:8080` or `http://localhost:8080` | **Do not** run `docker build`, `docker pull`, or `docker buildx bake` on the agent machine — that only affects the agent's local daemon, not the remote platform. Tell the user they must build or load the image **on the target host** (the machine whose Docker daemon runs the GPU job steps). Report with the template in `references/reporting.md`, then append **Report follow-up — missing image (remote platform)** below. Stop; do not retry submit until the user confirms the image is available on the target. |
| **Local platform** — default URL only (`127.0.0.1:8080` / `localhost:8080`) | Build or pull on **that same host** where `nemo services run` and Docker share a daemon. See build commands below and `docker/unsloth/README.md` (unsloth) or automodel docker docs. Set env vars **before** starting/restarting the platform. |

Image env vars are read when the platform starts (not per job):

```bash
export NHX_IMAGE_REGISTRY=<registry>
export NHX_IMAGE_TAG=<tag>
```

**Automodel** — also set `NHX_AUTOMODEL_IMAGE_REGISTRY=$NHX_IMAGE_REGISTRY`.

**Unsloth** — set `NHX_UNSLOTH_TRAINING_IMAGE` (and optionally `NHX_UNSLOTH_TASKS_IMAGE`) to the full built ref, then restart Helix services so the env var takes effect.

### Build on the target host (unsloth)

Run on the **platform host** (SSH, console, or CI on that box — not from the agent when the platform is remote):

```bash
cd /path/to/nemo-helix

# Local build (platform and Docker on the same machine)
docker buildx bake \
  -f docker-bake.hcl \
  nhx-unsloth-training \
  --load \
  --set "*.platform=linux/amd64"

export NHX_UNSLOTH_TRAINING_IMAGE="${IMAGE_REGISTRY:-my-registry/nemo-helix-dev}/nhx-unsloth-training:${BAKE_TAG:-local}"
# Restart platform so the env var is picked up
nemo services restart
```

Or push to a registry the target can pull from — see **Option B** in `docker/unsloth/README.md` — then set `NHX_UNSLOTH_TRAINING_IMAGE` to that full ref before restart.

After the image is on the target, re-submit the same job JSON (use a fresh `output.name` if a prior partial run already registered an adapter).

### Report follow-up — missing image (remote platform)

When submit or poll returns a missing-image error and the base URL is **user-overridden**, start with the **Report to user** template in `references/reporting.md` (status `error`, **Output adapter fileset (planned):**, Notes quoting the pull error and naming the target host). Then append these sections:

**What you need to do on the target host** — build or load the training image on the machine running the NeMo Helix (where `docker info` works for the platform's daemon), set `NHX_UNSLOTH_TRAINING_IMAGE` or automodel image env vars, and restart Helix services. Full steps: `docker/unsloth/README.md` (unsloth) or automodel docker docs.

**Re-submit after the image is available:**

```bash
export NHX_BASE_URL=<user's platform URL>
cd /path/to/nemo-helix
nemo customization <plugin> --job-json /tmp/job.json --workspace default [--profile <gpu-profile>]
```

Then poll until terminal status. Offer to re-submit once the user confirms the image is on the target — do not attempt a local Docker build from the agent for a remote platform.

## W&B / integrations not working

Job JSON has `integrations.wandb` (and/or `integrations.mlflow`) but tracking fails or never starts. Full setup: `references/integrations-setup.md`.

| Symptom / log excerpt | Likely cause | Fix |
|-----------------------|--------------|-----|
| Training logs **omit** `[launcher]` lines; entrypoint is the training module directly (e.g. `Running main process: /opt/venv/bin/python [-m nhx.unsloth.tasks.training]` with **no** preceding `Fetching secret wandb-api-key`) | **jobs-launcher** binary missing or `launcher_tool_path` wrong — secrets are never injected | Build launcher on the **platform host**, set absolute `launcher_tool_path`, restart services. See § **jobs-launcher missing** below and `integrations-setup.md` § **jobs-launcher**. |
| `wandb: ERROR` / HTTP 401 / `permission denied` after launcher lines present | Platform secret `wandb-api-key` has wrong or placeholder value; local `wandb login` cache is **not** used | `uv run nemo secrets update wandb-api-key --value "$WANDB_API_KEY" --workspace default` (or `--from-file -`). Re-submit. |
| `RuntimeError: WandbCallback requires wandb to be installed` (unsloth) | `nhx-unsloth-training` image lacks `wandb` | Rebuild image with `nhx-unsloth[integrations]` extra; set `NHX_UNSLOTH_TRAINING_IMAGE`; restart platform. See **Missing training images**. |
| Compile/submit warning: `integrations.wandb is configured but api_key_secret is missing` | Job JSON has `wandb` block without `api_key_secret` | Add `"api_key_secret": "default/wandb-api-key"` (or your secret ref). |
| MLflow run never appears; W&B works | `tracking_uri` unreachable from container (`localhost`, wrong port) | Use `docker0` host IP + published port (e.g. `http://${DOCKER_HOST_IP}:5001`). See `integrations-setup.md` § **`tracking_uri`**. |

### jobs-launcher missing (W&B secret not injected)

The Docker executor wraps the training entrypoint with **jobs-launcher** only when `launcher_tool_path` points to a built binary on the platform host. If the path is missing or still the default stub, the container runs training **without** secret injection — `WANDB_API_KEY` never reaches the process even when `integrations.wandb.api_key_secret` is set in job JSON.

**Working** — launcher fetched the secret before training:

```text
[launcher] 2026/06/11 22:03:03 Fetching secret wandb-api-key from workspace default...
[launcher] 2026/06/11 22:03:03 Successfully fetched secret wandb-api-key and mapped to WANDB_API_KEY
[launcher] 2026/06/11 22:03:03 Injected 1 secret(s) as environment variables
[launcher] 2026/06/11 22:03:03 Running main process: /opt/venv/bin/python [-m nhx.unsloth.tasks.training]
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
cd /path/to/nemo-helix/services/core/jobs/jobs-launcher
./build-manual.sh linux amd64
```

Set `jobs.executors.docker.launcher_tool_path` in `~/.nemo/config.yaml` to the **absolute** path of the built `jobs-launcher` binary, then `uv run nemo services restart`. Re-submit with a fresh `output.name`.

## Unsloth submit errors

| Error / symptom | Cause | Fix |
|-----------------|-------|-----|
| `Unsloth training requires platform.runtime: docker` | Platform not configured for Docker GPU jobs | Start platform with Docker runtime and a GPU execution profile |
| Unknown execution profile | Default `gpu` profile missing or wrong | Re-list profiles; pass `--profile <exact-name>` on submit |
| Missing `nhx-unsloth-training` image / `Failed to pull image` / `manifest unknown` | Image not on the **platform host's** Docker daemon | **Missing training images** above |
| `torch.cuda.is_available()` False in training step logs | GPU not exposed to the container step | Confirm the execution profile is GPU-backed; check platform Docker GPU setup |
| Job stuck in `active` after training step completes | Upload / model-entity steps still running | Keep polling (`SKILL.md` § **Workflow** step 6) |


## Runtime failures

For a job that starts but fails or misbehaves while training. The poll script's `ALERT:` and `WARN:` lines point here; for a failed job, read `nemo jobs get-logs <job>` for the first error before acting.

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `CUDA out of memory` in the logs, or a training step exits with code 137 | Batch, sequence length or model too large for the GPU | Halve `micro_batch_size` (raise accumulation to keep the global batch), then turn on activation checkpointing (`activation_checkpointing: true`; unsloth `use_gradient_checkpointing`), then shorten `max_seq_length`. GRPO: also lower `num_generations_per_prompt`, keeping the batch divisible. Sizing tables: `batch-sizing.md` |
| NCCL timeout or a collective-operation watchdog error | One rank stalled or died, so the others waited; usually that rank ran out of memory first | Find the first error across all ranks' logs and fix that. If none, retry on one node to rule out the interconnect |
| `ALERT: train_loss is nan`, or loss that spikes then turns NaN | Learning rate too high, fp16 overflow, or broken rows | Cancel. Lower the learning rate 2–5×, train in `bf16`, and check the data for empty labels |
| Submit or compile rejects the batch: `global_batch_size (…) must be divisible by …`, `batch_size (…) must be divisible by micro_batch_size (…) * …`, or `Total GPUs (…) must be divisible by …` | The batch or parallel layout does not split evenly over the GPUs | Fix the numbers per the constraint in `batch-sizing.md`; for GRPO, `hyperparameters-rl.md` § **GRPO-specific** |
| `ALERT: every prompt group got the same reward …` (GRPO) | Every rollout in a group scores the same: the task is too hard or too easy for the model, the reward is broken, or sampling is near-greedy | Cancel. Check the environment's reward on a few rows by hand, keep `temperature` near 1.0, and adjust the data's difficulty |
| `WARN: train_loss is …, below 0.2` or `WARN: val_loss rose …` | The model is memorizing a small dataset | Fewer epochs, a lower learning rate or more data. The base-vs-tuned score on the test split decides whether it hurt |

## Older jobs missing from a backend's job list

`GET /apis/customization/v2/workspaces/{workspace}/automodel/jobs` (and `/unsloth/jobs`, `/rl/jobs`) returns jobs whose spec stores `backend`. Jobs submitted before that field existed appear only in `nemo jobs list` (`GET /apis/jobs/v2/workspaces/{workspace}/jobs`).

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
| Create dataset fileset | `nemo files filesets create <name> --workspace default --purpose dataset` (a new name when the data changes: `SKILL.md` § **Workflow** step 2) |
| Create HF weights fileset (public) | `nemo files filesets create <name> --workspace default --purpose model --exist-ok --storage '{"type":"huggingface","repo_id":"<repo>","repo_type":"model","revision":"main"}'` |
| Create HF weights fileset (gated) | Same as above plus `"token_secret":"hf-token"` — see § **Gated HuggingFace models** |
| Upload | `nemo files upload <local> <fileset> --workspace default --remote-path train.jsonl` |
| List files | `nemo files list <fileset> --workspace default` |
| Create model | `nemo models create <name> --workspace default --exist-ok --input-data '<json>'` |
| Poll job | `nemo jobs get-status <automodel\|unsloth\|rl>-<job-id>` |

Automodel:

| Action | Command |
|--------|---------|
| Submit | `nemo customization automodel --job-json <job.json> --workspace default` |
| Status | `nemo jobs get-status <name>` (submit `"name"`, verbatim: already `automodel-…`) |
| Live schema | `nemo customization automodel explain` |

Unsloth:

| Action | Command |
|--------|---------|
| Submit | `nemo customization unsloth --job-json <job.json> --workspace default [--profile P]` |
| Status | `nemo jobs get-status <name>` (submit `"name"`, verbatim: already `unsloth-…`) |
| Live schema | `nemo customization unsloth explain` |

