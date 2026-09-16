<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nhx-unsloth container images

Unsloth customization jobs use the shared CPU image for file_io / model_entity
steps and a dedicated GPU image for training.

| Image | Dockerfile | Role |
|-------|------------|------|
| `nhx-customizer-tasks` | `docker/Dockerfile.nhx-customizer-tasks` | Shared CPU steps (`file_io`, `model_entity`) |
| `nhx-unsloth-training` | `docker/Dockerfile.nhx-unsloth-training` | NGC PyTorch base + Unsloth ML stack + platform glue. ENTRYPOINT is `/opt/venv/bin/python`. |

Bake file: **`docker-bake.hcl`** at the Platform repo root (`context = "."`). Run all commands from the Platform repo root.

Tags use the same bake variables as automodel (`IMAGE_REGISTRY`, `BAKE_TAG`; defaults in `docker-bake.hcl`):

- Default: `${IMAGE_REGISTRY}/nhx-unsloth-training:${BAKE_TAG}` (e.g. `…/nhx-unsloth-training:local` with `--load`)
- CPU steps: `${IMAGE_REGISTRY}/nhx-customizer-tasks:${BAKE_TAG}` via `get_tasks_image()` (override with `NHX_CUSTOMIZER_TASKS_IMAGE` or `NHX_UNSLOTH_TASKS_IMAGE`)

---

## Build & push (from Platform repo root)

```bash
cd /path/to/Platform

# --- Option A: local build (loads into the local daemon, no registry needed) ---
docker buildx bake \
  -f docker-bake.hcl \
  nhx-unsloth-training \
  --load \
  --set "*.platform=linux/amd64"
# Result: `${IMAGE_REGISTRY}/nhx-unsloth-training:${BAKE_TAG}` in `docker images`.

# --- Option B: push to a registry ---
docker login nvcr.io
export IMAGE_REGISTRY="my-registry/nemo-helix-dev"
export BAKE_TAG="$(git rev-parse --short HEAD)"
docker buildx bake \
  -f docker-bake.hcl \
  nhx-unsloth-training \
  --push \
  --set "*.platform=linux/amd64"
# Result: `${IMAGE_REGISTRY}/nhx-unsloth-training:${BAKE_TAG}` in the registry.
```

> **Prebuilt wheels (local builds).** The image installs `mamba-ssm` +
> `causal-conv1d` from the shared `causal-conv1d-wheel` / `mamba-ssm-wheel`
> bake contexts (same wheels as `nhx-automodel-base`). A **local** bake must
> build those wheels first — prepend `USE_LOCAL_WHEELS=1` (or point
> `WHEELS_REGISTRY`/`WHEELS_TAG` at prebuilt wheels), otherwise bake tries to
> pull `${WHEELS_REGISTRY}/causal-conv1d-wheel:${WHEELS_TAG}` and fails. To
> avoid recompiling on every image build, build the wheels once with
> `USE_LOCAL_WHEELS=1 docker buildx bake -f docker-bake.hcl nhx-automodel-gpu-wheels`.

The build pulls the NGC PyTorch base, then:

1. `uv pip install unsloth --torch-backend=auto transformers==5.5.0 huggingface-hub==1.5.0` with
   `preserve_base_torch.txt` overrides so the NGC base's PyTorch + CUDA are not
   replaced. Unsloth's resolver still pulls `unsloth_zoo`, trl, peft,
   accelerate, datasets, bitsandbytes, and xformers. **transformers is pinned
   explicitly** to `5.5.0` (override at build time via
   `--build-arg TRANSFORMERS_VERSION=...`).
1b. bitsandbytes — compiled from source against the NGC CUDA 13.1 toolkit
    (PyPI wheels only ship through cuda130), replacing the wheel from step 1.
1c. mamba-ssm + causal-conv1d — installed from the prebuilt `cu13.1.1` / `cp312`
    wheels shared with `nhx-automodel-base` (see the **Prebuilt wheels** note
    above). Required by hybrid Mamba/SSM models (e.g. NVIDIA Nemotron-H `*-A3B`).
1d. Flash Attention 2 — **not currently installed** (commented TODO in the
    Dockerfile). Unsloth does not depend on it; without it you may see
    `FA2 = False` / `Xformers = None` on newer CUDA stacks.
2. Editable install of the platform glue: `nemo-helix-sdk`,
   `nemo-helix-plugin`, `nhx-common`, `nhx-unsloth`.

We considered using the official `unsloth/unsloth` image as a base. We
didn't because it's 13 GB, bundles Jupyter Lab + SSH + Unsloth Studio
(which we don't need for a non-interactive job runner), and we'd still
need to layer the platform glue on top. Building on NGC PyTorch keeps
the image smaller and the entrypoint/user-config under our control.

---

## Local smoke test (no platform, no GPU)

```bash
# Use the same tag the bake produced. For local builds that's the bare name.
IMAGE=nhx-unsloth-training:local

# CMD prints the training help banner — proves entrypoint + the ML stack import cleanly.
docker run --rm "$IMAGE"

# Extra args replace CMD; include `-m nhx.unsloth.tasks.training` or you get plain `python --help`.
docker run --rm "$IMAGE" \
  -c "import unsloth, trl, peft, bitsandbytes; print('ok')"
```

---

## GPU pod runbook — end-to-end

The same runbook works whether the GPU pod is a developer's interactive
node, a CI cluster runner, or a customer's air-gapped lab.

### 0. Prereqs on the host

- NVIDIA driver compatible with the NGC PyTorch base (CUDA 13.1 at the
  time of writing; check `docker/automodel/Dockerfile.nhx-automodel-base` for the latest
  pin if unsure).
- `nvidia-container-toolkit` installed so Docker can mount GPUs.
- Network access to your image registry (`nvcr.io` by default).
- A running NeMo Helix install (`make bootstrap` + `nemo services run`)
  with `platform.runtime: docker` configured. See top-level `AGENTS.md` for setup.

### 1. Build the image

From the Platform repo root, on a machine with Docker buildx.

Pick **one** of the following depending on where the GPU host will pull from:

**A) Same host builds and runs the platform** (e.g. the DinD GPU pod under
`services/unsloth/scripts/gpu-test/`). The bare local tag is enough:

```bash
docker buildx bake \
  -f docker-bake.hcl \
  nhx-unsloth-training \
  --load \
  --set "*.platform=linux/amd64"
# → nhx-unsloth-training:local in the local daemon.
```

**B) Push to a registry the GPU host will pull from**:

```bash
export IMAGE_REGISTRY="my-registry/nemo-helix-dev"
export BAKE_TAG="$(git rev-parse --short HEAD)"

docker buildx bake \
  -f docker-bake.hcl \
  nhx-unsloth-training \
  --push \
  --set "*.platform=linux/amd64"
# → ${IMAGE_REGISTRY}/nhx-unsloth-training:${BAKE_TAG} in the registry.
```

**C) Air-gapped GPU host** (save + `scp` + `docker load`):

```bash
export IMAGE_REGISTRY="my-registry/nemo-helix-dev"
export BAKE_TAG="$(git rev-parse --short HEAD)"
# The image needs the mamba-ssm + causal-conv1d wheel images as build contexts.
# A direct `docker buildx build` can't resolve the bake `*_wheel_context()`
# functions, so prefer `docker buildx bake nhx-unsloth-training` (with
# USE_LOCAL_WHEELS=1) which wires them automatically. If you must use
# `docker buildx build`, pass both contexts explicitly:
docker buildx build \
  -f docker/Dockerfile.nhx-unsloth-training \
  --output type=docker,dest=/tmp/nhx-unsloth-training.tar \
  --target runtime \
  -t "${IMAGE_REGISTRY}/nhx-unsloth-training:${BAKE_TAG}" \
  --build-context "platform-workspace=path-to-platform-workspace" \
  --build-context "causal-conv1d-wheel-image=docker-image://${WHEELS_REGISTRY}/causal-conv1d-wheel:${WHEELS_TAG}" \
  --build-context "mamba-ssm-wheel-image=docker-image://${WHEELS_REGISTRY}/mamba-ssm-wheel:${WHEELS_TAG}" \
  .

scp /tmp/nhx-unsloth-training.tar gpu-pod:/tmp/
ssh gpu-pod docker load -i /tmp/nhx-unsloth-training.tar
```

### 2. Point the platform at your tag

On the host running `nemo services run`, set the full image ref. For a local
build this is just the bare name:

```bash
# Local bake (--load; matches docker-bake.hcl defaults):
export NHX_UNSLOTH_TRAINING_IMAGE="${IMAGE_REGISTRY:-my-registry/nemo-helix-dev}/nhx-unsloth-training:${BAKE_TAG:-local}"

# Or, when you pushed (Option B above):
export NHX_UNSLOTH_TRAINING_IMAGE="${IMAGE_REGISTRY}/nhx-unsloth-training:${BAKE_TAG}"

# Restart so the env var takes effect.
nemo services restart
```

Or persist in `~/.nemo/config.yaml`:

```yaml
unsloth:
  training_image: my-registry/nemo-helix-dev/nhx-unsloth-training:local
```

### 3. Prepare model + dataset filesets

```bash
# 0.5B Qwen — fast enough for a smoke test on a single GPU.
nemo files filesets create base-qwen-05b -w default
# Push your model weights (or use an existing entity that already points at a fileset).
nemo files upload <path-to-qwen-checkpoint> --fileset default/base-qwen-05b -w default

nemo models create base-qwen-05b \
  -w default \
  --fileset default/base-qwen-05b \
  --finetuning_type all_weights

# Tiny chat dataset.
nemo files filesets create unsloth-smoke-dataset -w default
nemo files upload ./smoke-dataset.jsonl --fileset default/unsloth-smoke-dataset -w default
```

### 4. Write a smoke job

Save as `unsloth-smoke.json`:

```json
{
  "name": "qwen-unsloth-smoke",
  "model": {
    "name": "default/base-qwen-05b",
    "max_seq_length": 1024,
    "load_in_4bit": true,
    "dtype": "auto"
  },
  "dataset": {
    "path": "default/unsloth-smoke-dataset",
    "text_field": "text"
  },
  "training": {
    "finetuning_type": "lora",
    "lora": {"rank": 16, "alpha": 16}
  },
  "schedule": {
    "max_steps": 20,
    "warmup_steps": 2,
    "lr_scheduler_type": "linear",
    "logging_steps": 1
  },
  "batch": {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4
  },
  "optimizer": {
    "learning_rate": 2e-4,
    "optim": "adamw_8bit"
  },
  "hardware": {
    "gpus": "0",
    "precision": "bf16"
  },
  "output": {
    "name": "qwen-unsloth-smoke-out",
    "save_method": "lora"
  }
}
```

### 5. Submit and watch

```bash
# Sanity-check schema + planned compilation locally.
nemo customization unsloth explain
cat unsloth-smoke.json | jq .

# Submit. Captures the job id from the JSON response.
JOB_ID=$(
  nemo customization unsloth submit unsloth-smoke.json -w default --json \
    | jq -r '.id'
)
echo "Submitted: $JOB_ID"

# Tail the status — should go PENDING → ACTIVE → COMPLETED across the four steps.
nemo jobs status "$JOB_ID" -w default

# Stream logs from each step.
nemo jobs logs "$JOB_ID" -w default --step model-and-dataset-download --follow
nemo jobs logs "$JOB_ID" -w default --step training --follow
nemo jobs logs "$JOB_ID" -w default --step model-upload --follow
nemo jobs logs "$JOB_ID" -w default --step model-entity-creation --follow

# Verify the output entity (adapter, since save_method=lora).
nemo models adapters list -w default --model_name base-qwen-05b
nemo models adapters retrieve qwen-unsloth-smoke-out \
  --model_name base-qwen-05b -w default
```

### 6. Common gotchas

| Symptom | Fix |
|---|---|
| `compile()` errors with "platform.runtime: docker" | Set `platform.runtime: docker` in `~/.nemo/config.yaml` and restart services. |
| `compile()` errors with "Docker daemon unreachable" | Confirm `docker info` works as the user running `nemo services`. |
| First job step errors with `Model 'X' has no fileset attached` | Attach a fileset to the model entity (`nemo models update --fileset ...`). |
| `training` step errors with `bitsandbytes`/CUDA mismatch (`libbitsandbytes_cuda131.so` not found) | Rebuild `nhx-unsloth-training` — the image compiles bitsandbytes from source against NGC CUDA 13.1 (same pattern as `nhx-automodel-base`). Override `BNB_MAX_JOBS` at build time if nvcc OOMs. |
| `WandbCallback requires wandb to be installed` | Rebuild `nhx-unsloth-training` — the image installs `wandb` and `mlflow-skinny` for integrations. |
| `training` step OOMs on a small GPU | Reduce `model.max_seq_length` and / or set `model.load_in_4bit: true`. |
| `model-entity-creation` errors with "Adapter already exists" | Pick a fresh `output.name` (the unsloth compiler is "always create"; no overwrite). |
| Step config not picked up (`NEMO_JOB_STEP_CONFIG_FILE_PATH is not set`) | The container was started outside the Jobs runner — only platform-driven submit populates this. |

### 7. Cleanup

```bash
# Remove the smoke adapter + fileset.
nemo models adapters delete qwen-unsloth-smoke-out --model_name base-qwen-05b -w default
nemo files filesets delete qwen-unsloth-smoke-out -w default
```

---

## Architecture notes

- **Step layout** mirrors `docker/automodel/`. The compiler in
  `nhx.unsloth.app.jobs.compiler` emits the same 4-step
  `PlatformJobSpec` shape automodel emits; the only difference is the
  image and the training entrypoint module.
- **`nemo_automodel` is intentionally not in this image** — unsloth is a
  separate ML stack. If you need both backends on the same cluster, run
  both images side by side; jobs from each backend route to their own
  `nhx-{backend}-training` image via env-var overrides.
- **transformers + huggingface-hub pins** — the training image pins `transformers==5.5.0`
  and `huggingface-hub==1.5.0` in
  `docker/Dockerfile.nhx-unsloth-training` (compatible with unsloth's upstream
  blocklists). Other HF deps (trl, peft, bitsandbytes, etc.) still come from
  unsloth's resolver. **PyTorch + CUDA** stay on the NGC base stack via
  `--system-site-packages` and `preserve_base_torch.txt` / `no_override_requirements.txt`
  overrides (same impossible-marker pattern as automodel).
- **bitsandbytes** — compiled from source in the image (v0.49.1, same approach as
  `nhx-automodel-base`) because NGC 26.02 is CUDA 13.1 and PyPI only ships
  prebuilt libs through cuda130.
