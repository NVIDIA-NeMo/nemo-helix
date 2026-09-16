<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nhx-automodel container images

All Automodel Docker build files live under **`docker/automodel/`** (wheel vendor script: `docker/base/build-ffmpeg-vendor.sh`).

Two images for the **nhx-automodel** customization backend, plus the shared **`nhx-customizer-tasks`** CPU image used by all customization backends. Published as flat repo names under **`my-registry/nemo-helix-dev/nhx-*`** (no nested `nhx/...` path — some registries reject that on push).

| Image | Dockerfile | Role |
|-------|------------|------|
| `nhx-automodel-base` | `docker/automodel/Dockerfile.nhx-automodel-base` | PyTorch 26.05 + Automodel + `mamba-ssm` / `causal-conv1d` wheels |
| `nhx-customizer-tasks` | `docker/Dockerfile.nhx-customizer-tasks` | Shared CPU tasks (`file_io`, `model_entity`, `model_spec`, LoRA sidecar) |
| `nhx-automodel-training` | `docker/automodel/Dockerfile.nhx-automodel-training` | Training (`nhx.automodel.tasks.training`) and retrieval mining (`nhx.automodel.tasks.retrieval_mine`) |

Full references (default tag `local`):

- `my-registry/nemo-helix-dev/nhx-automodel-base:local`
- `my-registry/nemo-helix-dev/nhx-customizer-tasks:local`
- `my-registry/nemo-helix-dev/nhx-automodel-training:local`

Bake file: **`docker-bake.hcl`** at the Platform repo root (`context = "."`). Run all commands from the Platform repo root.

## `docker buildx bake --print`

`--print` only parses the HCL and prints JSON. A **0.0s FINISHED** result is normal — no image is built. Use it to verify targets, tags, and platforms before a real build.

## Prerequisites

1. **CUDA extension wheels** (`causal-conv1d-wheel`, `mamba-ssm-wheel`) - built from this directory or pulled from NGC. The wheel Dockerfile and uv locks live under `docker/locks/` (ported from `nhx`).

2. **Base image tag** - after building the base, set `BASE_TAG_AUTOMODEL` (or push to `BASE_REGISTRY`) before building tasks/training.

## Build wheels and push to NGC (from Platform root)

```bash
cd /path/to/Platform

docker login nvcr.io

export WHEELS_TAG="$(git rev-parse --short HEAD)"
# Bake variables (WHEELS_REGISTRY, WHEELS_TAG, IMAGE_REGISTRY) are overridden via env, not --set.
# Example:
#   export WHEELS_REGISTRY=my-registry/nemo-helix-dev
#   export IMAGE_REGISTRY=my-registry/nemo-helix-dev

docker buildx bake --print -f docker-bake.hcl nhx-automodel-gpu-wheels

docker buildx bake \
  -f docker-bake.hcl \
  nhx-automodel-gpu-wheels \
  --push \
  --set "*.platform=linux/amd64"
```

Override platform: `export BUILD_PLATFORM=linux/amd64` or `--set "*.platform=linux/amd64"`.

## Build automodel images (from Platform root)

```bash
cd /path/to/Platform

export WHEELS_TAG="${WHEELS_TAG:-3fd6986ff173b598446ffac06d9be3f84b482495}"
export BAKE_TAG="${WHEELS_TAG}"

docker buildx bake \
  -f docker-bake.hcl \
  nhx-automodel-base-builder \
  --push \
  --set "*.platform=linux/amd64"

docker buildx bake \
  -f docker-bake.hcl \
  nhx-automodel \
  --push \
  --set "*.platform=linux/amd64"
```

To use wheels already published without rebuilding, `export WHEELS_TAG=<existing-tag>` and matching `BAKE_TAG`.

Override registry: `export WHEELS_REGISTRY=...` and `export IMAGE_REGISTRY=...` before bake.

## Tasks / training runtime (platform glue)

**Base (`nhx-automodel-base`):** NGC PyTorch 26.05, Automodel `uv sync --locked`, pinned `transformers`/`torch`.

**Automodel cherry-picks:** Platform-specific patches under `docker/automodel/cherry-picks/` are applied after `update_pyproject_pytorch.sh` and before `uv sync`. Re-pin or drop patches when upstream `r0.x.y` absorbs the same changes.

| Patch | Purpose |
|-------|---------|
| `3d98f6e3.diff` | Drop old media deps (`decord`, `imageio-ffmpeg`) from Automodel extras, remove Automodel's unconditional `opencv-python-headless` dependency, and stop selecting `mistral-common`'s image extra for text training. The container skips Automodel's VLM extra, so FFmpeg-bearing `av` / `opencv-python-headless` wheels are not installed. |

**Customizer tasks image (`nhx-customizer-tasks`):** `uv sync --package nhx-customization-common --package nhx-models --no-dev --inexact` from the customizer workspace slice (`docker/customizer/`). Hosts shared CPU steps (`file_io`, `model_entity`, `model_spec`, LoRA sidecar) for all customization backends.

**Training image (`nhx-automodel-training`):** GPU training (`nhx.automodel.tasks.training`) and retrieval mining (`nhx.automodel.tasks.retrieval_mine`). Platform glue is installed with `uv pip install --overrides /app/docker/automodel/no_override_requirements.txt -e "/app/services/automodel[training]"` (alongside its platform dependencies), then `uv pip install --no-deps -e /opt/Automodel` re-pins `nemo_automodel` from the base clone (not PyPI). `uv sync` is not used here: it upgrades `transformers` and breaks `PreTrainedModel`.

**Retrieval mining:** Data Designer's `retrieval-prepare` stages the encoder into shared job storage with `nhx-customizer-tasks`, then the training image loads it with `HF_HUB_OFFLINE=1`. All retrieval steps share one container-backed profile (`data_designer.job_executor_profile`, or `--profile`).

## Runtime

Entrypoint is `/opt/venv/bin/python` on both images. The compiler routes CPU steps to `nhx-customizer-tasks` and the GPU training step to `nhx-automodel-training` (see `nhx.automodel.app.jobs.compiler` and `nhx.automodel.images`). Local smoke:

```bash
# Customizer tasks image — default CMD prints file_io help.
docker run --rm my-registry/nemo-helix-dev/nhx-customizer-tasks:local

# Automodel CPU step (file_io).
docker run --rm my-registry/nemo-helix-dev/nhx-customizer-tasks:local \
  -m nhx.customization_common.tasks.file_io --service-source automodel --service-name customizer --help

# Training image — default CMD prints training help.
docker run --rm my-registry/nemo-helix-dev/nhx-automodel-training:local

# Training image — retrieval hard-negative mining.
docker run --rm my-registry/nemo-helix-dev/nhx-automodel-training:local \
  -m nhx.automodel.tasks.retrieval_mine --help
```

The job compiler resolves `nhx-customizer-tasks` and `nhx-automodel-training` under `NHX_AUTOMODEL_IMAGE_REGISTRY` (default `my-registry/nemo-helix-dev`). See `nhx.automodel.images`.
