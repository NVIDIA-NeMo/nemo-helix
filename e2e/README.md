<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# E2E Test Harness

The E2E test harness runs tests against one of two **platforms**: **docker** (quickstart: Docker + config file) or **kubernetes** (already-deployed cluster). You select the platform at run time with `--docker` or `--kubernetes`. A config file is only used for docker (default: `e2e/quickstart/default.yaml`; auth: `e2e/quickstart/auth.yaml` when `--feature auth`). Tests declare which platform(s) they support via the `platform` marker and optional features (e.g. auth) via the `feature` marker.

## Test Organization

E2E tests are organized by functional service workflows that represent complete user scenarios.

1. Tests declare which platform(s) they run on with `@pytest.mark.platform("docker", "kubernetes")` (or no marker to run on both).
2. Tests that require auth use `@pytest.mark.feature("auth")` and run when you pass `--feature auth`.
3. Tests that require GPU use `@pytest.mark.feature("gpu")` and run when you pass `--feature gpu`.
4. Tests that require both auth and GPU use `@pytest.mark.feature("auth", "gpu")` and run when you pass `--feature auth --feature gpu`.
5. Tests known to fail on Astra use `@pytest.mark.skip_on_astra`; they are skipped when `NHX_E2E_ON_ASTRA=1` is set (e.g. in e2e-astra / e2e-astra-gpu CI jobs).

## Platforms

1. **Docker** (`--docker`): Runs the NHX API in a Docker container (quickstart) using a config file. Config defaults to `e2e/quickstart/default.yaml` (or `auth.yaml` when `--feature auth`); override with `--config path/to/config.yaml`.

2. **Kubernetes** (`--kubernetes`): Connects to an already-deployed cluster. Requires `--cluster-url` or `NHX_E2E_CLUSTER_URL`. No local backend is started; OpenShift is treated as kubernetes.

## Building Images

Before running e2e tests, build the required Docker images:

```bash
# Build all images needed for E2E tests
make docker/build

# Build a specific image
make docker/nhx-api

# Build multiple specific images
make docker/nhx-api docker/nhx-tasks

# Build with custom registry and tag
make docker/build REGISTRY=my-registry TAG=latest
```

## Running Tests

### Docker

Pass `--docker` to run quickstart via Docker + config:

```bash
# Run all e2e tests with Docker (uses e2e/quickstart/default.yaml)
uv run pytest e2e --docker -v

# Run specific test file
uv run pytest e2e/test_hello_world.py --docker -v

# Docker with custom config
uv run pytest e2e --docker --config=path/to/config.yaml -v

# Run with verbose output and no capture
uv run pytest e2e --docker -v -s
```

### Docker/quickstart already running (e.g. `nemo quickstart up`)

If quickstart is already running outside the harness (e.g. you ran `nemo quickstart up`), pass `--docker --cluster-url=...` to connect without starting a backend. Config and Docker are not used; tests run against the given URL.

```bash
uv run pytest e2e --docker --cluster-url=http://localhost:8080 -v
```

### Kubernetes (Already-Deployed Cluster)

```bash
# Run e2e tests against an already-deployed cluster (requires --cluster-url)
uv run pytest e2e --kubernetes --cluster-url=https://my-cluster.example.com -v

# Via environment variable
NHX_E2E_CLUSTER_URL=https://my-cluster.example.com uv run pytest e2e --kubernetes -v
```

Use this for staging, production, or a locally deployed NHX instance (e.g., minikube). OpenShift is treated as kubernetes.

### Feature: Auth

Tests that require authentication use the `feature("auth")` marker. Run them by passing `--feature auth`:

```bash
# Run only auth-required tests (tests with no feature marker are skipped when --feature is passed)
uv run pytest e2e --docker --feature auth -v

# Run only workspace auth tests
uv run pytest e2e/test_workspaces.py --docker --feature auth -v
```

When no `--feature` is passed, only tests with **no** feature marker run. When `--feature` is passed, only tests that have that feature (and any other required features) run; tests with no feature marker are skipped. If a test has multiple feature markers (e.g. `feature("auth", "gpu")`), pass all of them (e.g. `--feature auth --feature gpu`).

### Feature: GPU

GPU-requiring tests use `feature("gpu")`. Use a GPU-capable config for docker (e.g. `--config e2e/quickstart/gpu.yaml` if available) and pass `--feature gpu`:

```bash
uv run pytest e2e --docker --feature gpu --config=path/to/gpu-config.yaml -v
```

Service-specific GPU tests can require an additional feature marker. Safe Synthesizer GPU tests use `feature("gpu", "safe-synthesizer")`.

### Feature: Customizer (Automodel, Unsloth, RL)

Plugin-backed customization tests live in `e2e/customizer/tests/` and use backend-specific
feature markers plus a suite marker:

- **`smoke`** (`test_smoke.py`): checks functionality, not quality. Each backend trains a few
  steps on a small dataset (the first 64 rows of each shared JSONL file, or a dedicated small
  asset), and the job auto-deploys its output: the test passes an unbound vLLM deployment config
  as the job's `deployment_config`, waits for the deployment, sends one chat completion (or one
  embeddings request), and deletes it. A LoRA adapter is served from its base model's deployment.
- **`uplift`**: longer runs on larger datasets that check the tuned model beats the base. SFT
  and DPO train for one epoch, serve the base and tuned models on vLLM (the models service's
  default vLLM image), and score both. The embedding test runs the validated NVDocs recipe
  (3 epochs on 4 GPUs) on the full mined dataset. GRPO validates on the same holdout prompts
  before and after training and requires both `val_accuracy` and `train_reward` to increase.

| Test | Backend / mode | Base model | Features |
|---|---|---|---|
| `test_smoke.py` | Automodel and Unsloth SFT (LoRA, all-weights), NeMo RL DPO (Kubernetes only) | `Qwen/Qwen3-0.6B` | `gpu`, backend, `smoke` |
| `test_smoke.py` | NeMo RL GRPO, `wheels-v1` `math_with_judge` environment (Kubernetes only) | `Qwen/Qwen3-0.6B` | `gpu`, `rl`, `grpo`, `smoke` |
| `test_smoke.py` | Automodel `bi_encoder`, all-weights, served for embeddings | `nvidia/Nemotron-3-Embed-1B-BF16` | `gpu`, `automodel`, `smoke` |
| `test_automodel.py` | Automodel SFT, LoRA and all-weights | `Qwen/Qwen3-0.6B` | `gpu`, `automodel`, `uplift` |
| `test_unsloth.py` | Unsloth SFT, LoRA and all-weights | `Qwen/Qwen3-0.6B` | `gpu`, `unsloth`, `uplift` |
| `test_rl_dpo.py` | NeMo RL DPO (Kubernetes only) | `Qwen/Qwen3-0.6B` | `gpu`, `rl`, `uplift` |
| `test_rl_grpo.py` | NeMo RL GRPO, `wheels-v1` `math_with_judge` environment (Kubernetes only); `val_accuracy` and `train_reward` must increase | `Qwen/Qwen3-0.6B` | `gpu`, `rl`, `grpo`, `uplift` |
| `test_automodel_embedding.py` | Automodel `bi_encoder`, all-weights on 4 GPUs, scored with `retrieve-eval`; requires nDCG@10 uplift >= 0.05 | `nvidia/Nemotron-3-Embed-1B-BF16` | `gpu`, `automodel`, `embedding`, `uplift` |
| `test_automodel_nemotron.py` | Automodel LoRA on 4 GPUs (expert parallel 4), served on 4 GPUs | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `gpu`, `automodel`, `h100`, `uplift` |

A test runs only when all of its features are selected:

```bash
# Against a Kubernetes cluster (review cluster in CI, or local minikube)
uv run pytest e2e/customizer/tests/ --kubernetes --feature gpu --feature automodel --feature unsloth --feature rl --feature smoke -v
uv run pytest e2e/customizer/tests/ --kubernetes --feature gpu --feature automodel --feature unsloth --feature rl --feature uplift -v
uv run pytest e2e/customizer/tests/ --kubernetes --feature gpu --feature rl --feature grpo --feature smoke -v
uv run pytest e2e/customizer/tests/ --kubernetes --feature gpu --feature automodel --feature embedding --feature uplift -v
uv run pytest e2e/customizer/tests/ --kubernetes --feature gpu --feature automodel --feature h100 --feature uplift -v
```

**CI schedule** (Platform-Deploy `.github/workflows/docker.yaml`, `Run Kubernetes/Docker Customizer GPU E2E tests`):

| Trigger | Suite | Failure handling |
|---|---|---|
| Nightly (`0 9 * * *`), Platform-Deploy PRs, `workflow_dispatch` with `run-e2e` (default) | `smoke` | Fails the run |
| Weekly (`0 11 * * 6`), `workflow_dispatch` with `run-e2e` and `customizer-suite: uplift` | `uplift` | Reported in the E2E summary; does not fail the run |

On Kubernetes, smoke runs install the platform with auth enabled
(`e2e/k8s/values/auth-overlay.yaml`), so job pods authenticate as their service principals.
Uplift runs without auth because its eval calls the inference gateway without principal headers.

Docker runs skip `rl` and `grpo`. CI does not select `embedding` uplift (three epochs over the
full mined set take hours) or `h100` (needs 4x 80GB H100s).

GRPO needs OpenSandbox, which the platform chart does not install. The `customizer-grpo`
`workflow_dispatch` input installs it on the Kubernetes GPU runner
(`e2e/k8s/scripts/install_opensandbox_minikube.sh`), installs the platform with `sandboxClusterCapable=true`,
and selects `grpo`. It is off by default, including on schedules.

GRPO smoke trains every environment format (`wheels-v1`, `adapter-wheels-v1`, `native-v1`, and
`native-v1` reference-only) with DTensor full weights, Automodel full weights, and Automodel LoRA,
and serves each output. The `native-v1` formats install the server's requirements from a package
index, so they need sandbox internet. That setting is platform-wide, and turning it on stops
forcing the wheels formats offline, so the job runs in two phases: the wheels formats first, then
a Helm upgrade with `platformConfig.rl.sandbox_allow_internet=true` and the `native-v1` formats
(feature `grpo-internet`).

The GRPO uplift test (`test_rl_grpo.py`) and the `h100` test (`test_automodel_nemotron.py`) are
disabled in code with `pytest.mark.skip`; remove the marker to enable each one.

**Test assets.** SQuAD, HelpSteer3, the mined NVDocs data, the GRPO math datasets, and the
Qwen3-0.6B, Nemotron 3 Embed 1B, and Nemotron 3.5 Lightning snapshots are staged from
``s3://aire-e2e-assets/`` into platform filesets before tests run (see
``e2e/customizer/stage_assets.py`` and ``assets_manifest.json``). The Lightning snapshot is
~66 GB and is synced only by the `h100` test.

The GRPO environment packages are not published. They vendor the `nemo-gym`, `ray`, and `openai`
versions of the NeMo-RL commit the training image is built from, so CI builds them from the
checkout under test (`generate_grpo_assets.py --env-only`) and a pin bump cannot leave
them stale. That build is the one place CI pulls from outside S3: it clones NeMo-RL, downloads
wheels from PyPI, and the `ascii-tree` conversion fetches the environment from the Prime
Intellect hub and its dataset (`kalomaze/ascii-tree-mix-it1`, ~36 MB) from Hugging Face.

Generate the off-CI assets, then publish them with ``publish_assets_to_s3.sh``:

```bash
# Mined NVDocs data (needs a running platform with a GPU and the Data Designer plugin):
# embedding_nvdocs/ (every training row, all 20,909 eval queries) and embedding_nvdocs_smoke/ (512 rows).
NHX_BASE_URL=http://localhost:8080 PYTHONPATH=. uv run --frozen \
    python e2e/customizer/mine_embedding_data.py --workspace default

# GRPO datasets: grpo_math_smoke/ (64 / 16 prompts) and grpo_math_uplift/ (2,048 / 200 prompts).
# Also builds the environment packages locally; publish_assets_to_s3.sh does not upload them.
# Needs internet.
PYTHONPATH=. uv run --frozen python e2e/customizer/generate_grpo_assets.py
```

The embedding uplift recipe was validated on the full mined set; `--max-train-rows` and
`--max-eval-queries` subsample for quicker local runs, but then the uplift floor and
`warmup_steps` no longer apply.

**Environment variables for customizer tests:**

| Variable | Description | Default |
|---|---|---|
| `E2E_REQUIRE_UPLIFT` | Set to `1` to require tuned > base instead of tuned >= base - 0.02 (the embedding test always requires nDCG@10 uplift >= 0.05) | unset |
| `E2E_GPU_TEST_TIMEOUT` | Per-test timeout in seconds (the embedding test sets its own 12-hour timeout) | `5400` |
| `E2E_ASSETS_CACHE_DIR` | Local cache for assets synced from S3 | `e2e/customizer/.asset-cache` |
| `E2E_ASSETS_SYNC_TIMEOUT` | Timeout in seconds for each `aws s3 sync` | `1800` |
| `S3_BUCKET` / `S3_ENDPOINT_URL` | Override the asset bucket or S3 endpoint | manifest bucket |
| `HF_TOKEN` | Hugging Face token for models pulled at runtime | (optional for public repos) |

### Custom Registry and Tag (Docker)

```bash
uv run pytest e2e --docker --registry=my-registry --tag=v1.0.0 -v

NHX_E2E_REGISTRY=my-registry NHX_E2E_TAG=latest uv run pytest e2e --docker -v
```

### Custom Principal ID

```bash
uv run pytest e2e --docker --principal-id=user@example.com -v

NHX_E2E_PRINCIPAL_ID=user@example.com uv run pytest e2e --docker -v
```

## Makefile Targets

```bash
# Run e2e tests with docker
make test-e2e-docker

# Run e2e tests against local minikube (see "Kubernetes" section above for custom cluster via pytest)
make test-e2e-minikube
```

### Kubernetes configuration ownership

Platform-Deploy CI loads Kubernetes scripts and values from the checked-out NeMo Helix source at
`platform/e2e/k8s/`. This repository retains only the values used by its documented local workflows:
`e2e/k8s/values/default.yaml` and `e2e/k8s/values/minikube-auth.yaml`. Add or update CI scenario
configuration in `NVIDIA-NeMo/nemo-helix`, not here.

## Local Kubernetes: make run-kube

The top-level target **`make run-kube`** sets up, builds, and runs the entire NHX stack on a local Kubernetes cluster (minikube) in one command. It is useful for local e2e testing against a real deployment without running the step-by-step minikube workflow.

**What it does:**

1. Starts minikube if not already running (via the same setup as `make setup-minikube-gpu`).
2. Builds the docker-cpu images with a unique tag and loads them into minikube's Docker daemon.
3. Installs or upgrades the NHX Helm release using `e2e/k8s/values/default.yaml` and the built image tag.

**Prerequisites:** `minikube`, `docker`, `kubectl`, and `helm` installed; `NGC_API_KEY` set (for minikube setup when the cluster is created).

**Run the stack:**

```bash
make run-kube
```

The script prints the image tag (e.g. `local-<epoch>`) and an example pytest command. Run e2e tests against the cluster using the same registry and tag:

```bash
# Substitute the NHX_E2E_REGISTRY and NHX_E2E_TAG printed by make run-kube.
NHX_E2E_REGISTRY=docker.io/my-registry NHX_E2E_TAG=local-1700000000 uv run pytest e2e --kubernetes --cluster-url=http://localhost:80 -v
```

If your ingress is on a different host or port, set `NHX_E2E_CLUSTER_URL` or pass `--cluster-url`. See [CONTRIBUTING.md](../CONTRIBUTING.md) for more on local minikube and `make run-kube`.

## Minikube GPU (Local Development)

For local GPU testing against a full Kubernetes deployment, you can use minikube with GPU passthrough. This sets up a real Kubernetes cluster on your machine with GPU support, deploys the NHX platform via Helm, and runs GPU E2E tests against it.

### Prerequisites

- A machine with an NVIDIA GPU and `nvidia-smi` working
- Docker, minikube, kubectl, helm, and jq installed
- An NGC API key (`NGC_API_KEY`)

### Workflow

**1. Set required environment variables:**

```bash
export NGC_API_KEY='your-ngc-api-key'

# Optional: GitLab registry access (for CI images)
export GITLAB_TOKEN='your-gitlab-token'
export GITLAB_USER='your-gitlab-username'

# Optional: HuggingFace token (for model downloads)
export HF_TOKEN='your-hf-token'
```

**2. Set up the minikube cluster:**

```bash
make setup-minikube-gpu
```

This creates a minikube cluster with GPU support, configures ingress, and creates Kubernetes secrets (NGC, optional GitLab/HuggingFace). The cluster uses the profile `minikube` by default.

## Local auth-enabled Helm harness (CPU-only)

Use this when you want to validate the Helm chart and auth wiring locally without requiring a GPU or NGC-backed NIM operator.

What it proves:

1. The chart installs on minikube with `auth.enabled=true`
2. The embedded PDP is reachable from split pods via the API service, not `localhost`
3. Unauthenticated requests are rejected and authenticated requests succeed
4. The existing auth E2E suite can run against the chart deployment

### 1. Start a CPU-only minikube profile

```bash
./e2e/k8s/scripts/setup_local_minikube_cpu.sh
```

This creates a local profile named `minikube-auth` by default, enables ingress, exposes it on `http://localhost:30080`, and creates the minimal placeholder secrets the chart expects (`ngc-api`, `nvcrimagepullsecret`).

### 2. Build local images into minikube

```bash
eval "$(minikube -p minikube-auth docker-env)"
cd ../nemo-helix
CI_COMMIT_SHA=$(git rev-parse HEAD) BAKE_TAG=local IMAGE_REGISTRY=docker.io/my-registry docker buildx bake docker-cpu
```

### 3. Install the auth-enabled chart values

```bash
MINIKUBE_PROFILE=minikube-auth ./e2e/k8s/scripts/install_nhx_auth_e2e.sh
```

This installs the chart with `e2e/k8s/values/minikube-auth.yaml`, which:

- disables `k8s-nim-operator`
- enables auth with embedded PDP
- keeps the shared embedded PDP URL on the in-cluster API service for split pods
- lets the API pod itself loop back to `localhost` automatically
- keeps ingress enabled for local smoke tests

### 4. Run the auth smoke check and auth E2E tests

```bash
MINIKUBE_PROFILE=minikube-auth ./e2e/k8s/scripts/run_auth_e2e.sh
```

By default this:

- checks `/apis/auth/discovery` reports `"auth_enabled": true`
- verifies unauthenticated workspace listing returns `401`
- verifies an authenticated `X-NHX-Principal-Id` request returns `200`
- runs `e2e/test_workspaces.py` against the cluster

You can override the base URL or test selection:

```bash
NHX_E2E_CLUSTER_URL=http://localhost:30080 ./e2e/k8s/scripts/run_auth_e2e.sh e2e/test_secrets.py
```

**3. Install NHX via Helm:**

```bash
make install-helm-e2e
```

This installs the platform using `e2e/k8s/values/default.yaml`. By default the chart uses `nvcr.io` images and the chart's appVersion tag (e.g. `26.2.0`), which may require NGC access or may not match the images you want. To use a different registry and tag (e.g. GitLab CI images for `main`), set `NHX_E2E_REGISTRY` and `NHX_E2E_TAG` before running the install (see "Using a registry or main-branch images" below).

**4. Point your Docker CLI at minikube's Docker daemon (only if building images locally):**

```bash
eval $(minikube docker-env -p minikube)
```

This ensures that `docker build` commands run inside minikube's Docker daemon, making images immediately available to Kubernetes pods without pushing to a remote registry.

**5. Build all container images (only for local-image workflow):**

```bash
make docker/all
```

By default, images are tagged as `my-registry/<name>:local` (configured in `docker-bake.hcl`).

**6. Install or upgrade with your chosen images:**

- **Locally built images:** set `NHX_E2E_TAG=local` and `NHX_E2E_REGISTRY=my-registry`, then run `make install-helm-e2e` (or re-run it to upgrade). Pods will use the images you built in minikube's Docker daemon.
- **Registry images (e.g. main):** set `NHX_E2E_REGISTRY` and `NHX_E2E_TAG` to your registry and tag, then run `make install-helm-e2e`. See "Using a registry or main-branch images" below.

**7. Run the GPU E2E tests:**

```bash
# General GPU tests
make test-e2e-minikube-gpu

# Backend customization tests (see "Feature: Customizer" above for the embedding and h100 tests)
uv run pytest e2e/customizer/tests/ --kubernetes --feature gpu --feature automodel -v
uv run pytest e2e/customizer/tests/ --kubernetes --feature gpu --feature unsloth -v
uv run pytest e2e/customizer/tests/test_rl_dpo.py --kubernetes --feature gpu --feature rl -v
```

- `test-e2e-minikube-gpu` runs the GPU suite against the minikube cluster (with cluster URL set).
- Backend customization tests live under `e2e/customizer/tests/` and stage datasets/models from S3 before uploading to platform filesets.

### Test your branch on minikube (build API and core images, run e2e)

Use this sequence when you have code changes (e.g. on a branch) and want to build the NHX API and core images, load them into minikube, update the Helm deployment, and run e2e. Steps 1–5 run on the **host** where minikube is running; step 6 can run on the host or from a pod that can reach the cluster.

1. **Point Docker at minikube** (on host):

   ```bash
   eval $(minikube docker-env -p ${MINIKUBE_PROFILE:-minikube})
   ```

2. **Build the nhx-api and nhx-core images.** The install script applies the same registry/tag to both, so both must exist in minikube. Default bake tag is `local` with registry `my-registry`:

   ```bash
   make docker/nhx-api docker/nhx-core
   ```

   Images will be `my-registry/nhx-api:local` and `my-registry/nhx-core:local`.

   **Which image to rebuild for local code changes:** The API pod uses the **nhx-api** image; the controller pod uses the **nhx-core** image. Rebuild the image that matches your changes.

3. **Set registry and tag for Helm** (use the same as bake defaults for local builds):

   ```bash
   export NHX_E2E_REGISTRY=my-registry
   export NHX_E2E_TAG=local
   ```

4. **Install or upgrade Helm** so the API (and core) deployment use your images:

   ```bash
   make install-helm-e2e
   ```

   This updates the release; the API and core pods will roll with the new image.

5. **Optional: force pod restart** if the deployment didn’t roll (e.g. same tag):

   ```bash
   kubectl rollout restart deployment nemo-helix-api -n default
   kubectl rollout restart deployment nemo-helix-core-controller -n default
   kubectl rollout status deployment nemo-helix-api -n default --timeout=300s
   ```

6. **Run the models e2e test** (cluster URL is minikube IP):

   ```bash
   CLUSTER_URL="http://$(minikube ip -p ${MINIKUBE_PROFILE:-minikube})"
   uv run pytest e2e/test_models.py --kubernetes --feature gpu --cluster-url="$CLUSTER_URL" -v -s
   ```

   Or use the make target (runs all e2e with kubernetes and GPU):

   ```bash
   make test-e2e-minikube-gpu
   ```

To use a **custom tag** (e.g. a commit SHA) instead of `local`, build both images with that tag and set `NHX_E2E_TAG` to it before `make install-helm-e2e`:

```bash
BAKE_TAG=my-branch-abc123 make docker/nhx-api docker/nhx-core
export NHX_E2E_REGISTRY=my-registry
export NHX_E2E_TAG=my-branch-abc123
make install-helm-e2e
```

### Using a registry or CI-built images

The default chart values use `nvcr.io/nvidia/nemo-microservices/` images with the chart's appVersion tag (e.g. `26.2.0`). Those images may not be public or may not match the branch you want to test. To use images from your GitLab registry (e.g. from a `main` or branch pipeline):

1. **Use the commit SHA as the tag.** CI does not create a `main` tag; branch pipelines tag images with `CI_COMMIT_SHA`. Pick a commit from the branch you want (e.g. from the latest pipeline on `main` in GitLab → Pipelines → copy the commit SHA).

2. **Set registry and tag** when installing or upgrading:

   ```bash
   export NHX_E2E_REGISTRY='gitlab-master.nvidia.com:5005/aire/microservices/nhx'   # or your project's CI_REGISTRY_IMAGE
   export NHX_E2E_TAG='<commit-sha>'   # e.g. abc123def from a main or branch pipeline
   make install-helm-e2e
   ```

3. **If the registry requires auth**, ensure the cluster can pull:
   - For **GitLab**: set `GITLAB_TOKEN` and `GITLAB_USER` before running `make setup-minikube-gpu` so the setup script creates the `gitlab-imagepull` secret; the install script adds it to the release when `GITLAB_TOKEN` is set.
   - For **NGC**: the setup script creates `nvcrimagepullsecret` from `NGC_API_KEY`; the chart uses it by default.

4. **If you already installed** with the default images, run `make install-helm-e2e` again with `NHX_E2E_REGISTRY` and `NHX_E2E_TAG` set; Helm will upgrade the release with the new image locations.

### Cleanup

```bash
minikube delete -p "${MINIKUBE_PROFILE:-minikube}"
```

This runs `minikube delete` on the profile, removing the cluster and all its resources.

### Customization

| Environment Variable | Description                                                  | Default                                          |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `MINIKUBE_PROFILE`   | Minikube profile name                                        | `minikube`                                       |
| `NHX_E2E_REGISTRY`   | Image registry for NHX services (used by `install-helm-e2e`) | (none — uses chart defaults, e.g. `nvcr.io/...`) |
| `NHX_E2E_TAG`        | Image tag for NHX services (used by `install-helm-e2e`)      | (none — uses chart appVersion, e.g. `26.2.0`)    |
| `NHX_E2E_REGISTRY_HOST`  | Registry host for optional minikube pre-pull login; defaults to the host parsed from `NHX_E2E_REGISTRY` | (parsed) |
| `NHX_E2E_REGISTRY_USER`  | Registry username for optional minikube pre-pull login | `$USER` |
| `NHX_E2E_REGISTRY_TOKEN` | Registry token/password for optional minikube pre-pull login | (none) |
| `HELM_CHART`         | Override the Helm chart source                               | `platform/k8s/helm`                             |
| `HELM_VALUES_FILE`   | Override the Helm values file                                | `e2e/k8s/values/default.yaml`                    |
| `HELM_EXTRA_ARGS`    | Additional `helm install/upgrade` arguments                  | (none)                                           |

## Test Structure

```
e2e/
├── quickstart/
│   ├── default.yaml                # Default config (used when --docker)
│   └── auth.yaml                   # Auth config (used when --docker --feature auth)
├── customizer/
│   ├── assets_manifest.json        # Dataset/model catalog for S3 asset publishing
│   ├── stage_assets.py             # S3 sync + SDK upload for datasets/models
│   ├── customizer_jobs.py          # Job submit, vLLM deploy helpers
│   ├── customizer_eval.py          # Deterministic uplift scoring
│   ├── customization_helpers.py    # Shared helpers for customizer E2E tests
│   ├── generate_customizer_data.py # Regenerate datasets from the manifest
│   ├── download_hf_model_snapshot.py
│   ├── publish_assets_to_s3.sh     # Off-CI S3 publish workflow
│   ├── tests/
│   │   ├── conftest.py             # S3 staging fixtures
│   │   ├── test_automodel.py       # Automodel uplift (lora + all_weights)
│   │   ├── test_unsloth.py         # Unsloth LoRA uplift
│   │   └── test_rl_dpo.py          # RL DPO uplift
│   └── testdata/
│       ├── prompt_completion/
│       ├── chat_format/
│       └── dpo/
├── k8s/scripts/                    # Kubernetes helper scripts (minikube setup, helm install, log collection)
├── conftest.py                     # Pytest fixtures (backend, sdk, workspace)
├── README.md                       # This file
├── test_*.py                       # Other test files
└── test_workspaces.py              # Workspace tests with auth (use --feature auth)
```

## Configuration (Quickstart Only)

Config is only used when running with `--docker`. The default config is `e2e/quickstart/default.yaml`; when `--feature auth` is set, `e2e/quickstart/auth.yaml` is used. Override with `--config path/to/config.yaml`.

Config files are merged with platform defaults at runtime. For example, the default quickstart config enables auth and sets `platform.base_url` for the Docker network.

## Fixtures

The test harness provides these fixtures:

### Core Fixtures (from `e2e/conftest.py`)

- **`e2e_config`** (session scope): The current E2EConfig (docker config or kubernetes sentinel)
- **`backend`** (session scope): The Docker backend for docker platform; `None` for kubernetes
- **`sdk`** (session scope): NeMoHelix SDK client with authentication
- **`workspace`** (function scope): Unique workspace for each test (auto-cleanup)
- **`image`** (session scope): Function to build fully qualified image paths
- **`principal_id`** (session scope): Authentication identity for tests
- **`registry`** / **`tag`** (session scope): Docker registry and image tag

### Testing Utilities (from `nhx.testing`)

For authentication-enabled tests, use these utilities:

```python
from nhx.testing import as_user, short_unique_name, unique_email

# Create SDK client as a specific user
user_sdk = as_user(sdk, unique_email("test-user"))

# Generate unique names with length constraints
workspace_name = short_unique_name("my-workspace")  # e.g., "my-workspace-a1b2c3d4"

# Generate unique email addresses
email = unique_email("admin")  # e.g., "admin-12345678@example.com"
```

### Mocking Inference Calls

For tests that need to mock LLM responses without real inference backends, use `add_mock_provider`:

```python
from nhx.testing import add_mock_provider

def test_with_mock_llm(sdk: NeMoHelix, workspace: str):
    provider = add_mock_provider(
        sdk,
        workspace=workspace,
        name="my-judge",
        mock_response_body={"choices": [{"message": {"content": "mock response"}}]},
    )
    # provider.name is auto-prefixed with "igw-mock-" for routing
    response = sdk.inference.gateway.provider.post(
        "v1/chat/completions",
        name=provider.name,
        workspace=workspace,
        body={"model": "test", "messages": []},
    )
```

See [Mock Provider README](../services/core/inference-gateway/src/nhx/core/inference_gateway/api/mock_provider/README.md) for complete documentation.

## Writing Tests

### Basic Test

```python
def test_my_endpoint(sdk: NeMoHelix, workspace: str):
    """Test an endpoint."""
    response = sdk._client.get(f"/v1/workspaces/{workspace}/my-service/endpoint")
    assert response.status_code == 200
```

### Test with Authentication

```python
import pytest
from nhx.testing import as_user, unique_email
from nemo_helix import PermissionDeniedError

@pytest.mark.platform("docker")
@pytest.mark.feature("auth")
def test_access_control(sdk: NeMoHelix, workspace: str):
    """Test role-based access control.

    Run with: pytest e2e --docker --feature auth
    """
    # Create SDK as different user
    viewer_email = unique_email("viewer")
    viewer_sdk = as_user(sdk, viewer_email)

    # Add viewer to workspace
    sdk.workspaces.members.create(
        workspace=workspace,
        principal=viewer_email,
        roles=["Viewer"]
    )

    # Viewer can read but not write
    workspace_obj = viewer_sdk.workspaces.retrieve(workspace)
    assert workspace_obj is not None

    with pytest.raises(PermissionDeniedError):
        viewer_sdk.workspaces.update(workspace, description="Should fail")
```

### Platform and Feature Markers

Use **`@pytest.mark.platform("docker", "kubernetes")`** to restrict tests to specific platforms. No marker means the test runs on both. Use **`@pytest.mark.feature("auth")`**, `"gpu"`, `"automodel"`, `"unsloth"`, `"rl"`, `"grpo"`, `"embedding"`, or `"h100"` for tests that require a feature; run with `--feature <name>` to include them:

```python
@pytest.mark.platform("docker")
def test_docker_only(sdk: NeMoHelix, workspace: str):
    """This test only runs with docker."""
    ...

# Require multiple features (AND semantics)
@pytest.mark.feature("gpu", "automodel")
def test_automodel_workflow(sdk: NeMoHelix, workspace: str):
    """Runs only with: --feature gpu --feature automodel"""
    ...

# Mark all tests in a module (e.g. auth-required)
pytestmark = [pytest.mark.platform("docker"), pytest.mark.feature("auth")]
```

### Waiting for Job Completion

```python
from nhx.testing.e2e import wait_for_platform_job

def test_job_lifecycle(sdk: NeMoHelix, workspace: str):
    """Test job creation and completion."""
    # Create job
    response = sdk._client.post(
        f"/apis/hello-world/v2/workspaces/{workspace}/jobs",
        json={"name": "test-job", "spec": {"message": "Hello"}}
    )
    assert response.status_code == 201

    # Wait for completion
    job = wait_for_platform_job(sdk, "test-job", workspace)
    assert job.status == "completed"
```

## Authentication Testing

### Default Test Users

The testing utilities provide default test user emails:

```python
from nhx.testing import TEST_USER_EMAIL, TEST_ADMIN_EMAIL

# TEST_USER_EMAIL = "user@example.com"
# TEST_ADMIN_EMAIL = "admin@example.com"
```

### Principal ID

For authentication, tests use a principal ID (X-NHX-Principal-Id header). The default is `e2e-test-user@example.com`, but this can be overridden via:

- `--principal-id` CLI argument
- `NHX_E2E_PRINCIPAL_ID` environment variable

### Auth-Enabled Runs

Run auth-required tests with `--docker --feature auth` (uses `e2e/quickstart/auth.yaml`). The default config is `e2e/quickstart/default.yaml`.

## Best Practices

- **Organize tests by functional service workflow**: Group tests that exercise complete user workflows together.
- **Use platform and feature markers**: Mark tests with `platform("docker", "kubernetes")` and `feature("auth")`, `feature("gpu")`, `feature("gpu", "automodel")`, or `feature("gpu", "unsloth")` as needed. Use compound features (AND semantics) to create sub-suites that run in dedicated CI jobs.
- **Mark Astra-known failures**: Use `@pytest.mark.skip_on_astra` (or `pytestmark = [pytest.mark.skip_on_astra]` in a conftest) for tests that are known to fail on Astra; set `NHX_E2E_ON_ASTRA=1` when running on Astra to skip them.
- **Keep E2E tests to a minimum**: Focus on critical customer workflows, not exhaustive testing.
- **Select one platform per run**: Use `--docker` or `--kubernetes`; use `--feature auth` (or `gpu`) to include feature-gated tests.
