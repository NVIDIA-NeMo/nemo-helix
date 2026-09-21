<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run the Gym resource-server matrix

Use this manual test to verify that NeMo Platform can launch and complete a Gym evaluation job for each resource server in the checked-in test matrix. Normal E2E and CI runs skip this matrix.

## Matrix scope

The 47 pairs were derived from the agent configurations shipped with NeMo Gym v0.5.0. An explicitly composed pair means that Gym provides a Hydra configuration that connects that particular agent to that resource server. The matrix tests those known pairings; it does not assume that every agent can work with every resource server. BigCodeBench is excluded because it dynamically installs a separate evaluation environment with system dependencies such as GDAL.

## Prerequisites

- Docker is running.
- The repository is bootstrapped with `make bootstrap`.

## Build the Gym task image

Build and load the local image into Docker:

```bash
IMAGE_REGISTRY=my-registry \
BAKE_TAG=local \
make docker-load TARGET=nmp-gym-tasks-docker
```

Build the image's smoke-test stage to verify the Gym, Ray, and tokenizer imports:

```bash
IMAGE_REGISTRY=my-registry \
BAKE_TAG=local \
make docker-build TARGET=nmp-gym-tasks-smoke-test
```

The matrix and the running Evaluator must both use `my-registry/nmp-gym-tasks:local`.

## Create the Python 3.13 test environment

Synchronize the repository development environment into a Python 3.13 virtual environment:

```bash
uv python install 3.13
uv venv --python 3.13 --clear /tmp/nemo-platform-e2e-client-py313
env -u VIRTUAL_ENV \
  UV_PROJECT_ENVIRONMENT=/tmp/nemo-platform-e2e-client-py313 \
  uv sync --frozen --all-packages
```

## Start NeMo Platform

Start all services and controllers from the Python 3.13 environment:

```bash
NMP_BASE_URL=http://localhost:8080 \
NMP_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX=igw-mock- \
NEMO_EVALUATOR_GYM_TASKS_IMAGE=my-registry/nmp-gym-tasks:local \
/tmp/nemo-platform-e2e-client-py313/bin/nemo services run \
  --service-group all \
  --controller-group all \
  --host 0.0.0.0 \
  --port 8080
```

These environment variables select the local platform, enable the mock inference provider used by the tests, and force Gym jobs to use the locally built image. `NMP_JOBS_EXECUTORS`, `NMP_JOBS_ENABLE_SUBPROCESS_EXECUTOR`, `NMP_IMAGE_REGISTRY`, and `NMP_IMAGE_TAG` are not required for this test setup.

Leave this process running while executing the matrix.

## Run one resource server

Start with one job:

```bash
/tmp/nemo-platform-e2e-client-py313/bin/python \
  plugins/nemo-evaluator/scripts/gym-matrix/run.py \
  --image my-registry/nmp-gym-tasks:local \
  --server arc_agi \
  --workers 1
```

## Run the complete matrix

The worker count limits concurrent pytest workers and NMP jobs. Use one worker locally so that Gym jobs and their Ray clusters run sequentially:

```bash
/tmp/nemo-platform-e2e-client-py313/bin/python \
  plugins/nemo-evaluator/scripts/gym-matrix/run.py \
  --image my-registry/nmp-gym-tasks:local \
  --workers 1
```

The complete sequential run takes approximately two to three hours. More than one worker starts multiple Ray clusters concurrently and can exhaust Docker Desktop's CPU and memory allocation.

## How it works

The script:

1. Extracts bundled `example.jsonl` datasets from the Gym task image into a temporary directory.
2. Sets the opt-in environment variable that enables the manual-only pytest module.
3. Runs the matrix with the current Python 3.13 interpreter and pytest-xdist.
4. Deletes the temporary datasets when pytest exits.

Each parameterized test submits one Evaluator job to NMP, waits for completion, validates the persisted Gym trial and reward, and cleans up the job and temporary workspace.

Matrix entries that require a GPU, an external service, or a real policy model are skipped before an NMP job is submitted.
