<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Taskset runs with workload identity, on local kind

The nightly taskset-run E2E's `auth-on-exchange-true` row (`.github/workflows/nightly-taskset-runs-e2e.yaml`), run on
your machine. It runs `e2e/test_taskset_runs.py` on the Authentik demo (`contrib/auth/authentik`) with workload token
exchange, so every Job runs as the `nemo-helix-jobs` ServiceAccount on behalf of the submitter. Use it to
reproduce a Job-identity bug, such as a Job's 403 on the Entity Store (NVBug 7083853), or to test its fix.

Auth on without token exchange doesn't reach these bugs: there, Jobs act as the submitter. The nightly covers
that in its `auth-on-exchange-false` row, on the platform chart rather than this demo: the demo's own services
authenticate through token exchange, so it doesn't run without it.

## Prerequisites

- Docker, with at least 8 GB of memory: Authentik, the platform, the builder, OpenSandbox and two sandboxes fit
  in 8 GB.
- `kind`, `kubectl`, `helm`, `uv`, `openssl`, `curl`, `python3`, and bash 4 or later (`brew install bash` on
  macOS).
- A checkout with the code you want to test. The test, the driver and the scripts come from the checkout; the
  platform comes from the images.

## 1. Images

The platform images must match your node's architecture:

- **amd64:** use CI's published images. Set `NHX_E2E_IMAGE_REGISTRY=ghcr.io/nvidia-nemo/nemo-helix` and set
  `NHX_E2E_IMAGE_TAG` to a commit SHA on `main`. Log in to `ghcr.io` first.
- **arm64 (Apple silicon):** CI publishes amd64 images only, and an arm64 node can't run them. Build the four
  images locally (15–25 min the first time):

  ```bash
  tag="$(git rev-parse --short=10 HEAD)-arm64"
  make docker-load DOCKER_TARGET="nhx-api-docker nhx-tasks-docker nhx-builder-tasks-docker nhx-kaniko-docker" \
      DOCKER_PLATFORMS=linux/arm64 IMAGE_REGISTRY=local BAKE_TAG="${tag}"
  ```

  Then set `NHX_E2E_IMAGE_REGISTRY=local`, `NHX_E2E_IMAGE_TAG="${tag}"` and `NHX_E2E_LOAD_IMAGES=1`, so the setup
  loads the images into kind.

## 2. Set up the cluster

```bash
NHX_E2E_IMAGE_REGISTRY=local NHX_E2E_IMAGE_TAG="${tag}" NHX_E2E_LOAD_IMAGES=1 \
    e2e/k8s/scripts/setup_authentik_taskset_runs.sh
```

On an existing cluster this takes about 3 minutes; a new one takes longer. The setup creates the kind cluster
`nhx-authentik-reuse`, or reuses it. It then installs Authentik, the platform, OpenSandbox and the builder, and
starts the taskset-run controller. The gateway listens on `https://127.0.0.1:18082`, and a restart loop keeps it
up. The setup also makes `nemo-setup` the platform admin and creates the builder's secrets in the `system`
workspace.

The setup is idempotent: rerun it after a code or image change. Set `KIND_CLUSTER_NAME` for a different cluster.
Secrets go to `~/.cache/nhx-e2e/<cluster>/` (0700) and are never printed; `NHX_E2E_STATE_DIR` changes where.
These are the builder registry's password, the signing key, app passwords and tokens.

## 3. Run the test

```bash
state=~/.cache/nhx-e2e/nhx-authentik-reuse
set -a; source "${state}/env"; set +a   # KUBECONFIG, kube context, gateway URL, gateway CA
e2e/k8s/scripts/authentik_user_token.sh nemo-user "${state}/nemo-user.token"
NHX_E2E_ACCESS_TOKEN="$(cat "${state}/nemo-user.token")" NHX_E2E_WORKLOAD_TOKEN_EXCHANGE=true \
    _TYPER_FORCE_DISABLE_TERMINAL=1 E2E_SERVICES_LOG_DIR="${state}/runs/latest" \
    uv run --frozen pytest e2e/test_taskset_runs.py --kubernetes --cluster-url "${NHX_E2E_CLUSTER_URL}" \
    --feature taskset-runs -m taskset_runs_e2e -v --no-cov --junitxml="${state}/runs/latest/junit.xml"
```

- **The user:** `nemo-user` is a non-admin editor, the case the Job-identity bugs are about. To see whether an
  admin submitter changes the outcome, run as `nemo-setup`.
- **Run time:** about 5 minutes when the image build fails; 10–15 minutes when the run completes.
- **The cancel test:** add `-m "taskset_runs_e2e or taskset_runs_cancel_e2e" --feature taskset-runs-cancel` to
  also run it, as the nightly does. It builds its own images and adds up to 15 minutes.
- **The token:** it expires after 1 hour, so mint a new one for each run.

## 4. Read the result

The evidence JSON in `E2E_SERVICES_LOG_DIR` records each step. These are the outcomes to expect:

- `[image-build] task-image-build-…: PermissionDeniedError: HTTP 403: Not allowed to read entities in workspace …`
  is NVBug 7083853, as on `main` until its fix lands. The test reads the error from the failed Job's pod log,
  because the build stays "building" until `JOB_GRACE` (2 min) runs out and `runs submit` times out without the
  cause. The evidence's `job_pods` shows who each pod ran as.
- `[identity] …` means the run didn't exercise the row's Job identity: discovery disagrees with
  `NHX_E2E_WORKLOAD_TOKEN_EXCHANGE`, or a Job pod ran as another ServiceAccount, without the workload token with
  exchange on, or with it with exchange off.
- A task Job that finishes while the run's result is missing, with a 403 on an Entity Store write, is NVBug
  6993994. That bug is reachable only once 7083853 is fixed.

## 5. Tear down

```bash
kill "$(cat ~/.cache/nhx-e2e/nhx-authentik-reuse/port-forward.pid)"
kind delete cluster --name nhx-authentik-reuse
```

## Friction

- **The cluster's secrets live in the checkout that created it.** `run.sh` keeps them in
  `contrib/auth/authentik/.generated/`, which is git-ignored and per checkout: the Authentik bootstrap token, the
  client secrets, the kubeconfig and the gateway CA. A second checkout makes its own, and the static tests in
  `tests/auth_idp/static` do too. Set up an existing cluster from another checkout and those secrets don't match
  it. Run the setup from the checkout that created the cluster, or copy that checkout's `.generated/` first.
- **Only one gateway port-forward fits on port 18082.** The setup stops `run.sh`'s port-forward and its own
  earlier loop, but not one started by another tool. Check with `lsof -iTCP:18082 -sTCP:LISTEN`.
- **`run.sh up k8s` resets the values.** Its `helm upgrade` has no `--reuse-values`, so it drops the taskset
  values. The setup reapplies them right after; never run `run.sh up k8s` alone on this cluster.
- **`run.sh` registers a NeMo CLI context.** It adds `authentik-k8s` to your `nemo` config.
- **The password grant rejects a user's own password.** Authentik accepts an app password instead. The token
  script mints one through the Authentik API, with the bootstrap token, and caches it in the state directory.
- **No demo user is the platform admin.** The chart sets `auth.admin_email: admin@example.com`, so the setup gives
  that email to `nemo-setup`. The builder's `system` secrets need a platform admin.
- **`helm --wait` can hang.** It waits out a hook's whole timeout even when every pod is ready, so the setup waits
  on `kubectl rollout status` instead.
- **A failed image build is slow to surface.** It looks like a submit timeout, of 240 s or 900 s depending on the
  branch. The test now names the Job's error, but only after the timeout.
- **A cluster from before the non-root chart breaks the files service.** In a cluster first installed before
  #3626, root pods created the files volume's directories. The platform now runs as uid 1000, and kind's
  `local-path` volumes ignore `fsGroup`, so every fileset create fails with HTTP 500. The cause shows in the API log
  as `PermissionError` on `/vol/files/filesets/...`. Recreate the cluster, or give the volume to uid 1000 on the
  node:

  ```bash
  pv="$(kubectl -n nemo-authentik get pvc nemo-helix-core-storage -o jsonpath='{.spec.volumeName}')"
  docker exec nhx-authentik-reuse-control-plane \
      chown -R 1000:1000 "/var/local-path-provisioner/${pv}_nemo-authentik_nemo-helix-core-storage"
  ```
