<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GKE bring-up (Phase 1)

The scaled-evals plugin on a GKE cluster, in its own namespace, on managed
substrate: **GCS** for artifacts, **Cloud Build** for task images, **GAR** for
the registry.

The manifests name no cloud project. Every project-specific value —  bucket,
service accounts, registry paths, image tag —  comes from `local.env`, which is
untracked; `local.env.example` is the template. `apply.sh` substitutes them and
refuses to apply while any is unresolved.

```bash
cd plugins/_temporary-scaled-evals/deploy/k8s
cp local.env.example local.env   # then fill it in
./apply.sh --render      # print substituted manifests, touch nothing
./apply.sh               # deploy
./smoke.sh               # create -> upload -> Cloud Build -> ready, then verify GAR
./eval-smoke.sh          # execute through Platform Jobs and sandbox-k8s
kubectl delete ns nemo-platform-scaled-evals
```

## Blast radius

Everything lands in **`nemo-platform-scaled-evals`**. The name deliberately
avoids a bare `scaled-evals-*` prefix, which in a shared project may already
belong to a standalone deployment, its staging release, and CI's per-MR preview
namespaces — nothing here should be mistakable for those, and `kubectl delete
ns` removes all of it.

Two things live outside the namespace and may be shared with such deployments:

- the GCP service account (`SE_GCP_SERVICE_ACCOUNT`), which gains one additional
  Workload Identity member (purely additive — existing members are untouched);
- the GAR repository holding `SE_APP_IMAGE` and `SE_TASK_IMAGE_REGISTRY`, which
  gains two new image paths.

## GCP prerequisites

Not created by `apply.sh`, because they are project-level and want a deliberate
operator. Run once:

```bash
set -a; . ./local.env; set +a          # PROJECT/SA/BUCKET come from local.env
PROJECT=$SE_GCP_PROJECT
SA=$SE_GCP_SERVICE_ACCOUNT
BUILD_SA=${SE_CLOUD_BUILD_SERVICE_ACCOUNT##*/}   # strip projects/.../serviceAccounts/
BUCKET=$SE_GCS_BUCKET
NS=nemo-platform-scaled-evals

# Note when choosing SE_GCS_BUCKET: GCS caps bucket names at 63 characters. Over
# the cap, the API rejects the create with a misleading "Use of this bucket name
# is restricted", so abbreviate rather than spelling out the namespace.

# 1. Artifact bucket, scoped to this deployment alone.
gcloud storage buckets create "gs://$BUCKET" --project="$PROJECT" --location=us-central1 \
  --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$SA" --role=roles/storage.objectAdmin

# Cloud Build fetches the task pack from gs:// as *itself*, not as the control
# plane, so it needs its own read grant. Without this, finalize fails on source
# fetch rather than on anything scaled-evals logs.
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$BUILD_SA" --role=roles/storage.objectViewer

# 2. Let this namespace's service accounts impersonate the GCP SA.
for KSA in scaled-evals-control-plane scaled-evals-gar-registry-auth; do
  gcloud iam service-accounts add-iam-policy-binding "$SA" --project="$PROJECT" \
    --role=roles/iam.workloadIdentityUser \
    --member="serviceAccount:$PROJECT.svc.id.goog[$NS/$KSA]"
done
```

The SA additionally needs `roles/cloudbuild.builds.editor` at project level and
`roles/iam.serviceAccountTokenCreator` on itself — the latter is what makes GCS
V4 signed URLs possible.

## Push the image

The application image is the same one the compose stack builds, for amd64. It
also carries the Platform Jobs launcher used by task pods. It can go to a new
image path inside an existing GAR repository: push and pull rights
are granted on the repository, not per image path —

- push: your user needs `roles/artifactregistry.writer` on the repo (check with
  `gcloud artifacts repositories get-iam-policy <repo> --location=<region>`).
  `gcloud auth configure-docker` then authenticates docker as you;
- pull: if the node service account holds `roles/artifactregistry.reader` on the
  same repo, nodes pull directly and nothing here needs `imagePullSecrets`;
- task images: Cloud Build pushes as its own service account, which must also be
  a writer on the repo.

Note these are **repository-scoped** grants. Looking only at project-level IAM
will suggest, wrongly, that nobody can push.

```bash
set -a; . ./local.env; set +a
TAG=dev-$(git rev-parse --short HEAD)
REPO=$SE_APP_IMAGE
gcloud auth configure-docker "${REPO%%/*}"    # the registry host

# HARBOR_EXTRA_INDEX_URL is not optional in practice: harbor is on PyPI but
# sandbox-k8s is not published there, so with the default empty value the
# harbor stage fails to resolve it. Point this at an index carrying both.
docker buildx build --platform linux/amd64 \
  -f plugins/_temporary-scaled-evals/deploy/compose/Dockerfile \
  --build-arg HARBOR_EXTRA_INDEX_URL="$SE_HARBOR_INDEX_URL" \
  -t "$REPO:$TAG" --push .          # from the repo root

# then point the stack at it: bump the tag in local.env, not kustomization.yaml,
# which carries a ${SE_APP_IMAGE_TAG} placeholder
sed -i.bak "s|^SE_APP_IMAGE_TAG=.*|SE_APP_IMAGE_TAG=$TAG|" local.env && rm local.env.bak
```

## What differs from the compose stack

| | compose | here |
|---|---|---|
| task image build | in-cluster BuildKit | **Cloud Build** (`BUILDKIT_ENABLED=false`) |
| artifacts | RustFS, S3 + HMAC keys | **GCS**, Workload Identity tokens |
| registry | in-cluster `registry:2`, insecure | **GAR**, credentials refreshed half-hourly |
| Postgres | compose service, named volume | in-namespace pod, 10Gi `standard-rwo` PVC |
| execution process | Platform Jobs subprocess profile | **Platform Jobs Kubernetes Job** profile |

Everything else — the image, the settings, the plugin's own startup migration
and bucket creation — is identical.

Two consequences worth knowing:

- **GAR needs auth even to read a manifest**, and scaled-evals resolves the
  digest of every image it builds. A CronJob mints an access token from the
  metadata server every 30 minutes and writes it into
  `secret/scaled-evals-gar-registry-auth`; `apply.sh` primes it once up front so
  the first build does not wait.
- **Postgres needs a real disk here, not `emptyDir`.** It had one, on the theory
  that startup migrations made a restart survivable. They do not: migrations run
  when the *API* starts, so when the autoscaler deleted the database pod for a
  node scale-down, the schema was gone and nothing rebuilt it — 30 minutes of
  503s and a crash-looping worker. On an autoscaled pool that is a scheduled
  outage, not a restart risk.

## The evaluation runtime

`SANDBOX_K8S_ENABLED=true`, and `./eval-smoke.sh` runs a real evaluation:
Cloud Build produces the `hello-world` task image, Harbor launches a sandbox
pod through the agent-sandbox controller, the `oracle` agent applies the task's
reference solution, and the verifier scores it 1.0. No model credentials are
involved — the oracle path makes no inference calls.

Three things make that work, and each is a place this diverges from the
standalone chart:

- **Harbor lives in the image.** It is not a dependency of the plugin, so the
  Dockerfile installs `harbor` plus `sandbox-k8s[harbor]` into `/opt/harbor` and
  overlays the same patched adapter the standalone image uses. Every catalog
  release marked `selectable` is built; `0.13.2` remains the default.
- **Sandboxes share the control-plane namespace.** `sandbox-rbac.yaml` is
  therefore all namespace-scoped. The plugin owns the baseline Sandbox CR
  create/read/delete verbs directly rather than depending on optional
  user-facing ClusterRoles. There is no isolation between the control plane
  and evaluated code — fine for a lab namespace, not for multi-tenancy.
- **The CRDs and controller are a cluster prerequisite.** Nothing here installs
  them. The cluster must already run `agent-sandbox-controller` (conventionally
  in `agent-sandbox-system`) with `sandboxes.agents.x-k8s.io` registered; on a
  cluster without them, every evaluation fails at launch.

Multi-service tasks must declare their service images as
`environment.kwargs.sidecars` in the Harbor profile. The adapter restores a
packed Compose declaration for Harbor validation, resolves service names to
pod loopback aliases, waits for configured ports, and routes collect hooks and
artifact downloads to the named sidecar. Separate verifier pods deliberately
omit those sidecars so they can restore and start their own service state.

Two compatibility controls are opt-in because they are not neutral:

- Set `environment.kwargs.enable_ipv6_loopback: true` only for tasks that bind
  `::1`; it adds deterministic pod sysctls for IPv6 loopback.
- A task that must move temporary files off `/tmp` should set
  `environment.kwargs.persistent_env.TMPDIR` and create that directory in its
  own task-specific `setup_command`. Do not put that redirect in a shared
  profile: tasks that explicitly manage `/tmp` permissions can break when their
  scratch directory moves.

## Viewing a run

`kubectl port-forward -n nemo-platform-scaled-evals deploy/scaled-evals-api 8080:8080`,
then query the API under `/apis/scaled-evals/v1` — `evaluations/<id>` for status,
`.../artifacts` to list outputs, `.../archive` for a signed tarball URL, and
`.../logs` or `.../events` to follow a run. The `scaled-evals` CLI works against
the same port-forward.

## Deliberate omissions

Present in the standalone chart, absent here, all out of Phase 1: the external
identity provider and its auth-router/agent pair, ingress, resource quotas, the
image-signing admission canary, and the hosted vault/externalsecret. Also skipped: the chart's metadata-egress NetworkPolicies,
which allow-list the metadata server only — a Phase-1 stack that has to reach
the Kubernetes API from the same pod is better off with no policy than a subtly
wrong one.

Secrets are generated by `apply.sh` on first run and never committed. Re-running
leaves them alone, so the Fernet key stays stable and stored credentials remain
decryptable.
