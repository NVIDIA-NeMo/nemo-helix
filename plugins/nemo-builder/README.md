<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Builder Plugin

In-cluster container image builds for the NeMo Helix. A caller submits Dockerfiles whose build
context is already in a Files fileset. The plugin builds them with kaniko in a locked-down
namespace, pushes them to a registry, signs them with cosign, and records the digest the registry
serves as a `ContainerImage` entity that consumers can pin.

**Status: proof of concept.** It runs end to end on minikube. It is not a uv workspace member, is
not in the Helm chart, and has the limitations listed under [Constraints](#constraints).

## Surfaces

- **REST:** `POST /apis/builder/v2/workspaces/{workspace}/builds` submits a build;
  `GET .../container-images` and `GET .../container-images/{name}` read the results.
- **Entity:** `ContainerImage` (`entity_type="container_image"`), one row per image.
- **Controller:** `BuilderController`, registered under `nemo.controllers`. It moves each row from
  `pending` to `ready` or `failed`.
- **Step program:** `nhx-build {fetch|supervise|push}`, which the build job's steps run. It ships
  in the `nhx-build` image (`docker/Dockerfile`).

There is no CLI or SDK accessor yet; use the REST API.

## How it works

A submitted build set becomes one Jobs job with three steps, each under its own ServiceAccount.
The caller's Dockerfile runs in a fourth pod, the sandbox, which is not a job step and holds
nothing:

| | Runs as | Holds | Work volume | Runs caller code |
|---|---|---|---|---|
| `fetch` | `nhx-build-fetch` | a Files client, acting as the submitter | yes | no |
| `build` (runs `nhx-build supervise`) | `nhx-build-control` | permission to manage pods in `nhx-builds` | no | no |
| sandbox (kaniko) | no ServiceAccount token | nothing | its own context read-only, and one output directory per image | yes |
| `push` | `nhx-build-push` | the registry credential and the signing key | yes | no |

1. **Submit.** The request is resolved against the deployment's config into a plan: a job name,
   plus a row name, destination and system tag for each image. The system tag is a tag the build
   pushes alongside the caller's, and the one the reconciler looks up; its form is under
   [Names](#get-container-images-and-get-container-imagesname). Every check that doesn't need to
   ask another service runs here, or in the request schema before it. One `ContainerImage` row per
   image is created `pending`, then the job. Each row records a digest of the request, so
   submitting the same request again returns the same rows, and a different request under the
   same `name` and `revision` is refused.
2. **`fetch`** clears anything an earlier run of the job left behind, then copies each fileset, or
   just its `context_path`, onto the work volume as the submitting user. A request naming a
   fileset the submitter can't read fails here.
3. **`build`** creates one sandbox pod per distinct fileset and `context_path`. The sandbox runs
   kaniko once per image with `--no-push`, writing an OCI layout to that image's output directory,
   which it empties first. The sandbox has:
   - no ServiceAccount token, environment variables or secrets
   - a security context the `baseline` Pod Security Standard admits: root, but with only five
     capabilities, `RuntimeDefault` seccomp and no privilege escalation
   - public internet access only: a NetworkPolicy blocks private, link-local and cluster
     addresses, and DNS goes to public resolvers
   - a deadline the kubelet enforces, so it ends even if `supervise` is killed first

   An admission policy holds `nhx-build-control` to creating pods of exactly this shape.
   `supervise` mounts no volume itself; it reads each image's pass or fail from the sandbox's log,
   which only affects its own exit code and logs.
4. **`push`** treats each layout as untrusted. It refuses symlinks, special files, blobs that
   don't hash to their names, and anything but a single manifest. It then pushes each image under
   the caller's tag and the system tag with crane, and signs it by digest with cosign.
5. **The reconciler** reads each image's digest from the registry and marks its row `ready`.
   Nothing that ran the Dockerfile reports an image's digest.

One failed Dockerfile doesn't stop the rest of the set: the other images still build and publish,
and only the failed image's row ends `failed`. If every image in a set fails, `push` never runs.

Every step finds files on the work volume through one `WorkLayout` (in `steps.py`). Step configs
name filesets and images, never paths.

## API

All routes are under `/apis/builder/v2/workspaces/{workspace}`.

### `POST /builds`

```json
{
  "name": "hello",
  "revision": 1,
  "build_specs": [
    {
      "name": "main",
      "source": {"fileset": "hello-context", "context_path": null},
      "dockerfile": "Dockerfile",
      "platform": "linux/amd64",
      "output": {"repository": "demo/hello", "tag": "v1"}
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `name` | What is being built. Reused across rebuilds. Lowercase letters, digits and single hyphens, starting with a letter, and short enough that every name built from it fits in 63 characters. |
| `revision` | The caller's own version number for `name`, starting at 1. Use a new revision for every build. |
| `build_specs` | Up to 100 specs. |
| `build_specs[].name` | A label for the image, unique within the set. |
| `source.fileset` | The fileset holding the build context: `name` in this workspace, or `workspace/name`. |
| `source.context_path` | Optional subdirectory of the fileset to use as the context. Relative, with no `..`. Omit it to use the whole fileset. |
| `dockerfile` | Path relative to the context. Absolute paths and `..` are rejected. Defaults to `Dockerfile`. |
| `platform` | Passed to kaniko. Defaults to `linux/amd64`. |
| `output.repository`, `output.tag` | Where the image is published, within your workspace's part of the registry. Tags containing `--` or starting with `sha256-` are reserved for system tags and signatures. |

Every request model rejects fields it doesn't know, so a misspelt field is a `422` rather than
silently ignored.

Every image goes to the deployment's one registry, at
`<registry>/<repository_prefix>/<workspace>/<output.repository>:<output.tag>`. A caller can't name
a registry: whatever runs the image has to pull it, and one registry means one pull credential for
every workload that does. The workspace in the path keeps workspaces out of each other's
repositories. To publish somewhere else, copy the image out afterwards, for example with
`crane copy`.

A successful submit returns `201` with the job name and the `pending` rows. Poll the rows, not
the job: images in a set finish independently.

| Status | When |
|---|---|
| `201` | Rows created and job submitted, or the same request was submitted before and these are its rows and job. |
| `400` | Two specs would publish the same `repository:tag`, or the workspace name can't be a repository path component. |
| `403` | The caller may submit builds but may not create jobs in the workspace. |
| `409` | This deployment can't build: builds are switched off (`execution_enabled: false`), or `registry`, `push_credential_secret` or `signing_key` is unset. Or this `name` and `revision` were already submitted with a different request. |
| `422` | The body failed validation: for example a path outside the context or fileset, a set name that doesn't fit, more than 100 specs, duplicate spec names, a `revision` below 1, a repository or tag that isn't valid or is reserved, or an unknown field such as `output.registry`. |
| `502` | Another platform service refused the build. |
| `500` | Anything else. |

### `GET /container-images` and `GET /container-images/{name}`

`GET /container-images` accepts `page` and `page_size` (at most 100). `GET
/container-images/{name}` returns `404` for a name that doesn't exist. Each row is a
`ContainerImage`:

| Field | Meaning |
|---|---|
| `status` | `pending`, `ready` or `failed`. Only the reconciler changes it, and `ready` and `failed` are final. |
| `status_detail` | Why a row failed. |
| `registry`, `repository` | Where the image lives: the deployment's registry, and `<repository_prefix>/<workspace>/<output.repository>`. |
| `digest` | What the registry serves for this image, set once when the row becomes `ready`. Pin `<registry>/<repository>@<digest>`. |
| `manifest_digest` | The per-platform manifest when the tag resolves to an index; otherwise equal to `digest`. |
| `tag` | The system tag the reconciler resolved. |
| `signature` | Set once the row is `ready`, meaning a cosign signature was found. It is not checked against a key, so `verified_against` is always `null`. |
| `platform` | The platform the image was built for. |
| `provenance.built_by` | The build set, revision, job and system tag that produced the image, and `request_digest`, the digest of the request that did. |
| `source_digest` | Reserved; always `null` for now. |

Names are derived from the request:

| | Pattern | Example |
|---|---|---|
| Job | `<set>-<revision>` | `hello-1` |
| Row | `<set>-<revision>-<index>` | `hello-1-0` |
| System tag | `<workspace>--<set>-<revision>-<index>` | `default--hello-1-0` |

## The image reconciler

`BuilderController` is the only writer of a row's `status` and registry-derived fields. Every
`reconcile_interval_seconds` (default 10) it lists `pending` rows across all workspaces, and for
each one:

1. Waits until the row's job has finished. If the Jobs service can't be read, it tries again on
   the next cycle. A job that doesn't exist counts as failed once the row is five minutes old;
   before that, the job may simply not have been created yet, since rows are written first.
2. Looks up the row's system tag in the registry, over the OCI Distribution API, and computes the
   digest from the manifest bytes it's served. A `Docker-Content-Digest` header that disagrees is
   refused. If the tag resolves to an index, `manifest_digest` is the first manifest in it.
3. Checks that a cosign signature exists at the tag `sha256-<hex>.sig`.
4. Marks the row `ready` with its `digest`, `manifest_digest`, `tag` and `signature`. These are
   written once and never change.

It asks the registry even when the job failed, because a job that ends in error may still have
published some of its images. A row fails when:

- the job failed, and the image isn't in the registry, or is there unsigned
- the job succeeded, but after 10 passes the tag still doesn't resolve, or the image still has no
  signature, because an unsigned build doesn't count as a success

A registry that returns errors, or can't be reached, never fails a row: an outage says nothing
about the image, and `failed` is final. The row stays `pending` and the error is logged each
cycle. Each cycle reads every `pending` row, oldest first.

The reconciler authenticates with `registry_username` and `registry_password`, which only need
read access, and uses HTTPS unless `registry_insecure` is set. It answers whichever challenge the
registry sends: Bearer, by exchanging the credential for a token (GAR, Docker Hub, Harbor), or
Basic, by sending it directly (`distribution` with htpasswd). Bearer is preferred when a registry
offers both, and the token endpoint must be HTTPS. Supply the credential from a Kubernetes Secret
as `NEMO_BUILDER_REGISTRY_USERNAME` and `NEMO_BUILDER_REGISTRY_PASSWORD` rather than in the config
file; `deploy/local/platform.yaml` shows how. It is read once at startup, so a short-lived
credential, such as a GAR access token, needs the controller restarted before it expires.

Run one replica. A second can't corrupt a row, because every update is conditional on the version
it read, but it doubles the registry traffic and keeps its own attempt counts.

## Configuration

Operator settings live under `builder:` in the platform config, or in `NEMO_BUILDER_*` environment
variables. Callers can't set any of them.

```yaml
builder:
  namespace: nhx-builds
  work_pvc: nhx-build-work
  sandbox_image: gcr.io/kaniko-project/executor:debug
  registry: us-central1-docker.pkg.dev           # a host only, never host/path
  repository_prefix: my-project/my-repo          # images land at <prefix>/<workspace>/<repository>
  push_credential_secret: registry-push-credential  # a Kubernetes Secret in `namespace`
  signing_key: k8s://nhx-builds/cosign-key       # a cosign key reference
  reconcile_interval_seconds: 10
  execution_enabled: true                        # false refuses new builds; reads keep working
```

Three settings have no safe default: `registry`, `push_credential_secret` and `signing_key`. If any
is missing, submits fail with a `409` rather than as a build that dies in a pod.

The push credential is the operator's: a `kubernetes.io/dockerconfigjson` Secret in the build
namespace, with write access to `registry` under `repository_prefix`. The `push` step reads it
through the Kubernetes API as `nhx-build-push`, the only ServiceAccount that can, the same way it
reads the signing key. It is not a Secrets-service entry, because Jobs resolves those as the
submitter, and every submitter would then be able to read it. The reconciler's read-only
credential (`registry_username`, `registry_password`) is separate; see
[The image reconciler](#the-image-reconciler).

**Set `repository_prefix` to a path dedicated to builds.** Left empty, each workspace name is a
top-level namespace in the registry, and the push credential can write all of them: a workspace
named after the namespace your platform images live in would publish over them.

The platform also needs three Jobs execution profiles, one per step, named by `fetch_profile`,
`control_profile` and `push_profile` (`build-fetch`, `build-control` and `build-push` by default).
Each is a `cpu` profile on the `kubernetes_job` backend, and they must agree with the builder
config and the manifests:

- `namespace` is the builder's `namespace`, and `storage.pvc_name` is its `work_pvc`
- `service_account_name` is `nhx-build-fetch`, `nhx-build-control` or `nhx-build-push`
- `default_task_image` is the `nhx-build` image, and `launcher_image` an image that ships
  `/tools/jobs-launcher`, as the platform image does
- `node_selector` is the build node's label

The manifests in `deploy/` name the namespace, the ServiceAccounts, the work volume and the two
Secrets (`cosign-key` and `registry-push-credential`) literally. Rename any of them in the config
and the manifests must change with it.

`config/platform-config.minikube.yaml` is the quickstart's config. It works, but it is local-only
in the ways its header lists, auth being off among them; don't start a shared deployment from it.

The remaining settings are `sandbox_cpu`, `sandbox_memory`, `sandbox_dns_nameservers`,
`runtime_class` (for example `gvisor`), `registry_mirror`, `node_selector`, `registry_insecure`
and `signature_storage`. Each is described on `BuilderConfig` in `config.py`.

## Cluster setup

`deploy/` holds the build namespace and everything that constrains pods in it. Apply the whole
directory with `kubectl apply -f`:

| File | What it does |
|---|---|
| `00-namespace.yaml` | Creates `nhx-builds`, enforcing the `baseline` Pod Security Standard |
| `10-serviceaccounts.yaml` | The three step identities |
| `20-rbac.yaml` | Pod management for `nhx-build-control` only, and read access to the signing key and push credential for `nhx-build-push` only |
| `25-sandbox-admission.yaml` | Holds `nhx-build-control` to creating pods of the sandbox's shape; without it, pod creation there reaches both Secrets |
| `30-networkpolicy.yaml` | Sandbox egress: the public internet, minus private, link-local and cluster ranges |
| `40-work-volume.yaml` | The shared work volume |
| `negative-control.sh` | Checks that the namespace still refuses a pod with BuildKit's privileges |
| `sandbox-admission-check.sh` | Checks that `nhx-build-control` can create the sandbox's shape and nothing else |
| `sandbox-egress-probe.sh` | Checks, with real connections, what a sandbox can and can't reach |

The cluster needs:

- **Kubernetes 1.30 or later**, for the ValidatingAdmissionPolicy.

- **A CNI that enforces NetworkPolicy**, such as Calico. On one that doesn't, the sandbox's egress
  is unrestricted, and nothing reports it.
- **One labelled build node.** The work volume is `ReadWriteOnce`, so every build pod runs on the
  node labelled `nhx.nvidia.com/build-node=true`.

`30-networkpolicy.yaml` blocks the RFC1918, link-local and carrier-grade NAT ranges, the other
reserved ranges clusters use as private, and Azure's WireServer. Some private addresses are known
only to the cluster: Pod or Service ranges outside those blocks, GKE's privately used public
ranges, and the public addresses of nodes. Add those to its `except` list.

## Quickstart (minikube)

This runs the platform inside minikube, builds one image, and checks the result against the
registry. Run the commands from the repository root, in one shell: later steps use variables
earlier ones set.

You need Docker with the default buildx driver, minikube, `kubectl`, `jq`, and the `nemo` CLI
(`make bootstrap`; see [SETUP.md](../../SETUP.md)). The cluster below uses 3 CPUs and 5.5 GB of
memory. Sandboxes resolve names with `8.8.8.8` and `1.1.1.1`, so the network must allow DNS to
them; some corporate networks and VPNs don't.

**1. Cluster and build namespace.**

```bash
minikube start --driver=docker --cni=calico --cpus=3 --memory=5500
kubectl label node minikube nhx.nvidia.com/build-node=true
kubectl -n kube-system wait --for=condition=Ready pod -l k8s-app=calico-node --timeout=300s

kubectl apply -f plugins/nemo-builder/deploy/
plugins/nemo-builder/deploy/negative-control.sh         # must print PASS
plugins/nemo-builder/deploy/sandbox-admission-check.sh  # must print PASS
plugins/nemo-builder/deploy/sandbox-egress-probe.sh     # must print PASS
```

**2. Signing key and push credential.** The key pair goes in a temporary directory, not the
repository, and the private key has no password: it's for this cluster only. Keep
`$KEYS/cosign.pub` if you want to verify signatures later.

```bash
KEYS=$(mktemp -d) && chmod 777 "$KEYS"
docker run --rm -v "$KEYS:/work" -w /work -e COSIGN_PASSWORD= \
  ghcr.io/sigstore/cosign/cosign:v2.5.3 generate-key-pair
kubectl -n nhx-builds create secret generic cosign-key --from-file=cosign.key="$KEYS/cosign.key"
```

The local registry needs no login, so the push credential is an empty Docker config. Against a
real registry, use `kubectl create secret docker-registry` with `--docker-server` set to the
config's `registry`.

```bash
kubectl -n nhx-builds create secret generic registry-push-credential \
  --type=kubernetes.io/dockerconfigjson --from-literal=.dockerconfigjson='{"auths":{}}'
```

**3. Images.** Build the platform image with this plugin added, and the image the build steps run
in. Use `linux/amd64` instead of `linux/arm64` on an x86 machine. `Dockerfile.platform` starts
`FROM` the local `my-registry/nhx-api:local`, which only the default `docker` buildx driver can
see; with a `docker-container` builder selected, run `docker buildx use default` first.

```bash
# nhx-api, with the Studio UI stubbed out; the builder doesn't use it
mkdir -p /tmp/empty-studio/artifacts && touch /tmp/empty-studio/artifacts/.keep
docker buildx bake -f docker-bake.hcl nhx-api-docker --load \
  --allow=fs.read=/tmp/empty-studio --set '*.platform=linux/arm64' \
  --set nhx-api-docker.contexts.nhx-studio-ui=/tmp/empty-studio

docker buildx build --platform linux/arm64 -f plugins/nemo-builder/docker/Dockerfile.platform \
  --build-arg NHX_API_IMAGE=my-registry/nhx-api:local -t nhx-api-builder:local --load .
docker buildx build --platform linux/arm64 -f plugins/nemo-builder/docker/Dockerfile \
  -t nhx-build:local --load .

minikube image load nhx-api-builder:local
minikube image load nhx-build:local
```

`minikube image load` doesn't reliably replace an image already loaded under the same tag. When
you rebuild, use a new tag, and update `deploy/local/platform.yaml` and the config to match.

**4. Platform.** `deploy/local/` runs the platform services the builder needs in one pod with
SQLite. It also runs an anonymous registry at `registry.nhx-builds.svc.cluster.local:5000`, which
is the config's `registry`.

```bash
kubectl apply -f plugins/nemo-builder/deploy/local/
kubectl -n nhx-platform create configmap nemo-helix-config \
  --from-file=config.yaml=plugins/nemo-builder/config/platform-config.minikube.yaml
kubectl -n nhx-platform rollout status deploy/nemo-helix
kubectl -n nhx-platform port-forward svc/nemo-helix 8080:8080
```

In another terminal:

```bash
export NHX_BASE_URL=http://localhost:8080
```

**5. Build context.**

```bash
CTX=$(mktemp -d) && cat > "$CTX/Dockerfile" <<'EOF'
FROM docker.io/library/python:3.13-slim
RUN pip install --no-cache-dir six==1.16.0
RUN echo "hello from nemo-builder" > /hello.txt
EOF
nemo files filesets create hello-context --workspace default
nemo files upload "$CTX/" hello-context --workspace default
```

**6. Build.** The spec's `platform` defaults to `linux/amd64`. On an arm64 machine, its `RUN`
steps work only if the node can run amd64 binaries, as Docker Desktop's emulation does; otherwise
add `"platform": "linux/arm64"` to the spec.

```bash
curl -s -X POST "$NHX_BASE_URL/apis/builder/v2/workspaces/default/builds" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "hello",
    "revision": 1,
    "build_specs": [
      {
        "name": "main",
        "source": {"fileset": "hello-context"},
        "output": {"repository": "demo/hello", "tag": "v1"}
      }
    ]
  }' | jq '{job, images: [.images[] | {name, status}]}'
```

`kubectl -n nhx-builds get pods -w` shows the `fetch` pod, then `build` with its sandbox, then
`push`. Poll the image until it leaves `pending`, which takes a minute or two:

```bash
curl -s "$NHX_BASE_URL/apis/builder/v2/workspaces/default/container-images/hello-1-0" \
  | jq '{status, digest, tag, status_detail}'
```

```json
{
  "status": "ready",
  "digest": "sha256:4c9b935d53ba5b129d7d8d68a5da8034103817d319623a6591650dd2b306c49d",
  "tag": "default--hello-1-0",
  "status_detail": null
}
```

**7. Check the result.** The registry should serve the recorded digest, and the image should
contain the Dockerfile's output:

```bash
kubectl -n nhx-builds run check --rm -i --restart=Never --image=nhx-build:local --command -- sh -c '
  sleep 2
  crane digest --insecure registry.nhx-builds.svc.cluster.local:5000/default/demo/hello:v1
  crane export --insecure registry.nhx-builds.svc.cluster.local:5000/default/demo/hello:v1 - | tar -xO hello.txt'
```

To build again, submit with `"revision": 2`.

## Development

The plugin isn't a uv workspace member. The root `pytest.ini` and `ty` config put its `src/` on
the path instead, so the repository's own unit-test run collects these tests, and nothing needs
installing:

```bash
uv run --frozen pytest plugins/nemo-builder/tests
uv run --frozen ruff check plugins/nemo-builder && uv run --frozen ty check plugins/nemo-builder
```

The tests need no cluster, registry or running platform.

| File | What it holds |
|---|---|
| `service.py` | The REST routes |
| `schema.py` | The request models: `BuildSet`, `BuildSpec`, `FileSetSource`, `BuildOutput` |
| `plan.py` | `BuildPlan`: a request resolved against the deployment, with every submit-time check |
| `compile.py` | Turns a plan into the three-step `HelixJobSpec`, as a pure function |
| `submit.py` | The submit sequence: plan, then rows, then the job |
| `steps.py` | What each step's config contains, and `WorkLayout` |
| `entities.py` | `ContainerImage` |
| `controller.py` | The reconciler |
| `registry.py` | The OCI Distribution client the reconciler uses |
| `identity.py` | Image reference parsing and the system tag |
| `config.py` | `BuilderConfig` |
| `run/` | The step programs: `fetch.py`, `supervise.py`, `push.py` |
| `docker/` | The `nhx-build` step image, and the platform image with this plugin added |
| `deploy/` | The build namespace; `deploy/local/` adds the minikube platform and registry |
| `config/` | The platform config for the minikube quickstart |

## Constraints

- **Kubernetes only, with the platform inside the cluster.** Build pods call back to Files,
  Secrets and Jobs, so a control plane outside the cluster can't run builds.
- **One build node.** The work volume is `ReadWriteOnce`. A `ReadWriteMany` volume would lift this.
- **One registry per deployment.** Every image is published to `registry`, under the submitting
  workspace's path. Publishing anywhere else means copying the image out afterwards.
- **Each new request needs a new `revision`.** The same request again returns its rows; a
  different one under a used `name` and `revision` returns `409`.
- **Signatures are checked for presence, not validity.** Admission-time verification, on the
  cluster that runs the image, is what should check the key.
- **Signatures are stored as cosign's legacy `.sig` tag.** `signature_storage: referrers` is
  refused, because `push` runs cosign v2, which writes the tag, and the reconciler looks only there.
- **Base images come from public registries.** `registry_mirror` is wired but unset, so nothing
  bounds what a Dockerfile can pull `FROM`. Only a mirror the sandbox can reach over the public
  internet works without editing the NetworkPolicy, which blocks private addresses.
- **Upstream kaniko was archived** on 2025-06-03. The default `sandbox_image` still points at it.
- **One reconciler replica**, checking `pending` rows one at a time.
