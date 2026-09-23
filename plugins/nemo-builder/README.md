<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Builder Plugin

In-cluster container image builds for the NeMo Platform. A caller submits Dockerfiles whose build
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
- **Step program:** `nmp-build {fetch|supervise|push}`, which the build job's steps run. It ships
  in the `nmp-build` image (`docker/Dockerfile`).

There is no CLI or SDK accessor yet; use the REST API.

## How it works

A submitted build set becomes one Jobs job with three steps, each under its own ServiceAccount.
The caller's Dockerfile runs in a fourth pod, the sandbox, which is not a job step and holds
nothing:

| | Runs as | Holds | Work volume | Runs caller code |
|---|---|---|---|---|
| `fetch` | `nmp-build-fetch` | a Files client, acting as the submitter | yes | no |
| `build` (runs `nmp-build supervise`) | `nmp-build-control` | permission to manage pods in `nmp-builds` | no | no |
| sandbox (kaniko) | no ServiceAccount token | nothing | its own context read-only, and the output directory | yes |
| `push` | `nmp-build-push` | the registry credential and the signing key | yes | no |

1. **Submit.** The request is resolved against the deployment's config into a plan: a job name,
   plus a row name, destination and system tag for each image. Every check that can reject the
   request runs here. One `ContainerImage` row per image is created `pending`, then the job.
2. **`fetch`** copies each fileset, or just its `context_path`, onto the work volume as the
   submitting user. A request naming a fileset the submitter can't read fails here.
3. **`build`** creates one sandbox pod per distinct fileset and `context_path`. The sandbox runs
   kaniko once per image with `--no-push`, writing an OCI layout to the output directory. The
   sandbox has:
   - no ServiceAccount token, environment variables or secrets
   - a security context the `baseline` Pod Security Standard admits: root, but with only five
     capabilities, `RuntimeDefault` seccomp and no privilege escalation
   - public internet access only: a NetworkPolicy blocks private, link-local and cluster
     addresses, and DNS goes to public resolvers

   `supervise` mounts no volume itself; it learns each image's result from the sandbox's log.
4. **`push`** treats each layout as untrusted. It refuses symlinks, blobs that don't hash to their
   names, and anything but a single manifest. It then pushes each image under the caller's tag and
   the system tag with crane, and signs it by digest with cosign.
5. **The reconciler** reads each image's digest from the registry and marks its row `ready`.
   Nothing that ran the Dockerfile reports its own result.

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
      "output": {"registry": null, "repository": "demo/hello", "tag": "v1"}
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `name` | What is being built. Reused across rebuilds. |
| `revision` | The caller's own version number for `name`, starting at 1. Use a new revision for every build. |
| `build_specs[].name` | A label for the image, unique within the set. |
| `source.fileset` | The fileset holding the build context: `name` in this workspace, or `workspace/name`. |
| `source.context_path` | Optional subdirectory of the fileset to use as the context. Omit it to use the whole fileset. |
| `dockerfile` | Path relative to the context. Absolute paths and `..` are rejected. Defaults to `Dockerfile`. |
| `platform` | Passed to kaniko. Defaults to `linux/amd64`. |
| `output.registry` | Registry host. Omit it to use the deployment's `default_registry`. |
| `output.repository`, `output.tag` | Where the image is published. With the default registry, the deployment's `repository_prefix` is prepended. |

A successful submit returns `201` with the job name and the `pending` rows. Poll the rows, not
the job: images in a set finish independently.

| Status | When |
|---|---|
| `201` | Rows created and job submitted. |
| `400` | Two specs would publish the same `registry/repository:tag`. |
| `409` | This deployment can't build: builds are switched off (`execution_enabled: false`), `push_secret` or `signing_key` is unset, or an image has no registry. |
| `422` | The body failed validation, for example a Dockerfile path outside the context, duplicate spec names, or a `revision` below 1. |
| `500` | Anything else, including reusing a `name` and `revision` that were already submitted. |

### `GET /container-images` and `GET /container-images/{name}`

`GET /container-images` accepts `page` and `page_size`. Each row is a `ContainerImage`:

| Field | Meaning |
|---|---|
| `status` | `pending`, `ready` or `failed`. Only the reconciler changes it, and `ready` and `failed` are final. |
| `status_detail` | Why a row failed. |
| `registry`, `repository` | Where the image lives. |
| `digest` | What the registry serves for this image, set once when the row becomes `ready`. Pin `<registry>/<repository>@<digest>`. |
| `manifest_digest` | The per-platform manifest when the tag resolves to an index; otherwise equal to `digest`. |
| `tag` | The system tag the reconciler resolved. |
| `signature` | Set once the row is `ready`, meaning a cosign signature was found. It is not checked against a key, so `verified_against` is always `null`. |
| `provenance.built_by` | The build set, revision, job and system tag that produced the image. |
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
   the next cycle. A job that no longer exists counts as failed.
2. Looks up the row's system tag in the registry, over the OCI Distribution API, and takes the
   digest the registry reports (`Docker-Content-Digest`). If the tag resolves to an index,
   `manifest_digest` is the first manifest in it.
3. Checks that a cosign signature exists at the tag `sha256-<hex>.sig`.
4. Marks the row `ready` with its `digest`, `manifest_digest`, `tag` and `signature`. These are
   written once and never change.

It asks the registry even when the job failed, because a job that ends in error may still have
published some of its images. A row fails when:

- the job failed and the image isn't in the registry
- the job succeeded, but the tag still doesn't resolve, or the registry keeps returning errors,
  after 10 attempts
- the image is there but has no signature, because an unsigned build doesn't count as a success

The reconciler authenticates with `registry_username` and `registry_password`, which only need
read access, and uses HTTPS unless `registry_insecure` is set. Run exactly one replica: there is
no leader election.

## Configuration

Operator settings live under `builder:` in the platform config, or in `NEMO_BUILDER_*` environment
variables. Callers can't set any of them.

```yaml
builder:
  namespace: nmp-builds
  work_pvc: nmp-build-work
  sandbox_image: gcr.io/kaniko-project/executor:debug
  default_registry: us-central1-docker.pkg.dev   # a host only, never host/path
  repository_prefix: my-project/my-repo          # prepended to output.repository
  push_secret: registry-credential               # a Secrets entry, looked up in the submitter's workspace
  signing_key: k8s://nmp-builds/cosign-key       # a cosign key reference
  registry_username: oauth2accesstoken           # read-only, for the reconciler
  registry_password: <token>
  reconcile_interval_seconds: 10
  execution_enabled: true                        # false refuses new builds; reads keep working
```

Three settings have no safe default: `push_secret`, `signing_key`, and a registry for every image
(`default_registry`, or `output.registry` on each spec). If any is missing, submits fail with a
`409` rather than as a build that dies in a pod. The push credential is either a Docker config
JSON or `user:password`. A `user:password` credential is only ever sent to `default_registry`.

The platform also needs three Jobs execution profiles, one per step: `build-fetch`,
`build-control` and `build-push`, each naming its ServiceAccount. `config/platform-config.minikube.yaml`
is a complete, working config. `config/platform-config.example.yaml` shows the settings for GKE with
Artifact Registry.

The remaining settings are `sandbox_cpu`, `sandbox_memory`, `sandbox_dns_nameservers`,
`runtime_class` (for example `gvisor`), `registry_mirror`, `node_selector`, `registry_insecure`
and `signature_storage`. Each is described on `BuilderConfig` in `config.py`.

## Cluster setup

`deploy/` holds the build namespace and everything that constrains pods in it. Apply the whole
directory with `kubectl apply -f`:

| File | What it does |
|---|---|
| `00-namespace.yaml` | Creates `nmp-builds`, enforcing the `baseline` Pod Security Standard |
| `10-serviceaccounts.yaml` | The three step identities |
| `20-rbac.yaml` | Pod management for `nmp-build-control` only, and read access to the signing key for `nmp-build-push` only |
| `30-networkpolicy.yaml` | Sandbox egress: the public internet, minus private, link-local and cluster ranges |
| `40-work-volume.yaml` | The shared work volume |
| `negative-control.sh` | Checks that the namespace still refuses a pod with BuildKit's privileges |
| `sandbox-egress-probe.sh` | Checks, with real connections, what a sandbox can and can't reach |

The cluster needs:

- **A CNI that enforces NetworkPolicy**, such as Calico. On one that doesn't, the sandbox's egress
  is unrestricted, and nothing reports it.
- **One labelled build node.** The work volume is `ReadWriteOnce`, so every build pod runs on the
  node labelled `nmp.nvidia.com/build-node=true`.

`deploy/README.md` records what was measured on GKE.

## Quickstart (minikube)

This runs the platform inside minikube, builds one image, and checks the result against the
registry. Run the commands from the repository root. The cluster below uses 3 CPUs and 5.5 GB
of memory.

**1. Cluster and build namespace.**

```bash
minikube start --driver=docker --cni=calico --cpus=3 --memory=5500
kubectl label node minikube nmp.nvidia.com/build-node=true

kubectl apply -f plugins/nemo-builder/deploy/
plugins/nemo-builder/deploy/negative-control.sh       # must print PASS
plugins/nemo-builder/deploy/sandbox-egress-probe.sh   # must print PASS
```

**2. Signing key.** Keep `keys/cosign.pub` if you want to verify signatures later.

```bash
mkdir -p keys && chmod 777 keys
docker run --rm -v "$PWD/keys:/work" -w /work -e COSIGN_PASSWORD= \
  ghcr.io/sigstore/cosign/cosign:v2.5.3 generate-key-pair
kubectl -n nmp-builds create secret generic cosign-key --from-file=cosign.key=keys/cosign.key
```

**3. Images.** Build the platform image with this plugin added, and the image the build steps run
in. Use `linux/amd64` instead of `linux/arm64` on an x86 machine.

```bash
# nmp-api, with the Studio UI stubbed out; the builder doesn't use it
mkdir -p /tmp/empty-studio/artifacts && touch /tmp/empty-studio/artifacts/.keep
docker buildx bake -f docker-bake.hcl nmp-api-docker --load \
  --allow=fs.read=/tmp/empty-studio --set '*.platform=linux/arm64' \
  --set nmp-api-docker.contexts.nmp-studio-ui=/tmp/empty-studio

docker buildx build --platform linux/arm64 -f plugins/nemo-builder/docker/Dockerfile.platform \
  --build-arg NMP_API_IMAGE=my-registry/nmp-api:local -t nmp-api-builder:local --load .
docker buildx build --platform linux/arm64 -f plugins/nemo-builder/docker/Dockerfile \
  -t nmp-build:local --load .

minikube image load nmp-api-builder:local
minikube image load nmp-build:local
```

`minikube image load` doesn't reliably replace an image already loaded under the same tag. When
you rebuild, use a new tag, and update `deploy/local/platform.yaml` and the config to match.

**4. Platform.** `deploy/local/` runs the platform services the builder needs in one pod with
SQLite. It also runs an anonymous registry at `registry.nmp-builds.svc.cluster.local:5000`, which
is the config's `default_registry`.

```bash
kubectl apply -f plugins/nemo-builder/deploy/local/
kubectl -n nmp-platform create configmap nemo-platform-config \
  --from-file=config.yaml=plugins/nemo-builder/config/platform-config.minikube.yaml
kubectl -n nmp-platform rollout status deploy/nemo-platform
kubectl -n nmp-platform port-forward svc/nemo-platform 8080:8080
```

In another terminal:

```bash
export NMP_BASE_URL=http://localhost:8080
```

**5. Push credential and build context.** The minikube config names its push secret
`local-registry`. The local registry needs no login, so an empty Docker config works.

```bash
nemo secrets create local-registry --value '{"auths":{}}' --workspace default

mkdir -p hello && cat > hello/Dockerfile <<'EOF'
FROM docker.io/library/python:3.13-slim
RUN pip install --no-cache-dir six==1.16.0
RUN echo "hello from nemo-builder" > /hello.txt
EOF
nemo files filesets create hello-context --workspace default
nemo files upload ./hello/ hello-context --workspace default
```

**6. Build.**

```bash
curl -s -X POST "$NMP_BASE_URL/apis/builder/v2/workspaces/default/builds" \
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

`kubectl -n nmp-builds get pods -w` shows the `fetch` pod, then `build` with its sandbox, then
`push`. Poll the image until it leaves `pending`, which takes a minute or two:

```bash
curl -s "$NMP_BASE_URL/apis/builder/v2/workspaces/default/container-images/hello-1-0" \
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

The spec's `platform` defaults to `linux/amd64`. If your nodes can't run amd64 binaries, `RUN`
steps fail; set `platform` to match the nodes.

**7. Check the result.** The registry should serve the recorded digest, and the image should
contain the Dockerfile's output:

```bash
kubectl -n nmp-builds run check --rm -i --restart=Never --image=nmp-build:local --command -- sh -c '
  sleep 2
  crane digest --insecure registry.nmp-builds.svc.cluster.local:5000/demo/hello:v1
  crane export --insecure registry.nmp-builds.svc.cluster.local:5000/demo/hello:v1 - | tar -xO hello.txt'
```

To build again, submit with `"revision": 2`.

## Development

The plugin isn't a uv workspace member, so install it into the environment first, then run the
tools against the environment as installed.

```bash
uv pip install -e plugins/nemo-builder
uv run --no-sync pytest plugins/nemo-builder/tests
uv run --no-sync ruff check plugins/nemo-builder && uv run --no-sync ty check plugins/nemo-builder
```

The tests need no cluster, registry or running platform.

| File | What it holds |
|---|---|
| `service.py` | The REST routes |
| `schema.py` | The request models: `BuildSet`, `BuildSpec`, `FileSetSource`, `BuildOutput` |
| `plan.py` | `BuildPlan`: a request resolved against the deployment, with every submit-time check |
| `compile.py` | Turns a plan into the three-step `PlatformJobSpec`, as a pure function |
| `submit.py` | The submit sequence: plan, then rows, then the job |
| `steps.py` | What each step's config contains, and `WorkLayout` |
| `entities.py` | `ContainerImage` |
| `controller.py` | The reconciler |
| `registry.py` | The OCI Distribution client the reconciler uses |
| `identity.py` | Image reference parsing and the system tag |
| `config.py` | `BuilderConfig` |
| `run/` | The step programs: `fetch.py`, `supervise.py`, `push.py` |
| `docker/` | The `nmp-build` step image, and the platform image with this plugin added |
| `deploy/` | The build namespace; `deploy/local/` adds the minikube platform and registry |
| `config/` | Platform configs for minikube and for GKE |

## Constraints

- **Kubernetes only, with the platform inside the cluster.** Build pods call back to Files,
  Secrets and Jobs, so a control plane outside the cluster can't run builds.
- **One build node.** The work volume is `ReadWriteOnce`. A `ReadWriteMany` volume would lift this.
- **No registry allowlist.** A spec can publish to any registry it names. The push credential is
  only ever sent to `default_registry`, but the image is pushed wherever the spec says.
- **`push_secret` is looked up in the submitter's workspace**, so each submitting workspace needs a
  secret of that name, and the credential used is theirs rather than the operator's.
- **Registries that challenge with Basic auth** can be pushed to, but the reconciler only handles
  Bearer tokens and can't resolve images on them.
- **Each submission needs a new `revision`.** Reusing a `name` and `revision` returns `500`.
- **Signatures are checked for presence, not validity.** Admission-time verification, on the
  cluster that runs the image, is what should check the key.
- **Keep `signature_storage` at `tag`.** It's recorded on each row, but `push` always signs with
  cosign's default legacy `.sig` tag, and that tag is the only place the reconciler looks.
- **Base images come from public registries.** `registry_mirror` is supported but unset, so
  nothing bounds what a Dockerfile can pull `FROM`.
- **`context_path` and `fileset` aren't validated at submit.** A path that would leave the job's
  directory fails the build job rather than returning a `400`.
- **Upstream kaniko was archived** on 2025-06-03. The default `sandbox_image` still points at it.
- **One reconciler replica**, reading up to 200 `pending` rows per cycle.
