<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Builder Plugin

Container image builds for NeMo Helix. A caller submits Dockerfiles whose build context is already
in a Files fileset. The plugin builds them with kaniko in a locked-down namespace, and a credential
broker hands the push step a registry token for exactly that build's destinations and signs what it
pushed. The build's last step delivers the signature to the platform, which verifies it and
records the signed digest as a `ContainerImage` entity that consumers can pin.

**Status: proof of concept.** It runs end to end on minikube. It is not a uv workspace member, is
not in the Helm chart, and has the limitations listed under [Constraints](#constraints).

## Surfaces

- **REST:** `POST /apis/builder/v2/workspaces/{workspace}/builds` submits a build;
  `GET .../container-images` and `GET .../container-images/{name}` read the results; and
  `POST .../container-images/{name}/signature` is how a build's last step completes an image.
- **Entity:** `ContainerImage` (`entity_type="container_image"`), one row per image.
- **Controller:** `BuilderController`, registered under `nemo.controllers`: the failure sweep,
  which fails rows whose build job ended without completing them.
- **Programs:** `nhx-build {fetch|supervise|push}`, which the build job's steps run, and
  `nhx-build broker`, the credential broker, which runs as its own Deployment. Both ship in the
  `nhx-build` image (`docker/Dockerfile`), along with `nhx-build token-server`, a local-only stand-in
  for a registry's token service.

There is no CLI or SDK accessor yet; use the REST API.

## How it works

Images are built in this cluster, by the `execution` backend. A submitted build set becomes one
Jobs job with three steps, each under its own ServiceAccount. The caller's Dockerfile runs in a
fourth pod, the sandbox, which is not a job step and holds nothing:

| | Runs as | Holds | Work volume | Runs caller code |
|---|---|---|---|---|
| `fetch` | `nhx-build-fetch` | a Files client, acting as the submitter | yes | no |
| `build` (runs `nhx-build supervise`) | `nhx-build-control` | permission to manage pods in `nhx-builds` | no | no |
| sandbox (kaniko) | no ServiceAccount token | nothing | its own context read-only, and one output directory per image | yes |
| `push` | `nhx-build-push` | nothing: no RBAC, no Secret, no key | yes | no |
| credential broker | `nhx-build-broker`, in its own namespace | the registry credential and the signing key | no | no |

No step can read a credential through its identity, and that is the point. `jobs.create` is in the
Editor role and execution profiles are not authorized, so anyone who can submit a build can run a
raw job as any of these identities. For `nhx-build-push` that opens almost nothing: the broker
grants a job only its own `pending` rows' destinations, and a raw job has none -- unless it takes
the name of a build whose rows never got their job (see [Constraints](#constraints)).

1. **Submit.** The request is resolved into a plan: a job name, and a row name for each image. The
   backend then places each image -- registry, repository and system tag -- and refuses at submit
   anything it cannot honor. The system tag is a tag the build pushes alongside the caller's, and
   the one the broker resolves and signs; its form is under
   [Names](#get-container-images-and-get-container-imagesname). One `ContainerImage` row per image
   is created `pending`, then the job. Each row records a digest of the request, so submitting the
   same request again returns the same rows, and a different request under the same `name` and
   `revision` is refused.
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
     addresses -- the broker and the platform among them -- and DNS goes to public resolvers
   - a deadline the kubelet enforces, so it ends even if `supervise` is killed first

   An admission policy holds `nhx-build-control` to creating pods of this shape: the sandbox's
   label, the namespace's default ServiceAccount with no token, no environment, and only
   work-volume mounts under a job's slice. It doesn't check the image or the command.
   `supervise` mounts no volume itself; it reads each image's pass or fail from the sandbox's log,
   which only affects its own exit code and logs.
4. **`push`** treats each layout as untrusted. It refuses symlinks, special files, blobs that
   don't hash to their names, and anything but a single manifest. For each image it then:
   - asks the broker for credentials, presenting its pod's own token, and gets a registry token for
     its job's `pending` destinations;
   - pushes the caller's tag and the system tag with crane;
   - asks the broker to sign the image, naming only the row. The broker resolves the system tag
     itself, signs that digest with a key no step can read, and returns the signed payload and the
     signature;
   - checks that the broker signed the digest it pushed, and delivers the signature to the
     platform, as the submitter.
5. **The signature route** verifies the signature against the deployment's public key, checks that
   the signed payload names this row's repository and carries this row's workspace, image, job and
   request digest, and only then marks the row `ready` with the signed digest. Nothing that ran the
   Dockerfile reports an image's digest, and the step that carries the signature cannot forge it.
6. **The failure sweep** fails a row that is still `pending` when its job has ended: a job's last
   step delivers its signatures before it exits, so none is coming.

One failed Dockerfile doesn't stop the rest of the set: the other images still build, publish and
complete, and only the failed image's row ends `failed`. If every image in a set fails, `push` never
runs.

Every step finds files on the work volume through one `WorkLayout` (in `steps.py`). Step configs
name filesets and images, never paths, and no step config carries a credential or the name of one.

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
| `build_specs[].name` | The image's name within the set, unique in it: lowercase letters and digits, joined by `.`, `_` or `-`, at most 63 characters. It is a repository path component, because an image with no `output` is published under it. |
| `source.fileset` | The fileset holding the build context: `name` in this workspace, or `workspace/name`. |
| `source.context_path` | Optional subdirectory of the fileset to use as the context. Relative, with no `..`. Omit it to use the whole fileset. |
| `dockerfile` | Path relative to the context. Absolute paths and `..` are rejected. Defaults to `Dockerfile`. |
| `platform` | Passed to kaniko. Defaults to `linux/amd64`. |
| `output.repository`, `output.tag` | Optional. Where the image is published, within your workspace's part of the registry. Tags containing `--` or starting with `sha256-` are reserved for system tags and signatures. |

Every request model rejects fields it doesn't know, so a misspelt field is a `422` rather than
silently ignored.

Every image goes to the deployment's one registry. With an `output`, at
`<registry>/<repository_prefix>/<workspace>/<output.repository>:<output.tag>`; without one, at
`<registry>/<repository_prefix>/<workspace>/<name>/<build_specs[].name>`, under the system tag only.
A caller can't name a registry: whatever runs the image has to pull it, and one registry means one
pull credential for every workload that does. The workspace in the path keeps workspaces out of
each other's repositories.

A successful submit returns `201` with the job name and the `pending` rows. Poll the rows, not
the job: images in a set finish independently.

| Status | When |
|---|---|
| `201` | Rows created and job submitted, or the same request was submitted before and these are its rows and job. |
| `400` | Two specs would publish the same `repository:tag`, or the workspace name can't be a repository path component. |
| `403` | The caller may submit builds but may not create jobs in the workspace. |
| `409` | This deployment can't build: `registry`, `sandbox_image`, `credential_broker` or `signing_public_key` is unset. Or this `name` and `revision` were already submitted with a different request, or its job's name is held by a job this request didn't create; the rows it wrote are then failed. |
| `422` | The body failed validation: for example a path outside the context or fileset, a set or spec name that doesn't fit, more than 100 specs, duplicate spec names, a `revision` below 1, a repository or tag that isn't valid or is reserved, or an unknown field such as `output.registry`. |
| `502` | Another platform service refused the build. |
| `500` | Anything else. |

### `GET /container-images` and `GET /container-images/{name}`

`GET /container-images` accepts `page` and `page_size` (at most 100), and two filters: `job`, a job
name this submit returned, and `status`. `GET /container-images/{name}` returns `404` for a name
that doesn't exist. Each row is a `ContainerImage`:

| Field | Meaning |
|---|---|
| `status` | `pending`, `ready` or `failed`. `ready` is written only on a verified signature, and `failed` only by the failure sweep; both are final. |
| `status_detail` | Why a row failed. |
| `registry`, `repository` | Where the image lives. |
| `digest` | The digest the verified signature covers, set once when the row becomes `ready`. Pin `<registry>/<repository>@<digest>`. |
| `signature` | Set once the row is `ready`. `verified_against` names the public key the signature was verified with (`key:sha256:<fingerprint>`), and `signer` the signed claims that tied it to this row. |
| `platform` | The platform the image was built for. |
| `provenance` | The build set, revision, job and system tag that produced the image, and `request_digest`, the digest of the request that did. |

Names are derived from the request:

| | Pattern | Example |
|---|---|---|
| Job | `<set>-<revision>` | `hello-1` |
| Row | `<set>-<revision>-<index>` | `hello-1-0` |
| System tag | `<workspace>--<set>-<revision>-<index>` | `default--hello-1-0` |

### `POST /container-images/{name}/signature`

What the push step calls, as the submitter, with what the broker returned:

```json
{"payload": "<base64 of the signed payload>", "signature": "<base64 of the signature>"}
```

The payload is cosign's simple-signing payload, byte for byte as it was signed: the same bytes a
verifier finds beside the image in the registry. The route verifies, in order: the signature,
under the deployment's public key; that the payload names a digest; that it names this row's
repository; that its annotations are this row's workspace, image, build set, revision, job and
request digest; and that the row is `pending`. Then one write, conditional on the row's
version, makes it `ready`. The caller must hold `builder.container-images.complete`, which the
Editor role has, but what the route trusts is the signature.

| Status | When |
|---|---|
| `200` | The row is `ready` -- now, or already with this digest. Delivering the same signature again is safe. |
| `404` | No such row. |
| `409` | The row has already failed, or is `ready` with another digest. |
| `422` | The signature does not verify, or is not this row's. |

## The credential broker

`nhx-build broker` is the build plane's credential authority. It runs as the `nhx-build-broker`
Deployment in its own namespace, and its whole Kubernetes permission is `create` on TokenReviews
and `get` on pods in the build namespace. It has no platform identity.

**Identity.** A step presents its pod's token as the password of an HTTP Basic credential. The
broker asks the API server whose token it is and which pod it is bound to, then reads that pod
and requires: the build namespace; the ServiceAccount the step's profile runs as; the pod's UID;
Jobs' `managed_by` label; a step and profile its backend entitles; and a pod that has not finished
(a finished pod's token stays valid until the pod is deleted). The job and workspace are then the
labels Jobs wrote, which the step can't choose. Every refusal is logged with the check that failed,
and never with the token.

**Rows.** The broker reads the job's `pending` rows through the builder's own list route:
anonymously with platform auth off, and with auth on as the submitter, by exchanging the step's own
token at the platform's token exchange (`token_exchange`). The platform issues that token only
while Jobs' delegation for the pod is live.

**Entitlement.** A step never names a scope. What it gets is derived from its role and its job's
`pending` rows, bounded by `<repository_prefix>/<the job's workspace>/`: the rows only narrow that
bound. On the `execution` backend only the push step is entitled to anything -- pull and push on its
job's destinations, and signing its job's rows.

**Registry credentials.** The broker holds one credential for the registry and asks the registry's
own token service -- the realm in its `401` challenge, per the Distribution token specification --
for `repository:<destination>:pull,push`, for exactly the job's destinations. It hands on what the
registry issues as a Docker config `registrytoken`, which crane sends directly. How long the token
lives is the token service's choice; the broker reports it, from the token itself where it can be
read. Whether the registry narrowed the token is reported the same way; with
`registry_narrows_scope` set, a readable token wider than the request is refused. A registry that answers with a Basic challenge is refused: it cannot narrow, and the broker
does not hand out its own credential.

**Signing.** `POST /sign` names a row and nothing else. The broker checks the same entitlement,
and that the row's repository and system tag parse as this system writes them. It resolves the
system tag in the registry with a token it gets for itself, refuses an index, signs by digest with
cosign -- every annotation from the row, the transparency log upload off -- and returns the signed
payload and signature.

**Audit.** Every token issued, every signature -- before it is made, naming the digest, and again
with its outcome -- and every refusal of an identified step is a line on the broker's stdout,
prefixed `audit`, naming the pod and the job. If a token's or a signature's line can't be written,
nothing is issued and nothing is signed. A request that identifies no step, or that the broker
can't decide, is logged instead.

**Transport.** The broker serves plain HTTP, and its NetworkPolicy admits only job steps. It drops
a connection that goes quiet, reads nothing larger than a request needs, and serves a bounded
number of requests at once.

| Setting (`NHX_BROKER_CONFIG`) | Meaning |
|---|---|
| `platform_url` | Where the builder's routes are. |
| `registry_username`, `registry_password_file` | The broker's credential for the registry's token service. The file is re-read on every use: a mounted Secret, or a projected ServiceAccount token the kubelet rotates. |
| `signing_key`, `signing_key_password_file` | The cosign key: a file path, or a KMS URI such as `gcpkms://...`. |
| `step_token_audience` | The audience a step's token must be bound to. Unset with platform auth off; with it on, the workload identity audience Jobs projects for token exchange. |
| `token_exchange` | With platform auth on: `token_endpoint` (HTTPS, or HTTP to a loopback address), `client_id`, and optionally `audience` and `scope`, for reading rows as the submitter. |
| `port`, `request_timeout_seconds` | `8080` and `30` by default. |

The broker also reads the platform's `builder:` section, from `NHX_CONFIG_FILE_PATH`, so it and the
platform agree on the registry, the prefix and the push step's identity.

## The failure sweep

`BuilderController` is the only writer of `failed`. Every `sweep_interval_seconds` (default 30) it
lists `pending` rows across all workspaces, oldest first, and asks Jobs about each row's job:

- **The job has ended** -- completed, errored or cancelled -- and the row is still `pending`:
  `failed`, with the job's status as the detail.
- **The job doesn't exist**, and the row is older than `job_creation_grace_seconds` (default 300):
  `failed`. Rows are written before their job, so a submit that died between the two leaves rows
  nothing will build.
- **Otherwise** nothing: a running job's rows wait, and so does a row whose job can't be read
  right now.

It reads no registry and holds no credential. Every write is conditional on the version it read,
so it never undoes a `ready` that a signature wrote a moment earlier. A wedged sweep delays failures,
never successes.

## Configuration

Operator settings live under `builder:` in the platform config, or in `NEMO_BUILDER_*` environment
variables. Callers can't set any of them.

```yaml
builder:
  namespace: nhx-builds
  work_pvc: nhx-build-work
  sandbox_image: my-registry/nhx-kaniko:v1.25.19 # built from docker/Dockerfile.kaniko
  registry: us-central1-docker.pkg.dev           # a host only, never host/path; http://host for plain HTTP
  repository_prefix: my-project/my-repo          # images land at <prefix>/<workspace>/...
  registry_narrows_scope: false                  # true only for a registry measured to narrow tokens
  credential_broker: http://nhx-build-broker.nhx-build-broker.svc.cluster.local:8080
  signing_public_key: |                          # the broker's cosign.pub
    -----BEGIN PUBLIC KEY-----
    ...
  sweep_interval_seconds: 30
```

Four settings have no safe default: `registry`, `sandbox_image`, `credential_broker` and
`signing_public_key`. If any is missing, submits fail with a `409` rather than as a build that dies
in a pod.

**The registry must accept nested repository paths, and create a repository on its first push**, as
Artifactory, GAR, Harbor and Distribution do. Docker Hub and nvcr.io don't: each repository there
must be created first, and names have a fixed depth.

**Set `registry_narrows_scope` only for a registry measured to narrow tokens.** When it is set, the
broker refuses a token wider than it asked for. Narrowing is the token service's job, not the
registry's: the quickstart's stand-in narrows, as the token specification requires, and
Artifactory's token service doesn't.

**Set `repository_prefix` to a path dedicated to builds.** Left empty, each workspace name is a
top-level namespace in the registry, and the broker's registry credential can write all of them: a
workspace named after the namespace your platform images live in would publish over them.

The platform also needs three Jobs execution profiles, one per step, named by `fetch_profile`,
`control_profile` and `push_profile` (`build-fetch`, `build-control` and `build-push` by default).
Each is a `cpu` profile on the `kubernetes_job` backend, and they must agree with the builder
config and the manifests:

- `namespace` is the builder's `namespace`, and `storage.pvc_name` is its `work_pvc`
- `service_account_name` is `nhx-build-fetch`, `nhx-build-control` or `nhx-build-push`, and the
  push profile's must equal `push_service_account`, which the broker requires of a push pod
- `default_task_image` is the `nhx-build` image, and `launcher_image` an image that ships
  `/tools/jobs-launcher`, as the platform image does
- `node_selector` is the build node's label

`config/platform-config.minikube.yaml` is the quickstart's config. It works, but it is local-only
in the ways its header lists, auth being off among them; don't start a shared deployment from it.

The remaining settings are `sandbox_cpu`, `sandbox_memory`, `sandbox_dns_nameservers`,
`node_selector` and `job_creation_grace_seconds`. Each is described on `BuilderConfig` in
`config.py`.

## Cluster setup

`deploy/` holds the build namespace, everything that constrains pods in it, and the broker's
namespace. Apply the whole directory with `kubectl apply -f`:

| File | What it does |
|---|---|
| `00-namespace.yaml` | Creates `nhx-builds`, enforcing the `baseline` Pod Security Standard |
| `10-serviceaccounts.yaml` | The three step identities |
| `20-rbac.yaml` | Pod management for `nhx-build-control`: the only RBAC any step holds |
| `25-sandbox-admission.yaml` | Holds `nhx-build-control` to creating pods shaped like the sandbox -- its label, the namespace's default ServiceAccount with no token, work-volume mounts only -- which is what keeps any other pod from wearing a push step's identity |
| `30-networkpolicy.yaml` | Sandbox egress: the public internet, minus private, link-local and cluster ranges |
| `40-work-volume.yaml` | The shared work volume |
| `50-broker.yaml` | The broker's namespace, ServiceAccount and its two permissions, and a policy admitting only job steps to it |
| `negative-control.sh` | Checks that the namespace still refuses a pod with BuildKit's privileges |
| `sandbox-admission-check.sh` | Checks that `nhx-build-control` can create the sandbox's shape and nothing else |
| `sandbox-egress-probe.sh` | Checks, with real connections, what a sandbox can and can't reach |

The broker's Deployment is deployment-specific -- it mounts the registry credential, the key and
the platform's config -- so it isn't in `deploy/`; `deploy/local/broker.yaml` is the minikube one.

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

This runs the platform, the broker and a token-auth registry inside minikube, builds one image, and
checks the result. Run the commands from the repository root, in one shell: later steps use
variables earlier ones set.

You need Docker with the default buildx driver, minikube, `kubectl`, `openssl`, `jq`, and the `nemo`
CLI (`make bootstrap`; see [SETUP.md](../../SETUP.md)). The cluster below uses 3 CPUs and 5.5 GB of
memory. Sandboxes resolve names with `8.8.8.8` and `1.1.1.1`, so the network must allow DNS to
them; some corporate networks and VPNs don't.

**1. Cluster, build namespace and the broker's namespace.**

```bash
minikube start --driver=docker --cni=calico --cpus=3 --memory=5500
kubectl label node minikube nhx.nvidia.com/build-node=true
kubectl -n kube-system wait --for=condition=Ready pod -l k8s-app=calico-node --timeout=300s

kubectl apply -f plugins/nemo-builder/deploy/
plugins/nemo-builder/deploy/negative-control.sh         # must print PASS
plugins/nemo-builder/deploy/sandbox-admission-check.sh  # must print PASS
plugins/nemo-builder/deploy/sandbox-egress-probe.sh     # must print PASS
```

**2. Images.** Build the platform image with this plugin added, the image the build steps and the
broker run in, and the sandbox's kaniko image. Use `linux/amd64` instead of `linux/arm64` on an
x86 machine.
`Dockerfile.platform` starts `FROM` the local `my-registry/nhx-api:local`, which only the default
`docker` buildx driver can see; with a `docker-container` builder selected, run
`docker buildx use default` first.

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
docker buildx build --platform linux/arm64 -f plugins/nemo-builder/docker/Dockerfile.kaniko \
  -t nhx-kaniko:local --load plugins/nemo-builder/docker

minikube image load nhx-api-builder:local
minikube image load nhx-build:local
minikube image load nhx-kaniko:local
```

`minikube image load` doesn't reliably replace an image already loaded under the same tag. When
you rebuild, use a new tag, and update `deploy/local/` and the config to match.

**3. Keys.** Two keys, both for this cluster only, in a temporary directory rather than the
repository: the broker's cosign key, which signs every image, and the local token service's key,
which signs registry tokens. The registry trusts the second through a JWKS derived from it. Keep
`$KEYS/cosign.pub` to verify signatures later.

```bash
KEYS=$(mktemp -d)
docker run --rm -u "$(id -u):$(id -g)" -v "$KEYS:/work" -w /work -e COSIGN_PASSWORD= \
  ghcr.io/sigstore/cosign/cosign:v2.5.3 generate-key-pair
openssl ecparam -name prime256v1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt -out "$KEYS/token-server.pem"
docker run --rm -i nhx-build:local nhx-build token-server jwks < "$KEYS/token-server.pem" > "$KEYS/jwks.json"
```

**4. Registry, token service, broker and platform.** `deploy/local/` runs a registry at
`registry.nhx-registry.svc.cluster.local:5000` with token auth, the local stand-in for its token
service, the broker, and the platform services the builder needs in one pod with SQLite. The
platform's config goes in the platform's namespace and the broker's, since both read `builder:`.

```bash
kubectl apply -f plugins/nemo-builder/deploy/local/
for ns in nhx-platform nhx-build-broker; do
  kubectl -n "$ns" create configmap nemo-helix-config \
    --from-file=config.yaml=plugins/nemo-builder/config/platform-config.minikube.yaml
done
kubectl -n nhx-registry create secret generic nhx-token-server-key --from-file=key.pem="$KEYS/token-server.pem"
kubectl -n nhx-registry create configmap registry-token-jwks --from-file=jwks.json="$KEYS/jwks.json"
kubectl -n nhx-build-broker create secret generic nhx-build-broker-cosign-key \
  --from-file=cosign.key="$KEYS/cosign.key"
kubectl -n nhx-platform create configmap nhx-build-signing-public-key --from-file=cosign.pub="$KEYS/cosign.pub"

kubectl -n nhx-registry wait --for=condition=Ready pod --all --timeout=300s
kubectl -n nhx-build-broker wait --for=condition=Ready pod -l app=nhx-build-broker --timeout=300s
kubectl -n nhx-platform wait --for=condition=Ready pod -l app=nemo-helix --timeout=600s
kubectl -n nhx-platform port-forward svc/nemo-helix 8080:8080 >/dev/null &
export NHX_BASE_URL=http://localhost:8080
```

Pods waiting on a Secret or ConfigMap made after them start once it exists; wait on the pods, as
above, rather than on the rollouts, whose progress deadline can pass first. The port-forward runs
in the background until the shell exits.

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
`push`; `kubectl -n nhx-build-broker logs deploy/nhx-build-broker | grep audit` shows what the
broker granted and signed. Poll the image until it leaves `pending`, which takes a minute or two:

```bash
curl -s "$NHX_BASE_URL/apis/builder/v2/workspaces/default/container-images/hello-1-0" \
  | jq '{status, digest, verified_against: .signature.verified_against, status_detail}'
```

```json
{
  "status": "ready",
  "digest": "sha256:4c9b935d53ba5b129d7d8d68a5da8034103817d319623a6591650dd2b306c49d",
  "verified_against": "key:sha256:…",
  "status_detail": null
}
```

**7. Check the result.** The registry should serve the recorded digest, the signature should
verify against your public key, and the image should contain the Dockerfile's output. The registry
refuses anonymous reads, so the check uses a short-lived token for the pull-only
`nhx-registry-reader`:

```bash
DIGEST=$(curl -s "$NHX_BASE_URL/apis/builder/v2/workspaces/default/container-images/hello-1-0" | jq -r .digest)
TOKEN=$(kubectl -n nhx-registry create token nhx-registry-reader --audience nhx-local-token-server --duration 10m)
kubectl -n nhx-registry run check --rm -i --restart=Never --image=nhx-build:local \
  --env="TOKEN=$TOKEN" --env="DIGEST=$DIGEST" --env="PUB=$(cat "$KEYS/cosign.pub")" --command -- sh -c '
  set -e; export HOME=/tmp
  REG=registry.nhx-registry.svc.cluster.local:5000 REF=registry.nhx-registry.svc.cluster.local:5000/default/demo/hello
  crane auth login "$REG" -u reader -p "$TOKEN" >/dev/null
  crane digest --insecure "$REF:v1"
  echo "$PUB" > /tmp/cosign.pub
  cosign verify --key /tmp/cosign.pub --insecure-ignore-tlog=true --allow-http-registry "$REF@$DIGEST" >/dev/null
  crane export --insecure "$REF:v1" - | tar -xO hello.txt'
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
| `_perms.py`, `authz.py` | The routes' permissions, and the plugin's authz scope |
| `schema.py` | The request models: `BuildSet`, `BuildSpec`, `FileSetSource`, `BuildOutput` |
| `plan.py` | `BuildPlan`: a request resolved into names, then placed by the backend |
| `backend.py` | The backend interface: what a backend refuses, compiles, places, and entitles its steps to, and how its signatures are checked |
| `backends.py`, `execution.py` | The deployment's backend, and the `execution` backend |
| `compile.py` | Turns a plan into the three-step `HelixJobSpec`, as a pure function |
| `signing.py` | What the broker's signature says, and how the platform verifies it and ties it to a row |
| `submit.py` | The submit sequence: plan, then rows, then the job |
| `completion.py` | Verifying signatures, and the one write that makes a row `ready` |
| `controller.py` | The failure sweep |
| `broker.py` | The broker's rules: which step a pod is, and what its job may have |
| `registry_auth.py` | Registry tokens by the Distribution token specification |
| `client.py` | A typed client for the builder's own routes |
| `steps.py` | What each step's config contains, and `WorkLayout` |
| `entities.py` | `ContainerImage` |
| `identity.py` | Image reference parsing and the system tag |
| `config.py` | `BuilderConfig` |
| `run/` | The programs: `fetch.py`, `supervise.py`, `push.py`, `broker.py`, and the local `token_server.py`; `main.py`, the `nhx-build` entry point that runs each; `context.py`, what the steps share; and `tools.py`, which runs crane and cosign |
| `docker/` | The `nhx-build` image, and the platform image with this plugin added |
| `deploy/` | The build namespace and the broker's; `deploy/local/` adds the minikube platform, broker, registry and token service |
| `config/` | The platform config for the minikube quickstart |

## Constraints

- **Kubernetes only, with the platform inside the cluster.** Build pods call back to Files, Jobs
  and the builder, so a control plane outside the cluster can't run builds.
- **One build node.** The work volume is `ReadWriteOnce`. A `ReadWriteMany` volume would lift this.
- **One backend, and one registry, per deployment.** Every image is published to `registry`,
  under the submitting workspace's path. Publishing anywhere else means copying the image out
  afterwards.
- **Each new request needs a new `revision`.** The same request again returns its rows; a
  different one under a used `name` and `revision` returns `409`.
- **Platform auth on is verified only against fakes.** The broker's auth-on path -- a workload
  token, rows read as the submitter through token exchange -- is unit-tested, not yet run.
- **Execution profiles aren't authorized in Jobs**, so anyone who can submit a build can run a raw
  job under any of the builder's profiles. Three things follow, and closing all three needs Jobs
  to restrict these profiles to the builder:
  - **A raw job can take over a build whose job was never created.** The broker ties rows to their
    job by the job's name. A submit that finds the name taken fails its rows, but if a submit dies
    between writing its rows and creating its job, a raw job created under that name is granted
    those rows' destinations, and can have them signed and completed with its own image. Same
    workspace only.
  - **A sandbox-shaped pod can mount any job's slice of the work volume.** The admission policy
    can't tie a pod to the job that asked for it. Such a pod holds no credential and has no cluster
    network, but could rewrite another job's layout before that job's push step publishes it --
    and the broker would sign it as that job's.
  - **A raw `build-control` job can read and delete any build pod.** Its Role lets it read the log
    of any build pod in `nhx-builds`, another tenant's build output among them, and delete any
    build pod.
- **The broker serves plain HTTP.** Only job steps can reach it, but with platform auth on, a
  step's workload token crosses the pod network in the clear. Put TLS in front of it, as a service
  mesh does.
- **Registry tokens aren't revoked.** They live as long as the registry's token service says.
- **Signatures are stored as cosign's legacy `.sig` tag**, which the broker's cosign v2 writes.
- **Base images come from public registries.** Nothing bounds what a Dockerfile can pull `FROM`,
  and the NetworkPolicy blocks private addresses, so a private base image can't be pulled.
