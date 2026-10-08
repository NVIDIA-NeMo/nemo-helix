<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Builder Plugin

Container image builds for NeMo Helix. You submit Dockerfiles whose build context is in a Files
fileset. The plugin builds them with kaniko, in a sandbox pod that holds no credentials. Then it
pushes each image to the deployment's registry, signs it with your workspace's key, and records it
as a `ContainerImage` you can pin by digest.

**Status: proof of concept.** It runs end to end on minikube. It isn't a uv workspace member or in
the Helm chart, and it has the [limitations](#limitations) listed below.

## Quickstart (minikube)

This runs the platform with the builder, and a registry, inside minikube. It builds one image and
checks the result. Run the commands from the repository root, in one shell: later steps use
variables that earlier ones set.

You need Docker with the default buildx driver, minikube, `kubectl`, `openssl`, `jq`, and the `nemo`
CLI (`make bootstrap`; see [SETUP.md](../../SETUP.md)). The cluster uses 3 CPUs and 5.5 GB of
memory. Sandboxes resolve names with `8.8.8.8` and `1.1.1.1`, so your network must allow DNS to
them; some corporate networks and VPNs don't.

**1. Cluster.** The work volume is `ReadWriteOnce`, so every build pod runs on the node labelled
`nhx.nvidia.com/build-node=true`.

```bash
minikube start --driver=docker --cpus=3 --memory=5500
kubectl label node minikube nhx.nvidia.com/build-node=true
```

**2. Images.** Build the platform image with this plugin added, the image the build steps run in,
and the sandbox's kaniko image. On an x86 machine, use `linux/amd64` instead of `linux/arm64`.
`Dockerfile.platform` starts `FROM` the local `my-registry/nhx-api:local`, which only the default
`docker` buildx driver can see; if a `docker-container` builder is selected, run
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

`minikube image load` doesn't reliably replace an image already loaded under the same tag. When you
rebuild, use a new tag, and update `deploy/` and the config to match.

**3. Keys and a password.** All for this cluster only, kept in a temporary directory rather than the
repository: a signing key pair, a password for the registry, and the registry's password file. Keep
`$KEYS/signing.pub` to verify signatures later.

```bash
KEYS=$(mktemp -d)
openssl ecparam -name prime256v1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt -out "$KEYS/signing.key"
openssl pkey -in "$KEYS/signing.key" -pubout -out "$KEYS/signing.pub"
REGISTRY_PASSWORD=$(openssl rand -hex 16)
printf '%s\n' "$REGISTRY_PASSWORD" | docker run --rm -i httpd:2.4-alpine htpasswd -niB builder > "$KEYS/htpasswd"
```

**4. Deploy.** `deploy/` has one manifest per namespace:

| File | Namespace | What runs there |
|---|---|---|
| `builds.yaml` | `nhx-builds` | The build jobs and their sandboxes, with the step ServiceAccounts, their RBAC, and the work volume |
| `platform.yaml` | `nhx-platform` | The platform services the builder uses, in one pod with SQLite |
| `registry.yaml` | `nhx-registry` | A registry at `registry.nhx-registry.svc.cluster.local:5000` that takes a username and password |

```bash
kubectl apply -f plugins/nemo-builder/deploy/
kubectl -n nhx-platform create configmap nemo-helix-config \
  --from-file=config.yaml=plugins/nemo-builder/config/platform-config.minikube.yaml
kubectl -n nhx-registry create secret generic registry-htpasswd --from-file=htpasswd="$KEYS/htpasswd"

kubectl -n nhx-registry wait --for=condition=Ready pod --all --timeout=300s
kubectl -n nhx-platform wait --for=condition=Ready pod -l app=nemo-helix --timeout=600s
kubectl -n nhx-platform port-forward svc/nemo-helix 8080:8080 >/dev/null &
export NHX_BASE_URL=http://localhost:8080
```

Pods waiting on a Secret or ConfigMap made after them start once it exists, so wait on the pods, as
above, rather than on the rollouts, whose progress deadline can pass first. The port-forward runs in
the background until the shell exits.

**5. Your workspace's secrets.** The push step logs in to the registry, and signs, with three
platform secrets in the workspace that submits the build:

```bash
nemo secrets create builder-registry-username --workspace default --value builder
printf '%s' "$REGISTRY_PASSWORD" | nemo secrets create builder-registry-password --workspace default --from-file -
nemo secrets create builder-signing-key --workspace default --from-file "$KEYS/signing.key"
```

**6. Build context.**

```bash
CTX=$(mktemp -d) && cat > "$CTX/Dockerfile" <<'EOF'
FROM docker.io/library/python:3.13-slim
RUN pip install --no-cache-dir six==1.16.0
RUN echo "hello from nemo-builder" > /hello.txt
EOF
nemo files filesets create hello-context --workspace default
nemo files upload "$CTX/" hello-context --workspace default
```

**7. Build.** A spec's `platform` defaults to `linux/amd64`. On an arm64 machine, its `RUN` steps
work only if the node can run amd64 binaries, as Docker Desktop's emulation does; otherwise add
`"platform": "linux/arm64"` to the spec.

```bash
curl -s -X POST "$NHX_BASE_URL/apis/builder/v2/workspaces/default/builds" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "hello",
    "revision": 1,
    "build_specs": [
      {
        "source": {"fileset": "hello-context"},
        "output": {"repository": "demo/hello", "tag": "v1"}
      }
    ]
  }' | jq '{job, images: [.images[] | {name, status}]}'
```

`kubectl -n nhx-builds get pods -w` shows the `fetch` pod, then `build` with its sandbox, then
`push`. Poll the image until it leaves `pending`, which takes a minute or two:

```bash
curl -s "$NHX_BASE_URL/apis/builder/v2/workspaces/default/container-images/hello-1" \
  | jq '{status, digest, status_detail}'
```

```json
{
  "status": "ready",
  "digest": "sha256:4c9b935d53ba5b129d7d8d68a5da8034103817d319623a6591650dd2b306c49d",
  "status_detail": null
}
```

**8. Check the result.** The registry should serve the recorded digest, the signature should verify
against your public key, and the image should contain the Dockerfile's output:

```bash
DIGEST=$(curl -s "$NHX_BASE_URL/apis/builder/v2/workspaces/default/container-images/hello-1" | jq -r .digest)
kubectl -n nhx-registry run check --rm -i --restart=Never --image=nhx-build:local \
  --env="PASSWORD=$REGISTRY_PASSWORD" --env="DIGEST=$DIGEST" --env="PUB=$(cat "$KEYS/signing.pub")" --command -- sh -c '
  set -e; export HOME=/tmp
  REG=registry.nhx-registry.svc.cluster.local:5000 REF=registry.nhx-registry.svc.cluster.local:5000/default/demo/hello
  crane auth login "$REG" -u builder -p "$PASSWORD" >/dev/null 2>&1
  V1=$(crane digest --insecure "$REF:v1")
  [ "$V1" = "$DIGEST" ] || { echo "v1 is $V1, not the recorded $DIGEST" >&2; exit 1; }
  echo "$PUB" > /tmp/signing.pub
  cosign verify --key /tmp/signing.pub --insecure-ignore-tlog=true --allow-http-registry "$REF@$DIGEST" \
    >/dev/null 2>/tmp/verify.log || { cat /tmp/verify.log >&2; exit 1; }
  echo "v1 is $DIGEST, and its signature verifies"
  crane export --insecure "$REF:v1" - | tar -xO hello.txt'
```

On success it ends with `v1 is sha256:…, and its signature verifies` and `hello from nemo-builder`.
If the digests differ or the signature doesn't verify, it says so and exits non-zero.

To build again, submit with `"revision": 2`.

## Using the API

There's no CLI or SDK accessor yet. All routes are under `/apis/builder/v2/workspaces/{workspace}`.

### Submit a build: `POST /builds`

A request is a `BuildSet`: a name, a revision, and up to 100 `BuildSpec`s. The set runs as one Jobs
job, and each spec becomes one `ContainerImage` row.

```json
{
  "name": "hello",
  "revision": 1,
  "build_specs": [
    {
      "name": "main",
      "source": {"fileset": "hello-context", "archive": null, "context_path": null},
      "dockerfile": "Dockerfile",
      "platform": "linux/amd64",
      "output": {"repository": "demo/hello", "tag": "v1"}
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `name` | What is being built. Reused across rebuilds. Lowercase letters, digits and single hyphens, starting with a letter and ending in a letter or digit, at least two characters, and short enough that every name built from it fits in 63 characters. |
| `revision` | Your own version number for `name`, starting at 1. Use a new revision for every build. |
| `build_specs` | Up to 100 specs. |
| `build_specs[].name` | The image's name within the set, unique in it: lowercase letters and digits, joined by single `.`, `_` or `-`. It names the image's row, `<name>-<revision>.<spec>`, and an image with no `output` is published under it. One spec in a set may leave it out; its row is `<name>-<revision>`. |
| `source.fileset` | The fileset holding the build context: `name` in this workspace, or `workspace/name`. |
| `source.archive` | Optional. A tar archive in the fileset to build from, compressed or not: `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tbz2`, `.tar.xz` or `.txz`. Relative, with no `..`. |
| `source.context_path` | Optional subdirectory to use as the context: of the fileset, or of the archive if `archive` is set. Relative, with no `..`. Omit it to use the whole fileset or archive. |
| `dockerfile` | Path relative to the context. Absolute paths and `..` are rejected. Defaults to `Dockerfile`. |
| `platform` | Passed to kaniko. Defaults to `linux/amd64`. |
| `output.repository`, `output.tag` | Optional. Where the image is published, within your workspace's part of the registry. Tags containing `--` or starting with `sha256-` are reserved for system tags and signatures. |

A misspelt field is a `422`, never silently ignored.

One archive can hold several images' contexts, as a benchmark task with an environment and a test
image does. Name the archive in each spec, each with its own `context_path`. A build downloads and
unpacks an archive once, however many of its specs use it:

```json
{"name": "tests", "source": {"fileset": "tasks", "archive": "tb/hello-world.tar.gz", "context_path": "tests"}}
```

Every image goes to the deployment's one registry. With an `output`, it goes to
`<registry>/<repository_prefix>/<workspace>/<output.repository>:<output.tag>`. Without one, it goes
to `<registry>/<repository_prefix>/<workspace>/<name>/<build_specs[].name>`, or
`<registry>/<repository_prefix>/<workspace>/<name>` for the unnamed spec, under the system tag only.
You can't name a registry: whatever runs the image has to pull it, and one registry means one pull
credential for every workload.

A successful submit returns `201` with the job's name and the `pending` rows. Poll the rows, not the
job: images in a set finish independently.

| Status | When |
|---|---|
| `201` | Rows created and job submitted, or the same request was submitted before and these are its rows and job. |
| `400` | Two specs would publish the same `repository:tag`, or the workspace name can't be a repository path component. |
| `401` | Platform auth is on and the request names no one. |
| `403` | You may not submit builds in the workspace, or read Jobs' execution profiles, which every user can by default. Or you may, but may not create jobs there: the rows were already written, and stay `pending`. |
| `409` | The deployment can't build: `registry` or `sandbox.image` is unset, or the build's [Jobs execution profiles](#jobs-execution-profiles) are missing, aren't on Kubernetes, or disagree. Or the push step's secrets don't exist in the workspace, or you can't read them. Or this `name` and `revision` were already submitted with a different request, or the job's name is held by a job this request didn't create; the request's rows are then failed. |
| `422` | The body failed validation: for example a path outside the context or fileset, a set or spec name that doesn't fit, more than 100 specs, duplicate spec names, a `revision` below 1, a repository or tag that isn't valid or is reserved, or an unknown field such as `output.registry`. |
| `502` | Another platform service refused the build. If Jobs refused the job, the rows were already written, and stay `pending`. |
| `500` | Anything else. |

### Read the results: `GET /container-images` and `GET /container-images/{name}`

`GET /container-images` takes `page` and `page_size` (at most 100), and two filters: `job`, a job
name a submit returned, and `status`. `GET /container-images/{name}` returns `404` for a name that
doesn't exist. With platform auth on, a person needs only the permission to read an image; a service,
as the push step calls without workload token exchange, must also act for the image's submitter, or
gets `403`. Each row is a `ContainerImage`:

| Field | Meaning |
|---|---|
| `status` | `pending`, `ready` or `failed`. `ready` is written once the push step has pushed and signed the image; `failed` only when the submit that wrote the row was refused. The builder's routes never change either (but see [Security](#security)). |
| `status_detail` | Why a row failed. |
| `registry`, `repository` | Where the image lives. |
| `digest` | The digest the push step pushed and signed, set when the row becomes `ready`. Pin `<registry>/<repository>@<digest>`. |
| `platform` | The platform the image was built for. |
| `provenance` | The build set, revision, spec and job that produced the image, and `request_digest`, the digest of the request. `spec` is null for the unnamed spec. |

Names come from the request:

| | Pattern | Example |
|---|---|---|
| Job | `<set>-<revision>` | `hello-1` |
| Row | `<set>-<revision>.<spec>`, or `<set>-<revision>` for the unnamed spec | `hello-1.tests`, `hello-1` |
| System tag | `<workspace>--<row>` | `default--hello-1` |

## How a build runs

A submitted build set becomes one Jobs job with three steps, each under its own ServiceAccount. Your
Dockerfiles run in sandboxes, one pod per build context, which aren't job steps and hold nothing. The
steps are trusted (green): they run the builder's own code. The sandboxes are untrusted (red): they
run yours. What a sandbox writes reaches the steps two ways only: the layouts on the work volume, and
its log, which `build` reads for each image's result. The namespaces are the quickstart's: the build
namespace is whichever the [Jobs execution profiles](#jobs-execution-profiles) name. This is the
default sandbox provider; with [OpenSandbox](#opensandbox), `build` asks an OpenSandbox server for each
sandbox instead, and takes each image's result from its command's exit code rather than a log.

```mermaid
flowchart TB
    caller([Caller])
    subgraph platform ["platform (nhx-platform)"]
        api[Builder routes]
        entities["Entities: the ContainerImage rows"]
        jobs[Jobs]
        files[Files]
        secrets["Secrets: each workspace's<br/>registry credential and signing key,<br/>if the deployment uses them"]
    end
    subgraph builds ["build namespace (nhx-builds)"]
        subgraph trusted ["trusted: the job's steps"]
            fetch[fetch]
            build[build]
            push[push]
        end
        work[("work volume (PVC)")]
        subgraph untrusted ["untrusted: the caller's code"]
            sandbox["sandbox, one per build context:<br/>kaniko runs the caller's Dockerfiles"]
        end
    end
    registry[(registry)]

    caller -->|submit, poll| api
    api -->|writes and reads the rows| entities
    api -->|reads the profiles and creates the job, as the caller| jobs
    jobs -->|runs| fetch & build & push
    fetch -->|reads, acting for the submitter| files
    build -->|creates, reads its log, then deletes| sandbox
    fetch -.->|writes the contexts| work
    push -.-|reads the layouts| work
    work <-.->|"its own context, read-only;<br/>writes its images' layouts"| sandbox
    secrets -->|fetched by Jobs, acting for the submitter| push
    push -->|pushes, and signs if signing is on| registry
    push -->|reads its rows, and completes them| api

    style trusted fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style untrusted fill:#ffebee,stroke:#c62828,color:#b71c1c
```

```mermaid
sequenceDiagram
    actor Caller
    participant API as Builder routes
    participant Entities
    participant Secrets
    participant Jobs
    participant Fetch as fetch
    participant Build as build
    participant Sandbox as sandbox
    participant Push as push
    participant Registry

    Caller->>API: POST /builds
    API->>Jobs: read the execution profiles, as the caller
    Note over API: check the request, the deployment and the profiles
    API->>Secrets: check the caller can read each secret the push step uses
    API->>Entities: write one pending row per image
    API->>Jobs: create the job, as the caller
    API-->>Caller: 201, the job and its rows
    Jobs->>Fetch: run
    Note over Fetch: copy each build context onto the work volume, acting for the submitter
    Jobs->>Build: run
    loop each build context, one at a time
        Build->>Sandbox: create a pod
        Note over Sandbox: kaniko --no-push writes an OCI layout per image
        Sandbox-->>Build: each image's result, in the pod's log
        Build->>Sandbox: delete the pod
    end
    Jobs->>Push: run, with the workspace's registry credential and signing key, if used
    loop each image
        Push->>API: GET /container-images/{name}
        API-->>Push: the row: go on only if it's pending
        Note over Push: validate the layout
        Push->>Registry: push the caller's tag and the system tag, checking the pushed digest
        opt signing is on
            Push->>Registry: sign the digest, and store the signature beside the image
        end
        Push->>API: POST /container-images/{name}/complete, with the digest
        API-->>Push: the row, ready
    end
    Caller->>API: GET /container-images/{name}
    API-->>Caller: ready, with its digest
```

1. **Submit.** The request is resolved into a job name and a row name per image, and each image is
   placed: registry, repository and system tag. Anything the deployment can't honor is refused
   before anything is written, including a push step secret the caller can't read, and execution
   profiles the sandbox can't follow. Then one `pending` row per image is written, then the job.
   Each row records a digest of the request, so the same request submitted again returns the same
   rows, and a different request under the same `name` and `revision` is refused.
2. **`fetch`** clears what an earlier run left behind, then copies each fileset, or just its
   `context_path`, onto the work volume. It calls Files as `service:builder`, acting for the
   submitter: a fileset in a workspace where the submitter has no role fails here, and so the whole
   set does. It downloads each archive to the work volume and unpacks it, once, with Python's
   `data` filter, which keeps every entry inside the archive's directory. It refuses an archive that
   unpacks to more than 4 GiB or 100,000 entries.
3. **`build`** first deletes any sandboxes an earlier attempt of the same job left. Then it creates
   one sandbox per distinct fileset, archive and `context_path`, one at a time, and deletes each
   when it ends. The sandbox runs kaniko
   once per image with `--no-push`, writing an OCI layout to that image's output directory. A plain
   pod sandbox has:
   - no ServiceAccount token, no secret, and no credential in its environment
   - a security context the `baseline` Pod Security Standard admits: root, but with only five
     capabilities, `RuntimeDefault` seccomp and no privilege escalation
   - public DNS resolvers rather than cluster DNS
   - a deadline the kubelet enforces, so it ends even if `build` is killed first

   An [OpenSandbox](#opensandbox) sandbox gets its security context and DNS from the server's
   template instead, and the server enforces its deadline.
4. **`push`** treats each layout as untrusted, and checks what crane will read of it: an index with
   exactly one image manifest, and the index, the manifest and every blob it names as regular files,
   not symlinks or FIFOs, under real directories. The registry checks each blob against its digest
   as crane uploads it. For each image whose row is still `pending`, it pushes the caller's tag and
   the system tag with crane, goes on only if crane reports pushing the digest the index names, signs
   that digest unless signing is off, and asks the builder to complete the image.
5. **Completing** writes the row `ready` at the digest the push step names, in one write.

One failed Dockerfile, or one sandbox that can't run, doesn't stop the rest of the set: the other
images still build, push and complete. The failed image's row stays `pending` (see
[Reliability](#reliability)). If every image in a set fails, `push` never runs.

| | Runs as | Holds | Work volume | Runs your code |
|---|---|---|---|---|
| `fetch` | `nhx-build-fetch` | a Files client, acting for the submitter | yes | no |
| `build` (runs `nhx-build supervise`) | `nhx-build-control` | permission to manage pods in the build namespace, and the OpenSandbox key with that provider | no | no |
| sandbox (kaniko) | no ServiceAccount token | nothing | its own context read-only, and one output directory per image | yes |
| `push` | `nhx-build-push` | the workspace's registry credential and signing key, unless the deployment turned them off | yes | no |

## Publishing and signing

The push step is the only step that holds the registry credential or the signing key. Before it
starts, Jobs fetches three platform secrets from the job's workspace, acting for the submitter, and
gives them to the push step as environment variables. The step reads them, then removes them from
its environment, so no tool it runs inherits them. They are named by the deployment's settings; by
default:

| Secret | Holds |
|---|---|
| `builder-registry-username` | The username the push step logs in to the registry with. |
| `builder-registry-password` | That username's password or token. |
| `builder-signing-key` | A PEM private key, EC or RSA, unencrypted, that the push step signs every image with. |

A deployment can turn off the credential, the key, or both (see [Configuration](#configuration)).
The push step then pushes anonymously, or skips signing, and workspaces don't need those secrets.

For each image, the push step:

1. **Pushes** the caller's tag, then the system tag, with crane. crane logs in as the registry asks:
   with the credential itself, or with a token from the registry's token service. crane reads the
   layout itself, so the step stops there unless crane reports pushing the digest the layout's
   index names.
2. **Signs** the digest it pushed, in cosign's format, every annotation taken from the image's row:
   its workspace, name, build set, revision, job and request digest. It stores the signature beside
   the image, under `sha256-<hex>.sig`, where `<hex>` is the digest's hash and where `cosign verify`
   looks for it. Nothing goes to a transparency log.
3. **Completes** the image: `POST /container-images/{name}/complete` with `{"digest": "sha256:…"}`.

Verify an image with the workspace's public key:

```bash
cosign verify --key signing.pub --insecure-ignore-tlog=true <registry>/<repository>@<digest>
```

Add `--allow-http-registry` for a plain-HTTP registry. A signature shows only that something holding
the workspace's key signed the image; see [Security](#security) for what can.

`complete` needs `builder.container-images.complete`, which the Editor role has. With platform auth
on, the caller must also act for the image's submitter: a user's own token is refused. That doesn't
show the caller is the image's push step (see [Security](#security)). Completing an image already
`ready` at that digest returns it, so a retry is safe. The builder doesn't read the registry: it
records the digest the push step names.

| Status | When |
|---|---|
| `200` | The image is `ready` at the digest, now or already. |
| `403` | The caller lacks the permission, or doesn't act for the image's submitter. |
| `404` | No such image. |
| `409` | The image has already settled: `ready` at another digest, or `failed`. |
| `422` | The body is anything but a sha256 digest. |

## Deploying

### Configuration

Operator settings live under `builder:` in the platform config, or in `NEMO_BUILDER_*` environment
variables. Callers can't set any of them.

```yaml
builder:
  sandbox:                                       # the pods your Dockerfiles run in
    image: my-registry/nhx-kaniko:v1.25.19       # built from docker/Dockerfile.kaniko
    provider: kubernetes_pod                     # or opensandbox: see OpenSandbox, below
  registry: us-central1-docker.pkg.dev           # a host only, never host/path; http://host for plain HTTP
  repository_prefix: my-project/my-repo          # images land at <prefix>/<workspace>/...
  # The push step's platform secrets, in the submitting workspace. These are the defaults; null turns one off.
  registry_username_secret: builder-registry-username
  registry_password_secret: builder-registry-password
  signing_key_secret: builder-signing-key
```

`registry` and `sandbox.image` have no default; while either is unset, submits fail with a `409`
rather than as builds that die in a pod. Set `repository_prefix` to a path dedicated to builds: left
empty, each workspace name is a top-level namespace in the registry. The other sandbox settings are
`opensandbox`, `cpu`, `memory` and `dns_nameservers`, each described on `SandboxConfig` in `config.py`. Where the
sandboxes run isn't a builder setting: it comes from the
[Jobs execution profiles](#jobs-execution-profiles). From the environment, the `sandbox` section is
one JSON value, `NEMO_BUILDER_SANDBOX`: only its one-word settings can be set on their own.

With `registry_username_secret` and `registry_password_secret` both `null`, the push step pushes
without logging in; set both or neither. With `signing_key_secret` `null`, it publishes images
unsigned. Both suit a local registry rather than a shared one: anything that can reach a registry
open to anonymous pushes can overwrite any workspace's images, and an unsigned image can be checked
only against the digest its row records.

`config/platform-config.minikube.yaml` is the quickstart's config. It works, but it is local-only in
the ways its header lists, auth being off among them; don't start a shared deployment from it.

### Each workspace's secrets

A workspace that builds needs the secrets in [Publishing and signing](#publishing-and-signing), all
three unless the deployment turned some off, created with `nemo secrets create` as in step 5 of the
quickstart. Until it has them, its submits fail with a `409` naming what is missing. Any
member who can run jobs in the workspace and read its secrets can read these, through a job of their
own, and so can anything inside the cluster (see [Security](#security)).

Give each workspace a registry credential that can write only its own part of the registry,
`<repository_prefix>/<workspace>/`, where the registry allows it: a Harbor project and robot
account, or an Artifactory permission on that path. Artifact Registry grants permissions per
repository, and a single `repository_prefix` puts every workspace in one repository, so there a
workspace's credential can write every workspace's images. A credential that can write more lets
that workspace's builds, and members, overwrite other workspaces' images. Until secrets are safe
inside the cluster, these limits don't hold against a plain pod build's `RUN`.

### Registries

The registry must accept nested repository paths and create a repository on its first push, as
Artifactory, GAR, Harbor and Distribution do. Docker Hub and nvcr.io don't: each repository there
must be created first, and names have a fixed depth. Known differences:

| Registry | Credential | Note |
|---|---|---|
| GAR | `_json_key` and a service account key, or `oauth2accesstoken` and an access token | An access token expires within an hour, so the secret needs refreshing. |
| Harbor | A robot account | A robot's permissions are per project. |
| Artifactory | A username and an identity token | Writes are slow: a manifest has taken two minutes to be accepted. |
| nvcr.io | `$oauthtoken` and an NGC API key | Repositories are exactly `<org>/<team>/<image>`, so the workspace must be named for the team and `repository_prefix` for the org. |

### Jobs execution profiles

The platform needs three Jobs execution profiles, one per step, named by `fetch_profile`,
`control_profile` and `push_profile` (`build-fetch`, `build-control` and `build-push` by default).
Each is a `cpu` profile on the `kubernetes_job` backend. The sandboxes run in the build step's
namespace, on the fetch step's work volume and `node_selector`, so the profiles must agree with each
other and with the manifests:

- all three name the same `namespace`, and the fetch and push profiles the same `storage.pvc_name`;
  until they do, submits fail with a `409`
- `service_account_name` is `nhx-build-fetch`, `nhx-build-control` or `nhx-build-push`
- `default_task_image` is the `nhx-build` image, and `launcher_image` an image that ships
  `/tools/jobs-launcher`, as the platform image does
- with a `ReadWriteOnce` work volume, every `node_selector` names the one node all build pods run on

### OpenSandbox

With `sandbox.provider: opensandbox`, each sandbox is an OpenSandbox sandbox rather than a pod `build`
creates. `build` asks the server for it, runs each image's build as a command in it, and takes the
command's exit code as the image's result. The step image carries the SDK, the plugin's
`opensandbox` extra. A deployment needs:

- **An OpenSandbox server, `v0.2.3` or later, with a tenant whose namespace is the build namespace.**
  Sandboxes mount the work volume, which can be mounted only in its own namespace. Multi-tenant mode
  keeps other tenants' keys out of that namespace.
- **The tenant's key** in the Secret `sandbox.opensandbox.api_key_secret` names
  (`opensandbox-builder-api-key` by default), under `api-key`, in the build namespace. `build` reads
  it with its ServiceAccount, which `deploy/builds.yaml` lets read that one Secret. So can a raw job
  under the `build-control` profile (see [Security](#security)).
- **A server template that hardens the sandbox.** OpenSandbox takes a sandbox's security context,
  DNS and nodes from its BatchSandbox template, which applies to every sandbox the server creates,
  not from the request; `dns_nameservers` and the fetch profile's `node_selector` don't apply. The
  plain pod's posture fits in it:

  ```yaml
  spec:
    template:
      spec:
        automountServiceAccountToken: false
        enableServiceLinks: false
        containers:
          - name: sandbox
            securityContext:
              allowPrivilegeEscalation: false
              seccompProfile: {type: RuntimeDefault}
              capabilities: {drop: ["ALL"], add: ["CHOWN", "DAC_OVERRIDE", "FOWNER", "SETUID", "SETGID"]}
  ```

  With a ReadWriteOnce work volume on more than one node, pin sandboxes to its node in the template too.
- **A kaniko image with `/bin/sh`,** which OpenSandbox's startup script runs; `docker/Dockerfile.kaniko`'s
  has it.
- **The server's egress sidecar,** set server-wide by `[egress]` in its config: an `image`, such as
  `docker.io/opensandbox/egress:v1.1.7`, and `mode = "dns+nft"`, so that rules can name domains.
- **The build namespace at the `privileged` Pod Security standard.** The egress sidecar needs
  `NET_ADMIN`, which `baseline` refuses, and the SDK then sees only a timeout. `deploy/builds.yaml`
  sets `baseline`, for the plain pod. With this provider, raise it, and drop the `pods` and `pods/log`
  rules from `build-control`'s Role, which only the plain pod uses:

  ```bash
  kubectl label namespace nhx-builds pod-security.kubernetes.io/enforce=privileged --overwrite
  ```

```yaml
builder:
  sandbox:
    image: my-registry/nhx-kaniko:v1.25.19
    provider: opensandbox
    opensandbox:
      domain: opensandbox-server.opensandbox-system.svc.cluster.local   # the server's Service
    egress_allow: ["*.com", "*.org", "*.io", "*.dev"]                  # the default
```

Each sandbox is created with a deny-by-default egress policy. It may reach what `egress_allow` names,
by default any name under `*.com`, `*.org`, `*.io` or `*.dev`, which covers the common registries
and package indexes. It can reach nothing in a private range, the cluster's and the metadata
server's included, unless an address in `egress_allow` is in it: not the platform, and not a
registry inside the cluster. `build` reads back the policy each sandbox reports before running
anything in it, and fails the sandbox's images unless its default is to deny.

## Limitations

### Deployment

- **Kubernetes only, with the platform inside the cluster.** Submits fail with a `409` unless the
  build's execution profiles are on Kubernetes. Build pods call back to Files, Jobs and the builder,
  so a control plane outside the cluster can't run builds.
- **One build node.** The work volume is `ReadWriteOnce`. A `ReadWriteMany` volume would lift this.
- **One registry per deployment.** Every image is published to `registry`, under the submitting
  workspace's path. Publishing anywhere else means copying the image out afterwards.
- **Each new request needs a new `revision`.** The same request again returns its rows; a different
  one under a used `name` and `revision` returns `409`.

### Reliability

- **Nothing fails a row whose build fails.** Its row stays `pending`. So do the rows of a submit
  that died, or that Jobs refused, between writing them and creating its job, and the rows of a job
  whose push step never started, because a secret was deleted after the submit checked it. Poll
  with a deadline. Submitting the same request again reuses its rows.
- **One unreadable fileset fails the whole set.** `fetch` copies every context in one step, so one
  that can't be read stops every image in the set.
- **A sandbox has an hour.** The images from one context build one after another in one sandbox.
  `build` stops watching it after an hour, and the kubelet, or the OpenSandbox server, ends it five
  minutes later; its unfinished images fail.
- **`build` gives up on a dropped watch.** If its watch on a plain pod sandbox closes early, the
  sandbox's unfinished images fail. A sandbox whose `build` was killed ends at its deadline, and the
  job's next attempt deletes it before building.
- **The work volume fills up.** Jobs deletes a job's directory only when the push step's profile
  sets `cleanup_completed_jobs_immediately`, and only after the job's last step succeeds. The
  quickstart's profiles don't set it, so every build's contexts and layouts stay until someone
  deletes them; a failed build's stay either way.
- **`fetch` reads each file of a fileset whole into memory.** A build context with very large files
  can exhaust the step's memory. An archive is streamed to the work volume instead.
- **An archive unpacks to at most 4 GiB and 100,000 entries.** A larger one fails the whole set,
  as an unreadable fileset does.

### Security

- **Anything inside the cluster can read every workspace's secrets.** With platform auth off, any
  request can. With it on, and workload token exchange off, a request that names a known service,
  such as `service:jobs`, in its `X-NHX-Principal-*` headers, and no user it acts for, gets any
  secret in any workspace: the platform takes those headers on trust. A plain pod sandbox's `RUN`
  reaches the platform, so a Dockerfile built in one can read every workspace's signing key and
  registry credential, sign as any workspace, and push to its repositories. With exchange on, the
  platform refuses those headers, but the builder doesn't work with exchange on yet (below).
- **A signature shows only that something holding the workspace's key signed the image.** Any
  member who can run a job that reads the key can sign any image with it, and so can anything that
  can read it (above). A signature doesn't show that the platform built the image.
- **With platform auth off, anything can complete or rewrite an image.** Nothing then checks who
  calls the platform, so anything that reaches it, a plain pod build's `RUN` included, can mark a
  `pending` image `ready` at a digest of its choosing, or rewrite any image's row. The quickstart
  runs with auth off.
- **With platform auth on, anything inside the cluster can still complete or rewrite an image.**
  `complete` takes any caller that says it acts for the image's submitter. Without workload token
  exchange, the platform takes a request's identity from its `X-NHX-Principal-*` headers whenever it
  has them, with no token behind them: a gateway is meant to strip them from outside traffic, but
  nothing strips them inside the cluster. So anything that reaches the platform, a plain pod build's
  `RUN` included, can mark any of the submitter's `pending` images `ready` at a digest of its choosing.
  One that names a service can also rewrite any image's row through the platform's generic Entities
  route, including a `ready` row's digest, which `complete` never changes. Even a real workload
  token says only whom a step acts for, not which job it belongs to.
- **Platform auth on works only with workload token exchange off.** With exchange on, the jobs
  launcher can't fetch the push step's secrets: it fetches them with the step's delegated identity,
  which Jobs doesn't give a step under exchange, so the step never starts. Without exchange, the
  push step calls the builder as `service:builder`, acting for the submitter, and the two routes it
  calls admit a service only when it acts for the image's submitter.
- **`fetch` unpacks archives the request names.** It runs acting for the submitter and parses an
  archive anyone who can write to the fileset put there. Python's `data` filter keeps every entry
  inside the archive's directory, and refuses links out of it and devices; the caps bound what one
  archive writes to the work volume every build on the node shares.
- **Execution profiles aren't authorized in Jobs.** Anyone who can submit a build can run a raw job
  under any of the builder's profiles. Under `build-control` such a job can create any pod in
  `nhx-builds`: it can mount the whole work volume, with every build's contexts and outputs, and
  rewrite a layout another job's push step is about to publish; read any build pod's log; and
  delete any build pod. With the `opensandbox` provider, it can also read the OpenSandbox tenant's
  key, and with it create sandboxes without an egress policy, mount any of the work volume, and run
  commands in any build's sandbox. Closing it needs Jobs to restrict these profiles to the builder.
- **Anything that reaches an OpenSandbox sandbox can run commands in it.** execd, OpenSandbox's
  agent in the sandbox, checks no key. `build` reaches it through the server, which checks the
  tenant's, but any pod that can reach a sandbox's address can run commands in it as root, and
  rewrite the layout a push step is about to publish. A NetworkPolicy in the build namespace that
  admits only the OpenSandbox server would close this, on a network plugin that enforces it:
  Flannel, for one, doesn't.
- **Nothing restricts a plain pod sandbox's network.** A Dockerfile's `RUN` in one can reach anything
  a pod can: the platform, as any user or service it names, every other Service, and on a cloud
  cluster the node's metadata server. Use the `kubernetes_pod` provider only with Dockerfiles you
  trust. An [OpenSandbox](#opensandbox) sandbox reaches only what `egress_allow` names, and nothing
  in a private range.
- **Nothing bounds a sandbox's resources.** It requests CPU and memory but has no limits, and
  `nhx-builds` has no LimitRange or quota, so one Dockerfile can take all of the build node's CPU,
  memory, processes and disk. In the quickstart, that node also runs the platform and the registry.
- **Images built from one context share a sandbox.** Each one's `RUN` can reach the others' outputs,
  so one image's Dockerfile, or a base image it pulls, can alter another image of the same build.
- **Base images must be pullable without credentials.** The sandbox holds none, and nothing bounds
  what a Dockerfile can pull `FROM`.

### Not yet supported

- **KMS keys.** The signing key is a PEM in the platform's Secrets service, and the push step holds
  it while it runs.
- **Signatures in a transparency log, or in cosign's newer formats.** Signatures are cosign's legacy
  `.sig` images.
- **Multi-platform images.** Each spec builds one platform.

## Development

The plugin isn't a uv workspace member. The root `pytest.ini` and `ty` config put its `src/` on the
path instead, so the repository's unit tests collect these tests, and nothing needs installing:

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
| `backend.py` | The backend interface: what a backend refuses, compiles and places |
| `backends.py`, `execution.py` | The deployment's backend, and the `execution` backend |
| `compile.py` | Turns a plan into the three-step `HelixJobSpec`, as a pure function |
| `submit.py` | The submit sequence: plan, secrets, rows, then the job |
| `completion.py` | Who may complete an image, and the one write that makes its row `ready` |
| `signing.py` | What a signature says, cosign's format for storing it, and the key that makes it |
| `registry.py` | The registry calls the push step makes to store a signature |
| `registry_auth.py` | Logging in to a registry with a username and password, as its challenge asks |
| `client.py` | A typed client for the builder's routes, which the push step uses |
| `steps.py` | What each step's config contains, the push step's environment, and `WorkLayout` |
| `entities.py` | `ContainerImage` |
| `identity.py` | Registry hosts, repository paths and tags, and the system tag |
| `config.py` | `BuilderConfig` |
| `run/` | The programs: `fetch.py`, `supervise.py` and `push.py`; `main.py`, the `nhx-build` entry point; `utils.py`, what the steps share; `tools.py`, which runs crane for `push`; and the sandbox providers, `sandbox.py` with `pod_sandbox.py`, and `opensandbox_sandbox.py` with its SDK calls in `opensandbox_sdk.py` |
| `docker/` | The `nhx-build` image, the sandbox's kaniko image, and the platform image with this plugin added |
| `deploy/` | The quickstart's manifests, one per namespace |
| `config/` | The platform config for the minikube quickstart |
