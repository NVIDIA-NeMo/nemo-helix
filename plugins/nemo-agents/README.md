<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-agents plugin

A NeMo Platform plugin for building, registering, deploying, and invoking agents
as first-class managed resources.

The plugin supports two agent flows:

- **Platform-backed agents** use the Platform-managed `nemo-agents-spec-v1`
  `agent.yaml` contract. This is the recommended flow for new agents.
- **NVIDIA Agent Toolkit (NAT) workflows** use the legacy `nat-workflow-v1`
  configuration format and remain supported for existing agents.

Both flows provide:

- **CRUD** — store and version agent configs in the platform entity store
- **Deployment** — start/stop agent servers via an in-memory controller
  (subprocess mode, default), or as durable containers via the `nemo-deployments`
  plugin (`--mode docker`, or `--mode k8s` when a k8s executor is configured;
  k8s runtime reachability is still evolving)
- **Gateway** — reverse-proxy agent traffic through `/apis/agents/…/-/…`
- **CLI** — `nemo agents` subcommand for platform-managed workflows
- **Packaging** — containerize agents with a single `nemo agents package` command that progressively renders, builds, and publishes

---

## Platform-backed agents

For new agents, use the Platform-managed configuration contract. It supports
shared instructions, models, skills, tools, MCP servers, environment settings,
telemetry, and a selectable harness. NeMo Agents validates that contract and
executes it through the selected harness.

### Prerequisites

| Requirement | Notes |
|---|---|
| Python | `>=3.11,<3.15` |
| NeMo Platform | Installed and running with an inference provider and model configured |
| Model credentials | Set the credentials required by the selected provider; the examples use `NVIDIA_API_KEY` |
| Harness CLI and authentication | Install and authenticate the selected harness when required; for example, run `codex login` for Codex or complete the Claude CLI login flow |
| NeMo Relay CLI | Required for Claude and Codex; install it with `script/dev-install-fabric.sh` after installing the plugin |
| Hermes runtime | Required only for Hermes; use a separate Python 3.12 environment and set `ADAPTER_PYTHON` as described in the [Hermes example](examples/nemo-agent-config/README.md#hermes) |

Install the plugin from the repository root, after `uv sync`. This installs
Fabric, the Relay Python bindings, and the supported harness adapters. The NeMo
Relay CLI and Hermes harness runtime remain separate as noted above.

```bash
uv pip install -e plugins/nemo-agents/
```

Verify it loaded:

```bash
nemo --help   # should show "agents" under Plugins
```

If you are using Claude or Codex, install and verify the NeMo Relay CLI:

```bash
script/dev-install-fabric.sh
nemo-relay --version
```

> **Working directory:** Platform-backed examples use paths relative to the
> repository root. Run them from there unless an example says otherwise.

### How this differs from legacy NAT

| Area | Recommended flow | Legacy NAT workflows |
|---|---|---|
| Recommended use | New agents and multi-harness Platform flows | Existing NAT workflow integrations |
| Config contract | Platform-managed `nemo-agents-spec-v1` `agent.yaml` | NAT `nat-workflow-v1` workflow YAML |
| Runtime | Platform-managed execution through supported harness adapters | NVIDIA Agent Toolkit runtime |
| Local execution | Invoked through the Platform agent runtime | Delegated to `nat run` |
| Platform lifecycle | `nemo agents create`, `deploy`, and `invoke` | The same Platform lifecycle commands |
| Packaging | Automatically selects the Platform agent image pipeline | Automatically selects the NAT image pipeline |
| Compatibility status | First-class flow | Supported legacy flow |

---

### Calculator agent demo — DeepAgents + Relay

[`examples/nemo-agent-config/calculator-agent/agent.yaml`](examples/nemo-agent-config/calculator-agent/agent.yaml)
uses DeepAgents as its harness and routes `nvidia-nemotron-3-nano-30b-a3b`
through the Platform Inference Gateway. The agent answers arithmetic and
numeric comparison requests and records ATIF and ATOF telemetry with NeMo
Relay.

#### Step 1 — Configure and start the platform

Set the NVIDIA API key and local Platform URL from the repository root:

```bash
export NVIDIA_API_KEY="<your NVIDIA API key>"
export NMP_BASE_URL=http://localhost:8080
```

Start ClickHouse for Intake, then set up NeMo Platform without deploying the
default demo agent:

```bash
services/intake/scripts/spans/run_clickhouse.sh
nemo setup --auto --start-services --install-skills --no-deploy-agent
```

Confirm that the Platform is ready before continuing:

```bash
curl -fsS --connect-timeout 2 --max-time 5 \
  "$NMP_BASE_URL/health/ready" >/dev/null || {
  echo "NeMo Platform is not ready at $NMP_BASE_URL"
  exit 1
}
```

#### Step 2 — Create and deploy the agent

```bash
nemo agents create \
  --name calculator-agent \
  --agent-config plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml

nemo agents deploy \
  --agent calculator-agent \
  --name calculator-agent-deployment \
  --mode subprocess
```

`create` validates the config and registers the agent. `deploy` waits for the
deployment to reach `running` by default.

#### Step 3 — Invoke through the gateway

```bash
nemo agents invoke \
  --agent-deployment calculator-agent-deployment \
  --input "What is 144 divided by 12?"
```

The response content should be `12`.

#### Step 4 — Verify Relay telemetry

The config writes ATOF events beneath the deployment's artifacts directory:

```bash
find ~/.local/share/nemo/agents/system/default \
  -path "*calculator-agent-deployment*/artifacts/*" \
  -name "*.atof.jsonl" \
  -exec ls -lh {} \;
```

### Packaging agents as container images

`nemo agents package` automatically detects `nemo-agents-spec-v1` and selects
the Platform agent image pipeline. Packaging creates a runnable image and
bundles its runtime dependencies and ancillary files. It does not register,
create, or deploy the agent on Platform.

The packaging command runs locally; Platform services are not required.

| Requirement | Notes |
|---|---|
| Docker | A running Docker-compatible daemon |
| Container dependencies | Install with `uv sync --package nemo-agents-plugin --extra container` from the repository root |
| A released `nemo-platform` (Fabric only) | Fabric images pin the installed `nemo-platform` version and resolve it from an index. A source checkout reports a setuptools-scm version such as `0.3.0.post402.dev0+062f0ac6e8`, which no index serves, so packaging stops before building. See below. |

##### Packaging a Fabric agent from a source checkout

`nemo agents package` renders `uv pip install "nemo-platform[nemo-agents-plugin]==<version>"`
into the Fabric image, where `<version>` is whatever is installed on the build
host. A checkout reports something like `0.3.0.post402.dev0+062f0ac6e8`, which is
both a developmental release and a local build identifier — neither of which a
public index serves — so the command fails immediately:

```text
Error: The installed nemo-platform version '0.3.0.post402.dev0+062f0ac6e8' carries
a local build identifier and is a developmental release, so no package index serves
it. Fabric packaging pins this exact version inside the image, so the build would
fail while resolving it. Install a released nemo-platform to package an agent, or
set NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION=1 if your index serves this
version.
```

To package from a checkout, build a wheel and point `NEMO_AGENTS_WHEEL` at it.
The image installs that wheel instead of resolving the pin, so the version never
has to be one an index can serve:

```bash
uv build --package nemo-platform --wheel --out-dir dist && \
NEMO_AGENTS_WHEEL=LATEST nemo agents package \
  --agent plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml \
  --tag calculator-agent:local
```

`LATEST` resolves to the newest wheel in the checkout's `dist`, so the variable
survives the rebuild each commit forces — the version carries the commit, so a
wheel built before your last commit no longer describes your tree. An absolute
path still works and takes precedence, so a file actually named `LATEST` is
never mistaken for the sentinel. If the wheel's version differs from the one
this host runs, the build says so and continues: the wheel is installed instead
of the pin, so the pin never has to resolve.

It is an environment variable rather than a flag because packaging also runs as
a platform job, and a flag would only ever reach the CLI. The subprocess job
inherits it, so exporting it before `nemo services run` covers Studio-triggered
builds too:

```bash
export NEMO_AGENTS_WHEEL=LATEST
uv run nemo services run --service-group all --controllers jobs --port 8080
```

Packaging copies the wheel into the build context for the duration of the build
and removes it afterward. It does not overwrite an existing file with that name.
It applies to Fabric packaging only — NAT images install the packaged project
itself.

Supplying your own `--template` also skips the check, since a custom template
need not pin the contract version at all.

NAT packaging is unaffected — it installs the packaged project plus a published
`nvidia-nat` release.

#### Progressive pipeline

| Invocation | Stages | Result |
|---|---|---|
| `package --no-build` | Render | `Dockerfile` and, by default, `.dockerignore` |
| `package` | Render → validate → build | Local image |
| `package --publish --registry <registry>` | Render → validate → build → publish | Local image and registry push |

`--no-build` only renders files; it does not run package validation.

#### Package the calculator agent

Render a Dockerfile without building an image:

```bash
nemo agents package \
  --agent plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml \
  --no-build
```

This writes `Dockerfile` and `.dockerignore` beside `agent.yaml`. Build and tag
the image directly with:

```bash
nemo agents package \
  --agent plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml \
  --tag calculator-agent:local
```

To build and publish in one operation:

```bash
nemo agents package \
  --agent plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml \
  --tag calculator-agent:1.0.0 \
  --publish \
  --registry nvcr.io/my-org
```

If the agent is part of a Python project, enable project mode by passing its
`pyproject.toml`:

```bash
nemo agents package \
  --agent plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml \
  --pyproject pyproject.toml \
  --tag calculator-agent:local
```

Use `--skip-validation` only when a separate trusted pipeline has already
validated the exact inputs being packaged:

```bash
nemo agents package \
  --agent plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml \
  --tag calculator-agent:local \
  --skip-validation
```

This flag bypasses strict config loading, runtime translation and planning, and
referenced-artifact checks. It can produce an image that fails at startup.

#### Packaging an agent that already lives on the platform

The CLI above packages a directory on your machine. For an agent created
through Studio or `nemo agents create`, the source of truth is the
`{agent}-spec` fileset instead, and the `agents.package` job builds from that:

```bash
curl -X POST "$NMP_BASE_URL/apis/agents/v2/workspaces/default/jobs/package" \
  -H 'Content-Type: application/json' \
  -d '{"spec": {"agent": "my-agent", "tag": "my-agent:1.0"}}'
```

Poll it like any other platform job:

```bash
curl "$NMP_BASE_URL/apis/agents/v2/workspaces/default/jobs/package/<job>/status"
curl "$NMP_BASE_URL/apis/agents/v2/workspaces/default/jobs/package/<job>/logs"
```

The tag to hand to `nemo agents deploy --image` is published as a
`package_result` job result:

```bash
curl "$NMP_BASE_URL/apis/agents/v2/workspaces/default/jobs/package/<job>/results"
```

```json
{"image": "nemo-agents/default/my-agent:1.0", "agent": "my-agent", "published": ""}
```

Add `registry` to push the built image, which is what `--mode k8s` needs since a
cluster cannot pull from the build host's local daemon.

**The platform host must already be authenticated to the registry** — run
`docker login` there once, as an operator, before submitting a push. Credentials
are deliberately not accepted over this API, so every workspace pushing from a
given host shares that host's identity. Per-workspace credentials would need the
secrets service and are not wired up.

```bash
-d '{"spec": {"agent": "my-agent", "registry": "nvcr.io/my-org"}}'
```

`published` then carries the remote tag. The default remote reference is
`<registry>/<image>` — and because the local image is namespaced, that is
`<registry>/nemo-agents/<workspace>/<tag>`. Set `push_tag` to rename the
destination; it requires `registry` and must start with
`<registry>/nemo-agents/<workspace>/` — otherwise it could overwrite another
workspace's image, or redirect the push to a registry other than the one
declared.

The job downloads the agent's spec fileset into a temporary build context,
writes `agent.yaml` from the stored config, and runs the same Fabric build the
CLI runs.

From the CLI the same job is `nemo agents package-agent`. It is deliberately
*not* named `package`: the generated job command mounts onto the same Typer app
that already owns `nemo agents package`, and would shadow the local packaging
flags above.

| Limitation | Detail |
|---|---|
| Host build | The step runs as a **host subprocess**, not in a container — the Fabric Dockerfile needs a real Docker CLI for its BuildKit cache mounts. Submissions are rejected at POST time wherever no subprocess execution profile is registered (notably `runtime = kubernetes`). |
| Fabric only | `nemo-agents-spec-v1` agents only. NAT workflows build from a source checkout, so they stay on the CLI. |
| Shared registry identity | Pushing uses the platform host's own `docker login`, so it is not per-workspace. Omit `registry` to leave the image in the host daemon. `push_tag` must start with `<registry>/nemo-agents/{workspace}/` so one workspace cannot overwrite another's tag or redirect to a different registry on that shared host, but the underlying registry credentials remain shared across all workspaces on the host. |
| Constrained base image | `base_image_url`, `base_image_tag`, `python_version`, and `uv_version` are interpolated into the Dockerfile unescaped, so the API restricts them to a strict grammar: an optional registry host with port followed by `/`-separated path components for `base_image_url`, `[A-Za-z0-9_][A-Za-z0-9._-]{0,127}` for `base_image_tag`, and up to three dot-separated numbers for the two versions. `PackageAgentInput` in `openapi/openapi.yaml` is the source of truth. The CLI, whose caller already owns the host, is unrestricted. |
| Namespaced tags | Every image lands at `nemo-agents/{workspace}/{tag}`, derived or submitted, and `tag` may not contain `/`. Docker tags are daemon-global while the auth boundary here is the workspace, so an unqualified tag would let one workspace repoint another's image on the shared host. A workspace whose name is outside the Docker path grammar (entity names still allow `@` and `+`) is rejected at POST. |
| Managed `.dockerignore` | A `.dockerignore` in the spec fileset is discarded. Validation reads the staged tree off disk and Docker applies exclusions afterwards, so a user-supplied one could exclude `agent.yaml` or a referenced skill, pass validation, build, and fail only at container start. The CLI still honours a `.dockerignore` you wrote yourself. |

#### `agent.yaml` validation

Before a build, Platform agent packaging:

1. Loads `agent.yaml` using the strict `nemo-agents-spec-v1` schema.
2. Translates the Platform-managed config into the runtime configuration.
3. Runs the runtime planner to verify that it can resolve the execution plan.
4. Validates every referenced artifact included in the build context.

Packaging intentionally does not run harness environment checks. The machine
building an image does not necessarily have the harness binaries,
authentication, or Relay CLI that the resulting image will contain.

#### Build context and referenced artifacts

The build context is selected automatically:

| Mode | Build context | Config path in the image |
|---|---|---|
| Config-only | Parent directory of `agent.yaml` | `/workspace/<agent.yaml filename>` |
| Project | Parent directory of `pyproject.toml` | Original path relative to the project root under `/workspace` |

The complete context is copied to `/workspace`, so files referenced relative to
`agent.yaml` retain the same relationship inside the image.

Each `skills.paths` entry must:

- Be relative to `agent.yaml`.
- Resolve inside the selected build context.
- Reference a directory.
- Contain a readable `SKILL.md`.

#### Flag reference

**Pipeline control:**

| Flag | Default | Description |
|---|---|---|
| `--no-build` | `False` | Render the Dockerfile and `.dockerignore` without validating or building |
| `--publish` | `False` | Tag and push the built image |
| `--registry`, `-r` | None | Registry required by `--publish` |
| `--push-tag` | `<registry>/<tag>` | Override the fully qualified published tag |

**Source inputs:**

| Flag | Default | Description |
|---|---|---|
| `--agent`, `-c` | Required | Path to an agent YAML file; its format selects the packaging pipeline |
| `--pyproject` | None | Path to `pyproject.toml`; enables project mode |
| `--format` | `docker` | Packaging format; `whl` is reserved but not implemented |
| `--dockerfile` | Render automatically | Use an existing Dockerfile instead of rendering one |
| `--template` | Built-in | Use an external Jinja2 Dockerfile template |

**Build options:**

| Flag | Default | Description |
|---|---|---|
| `--tag`, `-t` | `<agent-name>-<agent-id>:<agent-version>` | Local image tag |
| `--platform` | Docker daemon's native platform | Target one platform, such as `linux/amd64`; multi-platform builds are not implemented |
| `--output`, `-o` | `<context>/Dockerfile` | Rendered Dockerfile path used with `--no-build` |
| `--skip-validation` | `False` | Bypass format-specific package validation |
| `--base-image-url` | `NEMO_AGENTS_BASE_IMAGE_URL` or built-in default | Base image repository |
| `--base-image-tag` | `NEMO_AGENTS_BASE_IMAGE_TAG` or built-in default | Base image tag |
| `--python-version` | `NEMO_AGENTS_PYTHON_VERSION` or built-in default | Python version installed in the image |
| `--uv-version` | `NEMO_AGENTS_UV_VERSION` or built-in default | uv release copied into the image |
| `--sandbox-runtime` | None | Add compatibility packages and users required by a supported sandbox runtime |

**Hardening overrides:**

| Flag | Default | Description |
|---|---|---|
| `--allow-root` | `False` | Disable the non-root runtime user |
| `--no-ignore` | Generates `.dockerignore` | Do not generate `.dockerignore` |

**OCI metadata overrides:**

| Flag | Default | Description |
|---|---|---|
| `--agent-version` | Project version or `YY.MM.DD` | Override the image and agent version |
| `--agent-author` | `git config user.name` | Override the image author |

The old shared environment variables have been replaced:

| Previous variable | Current variable |
|---|---|
| `NAT_BASE_IMAGE_URL` | `NEMO_AGENTS_BASE_IMAGE_URL` |
| `NAT_BASE_IMAGE_TAG` | `NEMO_AGENTS_BASE_IMAGE_TAG` |
| `NAT_PYTHON_VERSION` | `NEMO_AGENTS_PYTHON_VERSION` |

Additional environment variables:

| Variable | Effect |
|---|---|
| `NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION` | Set to `1` to let Fabric packaging pin an unpublished `nemo-platform` version (a local build identifier or a `.dev` release). Use only when the build's index serves that version. Does not apply when `nemo-platform` is not installed at all. |

There are no compatibility aliases. `NAT_VERSION` remains available only for
NAT workflow packaging.

#### Image tagging convention

When `--tag` is omitted, the image tag is:

```text
<agent-name>-<agent-id>:<agent-version>
```

| Component | Resolution |
|---|---|
| Agent name | `pyproject.toml` project name → Platform config `name` → config filename stem |
| Agent version | `--agent-version` → `pyproject.toml` project version → current date as `YY.MM.DD` |
| Agent ID | First 12 characters of a SHA-256 digest over the config, optional project file, and runtime/toolchain inputs |

Changing the config, project metadata, runtime contract, Relay version, base
image, Python version, or uv version changes the content-addressed agent ID.

#### OCI image labels

Generated Dockerfiles include standard OCI annotations:

| Label | Value |
|---|---|
| `org.opencontainers.image.title` | Resolved agent name |
| `org.opencontainers.image.version` | Resolved agent version |
| `org.opencontainers.image.authors` | Explicit author → Git user → `unknown` |
| `org.opencontainers.image.created` | Reproducible build timestamp when available |
| `org.opencontainers.image.description` | Project description → Platform config description → runtime-specific fallback |
| `org.opencontainers.image.revision` | Git revision when available |
| `org.opencontainers.image.source` | Git remote when available |
| `org.opencontainers.image.licenses` | Project SPDX license when available |

Agent-specific labels are:

| Label | Platform agent value |
|---|---|
| `com.nemo.agent.id` | Content-addressed 12-character ID |
| `com.nemo.agent.framework` | `nemo_platform_agent` |
| `com.nemo.agent.contract-version` | Release-matched packaging contract |

NAT images add a NAT version label as described in the NAT-specific packaging
section below.

#### Security defaults

| Default | Override |
|---|---|
| Non-root `agent` user with uid 1000 | `--allow-root` |
| `apt-get --no-install-recommends` and package-list cleanup | None |
| `.dockerignore` excludes credentials, Git data, caches, virtual environments, and build outputs | `--no-ignore` |

#### Rendering modes

Both modes copy the full build context to `/workspace`. Their Platform runtime
installation differs:

| Mode | Trigger | Install strategy |
|---|---|---|
| Config-only | No `--pyproject` | Install the release-matched `nemo-platform[nemo-agents-plugin]` runtime |
| Project | `--pyproject` provided | Install the release-matched runtime and the project together |

The image includes the supported harness adapters and dependencies, matching
NeMo Relay CLI and Python binding version `0.7.3`, a non-root `agent` user, and
the packaged agent server on port `8000`. The Hermes adapter is installed, but
the Hermes harness runtime remains excluded until its Python dependency
constraint is resolved.

#### Deploy the packaged calculator image

The calculator agent created in the preceding demo can be deployed from the
local image:

```bash
nemo agents deploy \
  --agent calculator-agent \
  --name calculator-agent-docker \
  --mode docker \
  --image calculator-agent:local \
  --timeout 300

nemo agents deployments get calculator-agent-docker

nemo agents invoke \
  --agent-deployment calculator-agent-docker \
  --input "What is 144 divided by 12?" \
  --timeout 300
```

Docker mode requires a configured Docker executor. Kubernetes mode uses the
same flow with `--mode k8s` and a registry-reachable image.

When `nemo agents create` registers a Platform-managed agent, it uploads the
directory containing `agent.yaml` to the `{agent-name}-ethos` fileset. Docker and
Kubernetes deployments stage that fileset beside `agent.yaml`, preserving
referenced skills and other text artifacts. If the fileset is unavailable, the
deployment falls back to the inline `agent.yaml`; referenced artifacts must then
be absent or deployment validation fails.

### Inspecting agent logs

For local subprocess deployments, `nemo agents logs` reads the runtime log
written by the in-memory runner. Use the agent name to select its most recent
deployment, or pass a deployment name directly:

```bash
# Print the full log for the calculator agent's latest deployment
nemo agents logs --agent calculator-agent

# Print the log for a specific deployment
nemo agents logs calculator-agent-deployment

# Tail the last 100 lines
nemo agents logs --agent calculator-agent --tail 100

# Follow new output until interrupted with Ctrl-C
nemo agents logs --agent calculator-agent --follow

# Print only the absolute log path
nemo agents logs --agent calculator-agent --path
```

Logs are stored under the Platform user-data directory:

```text
$NMP_DATA_DIR/agents/system/<workspace>/<deployment-name>.log
```

`$NMP_DATA_DIR` resolves first from the explicit environment variable, then
from `$XDG_DATA_HOME/nemo`, and finally to `~/.local/share/nemo`. The CLI must
run on the same host as the Platform service to read these local files.

### Model names

The `model` field in `agent.yaml` must use the Inference Gateway (IGW) entity
name. The models controller creates these names by normalizing slashes and dots
to hyphens:

| Provider model name | IGW entity name |
|---|---|
| `nvidia/nemotron-3-nano-30b-a3b` | `nvidia-nemotron-3-nano-30b-a3b` |

The calculator config therefore declares:

```yaml
models:
  default:
    provider: nvidia
    model: nvidia-nemotron-3-nano-30b-a3b
    api_key_env: NVIDIA_API_KEY
```

At runtime, the Platform supplies the Inference Gateway connection details for
that registered model.

### Cleanup (optional)

To remove the resources created during the calculator walkthrough:

```bash
nemo agents undeploy --agent calculator-agent
nemo agents delete calculator-agent

nemo inference providers delete nvidia-build
nemo secrets delete ngc-api-key
```

Stop the local Platform process with `Ctrl-C` in its terminal.

---

## NVIDIA Agent Toolkit (NAT) workflows

### Legacy NAT workflow compatibility

The existing NAT workflow experience remains available for agents authored as
`nat-workflow-v1` YAML. The material below is retained as the compatibility
guide while the recommended flow becomes the primary user-facing walkthrough.

### Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.11 | |
| NVIDIA Agent Toolkit runtime | installed by this plugin as `nvidia-nat-core >= 1.5.0, < 2.0` and `nvidia-nat-langchain >= 1.5.0, < 2.0` |
| NVIDIA Agent Toolkit eval/optimizer | installed by this plugin as `nvidia-nat-eval >= 1.5.0, < 2.0` and `nvidia-nat-config-optimizer >= 1.5.0, < 2.0` |
| NVIDIA API key | set `NVIDIA_API_KEY` |

Install the plugin from the repo root, after `uv sync`. This also installs the
required NAT runtime, eval, and config optimizer subpackages; no separate
`nvidia-nat[most]` install is needed.

```bash
uv pip install -e plugins/nemo-agents/
```

Verify it loaded:

```bash
nemo --help   # should show "agents" under Plugins
nat --help    # should show run, eval, optimize, start, …
```

> **Working directory:** All example commands that reference `examples/` use
> paths relative to the plugin directory.  Run them from `plugins/nemo-agents/`:
>
> ```bash
> cd plugins/nemo-agents/
> ```

---

### ReAct agent demo — Wikipedia search + datetime tools

`examples/react-agent.yml` uses `meta/llama-3.1-70b-instruct` with:

- `wiki_search` — searches Wikipedia (no API key needed)
- `current_datetime` — returns current UTC time

When deployed via the platform, the Inference Gateway URL is injected
automatically into the agent config — you only need to:

1. Create an `nvidia-build` inference provider pointing at NVIDIA Build
2. Create the agent and deploy it
3. Invoke through the gateway

#### NAT Step 1 — Start the platform

Run this in a **dedicated terminal** — it stays in the foreground.  Use a
separate terminal for all subsequent steps.

```bash
nemo services run
```

#### NAT Step 2 — Create an inference provider

In a new terminal, export the base URL once so all subsequent `nemo` commands
pick it up automatically:

```bash
export NMP_BASE_URL=http://127.0.0.1:8080
cd plugins/nemo-agents/
```

In production the `system/nvidia-build` provider is created automatically by
the platform seed job. For local development, create it manually:

```bash
# Store the API key as a secret
nemo secrets create ngc-api-key \
    --value "$NVIDIA_API_KEY"

# Create the model provider
nemo inference providers create nvidia-build \
    --host-url https://integrate.api.nvidia.com \
    --api-key-secret-name ngc-api-key
```

Wait for the models controller to discover served models and register model
entities:

```bash
nemo wait inference provider nvidia-build
```

#### NAT Step 3 — Create and deploy the agent

```bash
# Register the agent config with the platform
nemo agents create \
    --name react-agent \
    --agent-config examples/react-agent/react-agent.yml

# Deploy it.  ``deploy`` waits for the spawned subprocess to reach a
# terminal state (``running`` or ``failed``) by default and exits 0 only
# when the agent is actually serving — so the exit code reflects the
# real outcome instead of just "the API call succeeded".
nemo agents deploy --agent react-agent

# Container mode (docker): requires the nemo-deployments controller plus a
# configured docker executor (see agents.deployments / deployments.executors).
# Build an image first, then deploy with that tag:
#   nemo agents package --agent-config examples/react-agent/react-agent.yml --tag react-agent:local
#   nemo agents deploy --agent react-agent --mode docker --image react-agent:local
#
# --mode k8s needs a k8s executor and a registry-reachable image; in-cluster
# inference-gateway wiring is still evolving — prefer docker for local smoke.
```

The deploy command prints a status line each time the deployment changes
state:

```
Waiting for deployment 'react-agent-e5e29e05' (timeout=300s)...
  [  0s] status: pending
  [  1s] status: starting
  [ 38s] status: running
Deployment 'react-agent-e5e29e05' is running at http://127.0.0.1:49152
```

If the subprocess dies during startup, the command exits 1 with the failure
reason from the deployment entity (e.g. ``Process exited with code 1``).
Use ``nemo agents logs --agent react-agent`` to inspect the subprocess log
afterwards (see [Inspecting agent logs](#inspecting-agent-logs)).

For scripted pipelines that prefer to poll separately, pass ``--no-wait``
to restore the legacy fire-and-forget behaviour:

```bash
nemo agents deploy --agent react-agent --no-wait
nemo agents deployments wait --agent react-agent
```

#### NAT Step 4 — Invoke through the gateway

```bash
nemo agents invoke \
    --agent react-agent \
    --input "Who invented the telephone? Also, what time is it right now?"
```

Expected response:
```json
{
  "choices": [{
    "message": {
      "content": "Alexander Graham Bell invented the telephone. The current time is 2026-03-23 23:17:08 +0000.",
      "role": "assistant"
    }
  }]
}
```

The gateway URL is:
```
http://127.0.0.1:8080/apis/agents/v2/workspaces/default/agents/react-agent/-/v1/chat/completions
```

You can call it directly with any OpenAI-compatible client using the same path.

Requests without ``X-Nemo-Session-Id`` use a one-shot Fabric runtime that is
stopped when the response or response stream completes. To retain runtime
context across turns, send a stable session ID in that header; the registered
runtime then follows the Platform session lifecycle.

The agent is still running — continue to the [Evaluation](#evaluation) section
below, or see [Cleanup](#cleanup-optional) to tear everything down.

---

### Evaluation

Evaluation delegates to `nat eval`, which sends dataset questions to the
agent's `/generate/full` endpoint and scores responses with a judge LLM.

The agent must be deployed and running (see NAT Step 3 above) before evaluating.

```bash
nemo agents evaluate \
    --eval-config examples/test-eval.yml \
    --agent react-agent
```

The `--agent` flag resolves the running deployment endpoint automatically and
passes it to `nat eval --endpoint`.

Expected output:
```
=== EVALUATION SUMMARY ===
Workflow Status: COMPLETED (workflow_output.json)
Total Runtime: ~1.8s

Per evaluator results:
| Evaluator   |   Avg Score | Output File         |
|-------------|-------------|---------------------|
| runtime     |        ~0.9 | runtime_output.json |
```

A non-zero `Avg Score` and `Total Runtime` confirms requests reached the agent
successfully.  (The `avg_workflow_runtime` metric reports average seconds per
request, so the score varies with network latency.)

#### LLM-judge evaluation (requires a judge LLM)

`examples/calculator-agent/calculator-eval.yml` uses `tunable_rag_evaluator`
with an LLM judge. The judge's `model_name` is `${NEMO_DEFAULT_MODEL}`, which
resolves to whichever model your platform context has set as the default
(see `nemo_platform.config.get_context().default_model`); `base_url` and
`api_key` are auto-injected by the platform to route through the Inference
Gateway. Set the env var, or edit `llms.judge_llm.model_name` to pin a
specific VirtualModel registered in your workspace, then run:

```bash
export NEMO_DEFAULT_MODEL=nvidia-nemotron-3-super-120b-a12b   # or any registered VirtualModel
nemo agents evaluate \
    --eval-config plugins/nemo-agents/examples/calculator-agent/src/calculator_agent/calculator-eval.yml \
    --agent calculator-agent
```

The job pre-flights every LLM `model_name` against
`sdk.inference.virtual_models.retrieve` before invoking `nat eval`, so a
missing or mistyped model fails fast with a message naming the model and
suggesting recovery options instead of an opaque subprocess error.

---

### Packaging NAT workflows

NAT workflows use the same progressive pipeline, flags, build-context rules,
tagging convention, OCI labels, and security defaults described in
[Packaging agents as container images](#packaging-agents-as-container-images).
When `nemo agents package` detects a `nat-workflow-v1` config, it selects the
NAT image pipeline automatically.

Pass `--nat-version` to make the installed NAT runtime reproducible. The value
defaults to `NAT_VERSION` and then to the CLI's built-in version (`1.8.0`). This
option is valid only for NAT workflows.

From `plugins/nemo-agents/`, build the ReAct example with:

```bash
nemo agents package \
  --agent examples/react-agent/react-agent.yml \
  --nat-version 1.8.0 \
  --tag react-agent:local
```

#### NAT workflow validation

Before building, the shared validation stage applies these NAT-specific checks:

- The file is valid YAML that parses to a mapping.
- The top-level `workflow` key exists and is a mapping.
- `workflow._type` is present and non-empty. An unrecognized workflow type
  emits a warning so workflows supplied by additional NAT plugins can proceed.
- Every name in `workflow.tool_names` is defined in `functions` or
  `function_groups`.
- `workflow.llm_name` is defined in `llms`.

Multiple errors are collected and reported together. Use `--skip-validation`
only when you deliberately need to bypass these checks.

#### NAT image differences

The selected NAT image pipeline changes the generated image contract:

| Area | NAT behavior |
|---|---|
| Config-only install | `uv pip install "nvidia-nat[most]==${NAT_VERSION}"` |
| Project install | `uv pip install .` |
| Runtime command | `nat serve` starts the packaged workflow |
| Framework label | `com.nemo.agent.framework="nemo_agent_toolkit"` |
| Version label | `com.nemo.agent.nat-version=<NAT_VERSION>` |

The shared `NEMO_AGENTS_*` build variables also apply to NAT images. The
NAT-specific `NAT_VERSION` variable remains available for selecting the
runtime version.

### Agent config format

Agent configs are standard NAT workflow YAML files. The platform stores them
as `nat-workflow-v1` entities. All NAT component types are supported.

**ReAct agent with tools** (`examples/react-agent.yml`):

```yaml
functions:
  wiki:
    _type: wiki_search           # Wikipedia search, no API key
  clock:
    _type: current_datetime      # current UTC time

llms:
  llm:
    _type: openai
    api_key: not-used            # injected by platform at deploy time
    model_name: nvidia-nemotron-3-nano-30b-a3b  # IGW entity name
    temperature: 0.0

workflow:
  _type: react_agent
  tool_names: [wiki, clock]
  llm_name: llm
  parse_agent_response_max_retries: 3
```

#### base_url injection

When the controller deploys an agent, it calls `inject_gateway_url()` which
sets `base_url` via `setdefault` on each `openai`/`nim` LLM in the config.
**Do not set `base_url` in configs intended for platform deployment** — leave
it absent so the injected gateway URL takes effect.

The injected URL format:
```
{NMP_BASE_URL}/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1
```

---

### Performance tips

#### First-deploy cold start

The first `nemo agents deploy` after installing packages is noticeably slower
than subsequent deploys because Python compiles `.pyc` bytecache files on first
import. Pre-compiling NAT's dependencies eliminates this overhead:

```bash
python -m compileall -q $(python -c "import nat; print(nat.__path__[0])") 2>/dev/null
python -m compileall -q .venv/lib/ 2>/dev/null
```

This can cut 20--40 seconds off the first deploy.

---

### Notes and known limitations

- **`tool_calling_agent`** is broken with `langchain-openai==1.1.x` due to a
  missing `_DirectlyInjectedToolArg` import. Use `react_agent` instead.

- **`nat eval --endpoint` payload mismatch**: `nat eval` sends
  `{"input_message": query}` to `/generate/full`, but NAT's own
  `nat start fastapi` server expects `{"query": ...}` for `chat_completion`
  and similar workflow types.  This causes 422 errors on every request when
  `--endpoint` points at a locally-run agent server.  Evaluation via
  `--endpoint` is only reliable against a platform-deployed agent (where the
  gateway handles the translation).

- **IPv6 / localhost**: Start the platform with
  `NMP_BASE_URL=http://127.0.0.1:8080` to ensure agent subprocess processes
  can reach the platform. Python's `httpx` resolves bare `localhost` to IPv6
  `::1` on macOS, which does not match an IPv4-only listener.
