<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor environments

Read this when Harbor is the provider. It implements Steps 3 and 5 through 7 of
[the environment sub-flow](../SKILL.md) with Harbor's task format. Check the
installed Harbor's help and task schema before using a field; this guide was
checked against Harbor 0.20.0, and the
[Harbor environment documentation](https://docs.harborframework.com/core-concepts/tasks/environment)
describes the current release.

## Contents

- [Layout and reuse](#layout-and-reuse)
- [Sidecar services](#sidecar-services)
- [Point the agent at its backends](#point-the-agent-at-its-backends)
- [Tools the environment provides](#tools-the-environment-provides)
- [Network policy](#network-policy)
- [Export the end state](#export-the-end-state)
- [Reset](#reset)
- [Smoke task](#smoke-task)
- [Version notes](#version-notes)

## Layout and reuse

A task's `environment/` defines where the agent runs. Its `Dockerfile` builds the
agent container, which Harbor names `main`; an `environment/docker-compose.yaml`
adds sidecar services.

Build the shared base once per agent in the kit's `build/` directory, then give
every task its own copy: copy `build/` and the kit's data into the task's
`environment/` and add the task's own records, so the task builds on its own.
Reference a prebuilt image with `[environment].docker_image` only when it is
pushed to a registry the runtime can pull and pinned by digest. Never start a
task `FROM` a locally built kit tag: it exists only in the engine that built it,
and task evidence and Harbor's task checksum cover only the task directory.
Before running task controls, remove the local kit tag
(`docker image rm <kit-tag>`) so any task that still depends on it fails there,
not after handoff.
Keep reference solutions, expected state, and verifier helpers out of
`environment/`; anything there is visible to the agent.

## Sidecar services

Harbor merges `environment/docker-compose.yaml` over its own base compose file,
so declare only additions: sidecars and overrides such as `depends_on` for
`main`. Sidecars share a network with `main` and are reached by service name.

```yaml
services:
  main:
    depends_on:
      orders-api:
        condition: service_healthy

  db:
    build:
      context: ./db  # FROM postgres:16, then COPY seed/ /docker-entrypoint-initdb.d/
    environment:
      POSTGRES_PASSWORD: sandbox-only  # a sandbox value, never a real credential
      POSTGRES_DB: orders
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d orders"]
      interval: 2s
      timeout: 5s
      retries: 30

  orders-api:
    build:
      context: ./orders-api  # copied from the repository, never edited in place
    depends_on:
      db:
        condition: service_healthy
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 2s
      timeout: 5s
      retries: 15
```

Relative paths resolve against the task's `environment/` directory. Bake seed
data into a sidecar image, as above, rather than bind-mounting it: the image
carries the data to remote sandboxes and keeps every trial identical, while a
host bind mount works only where that host path exists.

Healthchecks keep the agent from starting before its services are ready, which
would otherwise show up as agent failures. Use `[environment.healthcheck]` in
`task.toml` for a readiness command that runs in `main` after start.

## Point the agent at its backends

Use `[environment.env]` to send the agent's existing clients to the sidecars and
to pass credentials by reference. Harbor resolves `${VAR}` and `${VAR:-default}`
from the host at runtime and does not copy other host variables into the
sandbox.

```toml
[environment.env]
ORDERS_API_URL = "http://orders-api:8000"
STRIPE_API_KEY = "${STRIPE_TEST_SECRET_KEY}"
```

Built-in agents forward their own model credentials during the agent phase;
other agent variables go through `--agent-env`. A separate verifier receives
neither agent-phase variables nor the agent's filesystem, only declared
artifacts. Keep provider credentials for remote sandboxes in the Harbor process
rather than the task.

## Tools the environment provides

Harbor treats tools as part of the agent. Provide tools from the environment
only when the agent expects them from outside itself, for example a coding agent
evaluated on a product's MCP server. Run the server as a sidecar and declare it:

```toml
[[environment.mcp_servers]]
name = "orders"
transport = "streamable-http"
url = "http://orders-mcp:8000/mcp"
```

Transports are `stdio` (with `command` and `args`), `sse`, and `streamable-http`
(with `url`). Harbor passes this configuration to compatible agents; that does
not prove the agent connected or called a tool. Prove discovery in an authorized
agent run, as described in the skill's Step 7.

## Network policy

Set network access per phase in `task.toml`:

| Phase | Typical setting | Why |
| --- | --- | --- |
| `[environment]` baseline | `no-network` when images are prebuilt; `public` only when setup must download | Setup runs under the baseline |
| `[agent]` during the run | `allowlist` with the model provider's host and any authorized vendor sandbox host | The agent needs its model and nothing else |
| `[verifier]` and `[verifier.environment]` | `no-network` | Grading must not depend on live services |

```toml
[environment]
network_mode = "public"  # agent setup downloads here; prefer no-network with a prebuilt image

[agent]
network_mode = "allowlist"
allowed_hosts = ["api.anthropic.com"]

[verifier]
environment_mode = "separate"
network_mode = "no-network"

[verifier.environment]
network_mode = "no-network"
```

Phase overrides need a sandbox that can switch network policy at runtime, and
several remote providers support allowlists only for single-container tasks.
On Docker, Harbor (0.20.0 and still 0.23.0) enforces both `no-network` and
`allowlist` through an egress-control sidecar that needs the kernel's nftables
`fib inet` support (`CONFIG_NFT_FIB_INET`). Docker Desktop's kernel can lack it; Harbor then
rejects the task with "network_mode='no-network' is not supported by
EnvironmentType.DOCKER environment" before anything runs. When that happens,
run on a Linux Docker host or a provider that supports the modes, or keep the
modes `public` for local proof and record isolation as unproven in the plan.
Never relax a policy silently, and never describe a public-network run as
isolated.

## Export the end state

Grade state, not the agent's report of it. Declare every file the verifier
needs as an artifact; a separate verifier receives only declared artifacts,
restored at their original source paths.

- **Files in `main`:** list their paths in `artifacts`.
- **State in a sidecar:** add a `[[verifier.collect]]` hook that runs in that
  service after the agent phase and writes a snapshot, then collect the snapshot
  with an artifact entry naming the same `service`.

```toml
artifacts = [
  { source = "/tmp/export/orders.json", service = "db" },
]

[[verifier.collect]]
service = "db"
command = "mkdir -p /tmp/export && psql -U postgres -d orders -Atc \"select coalesce(json_agg(o order by o.id), '[]') from orders o\" > /tmp/export/orders.json"
timeout_sec = 30
```

Run grading in a separate verifier: set `[verifier].environment_mode =
"separate"` and give the verifier its own image, for example a `tests/Dockerfile`
that installs grading dependencies and runs `COPY . /tests/`, so grading never
installs anything or reaches the network. Sidecar artifacts and collect hooks
need a Compose-capable sandbox.

For tables the case must leave unchanged, the verifier recomputes their content
digests from the export and compares them with the entries copied into the
task's verifier files, as [Starting data](starting-data.md#digests-for-preservation-checks)
describes.

Harbor records a failed collection in the trial's artifact manifest without
failing the trial. Write the verifier so a missing or unparseable export is an
explicit infrastructure error, never a pass and never an agent failure. When the
task follows trace-environment's per-check grammar, missing rows already read as
missing evidence.

## Reset

Harbor creates fresh containers for each trial, which resets `main` and every
sidecar whose state is seeded at build or start. Named volumes or host mounts
that persist between trials break that guarantee. External services, including
vendor sandboxes, do not reset with Harbor: document their reset command, run it
before each trial, and keep those trials serial until isolation is proven.

## Smoke task

Create the smoke task under the kit with Harbor's scaffolder, so it never joins
the suite's task set:

```bash
harbor task init <org>/<agent-slug>-environment-smoke \
  --tasks-dir .eval-author/environments/<agent-slug>/smoke \
  --description "Environment smoke check" --author "<actual author>"
```

- `instruction.md`: state that this task checks the environment and is not an
  agent evaluation.
- `solution/solve.sh`: for each dependency, one read and one write performed
  through the agent's own client code or CLI inside `main`, so the check covers
  the same configuration path the agent uses.
- `tests/test.sh`: read the exported state and emit one result per dependency
  confirming that exactly the smoke writes are present, plus one confirming the
  export itself was readable. A write that appears twice means state survived
  from an earlier trial.

Run the controls from the repository root with fresh job names:

```bash
harbor run -p .eval-author/environments/<agent-slug>/smoke/<agent-slug>-environment-smoke -a nop \
  --jobs-dir .eval-author/jobs --job-name <agent-slug>-env-smoke-nop-1
harbor run -p .eval-author/environments/<agent-slug>/smoke/<agent-slug>-environment-smoke -a oracle \
  --jobs-dir .eval-author/jobs --job-name <agent-slug>-env-smoke-oracle-1
harbor run -p .eval-author/environments/<agent-slug>/smoke/<agent-slug>-environment-smoke -a oracle \
  --jobs-dir .eval-author/jobs --job-name <agent-slug>-env-smoke-oracle-2
```

Expect NOP to export the untouched seed with the write checks failing, and both
reference-solution runs to pass. A failing reference solution means the
environment or the smoke script is broken; fix the environment rather than
loosening the checks. Record each job path and result in the plan; these runs
are environment proof, not task evidence receipts. The plan's status is `proven`
only when it names these three jobs.

## Version notes

Harbor 0.20.0 supports `[environment.env]`, `[[environment.mcp_servers]]`,
`[environment.healthcheck]`, `[environment].workdir`, per-phase network policy,
`[[verifier.collect]]`, and artifacts with a `service`. On Harbor 0.20.0 with
local Docker, this recipe was exercised end to end: a Postgres sidecar with a
healthcheck and baked seed data, `[environment.env]` settings reaching it by
service name, a collect hook and `service` artifact feeding a separate
verifier built from `tests/Dockerfile`, NOP exporting the untouched seed, and
two reference-solution runs each seeing exactly one smoke write. Later releases
add trial regrading (0.21) and simulated users and task-level trajectory loading
(0.22). Use a field only when the installed version's schema accepts it.
