<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The war-game victim, as a NeMo Platform agent

This is the shape Agent Hardener expects in production: an agent **registered on NeMo Platform**, not an
image you built by hand. It is a retail-banking support agent with three tools worth attacking —
`transfer_funds`, `read_customer_record` and `send_email`.

```
agent.yaml       the platform submission — harness, model, MCP tools, telemetry
ledger_mcp.py    the tools, as an MCP server that ships inside the image
Dockerfile       your image, registered alongside agent.yaml
agent-hardener.yaml  only for running the war-game standalone; `init` derives this in production
```

There is **no agent code**. Fabric constructs the agent from `agent.yaml`, and the DeepAgents adapter
does the Relay wiring when it sees the `telemetry:` block. Nothing in this bundle imports Relay or
Agent Hardener.

## Running it

```bash
export NMP_BASE_URL=http://localhost:8080

nemo agents package --agent plugins/nemo-agent-hardener/examples/relay-victim/agent.yaml \
                    --dockerfile plugins/nemo-agent-hardener/examples/relay-victim/Dockerfile --tag ledger:v1
nemo agents create  --name ledger --agent-config plugins/nemo-agent-hardener/examples/relay-victim/agent.yaml
nemo agents deploy  --agent ledger --image ledger:v1

nemo agent-hardener init --agent ledger --name ledger
nemo agent-hardener synth-benign --manifest-id ledger --yes
nemo agent-hardener run --manifest-id ledger
```

Standalone, skipping agent registration:

```bash
nemo agent-hardener run --config plugins/nemo-agent-hardener/examples/relay-victim/agent-hardener.yaml --env-file .env
```

## What makes it a victim

Three things, none of them code.

**A harness whose tool calls can be refused.** `deepagents` here; `hermes` also works. Claude and
Codex run Relay as a compiled gateway binary that cannot load the guardrail plugin — handing it one
is a fatal config error, so the gateway never starts.

**`telemetry:` with `atof.enabled: true`.** `provider: relay` alone emits no tool trace; ATOF has its
own flag. Agent Hardener reads that stream to prove the victim is instrumented, and fails the run up
front when a warm-up probe produces none — rather than reporting a clean bill of health for a
guardrail that was never consulted.

**Tools that are business operations.** A guardrail on `transfer_funds` is meaningful. A guardrail on
`Bash` is sandbox-policy territory, which is a different defender.

## What your image must provide

Beyond an ordinary Fabric image, three things — each marked `[war-game]` in the [Dockerfile](Dockerfile):

- **A `sandbox` user/group and `iproute2`.** The OpenShell sandbox needs both.
- **`/etc/nemo-relay/`, existing and writable by the runtime user.** Agent Hardener uploads each round's
  guardrails there before the victim starts. Missing directory → the run fails at deploy with
  `mkdir: cannot create directory '/etc/nemo-relay': Permission denied`, surfaced as a tar-extract
  error.
- **No `RUN --mount=type=cache`.** OpenShell builds through the Docker Engine API's classic builder,
  which rejects BuildKit-only syntax. Agent Hardener strips the flags for you, but not writing them is
  one less surprise.

You install nothing for the guardrail itself. Agent Hardener appends two lines to a copy of your
Dockerfile — `COPY openshell-shims/` and `ENV PYTHONPATH` — and the plugin rides in beside the
sandbox's egress shim.

## Why an author-supplied Dockerfile

`nemo agents package` can render one, and Agent Hardener falls back to that. Prefer your own: a rendered
Dockerfile pins the packaging machine's `nemo-platform` version — from a git checkout that is
something like `0.4.0.post96.dev0+789b8466e`, which no index serves — and a fixed `nemo-relay` that
an agent has no way to ask to change.

Registration uploads the whole directory holding `agent.yaml` into the `{agent}-ethos` fileset, so a
`Dockerfile` sitting beside it is already on the platform. Agent Hardener reads it back in preference.

`agent.yaml` is verified against the real platform code — `load_agent_config` +
`translate_agent_config` resolve it to adapter `nvidia.fabric.langchain.deepagents`, with the three
MCP tools and an ATOF file sink. Fabric's own translator emits `RelayObservabilityConfig(version=3)`,
which is the schema the war-game's uploaded `plugins.toml` matches.

> **The image is not built end to end.** The Dockerfile mirrors what `render_fabric_dockerfile` produces, but
> it has not been built here: the released `nemo-platform` on the index (0.3.0) predates the Fabric
> adapters, and the checkout's own version is a dev build no index serves. That gap is the same one
> the note above describes.
