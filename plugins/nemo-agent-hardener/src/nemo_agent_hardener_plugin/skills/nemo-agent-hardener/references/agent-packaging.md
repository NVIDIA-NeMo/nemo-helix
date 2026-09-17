<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Packaging an agent the war-game can score

What Route A (registered Fabric agent) and Route B (BYO image) each require of the files you
already have. Agent Hardener hardens the image you ship — it never builds an agent for you.

## Route A: the `agent.yaml`

`nemo agents create` accepts only `config_format: nemo-agents-spec-v1`. The parts that decide
whether a war-game can run:

```yaml
config_format: nemo-agents-spec-v1
name: my-agent
description: What this agent is for

instructions:
  system:
    content: |
      You are a retail bank's support agent. Only act on what the customer asked for.

default_harness: deepagents          # deepagents or hermes — claude/codex are refused at init

harnesses:
  deepagents:
    kind: deepagents
    model:
      provider: nvidia
      model: nvidia/nvidia/Nemotron-3-Nano-30B-A3B
    settings:
      deepagents: {}

mcp:                                 # custom tools are ALWAYS MCP — Fabric rejects a tools: key
  servers:                           # naming Python callables
    ledger:
      transport: stdio               # stdio runs inside the container; streamable_http is remote
      url: /usr/local/bin/python     # absolute — OpenShell replaces PATH
      args: ["/app/ledger_mcp.py"]

telemetry:                           # REQUIRED for a war-game: this is how Relay attaches
  enabled: true
  provider: relay                    # the only accepted provider
  output_dir: /home/sandbox/.agent-hardener/relay
  project: my-agent
  atof:
    enabled: true                    # defaults to false; without it the run has no evidence
    filename: events.atof.jsonl
```

- **Harness**: only `deepagents` and `hermes` are guardable. `claude`/`codex` run behind a
  compiled gateway with no interception point — `init` refuses them.
- **`telemetry` with `atof.enabled: true`** is the single most-missed block. Without it the victim
  answers happily and emits nothing, which preflight reports as an uninstrumented victim.
- Write real identifiers into MCP tool docstrings: the agent — and the benign-suite synthesizer —
  only knows what the descriptions say.
- Bounding runaway turns: `runtime.max_turns` on the harness maps to LangGraph's recursion limit
  for `deepagents` (omit it to keep the harness default).

## Both routes: the Dockerfile contract

The sandbox (OpenShell) runs the image with its own PATH, its own network policy, and a non-root
user. The image must therefore:

| Requirement | Why |
|---|---|
| a `sandbox` user and group | OpenShell runs the container as `sandbox`, not root |
| `iproute2` installed | the sandbox's egress enforcement needs it inside the image |
| `/etc/nemo-relay/` writable by the runtime user | each round's guardrails are uploaded there as `plugins.toml` |
| absolute paths in ENTRYPOINT/CMD | OpenShell replaces `PATH`; a bare `python` never resolves |
| exec-form ENTRYPOINT (`["…", "…"]`) | shell-form is not machine-readable — BYO derivation will ask for `--start-command` |
| no BuildKit `--mount=type=cache` | the sandbox builds with the classic Docker builder, which rejects them |
| serve on port 8000 | every war-game victim binds 8000; the port is read from ENV `PORT` or `EXPOSE` |

## Route B: the project bundle

`init --project-dir` (CLI) or the Studio archive upload takes a directory that holds the
Dockerfile plus whatever it COPYs. Derivation rules:

- ENV, EXPOSE, exec-form ENTRYPOINT/CMD are parsed; `$VAR` references are resolved from the
  image's own ENV.
- An env var whose value is **empty** in the Dockerfile (`ENV OPENAI_API_KEY=""`) is treated as a
  **secret name** — the name is carried, the value never is. Names in a committed `.env` count
  too.
- Hosts the project names (in ENV URLs, RUN commands) seed the egress list; everything else needs
  `--egress`.
- Two Dockerfiles → you must pick with `--dockerfile`. Vendored ones (under `.venv/`,
  `node_modules/`) are ignored. No Dockerfile → refused; Agent Hardener does not write one.
- What a project can never state about itself — `--harness` and `--relay-confirmed` — is always
  asked of you.

## Worked examples

`plugins/nemo-agent-hardener/examples/` has a complete, runnable victim for every guardable harness,
across both routes — see its README for the matrix. In short:

- Route A (registered Fabric agent): `relay-victim/` (deepagents) and `hermes-victim/` (hermes).
- Route B (BYO project): `langchain-victim/`, `langgraph-victim/`, and `other-victim/`
  (framework-free) — same two tools and server shape across all three, so the only difference is
  the Relay-attachment line described in [relay-attachment.md](relay-attachment.md).
