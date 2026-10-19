<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# research-agent

A minimal research agent built on the [NVIDIA NeMo Agent Toolkit (NAT)](https://github.com/NVIDIA/NeMo-Agent-Toolkit): a [Tavily](https://tavily.com/) search tool wired into a ReAct agent backed by an NVIDIA NIM-hosted LLM.

The workflow is declared in [`workflow.yml`](./workflow.yml); `main.py` is a thin wrapper that loads `.env` and invokes NAT's Python API.

## Setup

```bash
cp .env.example .env   # fill in NVIDIA_API_KEY and TAVILY_API_KEY
uv sync
```

## Run

Via the Python wrapper:

```bash
uv run main.py "What did NVIDIA announce at GTC 2026?"
```

Or directly via the NAT CLI (equivalent):

```bash
uv run nat run --config_file workflow.yml --input "What did NVIDIA announce at GTC 2026?"
```

## Customize

Edit `workflow.yml` to swap the model, change the search depth, or tweak the system prompt. To list every knob NAT exposes for a given component type:

```bash
uv run nat info components -t llm_provider -q nim
uv run nat info components -t function    -q tavily_internet_search
```

## Run on NeMo Helix

This example uses local paths to the monorepo's `nemo-helix`, `nemo-helix-plugin`, and `nemo-insights-plugin` packages. Run `uv sync` from this directory to create its standalone environment.

### Start services

From this directory:

```bash
set -a && source .env && set +a   # exposes NVIDIA_API_KEY / TAVILY_API_KEY
uv run nemo services run          # boots 18 services on :8080 — leave running
```

Confirm the plugin loaded: `nemo plugins list` should include `insights`, and the startup banner should list the `insights` service.

### Verify ClickHouse (for the `intake` service)

The platform's `intake` service persists spans/traces in ClickHouse. With Docker running,
`nemo services run` automatically provisions and reuses a local ClickHouse container for the
resolved NeMo data directory.
Set `NHX_INTAKE_CLICKHOUSE_URL` before startup to use an externally managed instance instead.

Verify the Intake read path after the platform starts:

```bash
curl -s http://localhost:8080/apis/intake/v2/workspaces/default/traces       # -> 200 with JSON
```

If this returns `503` and the service log reports that the Docker daemon is unavailable, start Docker
Desktop on macOS/Windows or the Docker service on Linux, then restart `nemo services run`.

### Configure a provider

In a second shell, register the NVIDIA Build provider and pick default and fast
models (press Enter at the fast prompt to reuse the default):

```bash
set -a && source .env && set +a
uv run nemo setup
```

### Register the workflow as an agent

`workflow.yml` is a valid NAT config and is accepted by the platform as-is. The model name must be the platform-registered entity ID (lowercase, hyphenated) — list available IDs with `uv run nemo models list`.

```bash
uv run nemo agents create --name research-agent \
  --agent-config workflow.yml \
  --description "Tavily-powered research agent"
uv run nemo agents deploy --agent research-agent
```

`deploy` blocks until the container reaches `running` and prints the deployment endpoint.

### Chat with the deployed agent

```bash
uv run nemo agents invoke --agent research-agent \
  --input "What did NVIDIA announce at GTC 2026?"
```

The response is an OpenAI-style chat completion. Inference goes through the platform's gateway, and Tavily search is invoked from inside the agent container.

### Inspect and tear down

```bash
uv run nemo agents list
uv run nemo agents deployments list
uv run nemo agents logs --agent research-agent

echo y | uv run nemo agents undeploy --agent research-agent
echo y | uv run nemo agents delete   research-agent
```

### Gotchas

- **Model name format.** NAT YAML accepts the `meta/llama-3.3-70b-instruct` style for direct NIM calls, but when the agent runs under the platform, NIM calls are routed through the inference gateway, which validates names as lowercase letters/digits/hyphens (2–63 chars). Use the hyphenated entity ID from `nemo models list` (e.g. `meta-llama-3-3-70b-instruct`).
- **Undeploy / delete prompt.** No `--yes` flag on this build — pipe `echo y |` for non-interactive use.
- **Editing `workflow.yml`.** The platform stores the YAML at create time. After edits, `undeploy` + `delete` + `create` + `deploy` to pick them up.
