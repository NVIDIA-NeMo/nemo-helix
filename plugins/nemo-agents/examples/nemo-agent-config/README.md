<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Agent Config

## Prerequisites

This directory contains NeMo Helix-managed `nemo-agents-spec-v1` configs for
NeMo Agents. Run the commands below from the repository root.

The base plugin installs Fabric, Relay support, and the supported harness
adapters. Install the extra for the harness you want to run; the
[installation matrix](../../README.md#harness-installation-matrix) lists the
available package expressions. Hermes is intentionally split out because its
runtime dependencies conflict with the NeMo Helix environment.

Set the credentials required by the selected model provider. The examples use
`NVIDIA_API_KEY`. Install and authenticate the selected harness CLI when
required; for example, run `codex login` for Codex or complete the Claude CLI
login flow.

For example, install Codex and its matching Relay CLI from the repository root:

```bash
uv sync --package nemo-agents-plugin --extra codex
nemo-relay --version
```

Shared agent capabilities live at the top level:

```yaml
instructions:
  system:
    content: You are a concise test assistant.

skills:
  paths: []

mcp:
  servers: {}

tools:
  blocked: []
```

`instructions.system` is the shared system prompt path for Claude, Codex,
DeepAgents, and Hermes. Adapter-specific options stay under
`harnesses.<name>.settings`; do not put prompt text there.

The selected harness is controlled by `default_harness`. To try another harness
from the same config today, edit `default_harness` before creating or invoking
the agent.

## Model parameters

Models can set optional `top_p` and `max_tokens` parameters:

```yaml
models:
  default:
    provider: openai
    model: openai/gpt-5.4
    top_p: 0.9
    max_tokens: 1024
```

`top_p` must be between 0 and 1, inclusive. `max_tokens` must be a positive
integer no larger than `18446744073709551615`. Omit either field to keep the
adapter's default. Support depends on the selected harness and model.
These fields also work under `harnesses.<name>.model`, which replaces
`models.default` for the selected harness; individual fields are not merged.

## Invoke

`agent.yaml` is the telemetry-neutral multi-harness example. Set
`default_harness` to the harness you want to validate, then create, deploy, and
invoke the agent through NeMo Helix.

```bash
make bootstrap-python
source .venv/bin/activate

export NVIDIA_API_KEY="<your NVIDIA API key>"
export NHX_BASE_URL=http://localhost:8080

if curl -fsS --connect-timeout 2 --max-time 5 \
  "$NHX_BASE_URL/health/ready" >/dev/null; then
  echo "Using the running NeMo Helix instance at $NHX_BASE_URL"
else
  nemo setup --auto --start-services --install-skills
fi

curl -fsS --connect-timeout 2 --max-time 5 \
  "$NHX_BASE_URL/health/ready" >/dev/null || {
  echo "NeMo Helix is not ready at $NHX_BASE_URL"
  exit 1
}

# If setup does not create a usable NVIDIA inference provider, follow
# Step 2 in plugins/nemo-agents/README.md before deploying.

nemo agents create \
  --name platform-agent \
  --agent-config plugins/nemo-agents/examples/nemo-agent-config/agent.yaml

nemo agents deploy \
  --agent platform-agent \
  --name platform-agent-deployment \
  --mode subprocess

nemo agents invoke \
  --agent-deployment platform-agent-deployment \
  --input "Reply with exactly: platform agent works"
```

Use a unique `--name` / deployment name for each harness, or delete the previous
agent and deployment before recreating them.

## Harness Notes

### Codex

Set `default_harness: codex` in `agent.yaml`. Authenticate Codex before
invoking:

```bash
codex login
```

In this example, Codex uses the shared Nemotron model through NeMo Helix IGW.

### DeepAgents

Set `default_harness: deepagents` in `agent.yaml`. In this example, DeepAgents
uses the shared Nemotron model through NeMo Helix IGW.

### Claude

Set `default_harness: claude` in `agent.yaml`. Authenticate Claude Code before
invoking:

```bash
claude
```

In this example, Claude uses its harness-local Anthropic model config.

### Hermes

Hermes Agent has dependencies that conflict with the NeMo Helix environment, so
use the repository helper to install Fabric's pinned Hermes source and matching
adapter in a separate Python 3.14 environment:

```bash
script/dev-install-hermes.sh
export ADAPTER_PYTHON="$PWD/.venv-hermes/bin/python"
```

Set `default_harness: hermes` in `agent.yaml`. For subprocess deployments,
export `ADAPTER_PYTHON` before starting NeMo Helix, or restart NeMo Helix after
exporting it. The NeMo Helix service launches the agent subprocess, so exporting
`ADAPTER_PYTHON` only in the later CLI shell is not enough.

### NOOA

Use Python 3.12 or 3.13 and install the NOOA harness:

```bash
uv sync --package nemo-agents-plugin --extra nooa --python 3.13
export NVIDIA_API_KEY="<your NVIDIA API key>"
```

Use [agent-nooa-coding.yaml](agent-nooa-coding.yaml) for CodingAgent or
[agent-nooa-bench.yaml](agent-nooa-bench.yaml) for BenchAgent with the invoke
commands above. Both examples call the NVIDIA model endpoint directly.

CodingAgent selects `workflow.target_id: nvidia.nooa.coding-agent`; BenchAgent
selects `kind: nooa-bench-agent` under `harnesses`. A config uses either
`workflow` or `default_harness` with `harnesses`, not both. Workflow models go
under `models.default` and target options under `workflow.settings`.

Helix enables Relay for streaming and sessions. Generated containers install the
NOOA extra and require Python 3.12 or 3.13. Run `make test-agents-nooa` to test
both adapters against a local model stub.

### Remote Agent

Use [agent-remote.yaml](agent-remote.yaml) to connect to a running agent; no
harness extra is needed. Set `harnesses.remote-agent.settings.base_url` to its
API root (including `/v1`), `models.default.model` to its model name, and
`REMOTE_AGENT_API_KEY` if authentication is required. For an unauthenticated
endpoint, remove `models.default.api_key_env` to skip credential lookup.

The endpoint must support SSE. Set `api_type` to `openai-responses` (default),
`openai-completions`, or `anthropic-messages`. Configure skills, MCP, and tool
policy on the remote agent.

For live streaming and sessions, use a Relay-instrumented OpenAI Responses or
Chat Completions endpoint and a running collector reachable by both services.
Update the example config:

```yaml
harnesses:
  remote-agent:
    kind: remote-agent
    settings:
      base_url: https://agent.example.com/v1
      api_type: openai-responses
      relay_streaming: true
telemetry:
  enabled: true
  provider: relay
  atof:
    enabled: true
    sinks:
      - type: stream
        name: nemo-fabric-stream
        url: https://collector.example.com
        transport: ndjson
```

Configure the remote agent to publish NDJSON ATOF events to the collector's
`/v1/atof` endpoint and carry `metadata.nemo_fabric_request_id` into its Relay
events. Use `header_env` on the sink if the collector requires authentication.
Helix connects to the collector without starting or stopping it.

## Relay Local Files

`agent-relay.yaml` enables Relay telemetry without Intake. It writes local ATIF
and ATOF artifacts under the deployment artifacts directory. Use
`agent-relay.yaml` with the invoke directions above; the artifact directory uses
the deployment name you pass to `nemo agents deploy`.

Confirm Relay emitted both ATIF and ATOF files:

```bash
find ~/.local/share/nemo/agents/system/default \
  -path "*platform-agent-deployment*/artifacts/*" \
  \( -name "*atif*" -o -name "*atof*" \) \
  -exec ls -lh {} \;
```

## Relay to Intake

`agent-relay-intake.yaml` enables Relay ATIF export to a locally running
NeMo Helix Intake API. With Docker running, Intake automatically provisions its
local ClickHouse container during platform startup. Use
`agent-relay-intake.yaml` with the invoke directions above.

Confirm Intake received ATIF-derived spans:

```bash
curl -g -s "http://127.0.0.1:8080/apis/intake/v2/workspaces/default/spans?page_size=20" \
  | jq '.data[] | {session_id, name, source, started_at}'
```

To inspect traces in Studio, run the web app with Intake enabled:

```bash
cd web
VITEST=true \
VITE_FF_INTAKE_ENABLED=true \
VITE_PLATFORM_BASE_URL=http://localhost:8080 \
pnpm --filter nemo-studio-ui start --host 127.0.0.1
```

Then open `http://localhost:5173/studio/workspaces/default/intake/traces`.

## Next Steps

- [Package the calculator agent as a container image](../../README.md#packaging-agents-as-container-images).
- [Deploy an agent](../../../../docs/agents/deploy-agents.mdx).
- [Review the agent configuration contract](../../../../docs/agents/index.mdx#agent-definition).
