<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Calculator Agent

This example provides two Platform-managed `nemo-agents-spec-v1` configurations
for comparing a DeepAgents calculator agent with and without tools:

| Config | Calculator server |
| --- | --- |
| `agent.yaml` | Disabled |
| `agent-with-mcp.yaml` | Enabled through MCP |

Both variants use the Platform Inference Gateway and record ATOF telemetry
through NeMo Relay.

## Prerequisites

Run these commands from the repository root:

```bash
export NVIDIA_API_KEY="<your NVIDIA API key>"
export NHX_BASE_URL=http://localhost:8080

make bootstrap-python
source .venv/bin/activate

command -v calculator-server
```

Start ClickHouse for Intake:

```bash
services/intake/scripts/spans/run_clickhouse.sh
```

Set up NeMo Helix without deploying the default demo agent:

```bash
nemo setup --auto --start-services --install-skills --no-deploy-agent
```

Use `nemo setup` without `--auto` for interactive provider and model selection.
Confirm the Platform is ready before continuing:

```bash
curl -fsS --connect-timeout 2 --max-time 5 \
  "$NHX_BASE_URL/health/ready" >/dev/null
```

## Run without MCP

Create and deploy the basic calculator agent:

```bash
nemo agents create \
  --name calculator-agent \
  --agent-config plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml

nemo agents deploy \
  --agent calculator-agent \
  --name calculator-agent-deployment \
  --mode subprocess
```

Invoke it:

```bash
nemo agents invoke \
  --agent-deployment calculator-agent-deployment \
  --input "What is 12 multiplied by 8?"
```

The response should report `96`. This configuration has no calculator server,
so its ATOF events should contain no calculator tool call.

## Run with MCP

Create and deploy the tool-enabled variant:

```bash
nemo agents create \
  --name calculator-agent-with-mcp \
  --agent-config plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent-with-mcp.yaml

nemo agents deploy \
  --agent calculator-agent-with-mcp \
  --name calculator-agent-with-mcp-deployment \
  --mode subprocess
```

Invoke it:

```bash
nemo agents invoke \
  --agent-deployment calculator-agent-with-mcp-deployment \
  --input "Use the calculator multiply tool to calculate 12 multiplied by 8. Do not calculate it yourself."
```

The response should report `96` after calling the calculator's `multiply`
tool.

## Verify telemetry

List the local ATOF artifacts for both deployments:

```bash
find ~/.local/share/nemo/agents/system/default \
  -path "*calculator-agent*-deployment*/artifacts/*" \
  -name "*.atof.jsonl" \
  -exec ls -lh {} \;
```

Inspect calculator tool calls in the MCP deployment's ATOF events:

```bash
find ~/.local/share/nemo/agents/system/default \
  -path "*calculator-agent-with-mcp-deployment*/artifacts/*" \
  -name "*.atof.jsonl" \
  -exec jq -c \
    'select(.category == "tool" and .scope_category == "start")
     | {name, arguments: .data}' {} +
```

The MCP deployment should include a `multiply` call. Running the same command
against the basic deployment should print no calculator tool calls.
