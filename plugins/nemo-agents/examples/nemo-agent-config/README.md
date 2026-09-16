<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Agent Config

## Prerequisites

This directory contains Platform-managed `nemo-agents-spec-v1` configs for
NeMo Agents. Run the commands below from the repository root.

The plugin installs Fabric, Relay Python bindings, and supported harness
adapters. The Relay CLI is separate. Hermes is intentionally split out because the Hermes
Agent runtime dependencies conflict with the Platform environment.

Set the credentials required by the selected model provider. The examples use
`NVIDIA_API_KEY`. Install and authenticate the selected harness CLI when
required; for example, run `codex login` for Codex or complete the Claude CLI
login flow.

For Claude or Codex, install and verify the Relay CLI:

```bash
script/dev-install-fabric.sh
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

## Invoke

`agent.yaml` is the telemetry-neutral multi-harness example. Set
`default_harness` to the harness you want to validate, then create, deploy, and
invoke the agent through Platform.

```bash
make bootstrap-python
source .venv/bin/activate

export NVIDIA_API_KEY="<your NVIDIA API key>"
export NMP_BASE_URL=http://localhost:8080

if curl -fsS --connect-timeout 2 --max-time 5 \
  "$NMP_BASE_URL/health/ready" >/dev/null; then
  echo "Using the running NeMo Platform instance at $NMP_BASE_URL"
else
  nemo setup --auto --start-services --install-skills --no-deploy-agent
fi

curl -fsS --connect-timeout 2 --max-time 5 \
  "$NMP_BASE_URL/health/ready" >/dev/null || {
  echo "NeMo Platform is not ready at $NMP_BASE_URL"
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

In this example, Codex uses the shared Nemotron model through Platform IGW.

### DeepAgents

Set `default_harness: deepagents` in `agent.yaml`. In this example, DeepAgents
uses the shared Nemotron model through Platform IGW.

### Claude

Set `default_harness: claude` in `agent.yaml`. Authenticate Claude Code before
invoking:

```bash
claude
```

In this example, Claude uses its harness-local Anthropic model config.

### Hermes

Hermes Agent has dependencies that conflict with the Platform environment, so
install it with the Fabric adapter in a separate Python 3.12 environment:

```bash
uvx uv@0.9.14 venv --python 3.12 .venv-hermes
uvx uv@0.9.14 --no-config pip install \
  --python .venv-hermes/bin/python \
  "nemo-fabric[relay]==0.3.0b1" \
  "nemo-fabric-adapters-hermes==0.3.0b1" \
  "hermes-agent==0.19.0"

export ADAPTER_PYTHON="$PWD/.venv-hermes/bin/python"
```

Set `default_harness: hermes` in `agent.yaml`. For subprocess deployments,
export `ADAPTER_PYTHON` before starting Platform, or restart Platform after
exporting it. The Platform service launches the agent subprocess, so exporting
`ADAPTER_PYTHON` only in the later CLI shell is not enough.

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
Platform Intake API. With Docker running, Intake automatically provisions its
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
