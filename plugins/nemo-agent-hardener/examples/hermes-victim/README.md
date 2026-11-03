<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The war-game victim, Hermes harness

The Hermes victim registers on NeMo Helix like `relay-victim`, with a Relay config
override for the guardrails uploaded by Agent Hardener.

```
agent.yaml       the platform submission — note there is NO telemetry: block, on purpose
ledger_mcp.py    the tools, as an MCP server that ships inside the image
Dockerfile       your image, with an isolated Hermes environment and Relay config override
```

**Why no `telemetry:` block.** If it declared `provider: relay`, the Fabric Hermes adapter would
repoint `HERMES_NEMO_RELAY_PLUGINS_TOML` at its own generated config and the guardrails Agent Hardener
uploads to `/etc/nemo-relay/plugins.toml` would never be read. Instead the Dockerfile sets that env
var for Hermes' built-in Relay integration. The pinned Hermes release no longer needs
`hermes plugins enable observability/nemo_relay`.

## Running it

```bash
export NHX_BASE_URL=http://localhost:8080

nemo agents package --agent plugins/nemo-agent-hardener/examples/hermes-victim/agent.yaml \
                    --dockerfile plugins/nemo-agent-hardener/examples/hermes-victim/Dockerfile \
                    --tag ledger-hermes:v1
nemo agents register  --name ledger-hermes --agent-config plugins/nemo-agent-hardener/examples/hermes-victim/agent.yaml
nemo agents deploy  --agent ledger-hermes --image ledger-hermes:v1

nemo agent-hardener init --agent ledger-hermes --name ledger-hermes --harness hermes
nemo agent-hardener synth-benign --manifest-id ledger-hermes --yes
nemo agent-hardener run --manifest-id ledger-hermes
```
