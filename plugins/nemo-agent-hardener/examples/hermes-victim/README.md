<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The war-game victim, Hermes harness

The second guardable harness (deepagents is the other), registered on NeMo Platform like the
`relay-victim` — but Hermes reaches Relay through an opt-in *plugin*, not the adapter's telemetry
wiring, and that changes two things.

```
agent.yaml       the platform submission — note there is NO telemetry: block, on purpose
ledger_mcp.py    the tools, as an MCP server that ships inside the image
Dockerfile       your image, with the two Hermes-only lines that replace the telemetry: block
```

**Why no `telemetry:` block.** If it declared `provider: relay`, the Fabric Hermes adapter would
repoint `HERMES_NEMO_RELAY_PLUGINS_TOML` at its own generated config and the guardrails Agent Hardener
uploads to `/etc/nemo-relay/plugins.toml` would never be read. Instead the Dockerfile sets that env
var itself and runs `hermes plugins enable observability/nemo_relay` — see the comments there.

## Running it

```bash
export NMP_BASE_URL=http://localhost:8080

nemo agents package --agent plugins/nemo-agent-hardener/examples/hermes-victim/agent.yaml \
                    --dockerfile plugins/nemo-agent-hardener/examples/hermes-victim/Dockerfile \
                    --tag ledger-hermes:v1
nemo agents create  --name ledger-hermes --agent-config plugins/nemo-agent-hardener/examples/hermes-victim/agent.yaml
nemo agents deploy  --agent ledger-hermes --image ledger-hermes:v1

nemo agent-hardener init --agent ledger-hermes --name ledger-hermes --harness hermes
nemo agent-hardener synth-benign --manifest-id ledger-hermes --yes
nemo agent-hardener run --manifest-id ledger-hermes
```
