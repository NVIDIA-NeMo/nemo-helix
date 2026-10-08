<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The war-game victim, framework-free (BYO, `--harness other`)

An agent with no framework at all — a plain OpenAI tool-calling loop — for the case Relay has no
ready-made integration (CrewAI, AutoGen, a homegrown loop, …). Two tools worth attacking
(`bash_executor`, `python_executor`) that really run what they're given.

```
agent.py         the loop; each tool call goes through nemo_relay.typed.tool_execute by hand
server.py        FastAPI serving loop — starts Relay, opens a scope per request
Dockerfile       your image; base nemo-relay, no integration extra
```

There is no middleware and no ToolNode to lean on: `tool_execute` **is** the interception point,
and the loop calls it for every tool. That is exactly what a `--harness other` victim must prove
it does — the run's tool-path preflight fails if any tool call skips it.

## Running it

```bash
export NMP_BASE_URL=http://localhost:8080

nemo agent-hardener init --project-dir plugins/nemo-agent-hardener/examples/other-victim \
    --name other-victim --harness other --relay-confirmed \
    --secrets INFERENCE_API_KEY \
    --egress inference-api.nvidia.com \
    --start-command "/usr/local/bin/python /app/server.py" \
    --binary "/usr/local/bin/python*"
nemo agent-hardener synth-benign --manifest-id other-victim --yes
nemo agent-hardener run --manifest-id other-victim
```

### Why the four extra flags

`init` derives what the Dockerfile *states* and warns about the rest rather than guessing. The
model host and API key live in `agent.py`, and this image installs to the system Python rather than
a venv — so four things cannot be derived, and each fails differently if omitted: the credential
name (victim starts, then fails its first model call), the egress host (default-deny sandbox drops
model traffic mid-run), the absolute start command (OpenShell replaces `PATH`, so bare `python`
never resolves), and the interpreter glob (the egress policy would match no process, granting
nothing).

If your own project declares these in the Dockerfile, `init` picks them up and you pass nothing.
