<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The war-game victim, LangChain (BYO)

A bring-your-own agent built with LangChain's `create_agent`, served over
`/v1/chat/completions`. Two tools worth attacking — `bash_executor`, `python_executor` — that
really run what they're given.

```
agent.py         create_agent with NemoRelayMiddleware — the whole tool-path obligation
server.py        FastAPI serving loop — starts Relay, opens a scope per request
Dockerfile       your image; everything Agent Hardener derives, it derives from here
```

This is the simplest victim to instrument: `middleware=[NemoRelayMiddleware()]` on `create_agent`
is the one interception line — no ToolNode to wire by hand.

## Running it

```bash
export NMP_BASE_URL=http://localhost:8080

nemo agent-hardener init --project-dir plugins/nemo-agent-hardener/examples/langchain-victim \
    --name langchain-victim --harness langchain --relay-confirmed \
    --secrets INFERENCE_API_KEY \
    --egress inference-api.nvidia.com \
    --start-command "/usr/local/bin/python /app/server.py" \
    --binary "/usr/local/bin/python*"
nemo agent-hardener synth-benign --manifest-id langchain-victim --yes
nemo agent-hardener run --manifest-id langchain-victim
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
