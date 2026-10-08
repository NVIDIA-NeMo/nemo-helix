<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The war-game victim, bring-your-own

The other intake path: no platform registration, no Fabric spec — just the image your own
Dockerfile builds. A hand-built LangGraph `StateGraph` with two tools worth attacking
(`bash_executor`, `python_executor`), served over `/v1/chat/completions`.

```
agent.py         the graph — Relay's wrapper on the ToolNode is THE interception line
server.py        FastAPI serving loop — starts Relay, opens a scope per request
Dockerfile       your image; everything Agent Hardener derives, it derives from here
agent-hardener.yaml  only for running standalone; `init --project-dir` derives this
```

Unlike the Fabric example, the Relay wiring here is the author's job — both obligations are marked
in the source, and the skill's `relay-attachment.md` reference explains each.

## Running it

```bash
export NMP_BASE_URL=http://localhost:8080

nemo agent-hardener init --project-dir plugins/nemo-agent-hardener/examples/langgraph-victim \
    --name langgraph-victim --harness langgraph --relay-confirmed \
    --secrets INFERENCE_API_KEY \
    --egress inference-api.nvidia.com \
    --start-command "/usr/local/bin/python /app/server.py" \
    --binary "/usr/local/bin/python*"
nemo agent-hardener synth-benign --manifest-id langgraph-victim --yes
nemo agent-hardener run --manifest-id langgraph-victim
```

### Why the four extra flags

`init` derives what the Dockerfile *states* and warns about the rest rather than guessing. Here it
cannot see four things, and each one fails differently if you leave it out:

| Flag | Why it can't be derived | What happens without it |
|---|---|---|
| `--secrets INFERENCE_API_KEY` | the key name lives in `agent.py`, not in an `ENV` or a committed dotenv | the victim starts, then fails its first model call |
| `--egress inference-api.nvidia.com` | the model host is a Python default in `agent.py`, not named in the Dockerfile | the sandbox is default-deny, so model traffic is dropped mid-run |
| `--start-command "/usr/local/bin/python …"` | the image installs to the system Python, so there is no venv to derive an absolute path from | OpenShell replaces `PATH`, so a bare `python` never resolves |
| `--binary "/usr/local/bin/python*"` | same reason — the default glob points at a venv this image does not have | the egress policy matches no process, granting nothing |

If your own project declares these in the Dockerfile (`ENV OPENAI_API_KEY=""`, a `base_url` in an
`ENV`, an exec-form `ENTRYPOINT` with an absolute path), `init` picks them up and you pass nothing.

`--relay-confirmed` is honest here because `agent.py` and `server.py` really do attach Relay; on
your own project, verify before you promise — the preflight will catch a false claim, but only
after a container build.
