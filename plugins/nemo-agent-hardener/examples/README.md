<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Hardener example victims

A complete, runnable victim for every harness Agent Hardener can guard, across both intake paths. A
"victim" is the agent a war-game attacks: it only has to be reachable over
`/v1/chat/completions` and have its tool calls routed through NeMo Relay, so a guardrail can refuse
one.

Agent Hardener can only guard a harness whose tool calls pass through Relay. That is **deepagents**,
**hermes**, **langchain**, **langgraph**, and framework-free (**other**) agents that wire Relay
themselves. Claude and Codex run Relay as a compiled gateway that cannot load the guardrail plugin,
so they are refused at `init` — there is no example for them because there is no way to make one
guardable.

| Example | Harness | Intake | How Relay attaches |
|---|---|---|---|
| `relay-victim/` | deepagents | registered (Route A) | `telemetry:` block in `agent.yaml` — the adapter wires it, no code |
| `hermes-victim/` | hermes | registered (Route A) | opt-in Hermes plugin, enabled in the Dockerfile; no `telemetry:` block |
| `langchain-victim/` | langchain | BYO (Route B) | `middleware=[NemoRelayMiddleware()]` on `create_agent` |
| `langgraph-victim/` | langgraph | BYO (Route B) | `create_tool_node` / `awrap_tool_call` on a hand-built `ToolNode` |
| `other-victim/` | other | BYO (Route B) | `nemo_relay.typed.tool_execute` called by hand, per tool call |

**Registered vs BYO.** A registered agent is a Fabric spec (`nemo-agents-spec-v1`) the platform
builds and serves — you hand Agent Hardener the agent name (`init --agent`). A BYO agent is an image
your own Dockerfile builds — you hand Agent Hardener the project directory (`init --project-dir`), and
everything the Dockerfile states is derived. The registered examples ship an `agent.yaml`; the BYO
examples ship `agent.py` + `server.py`.

**The common victim, on purpose.** The three BYO examples share the same two tools
(`bash_executor`, `python_executor`) and the same server shape, so the *only* thing that differs
between them is the Relay-attachment line. Read them side by side to see what each framework asks
of you. The two registered examples share the MCP `ledger` tool server for the same reason.

Each subdirectory has its own README with the exact `nemo` commands. Start from the one that
matches the agent you actually have; if you are unsure, the `nemo-agent-hardener` skill's
`references/relay-attachment.md` walks the attachment decision.
