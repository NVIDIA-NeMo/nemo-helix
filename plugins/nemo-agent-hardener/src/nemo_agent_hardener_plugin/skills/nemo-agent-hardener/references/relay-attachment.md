<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Attaching NeMo Relay so the war-game can guard tool calls

Scope: exactly what Agent Hardener needs and verifies. For attachment modes and frameworks beyond
these, use NeMo Relay's own documentation and published skills — they are the authoritative
integration guide; do not improvise wiring.

A Fabric agent (Route A) needs **none of this in code** — the `telemetry: {provider: relay,
atof: {enabled: true}}` block in `agent.yaml` makes the adapter do all of it. This reference is
for Route B (BYO).

## The two obligations — both, not one

1. **Start Relay and let it discover its config**, once, at startup:

   ```python
   import nemo_relay
   from nemo_relay.plugin import PluginConfig

   await nemo_relay.plugin.initialize(PluginConfig())
   ```

   The empty `PluginConfig()` means "discover": Relay layers `/etc/nemo-relay/plugins.toml` over
   it — and that uploaded file is how Agent Hardener's guardrails *and* its ATOF telemetry sink reach
   the victim. Skip this call and every guardrail is inert.

2. **Put Relay in the tool path.** Per framework:

   | Framework | The interception line |
   |---|---|
   | LangChain `create_agent` | `create_agent(model=…, tools=…, middleware=[NemoRelayMiddleware()])` |
   | Hand-built LangGraph graph | `graph.add_node("tools", create_tool_node(TOOLS))` — `from nemo_relay.integrations.langgraph import create_tool_node` |
   | DeepAgents library | `create_deep_agent(**add_nemo_relay_integration(kwargs))` — `from nemo_relay.integrations.deepagents import add_nemo_relay_integration` |
   | Hermes | no code: `export HERMES_NEMO_RELAY_PLUGINS_TOML=/etc/nemo-relay/plugins.toml` and `hermes plugins enable observability/nemo_relay` in the Dockerfile |

   Install the matching extra in the image: `pip install "nemo-relay[langgraph]"` (or
   `[langchain]`). `create_tool_node` preserves ToolNode's argument injection, parallel
   execution, error handling, `Command` results and interrupts; on a nemo-relay wheel that
   predates it (0.8.x from PyPI) the equivalent is
   `ToolNode(TOOLS, awrap_tool_call=NemoRelayMiddleware().awrap_tool_call)`.

3. *(Attribution, recommended)* pass Relay's callback handler on each invocation so a turn's
   scopes nest under one root:

   ```python
   await graph.ainvoke(state, {"callbacks": [NemoRelayCallbackHandler()]})
   ```

## The trap: instrumented-looking but unguarded

On a hand-built LangGraph graph, `NemoRelayCallbackHandler()` alone gives **visibility only** —
LLM scopes appear in the telemetry while tool calls execute outside Relay, so a guardrail is
never consulted. The victim looks instrumented, `--relay-confirmed` blesses it, and every attack
scores as unblocked while the defenses were never in the fight. The interception is the tools
node (`create_tool_node`), not the callback.

## How Agent Hardener verifies the promise

Two checks, two moments; each failure names its cause:

| Error text contains | It means | Fix |
|---|---|---|
| "wrote no Relay telemetry … the file was never created" | Relay never started | obligation 1: `initialize(PluginConfig())` at startup |
| "still empty after 30s; Relay is not attached, or its ATOF sink is disabled" | Relay started but emits nothing | the uploaded `plugins.toml` was not discovered — check `/etc/nemo-relay/` exists and is writable in the image |
| "emitted no new Relay events in 30s (N older records present)" | this invocation went untraced | the serving path bypasses the instrumented agent object |
| "Relay recorded no tool call at all" (after round 1) | tool calls bypass Relay — the trap above | obligation 2: `create_tool_node` / middleware in the tool path |

A worked, verified example of all of it: `plugins/nemo-agent-hardener/examples/langgraph-victim/` —
`agent.py` (tools node wiring) and `server.py` (`initialize()` + per-request scope + callback).
