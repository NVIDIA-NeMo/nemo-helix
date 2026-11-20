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

## The obligations — all of them, not one

1. **Start Relay and let it discover its config**, once, at startup:

   ```python
   import nemo_relay
   from nemo_relay.plugin import PluginConfig

   async with nemo_relay.plugin.activate(PluginConfig()):
       ...  # serve requests inside the block (e.g. a FastAPI lifespan)
   ```

   The empty `PluginConfig()` means "discover": Relay layers `/etc/nemo-relay/plugins.toml` over
   it — and that uploaded file is how Agent Hardener's guardrails *and* its ATOF telemetry sink reach
   the victim. Skip this call and every guardrail is inert. The activation owns the plugin host:
   keep it open for the process lifetime. A bare `await nemo_relay.plugin.initialize(...)` that
   discards its result lets the host shut down, and telemetry is then lost at random.

2. **Put Relay in the tool path.** Per framework:

   | Framework | The interception line |
   |---|---|
   | LangChain `create_agent` | `create_agent(model=…, tools=…, middleware=[NemoRelayMiddleware()])` |
   | Hand-built LangGraph graph | `graph.add_node("tools", create_tool_node(TOOLS))` — `from nemo_relay.integrations.langgraph import create_tool_node`; or, to keep your own `ToolNode` options, `ToolNode(TOOLS, awrap_tool_call=NemoRelayMiddleware().awrap_tool_call)` (what `examples/langgraph-victim` does) |
   | DeepAgents library | `create_deep_agent(**add_nemo_relay_integration(kwargs))` — `from nemo_relay.integrations.deepagents import add_nemo_relay_integration` |
   | Hermes | no code: `ENV HERMES_NEMO_RELAY_PLUGINS_TOML=/etc/nemo-relay/plugins.toml` in the Dockerfile (what `examples/hermes-victim` does). Do not run `hermes plugins enable observability/nemo_relay`: current Hermes removed that plugin, and enabling it fails the image build |

   Install the matching extra in the image: `pip install "nemo-relay[langgraph]"` (or
   `[langchain]`). `create_tool_node` preserves ToolNode's argument injection, parallel
   execution, error handling, `Command` results and interrupts.

3. **Put Relay in the LLM path** (hand-built LangGraph graph only — `create_agent` middleware and
   the other rows above already do it). Send each model call through `nemo_relay.llm.execute`
   instead of calling the model directly:

   ```python
   import nemo_relay
   from langchain_core.messages import SystemMessage, messages_from_dict, messages_to_dict
   from nemo_relay.integrations.langchain._serialization import LANGCHAIN_MODEL_RESPONSE_KEY, LangChainCodec

   async def model(state: MessagesState) -> dict[str, list]:
       messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
       codec = LangChainCodec()
       request = nemo_relay.LLMRequest({}, {"messages": messages_to_dict(messages), "model": MODEL})

       async def call(request: nemo_relay.LLMRequest) -> Any:
           reply = await llm.ainvoke(messages_from_dict(request.content["messages"]))
           return {LANGCHAIN_MODEL_RESPONSE_KEY: {"messages": messages_to_dict([reply])}}

       result = await nemo_relay.llm.execute(MODEL, request, call, model_name=MODEL, codec=codec, response_codec=codec)
       return {"messages": messages_from_dict(result[LANGCHAIN_MODEL_RESPONSE_KEY]["messages"])}
   ```

   A bare `await llm.ainvoke(...)` in the model node records no LLM span. Guardrails on tool calls
   still fire, but each attack's trace then has no prompt and no model steps, so the traces
   guardrails defender cannot match an attack to its trace and writes no rail from it.

4. *(Attribution, recommended)* pass Relay's callback handler on each invocation so a turn's
   scopes nest under one root:

   ```python
   await graph.ainvoke(state, {"callbacks": [NemoRelayCallbackHandler()]})
   ```

## The trap: instrumented-looking but unguarded

On a hand-built LangGraph graph, `NemoRelayCallbackHandler()` alone gives **visibility only** —
graph scopes appear in the telemetry while tool calls execute outside Relay, so a guardrail is
never consulted. The victim looks instrumented, `--relay-confirmed` blesses it, and every attack
scores as unblocked while the defenses were never in the fight. The interception is the tools
node (`create_tool_node`), not the callback.

The half-way version of the same trap: the tools node is wired but the model node calls the model
directly. Tool calls are guarded, but the callback handler records no LLM call either, so the
traces have no LLM events and no attack can be matched to its trace. Obligation 3 closes it.

## How Agent Hardener verifies the promise

Two checks, two moments; each failure names its cause:

| Error text contains | It means | Fix |
|---|---|---|
| "wrote no Relay telemetry … the file was never created" | Relay never started | obligation 1: `plugin.activate(PluginConfig())` at startup, held open |
| "still empty after 30s; Relay is not attached, or its ATOF sink is disabled" | Relay started but emits nothing | the uploaded `plugins.toml` was not discovered — check `/etc/nemo-relay/` exists and is writable in the image |
| "emitted no new Relay events in 30s (N older records present)" | this invocation went untraced | the serving path bypasses the instrumented agent object |
| "Relay recorded no tool call at all" (after round 1) | tool calls bypass Relay — the trap above | obligation 2: `create_tool_node` / middleware in the tool path |

A worked example of all of it: `plugins/nemo-agent-hardener/examples/langgraph-victim/` —
`agent.py` (tools node and `llm.execute` model node) and `server.py` (`activate()` + per-request
scope + callback).
