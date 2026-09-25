# NAT to Fabric transpile spike

Stage 1 proof that an existing NAT (NVIDIA NeMo Agent Toolkit) agent can be migrated to a Fabric-native agent by translation, not by a permanent NAT compatibility shim. It reads a NAT workflow config and emits a Fabric `agent.yaml` for the LangChain Deep Agents harness, plus a migration report that carries MCP servers across and flags what needs auth.

## Why

A NAT agent can be moved to Fabric by translating it into a Fabric-native agent, rather than running its config through a permanent compatibility shim. This spike shows that path end to end for a non-trivial agent, and it makes the boundaries visible, including the tools that need more than a config rewrite.

## What it does

Input: `nat_agent/config.yml`, a deliberately non-trivial NAT topology.

```
reasoning_agent (workflow)
  └─ research_orchestrator (react_agent)
       ├─ math_agent (tool_calling_agent) -> mcp_math  (streamable-http, ${ENV} url)
       ├─ time_agent (tool_calling_agent) -> mcp_time  (stdio) + current_timezone
       ├─ jira_agent (tool_calling_agent) -> mcp_jira  (streamable-http, OAuth2)
       └─ code_generation (builtin)
```

Output: `fabric/agent.yaml` (Deep Agents harness) and `MIGRATION_REPORT.md`.

The transpiler does four things:

1. Walks the NAT composition graph. Sub-agents used as tools become Deep Agents subagents, so the topology survives instead of flattening to one agent. The `reasoning_agent` wrapper is unwrapped onto the executing agent.
2. Maps each NAT LLM to a Fabric model. The main agent's model becomes `models.default`; the rest are emitted as aliases.
3. Carries MCP servers across. NAT `mcp_client` function groups map one-to-one to Fabric `mcp.servers`. stdio becomes a command URL; streamable-http keeps its URL and any `${ENV}` reference.
4. Reports instead of guessing. Env-var URLs are listed for the user to set. NAT builtins are flagged as needing a prebuilt MCP equivalent. Auth that Fabric's adapter cannot carry today is called out.

## Result

Running the transpiler on the fixture produces a config that passes the Fabric contract check, carries all three MCP servers, preserves all three sub-agents, and flags one auth gap. See `MIGRATION_REPORT.md` for the generated findings.

Two findings worth reading before anyone promises "seamless":

- **Auth gap.** Fabric's Deep Agents adapter forwards only `transport` and `url` per MCP server. A `${ENV}` in the URL is the one credential path that reaches the server. NAT servers using OAuth2 providers or custom headers (the `mcp_jira` case here) cannot carry as-is. Closing that is Fabric adapter work.
- **Builtins.** NAT builtin tools (`current_timezone`, `code_generation`) are in-process Python, not MCP servers. Each needs a prebuilt MCP equivalent before it runs under Deep Agents.

## Scope and honesty

This is Stage 1. It does not run the migrated agent against a live Fabric runtime; that is Stage 2 and needs a Fabric environment plus an API key.

The transpiler reads the NAT YAML directly so it runs with only PyYAML. A production version resolves the config through NAT's `WorkflowBuilder.from_config()` to also recover default prompts (which live in NAT's Python, not the YAML) and each tool's resolved schema. The one place this shows up is the main agent's `system_prompt`, emitted here as a `[RESOLVE]` marker.

Validation is a self-contained structural check of the Fabric contract. For full JSON Schema validation, point `--schema` at a local Fabric checkout:

```bash
python3 transpile.py --schema /path/to/nemo-fabric/schemas/agent.schema.json
```

## Analyze mode

The same introspection also runs read-only. `--analyze` skips the Fabric emission and writes `ANALYSIS.md`: the agent's composition tree, models, tools, and anything that would need work to move. It's the low-friction on-ramp: install NeMo Platform, hand it a NAT workflow, and it tells you what your agent does and how it's composed, with no migration commitment.

```bash
python3 transpile.py --analyze                      # writes ANALYSIS.md
python3 transpile.py --in path/to/agent.yml --analyze
```

The migration report and the analysis both call out NAT features this transpiler does not carry (a top-level `middleware:` section such as NASSE maps to NeMo Relay; `memory`, `retrievers`, and similar need a Fabric or Platform equivalent). Those aren't dropped silently; they're listed as work that needs a home.

## Run

```bash
pip install pyyaml            # jsonschema optional, only for --schema
python3 transpile.py          # reads nat_agent/config.yml, writes fabric/agent.yaml + MIGRATION_REPORT.md
python3 transpile.py --analyze   # read-only analysis -> ANALYSIS.md
```
