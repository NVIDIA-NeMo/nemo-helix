# NAT agent analysis: research_orchestrator

Migrated from a NAT reasoning_agent wrapping react_agent 'research_orchestrator'.

## Composition

- **research_orchestrator** (react_agent, main agent)
  - tool: code_generation
  - **math_agent** (tool_calling_agent, sub-agent)
    - tool: mcp_math
  - **time_agent** (tool_calling_agent, sub-agent)
    - tool: mcp_time
    - tool: current_timezone
  - **jira_agent** (tool_calling_agent, sub-agent)
    - tool: mcp_jira

## Models

- `default`: nvidia / nvidia/llama-3.3-nemotron-super-49b-v1
- `worker_llm`: nvidia / nvidia/nemotron-3-nano-30b-a3b

## Tools

MCP servers:
- `mcp_math` (streamable-http)
- `mcp_time` (stdio)
- `mcp_jira` (streamable-http)
NAT builtins (would need an MCP equivalent to run under Deep Agents):
- `code_generation`
- `current_timezone`

## Summary

4 agent(s), 3 MCP server(s), 2 NAT builtin(s), 0 feature(s) needing another home, 0 unresolved item(s).
