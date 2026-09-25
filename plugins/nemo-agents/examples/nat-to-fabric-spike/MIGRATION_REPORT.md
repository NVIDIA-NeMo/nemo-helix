# NAT -> Fabric migration report

**Status: ready after 4 manual step(s)**

## MCP servers carried across
- mcp_math: streamable-http, url via env var
- mcp_time: stdio, no credentials
- mcp_jira: streamable-http, url via env var

## Environment variables to set before running
- `CORPORATE_MCP_JIRA_URL`
- `MATH_MCP_URL`
- `NAT_REDIRECT_URI`

## Auth requiring action (Fabric adapter gap)
- mcp_jira: NAT used auth_provider 'mcp_oauth2_jira' (_type mcp_oauth2). Deep Agents carries only ${ENV} URLs, so this needs a token-in-URL gateway or Fabric adapter OAuth2 support.

## Builtin tools requiring an MCP equivalent
- `code_generation`: NAT in-process tool. Needs a prebuilt MCP server equivalent before it runs under Deep Agents.
- `current_timezone`: NAT in-process tool. Needs a prebuilt MCP server equivalent before it runs under Deep Agents.

## Errors (must resolve)
- None.

## Features not carried (need a Fabric, Platform, or Relay home)
- None.

## Notes
- Unwrapped reasoning_agent onto 'research_orchestrator' as the main Deep Agent.
- Main agent 'react_agent' had no explicit system_prompt; its default lives in NAT Python and must be resolved via WorkflowBuilder.
- Main-agent direct tools (not sub-agents): code_generation.
