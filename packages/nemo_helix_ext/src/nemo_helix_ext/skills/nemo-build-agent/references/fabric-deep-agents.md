# Fabric Deep Agents contract

Use this reference when choosing the agent's implementation shape.

## Ownership boundary

NeMo Helix stores `agent.yaml` and translates it into a Fabric config. The
Fabric Deep Agents adapter constructs `create_deep_agent` from that normalized
config. It does not import a customer `agent.py` or accept Python callables in
`harnesses.<name>.settings`.

Use only these supported surfaces:

| Requirement | Delivery surface |
|---|---|
| System behavior and model | `instructions` and `models` in `agent.yaml` |
| Reusable instruction package | Agent Skill referenced by `skills.paths` |
| Executable custom tool | MCP server declared by `mcp.servers` |
| Specialized delegate | Declarative subagent under `harnesses.deepagents.settings.deepagents.subagents` |
| Remote background delegate | Supported asynchronous subagent configuration |

Never create `src/<package>/agent.py` as an entry point for this path. Fabric
will not load it.

## Custom tools

Expose local Python tools as an MCP server. The generated project should install
a console script:

```toml
[project.scripts]
support-tools = "support_agent.server:main"
```

Declare that executable in `agent.yaml`:

```yaml
mcp:
  servers:
    support-tools:
      transport: stdio
      url: support-tools
```

Keep credentials out of the config and source tree. Declare environment
variable names and resolve secrets through the selected NeMo environment.

For local testing, run the MCP server and its tools from the generated `uv`
project. For deployment, package the project as an image so the console script
is installed on the runtime `PATH`. Do not assume a nested project virtual
environment is visible to a Platform subprocess deployment.

## Ordering and approvals

Prompt instructions and subagent delegation do not guarantee execution order.
When several steps form one safety or transactional invariant, expose one MCP
operation that implements the state machine and tests the complete transition.
For example, `issue_verified_refund` should verify identity and issue the refund
inside one controlled operation instead of exposing two freely ordered tools.

Do not use `interrupt_on` for a deployed workflow until the selected Platform
runtime exposes and tests an end-to-end pause and resume contract. The current
adapter accepts the setting, but deployed invocation does not provide a verified
approve, edit or reject resume path. Configuration acceptance is not runtime
support.

A local compiled LangGraph subagent cannot be serialized through the current
Platform config. If the required behavior cannot be represented by a
deterministic MCP operation, a supported declarative subagent or a supported
remote agent, stop and report the adapter gap.

## Version boundary

Read the installed Fabric Deep Agents adapter descriptor and the NeMo Agents
config model before emitting adapter settings. The optional NeMo Agents plugin
supplies the adapter and its Deep Agents runtime dependency. Reuse the installed
versions and let the plugin dependency metadata select compatible versions. Do
not add a separate harness constraint to the generated customer project unless
its own code directly imports that API.
<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
