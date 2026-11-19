<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Linear implementation planner

A standalone LangChain Deep Agents agent that reads a Linear ticket, inspects
a public GitHub repository, and produces an implementation plan for a developer
or coding agent. It does not implement or publish changes.

The investigation approach is inspired by LangChain's public
[planning example](https://github.com/langchain-ai/deepagents/tree/main/examples/deploy-coding-agent/skills/planning).
The instructions and artifact contract here are newly authored for Helix.

## Input and output

Replace `ticket` in `input.example.json` with a real Linear issue ID or URL.
`repo`, `base_branch`, and `constraints` are optional caller inputs. When they
are omitted, the planner resolves the repo from the ticket and the repo's default
branch. Ambiguous requirements produce `needs_clarification`, not an invented plan.
The planner resolves and records the commit SHA itself.

The job's `output_workdir` contains `workspace/plan.json` and
`workspace/plan.md`. The JSON statuses are `ready`, `needs_clarification`, and
`failed`. Check job success and validate the artifact contract before using a
plan. Only `ready` indicates a complete plan; other statuses carry unresolved
questions or errors. The artifacts include the ticket reference, repository,
resolved base SHA, tasks, and acceptance criteria. They can be read directly or
consumed by another application. These contracts are instructions to the model,
not schemas enforced by the platform.

## Runtime requirements

- A container with the Helix execute-agent runtime, the compatible Deep Agents
  extra/adapter, Python 3, Git, and outbound access to GitHub and Linear.
  The config uses `deepagents.backend.type: local_shell` and `tools.enabled`;
  verify those capabilities in the chosen image. Run this inside a dedicated
  job container, not on a host:
  the shell tool is not a filesystem sandbox and planning-only behavior is a
  prompt instruction.
- Access to `internal-agents/azure-anthropic-claude-opus-5-5` on the hosted
  Helix instance. This route previously passed a Chat Completions tool-call
  probe. Deep Agents runtime compatibility still needs
  an end-to-end trial. The model workspace is distinct from the job workspace.
- A Linear API key with Read permission for the required team/issues, stored
  as a Helix secret and injected into the job as `LINEAR_API_KEY`.
  [Linear's read-only MCP endpoint](https://linear.app/docs/mcp) accepts API keys
  in the Bearer header. The YAML contains only the environment reference.
  Your chat application's Linear connection is not inherited by a Helix job.
- Public GitHub repositories for this initial demo. No GitHub credential is
  injected. Private repository support requires a separate credential design.

Set the execute job's inline environment secret reference, for example:

```json
{
  "environment": {
    "environment_spec": {
      "provider": "local",
      "secrets": {
        "LINEAR_API_KEY": "mmogallapall/linear-planner-token"
      }
    }
  }
}
```

`linear-planner-token` is a proposed secret name, not a provisioned credential.
The MCP tool allowlist expects `get_issue`, `list_comments`, and `get_document`;
confirm those names with the authenticated server before the first hosted run.
No agent is registered and no job is launched by adding these files.
