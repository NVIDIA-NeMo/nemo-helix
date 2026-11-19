<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Linear ticket planner

A standalone LangChain Deep Agents agent that turns a Linear ticket and relevant
comments into tasks, acceptance criteria, verification scenarios, and open
questions. It reads Linear through its read-only MCP endpoint and writes
`plan.json` and `plan.md` using the default filesystem backend.

No source checkout, GitHub access, shell backend, or `tools.enabled` field is
required. Shell execution and subagent delegation are blocked. Plans describe
ticket requirements; they do not claim to be verified against the codebase.

## Input and output

Replace `ticket` in `input.example.json` with a real Linear issue ID or URL.
`constraints` is optional. No repository, branch, or commit is required.

The job's `output_workdir` archive contains `plan.json` and `plan.md` at its root.
The JSON status is `ready`, `needs_clarification`, or `failed`. Check job success
and parse/validate the artifact contract before using a plan; these contracts
are instructions to the model, not schemas enforced by the platform.

Missing or conflicting requirements produce `needs_clarification`. Completed
or canceled tickets also require clarification unless the caller explicitly
requests a retrospective plan. A retrospective plan describes the requirements
without suggesting that the work is outstanding.

## Runtime requirements

- A Helix execute-agent image containing the compatible Deep Agents adapter
  and harness, plus outbound access to Linear and the model gateway. No Git
  executable or GitHub credential is needed.
- Access to `azure-anthropic-claude-opus-5-5` through local Helix in workspace
  `mmogallapall`. Use the shared `pr-compose:local` image, which includes Deep Agents.
  An earlier configuration passed a hosted trial on AIRCORE-1634. Local job
  `nemo-agents-plugin-e-ba9fepiq` also completed with a `ready` retrospective plan
  and both expected artifacts on 2026-11-19.
- A Linear API key with Read permission for the required team/issues, stored
  as a Helix secret and injected as `LINEAR_API_KEY`.
  [Linear's read-only MCP endpoint](https://linear.app/docs/mcp) accepts API keys
  in the Bearer header. The YAML contains only the environment reference.
  Your chat application's Linear connection is not inherited by a Helix job.

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

`linear-planner-token` is provisioned in `mmogallapall` for the tested setup.
For another workspace, supply its own secret reference.
The instructions use `get_issue` and `list_comments`; confirm those names with
the authenticated server during the first local run.

## Output limits

The model configuration explicitly requests an 8192-token output allowance. The
initial hosted trial read Linear successfully but repeatedly truncated file-writing
calls at the route's default 4096-token limit. The instructions now bound artifact
size and retries, and `runtime.max_turns: 20` bounds the Deep Agents graph steps. Set
`timeout_seconds: 300` on execute jobs for trials. Older runtimes may exceed this
deadline while cancelling an in-flight request, so it is not a hard wall-clock cap.

## Run locally

`agent.yaml` directly uses the local gateway route, an 8192-token output
allowance, and a bounded agent loop. The prepared `run.local.json` contains an
inline copy of this config, the trial input, shared image, and Linear secret
reference. See the [local job commands](../README.md#local-planner-and-reviewer-jobs).

Alternatively, register `agent.yaml` on local Helix and use `run.example.json`
with `--image pr-compose:local --profile gpu`. Replace its ticket placeholder
first. For a completed ticket, explicitly request a retrospective plan in the
input's `constraints`; otherwise the planner asks whether new work is intended.
