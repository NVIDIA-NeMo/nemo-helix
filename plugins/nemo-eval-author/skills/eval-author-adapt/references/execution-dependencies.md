<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Execution dependencies

Use this when the source shows that an eval depends on software or state outside
the agent process: desktop applications, licensed tools, hardware, or external
services. Read the relevant source evidence before asking questions. A tool name
alone does not establish where it executes.

## Record the execution setup early

Inventory the dependencies needed to perform the task and those needed to check
its result. This uses the same distinction as the software-requirements inventory
in `eval-author-trace-environment`; it does not invoke that flow or its candidate
proof gates. Keep these findings in `.eval-author/adaptation.md` and the created
task's README, with source references and unknowns clearly identified:

| Detail | What to establish |
|---|---|
| Software and location | App/tool name, version and OS when known; whether it runs in the container, on a desktop host, or on a remote machine |
| Access path | How the agent reaches it: CLI, API, MCP server, websocket proxy, GUI automation, or an unverified connection |
| Runtime needs | Required active desktop/session, display, hardware, authentication, and license provisioning; whether container distribution is permitted and supported |
| Starting state | Test data, files, application state, fixture versions, reset procedure, and ownership of temporary outputs |
| Result collection | How responses, ordered tool calls, failures, or final application state reach the verifier |
| Trial isolation | Whether each trial has its own session and data or must run serially against a shared resource |

An enterprise or site license does not itself establish permission to redistribute
binaries, headless operation, or container compatibility. Record known constraints
and leave unknowns unresolved; do not package proprietary software on that assumption.

## Choose a supported path, or document what remains unknown

Harbor's [task format](https://www.harborframework.com/docs/task-format) separates
environment setup from instructions and grading. Its network configuration can
permit access to services, but configuration alone does not prove reachability
or that a service can safely support repeatable trials. Check the installed
Harbor version and selected backend before writing configuration.

- **Software runs in a supported container:** package the required client and
  dependencies only when the OS, runtime, installation, and licensing requirements
  are understood. A Windows container is not automatically an interactive desktop.
- **Software stays on a desktop or remote machine:** keep it there and document
  a proposed connection through the customer's existing API, MCP gateway, or
  supported automation interface. Package only the compatible client-side parts.
  Verify the agent connection and network path when access becomes available;
  do not claim Harbor natively provisions that desktop.
- **Only an interactive or inaccessible setup is known:** create the task files
  and supported offline checks. Document the missing automation/session interface
  as an execution requirement. Do not invent a remote endpoint or replace the
  real application with a mock to claim a live eval works.

Keep container, agent, gateway, and application locations explicit. `localhost`
in a container does not normally refer to the developer's desktop. Authentication,
network routing, and an existing gateway's transport must be verified from the
actual execution environment; do not guess endpoint names or expose a local
service publicly to make a draft work.

Preserve the user's grading goal. A trace-based tool-sequence check can be
developed offline, while checking the resulting application state requires fresh
evidence. Label those separately. A recorded response or mock is a verifier test
fixture, not evidence that the desktop app executed a new request.

## Explain the consequence and keep building

For example, when the source documents an application running on a separate host:

> Your agent uses an application running on a separate machine. The Harbor task can hold
> the requests and grading checks, but its container doesn't automatically include
> that application session. I'll record the connection and setup it needs alongside
> the task. We can create the task and test supported checks now; a live run will
> need access to the app and a known starting state for each run.

If the source describes a gateway but not where its application runs, say so;
do not present the example's separate host as a discovered fact. Ask only for
the next fact needed to implement the affected connection, and reuse answers.
Task creation should continue while app access is unavailable.

At handoff, translate the remaining setup findings into actions using
[Explaining a task draft](explaining-a-task-draft.md#turn-remaining-gaps-into-an-actionable-handoff).
For each actual blocker, name the missing connection, dependency, or starting
state; explain why the eval needs it and what you will verify once it is supplied.
Ask for an existing setup guide or working example before requesting separate
technical details the user may not know. Treat unavailable documentation, an
unverified connection, and a confirmed incompatibility as different findings.

When live execution is available and authorized, verify reachability and result
collection, then exercise the task in a disposable test session with an agreed
reset procedure. Resetting the Harbor container does not reset an external app.
Keep trials serial until independent app sessions and state isolation are proven;
preserve conversation and application state across steps within one trial. Report the
external requirement on the run instructions so a task tied to a particular
desktop is not described as portable or independently reproducible.
