<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Execution dependencies

For execution failures, compatibility fixes, and reporting, also follow
[Execution recovery](execution-recovery.md). External application access does
not authorize bypassing the selected evaluation runner or replacing its verifier.

Use this in first-eval when evidence points to software or state
outside the agent process: desktop applications, licensed tools, hardware, or
external services. A tool name alone does not establish where it executes.
Stage timing follows [Milestone check-ins](milestone-checkins.md): learn the
requirements while defining scope, implement the supported environment later,
and verify it during validation. Scope does not require a live connection or
finished environment configuration.

## Learn the execution requirements at scope

Follow relevant setup documentation, runner/configuration, and working examples
to establish the dependencies for task execution and result checking. Record the
applicable details below with their source references in the calling workflow's
existing findings file, and carry them into the task README once it exists.
Keep unknowns explicit; these details are not a separate audit or a mandatory
inventory of every tool.

| Detail | What to establish |
|---|---|
| Software and location | App/tool, version and OS; container, desktop host, or remote machine |
| Access path | CLI, API, MCP server, websocket proxy, GUI automation, or an unknown interface |
| Runtime needs | Active desktop/session, display, hardware, authentication, licenses, and supported distribution |
| Starting state | Test data, files, application state, fixture versions, reset procedure, and temporary-output ownership |
| Result collection | How responses, ordered tool calls, failures, or final application state reach the verifier |
| Trial isolation | Separate sessions/data or serial execution against a shared resource |

An enterprise or site license does not itself establish redistribution rights,
headless operation, or container compatibility. Record known constraints instead
of packaging proprietary software on those assumptions.

## Configure a supported execution path during environment preparation

[`eval-author-environment`](../../eval-author-environment/SKILL.md) builds and
proves this path, using the requirements recorded here as its input; the
guidance below continues to apply to dependencies outside the container.

Use the requirements learned at scope. Check the installed Harbor version and
selected backend before configuring the
[task environment](https://www.harborframework.com/docs/task-format). Permitted
network access does not prove reachability or repeatable external sessions.

- **Supported container:** package the client and dependencies once their OS,
  runtime, installation, and licensing needs are understood. A Windows container
  is not automatically an interactive desktop.
- **Desktop or remote host:** document the existing API, MCP gateway, or supported
  automation interface and package only compatible client-side parts. The README
  must distinguish the container contents from the external application; a
  Dockerfile does not provision, license, or make that application reachable.
- **Interactive or inaccessible setup:** retain supported task files and offline
  checks, and document the missing automation/session interface. Do not invent an
  endpoint or substitute a mock application to claim live execution.

Keep container, agent, gateway, and application locations explicit. `localhost`
in a container does not normally refer to the developer's desktop. Verify
authentication, routing, and gateway transport from the actual execution environment;
do not expose a local service publicly to make a draft work.

A tool-sequence check can use recorded traces; checking resulting application
state needs evidence of that state. Preserve that difference in the dependency
plan so an offline verifier test is not mistaken for application execution.

## Explain the external requirement

Name what remains outside Harbor, how it is reached, and which task operation
requires it. For example, when the source establishes a separate application host:

> Your agent uses an application on a separate machine. The Harbor task holds
> requests and grading checks; the application session stays on that host. Live
> evaluation needs access to it and a known starting state for each trial.

If only a gateway is documented, keep the application location unknown. Seek an
existing setup guide or working example before requesting separate technical
details. Distinguish missing documentation, an unverified connection, and a
confirmed incompatibility. For each unresolved requirement, explain the affected
operation and the smallest useful input or implementation needed. Save the
remaining actions in the calling workflow's findings or task README.

## Verify external state and isolation

During validation, check reachability and result collection in a disposable test
session with an agreed reset procedure, preserving existing user sessions.
Resetting the Harbor container does not reset an external application. Keep
trials serial until separate app sessions and state isolation are proven;
preserve conversation and application state across steps within a trial. Include
the external requirements in rerun instructions so a task tied to a particular
desktop is not described as independently portable.
