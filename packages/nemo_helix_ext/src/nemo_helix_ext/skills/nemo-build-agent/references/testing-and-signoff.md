# Testing and signoff

Use this reference after the Ethos is approved and before registration.

## Derive one acceptance contract

Turn the approved Ethos examples and success criteria into one versioned test
case file in the generated project. Reuse those cases for local behavioral
tests, deployed invocations and evaluation. Do not create separate definitions
that can drift.

Each case should identify:

- input and expected outcome;
- allowed and forbidden side effects;
- required or forbidden tool calls;
- required ordering when observable;
- whether live credentials are required;
- the pass rule.

## Required local gates

1. Unit test every MCP operation, including invalid input, denied actions,
   upstream errors, timeouts and redacted error messages.
2. Contract test MCP schemas, tool discovery and structured outputs.
3. Run the Ethos cases against the assembled local agent shape.
4. Record trajectories when tool choice, approval or ordering is part of the
   result.
5. Run a separate live smoke test only when the required credentials and
   network access are available.

Mock business systems for ordinary tests. Never run representative production
side effects as evaluation. For a live side effect, use an approved sandbox or
test tenant and obtain explicit approval for the specific operation.

## Delivery reachability

A source file or passing import test does not prove that Fabric can call a
tool. Before registration, verify every required executable capability is
reachable from `agent.yaml` through `mcp.servers`, and every instruction package
is reachable through `skills.paths`.

After image deployment:

1. Invoke a case that requires each MCP server.
2. Confirm the expected tool appears in the trajectory and returns the expected
   schema.
3. Exercise one denied action and one upstream failure.
4. Confirm telemetry contains the invocation, tool and error or approval spans
   expected by the Ethos.

## Status language

Use these states consistently:

| State | Minimum evidence |
|---|---|
| Built | Approved Ethos, generated artifacts and required local tests pass |
| Onboarded | Fabric validation, registration, deployment and telemetry checks pass |
| Production candidate | Live representative checks pass in the target environment and every required Ethos threshold is met |

A skipped live test, missing telemetry or untested production integration
prevents `Production candidate` status. Report passed, failed and skipped checks
separately.
<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
