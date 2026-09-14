<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace-derived tool-call access

## Prerequisites

- One bounded ATIF v1.0-v1.7 trace, normalized through the trace-environment
  author workflow when needed.
- Python 3.11 or later; Harbor and Docker when proving a candidate task.
- A completed contextual privacy review before materializing mock calls.

## Purpose

A tool is a software dependency such as a CAD application, CLI, database, or
service. A tool call is an invocation recorded by ATIF: a function name, input,
and paired output. They are related but have different environment decisions.

The evaluation may receive the real tool through its normal environment, receive
a deterministic mock of recorded tool calls, or receive no access. The trace can
establish exact call cases without establishing that the underlying software is
available. The evaluation author therefore chooses access explicitly; the
compiler never silently replaces real software with a mock.

## Access states

Every observed function receives exactly one reviewed access state:

| Access | Meaning |
| --- | --- |
| `real` | The task author provides the real tool through the evaluation environment. Harbor execution must prove it is usable. |
| `mock` | The evaluation receives a callable surface backed by deterministic trace-derived input/output cases. A supported adapter is required. |
| `none` | The author intentionally provides neither the real tool nor a mock. Harbor execution determines whether the resulting task remains solvable. |

These states describe evaluation access, not whether the original tool exists on
the author's machine. Software availability and licensing remain in
`candidate.json` when they matter, but they do not decide tool-call replay.

## Data flow

```text
canonical source
  -> safe ATIF
  -> private/tool-call-inventory.json
  -> private/tool-call-plan.json
  -> author decisions: real | mock | none
  -> private/tool-access.json
  -> selected mock calls only
       task/environment/tool-call-fixtures/call-fixtures.json
       -> MCP adapter
            mcp-scenario.json
            mcp_replay.py
            integration.toml
```

The inventory, plan, access decision, and generation receipt remain private
construction evidence. Generated mock files enter the task tree and are covered
by the existing reproducibility, contamination, publication-review, and export
gates.

## Tool-call inventory

For each observed function, the inventory records:

- the original definition and a normalized name and input schema;
- every argument object, call ID, paired observation, trajectory path, and ATIF
  step ID;
- explicit read-only metadata when present, as context rather than a replay
  requirement; and
- uncertainties such as missing or ambiguous definitions, missing observations,
  non-object arguments, and redacted values.

The compiler accepts MCP-shaped definitions (`name` and `inputSchema`) and
OpenAI-shaped definitions (`function.name` and `function.parameters`). This is
an input normalization rule. It does not make MCP the fixture model.

## Deterministic mock support

The plan reports whether each observed function supports exact replay. A function
is supported when it has a usable definition, object inputs, exactly one textual
output for each call, no redacted values, and no conflicting outputs for the same
canonical input.

A mutation annotation does not block replay: the mock does not invoke the
underlying mutation. Instead, `tool_declared_mutating` warns the author that the
recorded output may not reproduce required side effects. Missing read-only
metadata similarly produces `side_effects_unproven`. The author may still choose
`mock`, but Harbor proof must demonstrate that output replay is sufficient for
the task.

Different outputs for the same input are not deterministic exact replay. They
require a future state machine, seeded real dependency, or a `real`/`none`
decision.

## Reviewed decision

The author provides a complete decision file before any mock is generated:

```json
{
  "schema": "nemo.eval_author.trace_environment_tool_access_decisions.v1",
  "decisions": [
    {
      "name": "cad.inspect_model",
      "access": "real",
      "adapter": null,
      "note": "The licensed CAD runtime is supplied by the task image."
    },
    {
      "name": "catalog.lookup",
      "access": "mock",
      "adapter": "mcp",
      "note": "The trace contains all deterministic lookup cases needed here."
    },
    {
      "name": "telemetry.publish",
      "access": "none",
      "adapter": null,
      "note": "Publishing is not required for the generalized task."
    }
  ]
}
```

The decision must name every inventoried function exactly once. `mock` is allowed
only when the plan reports exact replay support and currently requires the `mcp`
adapter. `real` and `none` do not select an adapter. The normalized decision is
immutable and bound to the inventory and plan digests.

## Neutral call fixtures and adapters

`call-fixtures.json` is transport-neutral. It contains selected function names,
definitions, exact inputs, recorded outputs, and evidence steps. An adapter makes
that logical callable surface available to an evaluation agent.

The initial adapter is MCP because Harbor can pass task-owned MCP servers to many
agent harnesses. MCP is the delivery mechanism, not the meaning of a tool call.
The generated stdio server implements `initialize`, `ping`, `tools/list`, and
`tools/call`. It matches only a function name plus canonical JSON input. Unknown
functions and unmatched inputs fail with a generic error and never reveal the
expected request. There is no live-service fallback.

Future adapters can consume the same neutral fixture contract to register native
functions, provide CLI shims, or expose HTTP endpoints without changing the
trace-derived evidence or author decision.

## Evaluation integration

Generating a mock is not sufficient to claim access. The task author must copy
the generated directory to `/opt/tool-call-fixtures` in the agent image and merge
`integration.toml` into `task.toml`. Candidate finalization rejects mock access
unless the task config contains the exact `trace-tool-call-replay` stdio MCP
server contract. Harbor NOP, Oracle, and negative-control runs remain the
authority that the evaluation can actually call it and that the task resets
between trials.

For `real`, the existing environment construction path supplies the software,
files, services, credentials, or hardware legitimately required by the task. For
`none`, no fixture is generated. Neither state is assumed successful before
Harbor proof.

## Privacy and limitations

Materialization requires completed contextual privacy review. Redaction markers
are unavailable evidence, not mock values. Generated inputs and outputs are
agent-inspectable in the initial stdio form, so they must not contain verifier
truth or private data.

Exact output replay does not synthesize unrecorded filesystem changes, database
updates, CAD artifacts, time progression, randomness, or other side effects. An
author who chooses `mock` despite a side-effect warning is asserting only that
the recorded output is sufficient for this bounded task; repeated Harbor proof
must validate that assertion.

## Next steps

- Follow the [trace-environment author workflow](../skills/eval-author-trace-environment/SKILL.md).
- Apply the [environment-integrity proof requirements](../skills/eval-author-trace-environment/references/environment-integrity.md).
