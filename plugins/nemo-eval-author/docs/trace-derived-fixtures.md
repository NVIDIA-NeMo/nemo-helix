<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace-derived tool-call access

## Prerequisites

- One bounded ATIF v1.0-v1.7 trace, normalized through the trace-environment
  author workflow when needed.
- Python 3.11 or later, `jsonschema>=4.23`, and `referencing>=0.28.4`; Harbor and Docker when proving a
  candidate task. Use the existing supported environment; do not install providers
  automatically. The plugin declares the schema dependency in its uv workspace.
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

Every observed scoped tool receives exactly one reviewed access state:

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
            launch-replay.sh
            integration.toml
```

The inventory, plan, access decision, and generation receipt remain private
construction evidence. Generated mock files enter the task tree and are covered
by the existing reproducibility, contamination, publication-review, and export
gates.

## Tool-call inventory

For each observed scoped tool, the inventory records:

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

Identity is the tuple `(trajectory_path, server, function_name)`, represented by a
stable `tool_id`. Definitions are resolved only inside that scope, never borrowed
from a parent, child, or sibling. Source normalization may preserve an explicitly
recorded server identity in `extra.tool_scope.server` on both the call and its
definition. Missing server identity stays null; do not invent a server from a
function name. Conflicting definitions within a scope stay ambiguous.

Access decisions should copy `tool_id` and `name` from the inventory. Name-only
decisions are accepted only when exactly one scoped tool has that name. Duplicate
names get distinct, deterministic `replay_name` aliases in the plan and MCP
listing. The transport-neutral fixtures retain the original name, scope, and
scoped call references. Identical schemas alone never justify merging scopes.

## Deterministic mock support

The plan reports whether each observed function supports exact replay. A function
is supported when it has a usable definition, object inputs, exactly one textual
output for each call, no redacted values, and no conflicting outputs for the same
canonical input.

The schema must declare an object root, use a supported JSON Schema dialect, and
validate every recorded argument object. Local references are supported;
unavailable references fail closed without network retrieval. Malformed schemas,
unsupported dialects, and argument mismatches are separate plan reason codes.

A mutation annotation does not block replay: the mock does not invoke the
underlying mutation. Instead, `tool_declared_mutating` warns the author that the
recorded output may not reproduce required side effects. Missing read-only
metadata similarly produces `side_effects_unproven`. The author may still choose
`mock`, but Harbor proof must demonstrate that output replay is sufficient for
the task.

Different outputs for the same input are not deterministic exact replay. They
require a future state machine, seeded real dependency, or a `real`/`none`
decision.

### Missing schemas: reviewed overrides

Recorded inputs and outputs can support a mock even when the source did not
capture a tool definition. Author a minimal **replay interface**, review it
against every recorded input, and supply it when creating the plan:

```bash
python <skill_dir>/scripts/trace_environment.py plan-tool-call-access \
  --task-dir <task-dir> --schema-overrides <reviewed-overrides.json>
```

The JSON contract is:

```json
{
  "schema": "nemo.eval_author.trace_environment_schema_overrides.v1",
  "inventory_sha256": "sha256:<digest-of-private/tool-call-inventory.json>",
  "reviewer_kind": "human",
  "entries": [
    {
      "tool_id": "<copy-from-inventory>",
      "input_schema": {
        "type": "object",
        "properties": {"item_id": {"type": "string"}},
        "required": ["item_id"]
      },
      "note": "Reviewed all recorded arguments; this schema describes bounded replay only.",
      "evidence": [
        {"trajectory_path": "$", "step_id": 2, "tool_call_id": "<copy-from-inventory>"}
      ]
    }
  ]
}
```

Use `agent` or `human` for the actual reviewer. Copy all call references for each
overridden tool, in inventory order. Overrides may also repair incomplete or
invalid source schemas, but must not invent outputs, omit observed arguments,
resolve redacted values, or claim the underlying software's full API. They do
not establish that ambiguous original definitions denote one software tool.
The override is a scoped replay contract only.

The helper validates the schema and evidence, retains an owner-private copy as
`private/tool-schema-overrides.json`, binds its digest into the plan, and carries
its provenance through access and fixtures. Source ATIF and inventory remain
unchanged. Review override contents along with the generated publication; do not
include private notes in a distributable mock. An existing plan is immutable:
retain that workspace and create a fresh one when the reviewed contract changes.

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

The decision must select every inventoried scoped tool exactly once. Include
`"tool_id": "<copy-from-inventory>"` when names are not unique. `mock` is allowed
only when the plan reports exact replay support and currently requires the `mcp`
adapter. `real` and `none` do not select an adapter. The normalized decision is
immutable and bound to the inventory and plan digests.

Candidate finalization and subsequent `check` require this complete pipeline
whenever safe ATIF contains calls. Missing function names block candidacy rather
than disappearing from the denominator. Pending and `no_candidate` workspaces may
retain partial inventories/plans without claiming access.

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

Agent harness, model provider, and fixture delivery are separate choices. A
Claude or Mistral model does not require a different fixture format. Select the
user's existing harness and its supported model configuration; do not switch the
user to Codex to obtain mock access. MCP is the currently implemented delivery
adapter, not a requirement that the original tool call came from MCP. A harness
without MCP needs an explicitly implemented native/CLI adapter; do not silently
replace a native call with a shell command or claim those future adapters exist.

## Evaluation integration

Generating a mock is not sufficient to claim access. The task author must copy
the generated directory to `/opt/tool-call-fixtures` in the agent image and merge
`integration.toml` into `task.toml`. Candidate finalization rejects mock access
unless the task config contains the exact `trace-tool-call-replay` stdio MCP
server contract. Harbor NOP, Oracle, and negative-control runs remain the
authority for task solvability and verifier discrimination, **not** for native
tool registration: those agents can solve via shell without discovering MCP.

The generated executable `launch-replay.sh` takes no registration arguments and
resolves its scenario relative to itself. This supports Harbor's current Codex
writer, which otherwise joins the executable and arguments into one executable
name. It does not patch or replace Harbor. Keep the launcher executable and retain
Python 3 and a POSIX shell in the image.

Before proof, run `check-runtime --task-dir <task-dir>` using the helper. This
checks the generic task integration without selecting a model or harness. Add
`--mock-agent <harbor-agent-name-or-module:Class>` to inspect the selected
harness's registration. The helper uses Harbor's factory, not an agent-name
allowlist. This imports and constructs provider/custom code, so use only trusted
installed harnesses. It never runs their emitted shell commands, installs an
agent, or calls a model. Constructor-specific options or unfamiliar APIs may
require a harness-specific execution check instead.

Runtime report v2 separates `mock_integration.registration` from
`mock_integration.execution`. Registration readers inspect literal JSON/TOML
configuration emitted by the installed writer, including map and list encodings:

| Registration status | Evidence and consequence |
|---|---|
| `verified` | The inspected payload retains the executable, arguments, and stdio transport. This is not proof that the harness loads it or calls the mock. |
| `unverified` | No agent selected, unavailable provider, unfamiliar writer/API, or missing evidence. Does not block generic task preflight; native mock access remains unproven. |
| `unsupported` | The recognized registration payload demonstrably drops or changes the required contract. Blocks this preflight for that integration until corrected. |

The default regression suite tests the installed Codex, Claude Code, and Mistral
Vibe writers with identical fixtures and protocol assertions. These are test
cases, not an allowlist. `valid: true` means no required preflight check failed;
it does not imply every integration is verified. Execution is always reported
`unverified` by this static probe, including for verified registration.

Also run the intended evaluation agent with the generated task configuration and
verify that it discovers and calls the mock. Retain its trace and the mock audit
log privately. Check a recorded input, an unseen input, and an unknown function
through the client. The repository's protocol integration tests exercise the
installed Harbor writers plus a real MCP client process; those tests are not a
substitute for an agent/image-specific run. Obtain approval for model spend when
it has not already been requested. If unavailable, report that integration as
unproven rather than equating a direct server invocation with agent access.

For `real`, the existing environment construction path supplies the software,
files, services, credentials, or hardware legitimately required by the task. For
`none`, no fixture is generated. Neither state is assumed successful before
Harbor proof.

## Privacy and limitations

Materialization requires completed contextual privacy review and no unresolved
privacy blocking reasons. Review attestation alone cannot clear a blocker such
as an image-only instruction. Rejection creates neither fixtures nor a receipt.
Redaction markers
are unavailable evidence, not mock values. Generated inputs and outputs are
agent-inspectable in the initial stdio form, so they must not contain verifier
truth or private data.

Exact output replay does not synthesize unrecorded filesystem changes, database
updates, CAD artifacts, time progression, randomness, or other side effects. An
author who chooses `mock` despite a side-effect warning is asserting only that
the recorded output is sufficient for this bounded task; repeated Harbor proof
must validate that assertion.

## Regression checks

From the repository root, in the existing uv environment:

```bash
uv run --frozen pytest plugins/nemo-eval-author/tests/test_tool_call_fixtures.py -v
```

The default suite covers every planner rejection reason, scoped identity,
reviewed overrides and tampering, privacy blockers, candidate decision
completeness, and actual Harbor registration followed by the same MCP
discovery/call checks for Codex, Claude Code, and Mistral Vibe. It also covers
custom import-path resolution and unverified versus unsupported integrations.
The live Codex test below is one optional harness-specific execution example,
not a requirement to use Codex. It never silently spends on a model:

```bash
TRACE_FIXTURE_LIVE_CODEX=1 \
TRACE_FIXTURE_CODEX_AUTH=<existing-auth-json> \
TRACE_FIXTURE_CODEX_MODEL=<authorized-model> \
uv run --frozen pytest \
  plugins/nemo-eval-author/tests/test_tool_call_fixtures.py::test_codex_agent_calls_generated_mock -v
```

That test executes Harbor's actual `Codex.run` with a host process bridge and a
real Codex process in a read-only sandbox. The bridge relocates provider-owned
container paths into the private test directory and tightens Harbor's
container-assuming sandbox-bypass flag to read-only for host safety; it does not
replace registration or tool dispatch. It checks both the recorded result and rejection of unseen
arguments through native MCP events, prohibits shell-based substitution, and
retains synthetic run evidence in pytest's temporary directory. Its private
temporary authentication copy is removed even on failure. The registered
executable and arguments stay as emitted. This proves the host agent/adapter
path, not a Docker image, Harbor trial lifecycle, or verifier isolation.

## Next steps

Scoped inventory, plan, access, and neutral fixtures use v2 contracts. Preserve
older evidence; regenerate in a new workspace from the retained source rather
than editing version strings or attaching old proof to changed artifacts.

- Follow the [trace-environment author workflow](../skills/eval-author-trace-environment/SKILL.md).
- Apply the [environment-integrity proof requirements](../skills/eval-author-trace-environment/references/environment-integrity.md).
