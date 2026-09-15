<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Audit Spec Measurement And Coverage Assumptions

The audit-spec scripts treat `audit.md` as the coverage denominator. Measurement
reads one task trace, compares it to the audit items, and writes per-task,
per-run, per-method artifacts. Aggregation reads those coverage artifacts and
writes one coverage report.

Current assumptions:

- Measurement input is an ATIF trajectory JSON file. `measure.py` accepts either
  `--trace <trajectory.json>` or a Harbor trial directory containing
  `agent/trajectory.json`.
- ATIF parsing is delegated to Harbor via
  `harbor.models.trajectories.Trajectory.model_validate_json`. The scripts do not
  maintain a separate ATIF parser or schema mirror.
- Harbor is a required dependency for measurement. Use `requirements.txt` when
  running `measure.py` outside a repository environment that already provides
  Harbor.
- Capability measurement may also consume a local skill-authored judgment
  sidecar with `--capability-judgments`. That sidecar is for non-tool evidence
  only; deterministic tool requirements still come from the ATIF trace. The
  sidecar's required `trace_sha256` binds its judgments to the exact ATIF bytes
  that were inspected, and measurement rejects a digest mismatch.
- Failure-case measurement follows the capability pattern with
  `--failure-case-judgments`. It also deterministically requires every
  `prohibited_tools` value to be absent from the ATIF trace.
- Measurement does not infer that a failure-case `trigger` occurred from its
  prose field. Authors must represent the trigger in `evidence_required` and
  judge that evidence from the trace; otherwise unrelated traces, especially a
  failure case containing only `tool_call` evidence, can overstate coverage.
- Reports are written under encoded path components:
  `<out-dir>/task=<task-id>/run=<run-id>/<method>/coverage.json` and
  `details.json`. The raw `task_id` and `run_id` remain in the JSON payloads.
  If no run id is provided by the CLI or Harbor result metadata, measurement
  derives one from the ATIF trajectory identity, then from the trace content
  digest.
- Aggregation input is one or more per-trace `coverage.json` files. `report.py`
  validates those files against `schemas/audit_coverage.schema.json`, rejects
  hard denominator mismatches against the current `audit.md`, and writes the
  aggregate report to the path supplied with `--out`.
- Aggregation treats audit status as review metadata, not denominator content.
  Status-only mismatches, such as measuring while an audit is `draft` and
  reporting after it becomes `approved`, are emitted in `warnings` without
  rejecting the input coverage file.
- Aggregation does not read ATIF traces, Harbor result files, or method-specific
  `details.json` files. It only unions stable audit item names from
  `coverage.json`.
- The aggregate report contains count summaries plus `uncovered_items`. Each
  uncovered item preserves the original audit item and adds generation-oriented
  context: the reason it is uncovered, a one-sentence focus, likely needed
  tools, and required evidence.
- The aggregate report records `measured_kinds` from the input coverage files.
  An uncovered item whose kind is absent from `measured_kinds` uses
  `reason: not_measured_by_any_method`; an uncovered item whose kind was measured
  uses `reason: not_covered_by_any_input_report`.
- Harbor's current `Trajectory` model is treated as the reader for now, but it
  may prove too strict for measurement if valid evaluator traces include producer
  extensions, omit fields that coverage does not consume, or move to a newer ATIF
  version before Harbor models are updated. If that happens, prefer contributing
  or adopting a permissive Harbor consumer reader before maintaining a local ATIF
  parser here.
- Richer deterministic predicates such as argument, output, state, verifier,
  and ordering checks remain out of scope. Keep additional gates inside each
  composite item method unless aggregation learns prerequisite-aware merging.

## Script Inventory

| Script | Use it to |
|---|---|
| `scripts/audit_spec/generate.py` | Create, reconcile, replace, or preview `.eval-author/audit.md` from `ETHOS.md` and reviewed item proposals |
| `scripts/audit_spec/measure.py` | Measure one ATIF trace or Harbor trial directory against `audit.md` and write coverage/details files for each selected method |
| `scripts/audit_spec/report.py` | Aggregate per-trace `coverage.json` files into one coverage report with uncovered audit items |
| `scripts/audit_spec/validate.py` | Validate the marked audit-spec block in `audit.md` |

Private shared helpers live in `scripts/audit_spec/_schema.py`,
`scripts/audit_spec/_markdown.py`,
`scripts/audit_spec/measurements/_composite.py`, and
`scripts/audit_spec/measurements/trace_tools.py`. Measurement methods live under
`scripts/audit_spec/measurements/`; v1 ships
`scripts/audit_spec/measurements/tool_calls.py` and
`scripts/audit_spec/measurements/capabilities.py`, plus
`scripts/audit_spec/measurements/failure_cases.py`.

## Schemas And Examples

| Artifact | Path |
|---|---|
| Shared coverage schema | `schemas/audit_coverage.schema.json` |
| Capability judgment input schema | `schemas/audit_capability_judgments.schema.json` |
| Failure-case judgment input schema | `schemas/audit_failure_case_judgments.schema.json` |
| Aggregate coverage report schema | `schemas/audit_coverage_report.schema.json` |
| Capability details schema | `schemas/audit_capabilities_details.schema.json` |
| Failure-case details schema | `schemas/audit_failure_cases_details.schema.json` |
| Tool-call details schema | `schemas/audit_tool_calls_details.schema.json` |
| Tool-call coverage example | `examples/schemas/tool_calls.coverage.json` |
| Tool-call details example | `examples/schemas/tool_calls.details.json` |
| Capability judgment input example | `examples/schemas/capability_judgments.json` |
| Capability coverage example | `examples/schemas/capabilities.coverage.json` |
| Capability details example | `examples/schemas/capabilities.details.json` |
| Failure-case judgment input example | `examples/schemas/failure_case_judgments.json` |
| Failure-case coverage example | `examples/schemas/failure_cases.coverage.json` |
| Failure-case details example | `examples/schemas/failure_cases.details.json` |
| Aggregate coverage report example | `examples/schemas/coverage_report.json` |

## Measurement Methods

| Method | Covers | Evidence |
|---|---|---|
| `tool_calls` | `tool` items | Matches each tool item's `name` against ATIF `steps[].tool_calls[].function_name`, including embedded subagent trajectories |
| `capabilities` | `capability` items | Requires every declared `required_tools` value and every `tool_call` evidence predicate to appear in the trace; non-tool evidence can be satisfied by a structured skill-authored judgment sidecar |
| `failure_cases` | `failure_case` items | Requires prohibited tools to be absent and every evidence predicate to be satisfied; `tool_call` evidence is deterministic and non-tool evidence uses a structured skill-authored judgment sidecar |

`capabilities` treats deterministic requirements as hard gates. A supplied
judgment can satisfy non-tool evidence kinds such as `user_intent`, `output`,
`outcome`, `policy_boundary`, and `verifier`, but it cannot cover a capability
when required tools or `tool_call` evidence are missing. Missing judgments are
reported as `unjudged` and leave the capability uncovered. Judgments bound to a
different trace digest are rejected before reports are written. Unknown evidence
kinds remain `unsupported`.

`failure_cases` uses the same evidence and judgment rules. It additionally treats
each declared prohibited tool as a hard gate. Other failure-case fields are
rubric context; conditions that must gate coverage belong in
`evidence_required`.
