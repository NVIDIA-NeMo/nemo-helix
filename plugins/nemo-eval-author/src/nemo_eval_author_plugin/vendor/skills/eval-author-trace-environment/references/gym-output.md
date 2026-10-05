<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Produce a native Gym task from trace evidence

**Experimental.** Use this extension when Gym is the selected output provider.
Any supported trace source can produce a Gym task. Follow shared Steps 1-5 in
[the skill](../SKILL.md), including privacy review, tool-access decisions,
ground-truth and software inventories, and `check-candidate`. A proposal may
provide context, but an audit report or uncovered tool is not a prerequisite.
Proposal-only requests still stop before construction.

Apply the shared [task fidelity review](task-fidelity.md) to the native task's
initial state, runtime setup, assertions, and solver-visible surfaces as well.

The existing candidate contract applies: complete text instruction, evidenced
requirements, recorded or explicitly reconstructed state, and objective binary
execution verification. A source rollout's reward or final answer is not an
independent oracle. Do not change these rules to fit a Gym template.

## Start from a rollout with ng_trajectory

Select one physical JSONL line from a private Gym rollout file. A record with
both `responses_create_params` / `response` and an `ng_trajectory` attachment
can follow this path after workspace initialization:

```bash
python <skill_dir>/../gym-to-atif/scripts/gym_to_atif.py \
  --input <rollouts.jsonl> --row 1 --output-dir <task-dir>/private/converted
python <skill_dir>/scripts/trace_environment.py prepare \
  --task-dir <task-dir> --atif <task-dir>/private/converted/trace.atif.json \
  --source-kind gym
```

The ATIF converter uses the Responses envelope, not the attachment. To inspect
native attachment evidence as well, use the sibling
[NeMo Compass loader](../../gym-to-atif/references/trace-intel-ingest.md) on the
same row and retain its output under `private/ng-evidence/`. Those normalized
files are not ATIF. An attachment-only export can be loaded for inspection but
cannot currently enter `prepare`; it needs a complete supported Responses
record, original ATIF, or a separately implemented and tested ATIF mapping.
Do not manufacture a Responses envelope from invocation history to bypass this
boundary. Continue shared privacy review and candidacy before scaffolding.

## Native workspace and scaffold

Use the existing Gym v0.6.0+ runtime in its separate Python 3.13.14+
environment. Inspect that installation's CLI help and request models as in the
[Gym authoring guide](../../eval-author-task-create/references/gym-tasks.md).
If it is unavailable, retain the candidate and report construction as blocked;
do not substitute Harbor or claim that the task runs.

Keep the shared evidence at the workspace root. Add:

```text
<task-dir>/
  private/gym-instruction.md
  private/gym-runs/<attempt-id>/
  gym/
    draft.json
    environments/<module>/manifest.yaml
    environments/<module>/config.yaml
    environments/<module>/data/example.jsonl
    resources_servers/<module>/app.py
    resources_servers/<module>/tests/verifier_cases.jsonl
    reproducibility.md
  gym-report.md
```

After `check-candidate` accepts `status: candidate` and contextual privacy
review is complete with no blocking findings, write the exact generalized
`candidate.json` instruction to `private/gym-instruction.md`. Review the
instruction and scaffold metadata for private values. Use a stable task name
and established author identity. From a shell with `umask 077`, call the sibling
native scaffolder directly:

```bash
/path/to/Gym/.venv/bin/python \
  <skill_dir>/../eval-author-task-create/scripts/gym_scaffold.py \
  --out <task-dir>/gym --name <module> \
  --instruction-file <task-dir>/private/gym-instruction.md \
  --description '<generalized scenario>' --author '<actual-author>'
```

Here `<skill_dir>` is the installed `eval-author-trace-environment` directory.
The sibling skill must be installed alongside it. The helper invokes Gym's
native scaffolder, refuses to overwrite a draft, and records `runnable: false`.
Choose a valid Python module name, replacing task-ID hyphens with underscores
(for example, `ledger-total` becomes `ledger_total`); the direct helper does not
normalize names or accept an `org/name` prefix. Do not use
`task_pipeline.py scaffold`: its audit-gap selector belongs to audit-derived
task creation and would require inventing a coverage report for this flow.

## Complete the scenario and tool access

Follow the native layout and implementation guidance under **Complete the task
without changing its intended capability** in the
[Gym authoring guide](../../eval-author-task-create/references/gym-tasks.md).
Use this workspace's `gym/` in place of that guide's audit draft directory.
Replace every example grader, expected answer, and verifier case. Map each
candidate requirement to an observable verifier assertion and a negative
control. Retain the trace step IDs and reconstruction assumptions in the report.

For observed tools, preserve the reviewed `real`, `mock`, or `none` decisions:

- `real`: expose the selected implementation through Gym's configured resources
  server; preserve application, licensing, and access requirements.
- `mock`: retain the generated `call-fixtures.json` and generation receipt as
  provenance. Implement a native resources-server adapter for those reviewed
  cases. Preserve exact argument matching and reject unknown calls; never return
  a recorded success for arbitrary arguments. Document any tool-name mapping.
- `none`: omit the callable surface and verify that the requested outcome is
  still achievable. Do not silently turn a tool-use task into answer copying.

`generate-mock-tool-calls` currently writes its evidence under
`task/environment/tool-call-fixtures/`; keep that location for shared digest
checks. Copy only the required safe fixture data into the Gym resources server.
Do not treat the generated Harbor `integration.toml`, MCP launcher, or
`[[environment.mcp_servers]]` configuration as native Gym registration. Prove
discovery and calls through the actual Gym composition, including mismatched
arguments and reset behavior. Fixture generation alone does not prove wiring.

Agent-visible tool data cannot contain verifier truth. Keep hidden expectations
out of `responses_create_params`, tool outputs, and agent-accessible files.
Check what the selected agent actually receives: a separate JSONL key alone
does not prove isolation. Restore per-session state between attempts and keep
the grading path free of live services and dependency downloads. Gym HTTP
service communication is not Harbor's separate no-network verifier container;
describe and test the actual trust boundary without claiming Harbor isolation.

Create and link `<task-dir>/README.md` when presenting the authored cases, using
[Suite review and rerun instructions](../../eval-author/references/suite-readme.md).
Include the native dataset rows, their readable inputs and grading explanations,
and the reviewed source trace as provenance. Refresh the inventory after native
validation or actual-agent runs, with links to their separate evidence and any
missing-trace or execution blockers.

## Validate and retain native evidence

Write `gym/reproducibility.md` using the **Version and rerun plan** in the
[Gym authoring guide](../../eval-author-task-create/references/gym-tasks.md).
Use candidate and trace provenance instead of an audit recommendation. Follow
[component dependency and packaging checks](../../eval-author-task-create/references/gym-packaging.md)
for the selected consumer. Record runtime and component pins, dataset and code
digests, reset steps, commands, agent/model settings, and unresolved limits.

From `<task-dir>/gym`, use the verified runtime:

```bash
/path/to/Gym/.venv/bin/gym env validate \
  --manifest environments/<module>/manifest.yaml --json
/path/to/Gym/.venv/bin/gym env test <module> --json
```

Retain stdout, stderr, exit status, exact commands and input digests under fresh
`private/gym-runs/<attempt-id>/` directories. A passing manifest or template
test is insufficient. Exercise the completed tools and verifier through the
native runtime, with known-correct, no-action, wrong-answer, and task-specific
negative controls. Run positive and negative controls at least twice from
restored state; verify all scored requirements fail for no action and pass for
the correct interaction. Include malformed input, cross-session state leakage,
and answer-copying controls where they affect this task. Retain all failures.

Use the native startup and `gym eval run --no-serve` procedure under **Validate,
run, and retain evidence** in the Gym authoring guide when actual-agent
execution is in scope and authorized. Keep output under `private/gym-runs/`.
Preserve the user's agent and model; scripted controls establish verifier
behavior, not model performance. Do not invent audit-gap closure or require an
audit report for this trace-derived workflow. Missing credentials leave actual
agent performance unmeasured without invalidating successful control evidence.

Fix defects and repeat affected validation against newly recorded inputs, with
at most three repair iterations. Preserve superseded artifacts and explain each
repair in the report. Do not weaken a requirement to obtain passing controls.

## Private handoff and reporting boundary

Write `gym-report.md` with the candidate decision, source kind, output provider
`gym`, state basis, requirement-to-check mapping, tool access, native task path,
reproducibility plan, retained evidence paths and digests, and remaining blockers.
Report manifest validation, verifier cases, repeated runtime controls, and
actual-agent runs separately as `passed`, `failed`, or `not_run`. Call the
task **unproven** until the completed scenario's native validation and repeated
controls pass; then describe precisely which behavior those checks establish.
Record whether a human supplied or reviewed the generalized task and Relevant
experience. Do not manufacture that review or infer model performance from
reference controls. Review the generated task files separately from the trace.

The shared `check-candidate` and `check` commands validate candidate metadata
and prepared evidence respectively; they do not validate this native product.
Leave the shared `summary.json` pending for a Gym candidate and explain that it
is the preparation record, while `gym-report.md` carries native results.
Keep `gym/draft.json` as the original scaffolding receipt, not a mutable proof
record. Do not run Harbor's `validate-task`, `probe`, `record-reproducibility`,
`record-run-inputs`, `record-validation`, candidate `finalize`, or publication
commands against Gym artifacts, or hand-write their proof JSON. For
`no_candidate`, use the shared no-candidate finalization and `check` normally.

This extension produces private native tasks. Automated Gym readiness,
batch-status aggregation, and digest-bound publication/export are not yet
implemented in `trace_environment.py`; report native batch results separately
with every selected trace in the denominator. Hand off the private task path,
report, and exact rerun instructions. The optional Helix packaging guide can
prepare a separate private package when selected, but it does not authorize
publication or upload. Never export the entire trace workspace.
