<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Review adapted eval coverage

Use after the user accepts the coverage audit offered at the end of adaptation,
or directly requests this review. The purpose is to compare the agent's intended
behavior with the newly created evals and deliver an actionable coverage report.
This review does not require the tasks to have run.

## Carry forward the work already done

Reuse the known Ethos path, selected task paths and IDs, adaptation findings,
remaining grading/setup requirements, and run artifacts. Inspect the current files
before relying on saved completion marks. Keep the agreed scope explicit; do not
silently include unrelated suites or historical runs from another agent version.
Keep mock, recorded replay, reference-solution runs, and real-agent runs distinct.

Follow the audit skill's Ethos pre-flight. If Ethos is missing, explain why the
audit needs it and use the bundled Local Ethos procedure to create and review it,
or use the file the user supplies. Preserve
the completed adaptation while that input is pending. Do not replace Ethos with
the task set, a report, or a placeholder contract.

## Establish what should be covered

Follow audit Steps 1–3 to draft, generate/reconcile, and validate `.eval-author/audit.md`.
Read the prescribed template and schema first. Derive the tools, capabilities,
and failure cases from Ethos and established runtime names; retain existing
reviewed audit items according to reconciliation rules. The created tasks help
map coverage but must not narrow the denominator to only what they already test.
Keep unresolved terminology or intent explicit rather than inventing tool names
or silently changing the Ethos.

## Compare the created evals with the audit items

Inspect each selected task's instructions, fixtures, written criteria, implemented
verifier, and recorded readiness. For each audit item, identify the relevant task
and step and distinguish:

- **Planned coverage:** the task asks for or creates the situation described by
  the audit item. Name the supporting instruction or fixture.
- **Grading implemented:** the actual checks assess the required behavior.
  Written rubrics and placeholders alone do not establish this. Identify partial
  checks and any evidence the grader still needs.
- **No matching case or unclear coverage:** explain what is missing or ambiguous
  and the consequence for testing the Ethos requirement.

Map failure cases to tasks that actually create their trigger and check the
expected response. Mentioning a failure in instructions is not sufficient.
Link the task evidence and retain contradictions between Ethos and source eval
criteria as decisions for the user; do not rewrite either during the audit.

Write the human-readable review to `.eval-author/adapted-eval-coverage.md`, with
the Ethos and audit paths, reviewed task scope, a per-item mapping, missing or
partial checks, and prioritized next actions. For each action, say what it would
enable and whether it needs a user decision, implementation, or execution access.
This is a task-coverage review, separate from the trace aggregation JSON schema.

## Add measured coverage only when evidence supports it

When suitable ATIF traces for the selected tasks are available, use audit Steps
4–5 to measure and aggregate them. Select their exact coverage inputs; do not
include stale or unrelated measurements through a broad directory scan. Report
which agent/run and measurement methods support the findings. Recorded or mock
evidence must not be presented as a new live run of the adapted evals.

If suitable traces are absent, deliver the validated audit specification and
task-coverage review now, and state that measured coverage is pending. Do not
fabricate trajectories, `coverage.json`, a measured percentage, or zero coverage
to represent “not measured.” Do not run agents solely to complete this audit
without the required execution authorization.

In the reply, summarize what the evals address, the most useful gaps to resolve,
and what remains unmeasured. Link the audit specification and review, plus the
aggregate report when one was produced. An accepted coverage audit ends with
these findings; further task proposals or creation follow only when requested.
