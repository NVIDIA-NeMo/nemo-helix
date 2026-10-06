<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Proposal-to-construction decision boundary

Use this with Step 5 and [candidate-record.md](candidate-record.md). A proposal
can identify useful work without establishing a dataset gap, a constructible
task, or a proven environment. Decide these separately:

| Question | Required evidence | What missing evidence means |
| --- | --- | --- |
| Does the benchmark need this scenario? | Reviewed intended behavior, inspected task inventory and audit results | Coverage or priority remains unknown; this alone does not block a separately requested trace-derived task. |
| Can this task be constructed faithfully? | Complete source-backed instruction and capability-defining constraints, objectively testable outcome, and reproducible required software | Missing requirements or required unavailable software block candidacy. Missing original world state can permit explicit reconstruction. |
| Does the generated task work? | Retained Harbor repeated NOP/Oracle runs and negative controls, or Gym native validation and repeated controls from the [Gym extension](gym-output.md) | Proof is unmeasured or failed. Artifact acceptance, diagnostic probes, and a proposed verifier do not establish executable success. |

## Decide without changing the task to obtain a candidate

First identify the requested capability and its constraints from the safe trace.
Do not fill a truncated request or an unavailable issue's acceptance criteria
from an agent's completion summary. Resolve later correction/confirmation roles
from recorded evidence; do not silently promote an ambiguous agent-labeled turn
to a human instruction. If a material requirement remains ambiguous, record the
specific missing requirement and `no_candidate`.

Next separate missing **state** from missing **requirements**. A complete request
to calculate totals from a ledger can support a new synthetic ledger even when
the original file is absent. The request determines the operation; the synthetic
fixture supplies new, internally consistent values. It cannot invent an unknown
refund policy, approval rule, or engineering acceptance criterion. Follow the
reconstruction provenance rules and declare `state_basis: "reconstructed"`.

Absence of independent truth about the historical answer is not automatically
absence of an objective oracle for a new synthetic task. Derive expected results
from source-backed requirements and the new fixture; keep historical
`ground_truth.availability` honest. Do not promote the observed answer to truth,
claim the synthetic result verifies historical correctness, or replace subjective
quality with exact text matching.

Preserve capability scope. Editing a JSON representation is not evidence that
an agent can perform the same edit in a required native application. If the
request depends on native geometry, application behavior, or actual side effects,
retain that requirement and reject unavailable required software. A portable
representation is appropriate only when it preserves the evidenced capability;
disclose the narrower scope and do not claim native equivalence. If that choice
cannot be justified from the trace and user scope, retain the uncertainty rather
than quietly simplifying the task.

Distinguish required task software from the author's proof infrastructure. A
missing local Docker daemon prevents local proof; it does not alone establish
that an otherwise specified task cannot be constructed. Record `not_run` and
`unproven` when proof prerequisites are unavailable. An unavailable application
required by the task still blocks candidacy. Continue the existing execution and
bounded repair protocol when prerequisites are available; do not finalize early
merely because a candidate record passes metadata checks.

## Reconcile the handoff

Re-evaluate each relevant proposal against these construction criteria. A
recommendation to gather audit evidence need not forbid a bounded reconstruction;
a recommendation to build does not waive missing instruction or native-runtime
requirements. In `candidate.json`'s existing `uncertainties`, explain material
disagreements with the proposal and name the evidence or remaining blocker.
Keep `reason_codes` specific. No extra approval, audit denominator, or new record
schema is required for this handoff.

Report artifact candidacy, diagnostic outcomes, and full proof separately. Keep
all attempted inputs and failures in comparative measurements. More candidates
is not itself an improvement: check that the instruction, task scope, and verifier
remain faithful before interpreting yield changes.
