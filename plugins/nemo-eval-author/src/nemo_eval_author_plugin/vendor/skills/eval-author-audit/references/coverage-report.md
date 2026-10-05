<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Writing the Audit coverage report

Write `.eval-author/audit-coverage-report.md` using the
[template](../templates/audit-coverage-report.md). This is the readable audit
output: explain what the audit intends to check, what existing tests address,
what selected runs demonstrate, what remains unknown, and what to do next. Adapt
the template to the available evidence and replace its authoring prompts before
presenting the report.

## Artifact names

Use the names in [Audit outputs](../SKILL.md#audit-outputs) consistently in
user-facing replies, artifact link labels, report references, and the proposal
handoff. On first introduction, include the filename and a brief explanation of
its role. Keep the full artifact name in later links so each opens a predictable
document. “Draft,” “approved,” “preliminary,” and “reviewed” describe status;
they do not change the artifact's name.

At specification and final-report review checkpoints, open with the actual
artifact updates and briefly explain each document's purpose beside its link,
before counts or findings. Keep this explanation in the turn-ending reply even
when the roles were explained earlier or in commentary. Correct names, filenames,
and status alone do not satisfy it. Claim only writes or validation actually
performed; reviewing an unchanged document does not imply an update.

Describe proposed work in plain language, such as “outline what your evaluations
should check and compare that with your existing tasks.” Use the artifact names
when referring to saved files, rather than as unexplained names for activities.

**Audit specification** (`audit.md`) defines the checks and required evidence.
**Audit coverage report** (`audit-coverage-report.md`) presents that scope and
the findings about existing tests and available run evidence. Its **Intended
coverage** section renders the specification for review; the two files remain
distinct. Use the Audit coverage report as the user's review document: at the
scope checkpoint, point to **Intended coverage** and ask whether the checks and
required evidence are right; at final review, point to the findings, evidence
limits, and next actions. Link the Audit specification as the structured source
behind the intended checks, without requiring a separate YAML review. Name the
section and decision in each review request rather than leaving the user to
choose between two file links. Avoid alternate artifact labels such as “coverage
specification,” “draft specification,” or “audit report.” In surrounding prose,
“specification” or “report” can be shortened once its referent is clear.

Call the generated aggregate **Coverage measurements (JSON)**
(`audit-coverage-report.json`), keeping it distinct from the Markdown report even
though their filename stems match. Link it only when it exists. **Audit progress**
(`audit-progress.md`) records workflow state and is not a findings report.

For example after creating the report and validating a draft:

> I created the Audit coverage report, which summarizes the proposed checks,
> what your selected evaluations test, and the gaps and evidence limits found.
> I also validated the draft Audit specification, the structured source that
> defines those checks and evidence requirements.
>
> Please review the Intended coverage section of the Audit coverage report.
> Are these the right checks and evidence requirements, or should anything change?

Link each name to its own file and include its filename on first introduction.
Use the actual review state; validation does not approve the specification.
At final review, ask about the Audit coverage report's findings. Preserve these
names when offering proposals; review status does not create another artifact.

## Sources and scope

Use the current `audit.md`, inspected evaluation definitions and verifiers, and
applicable measurement artifacts. Link the specification, Ethos, selected eval
locations, and JSON report when it exists. State the trace scope, methods run,
excluded or failed inputs, and whether the result is a specification only,
partial measurement, or measured audit. Distinguish confirmed absence of evals
from inaccessible or uninspected evals.

The specification defines the item names and evidence requirements. The JSON
report supplies measured counts and item coverage. Method-specific details and
judgments explain those results; keep generated JSON unchanged. Trace paths in
reports are references, not permission to open new sources: follow the audit's
source-selection rule before any further trace inspection.

## Coverage and test mappings

Summarize tools, capabilities, and failure cases separately. For each kind, show
the declared total and distinguish measured coverage, measured items not
demonstrated, and unmeasured items. Copy measured counts from applicable reports;
do not turn an unmeasured kind's zero covered count into a 0% coverage claim.
For a kind with no declared items, show that fact without a percentage.

Use a section named **Intended coverage** as the complete, readable catalog of
the scope being reviewed. State whether the user has agreed to that scope;
successful schema validation alone does not make it approved. Map every current
audit item by stable `name` and kind to:

- **Intended check and required evidence:** explain in ordinary language what
  must be demonstrated and what evidence would establish it. Summarize the
  specification's tool expectations, capability behavior, or failure trigger
  and expected safe response, together with its evidence requirements. Keep
  these proposed criteria distinct from observed results. A stable name or test
  mapping alone does not tell the user what they are agreeing to.
- **Existing tests:** links to inspected cases, instructions, or verifiers that
  address the item. This describes intended testing, not observed coverage. Use
  “no match in inspected tests” or “not inspected” where appropriate, and preserve
  the scope of that statement.
- **Measurement status:** covered, measured but not demonstrated, or unmeasured.
  `not_measured_by_any_method` is unmeasured; `not_covered_by_any_input_report`
  means the selected measurements did not demonstrate the item.
- **Observed evidence and limits:** relevant task/run IDs, method, measurement
  files, judgments, or trace references. The aggregate's `input_reports` links
  covered names to task/run identities and coverage files. State the reason for
  missing evidence rather than inventing a supporting run.

Group rows by kind when helpful, or use per-item lists when a table would become
too wide. Keep the complete catalog in this section rather than maintaining
separate scope and mapping lists that can drift. Preserve material criteria when
summarizing; link to the specification for detail instead of copying YAML or
JSON. A manual mapping from task definitions can be useful without traces, but
does not establish measured coverage. A covered item in one run does not erase
observed failures in others or establish agent quality.

## Findings, limitations, and next actions

Explain material gaps with evidence: a missing scenario in inspected tests, an
observed agent failure, or insufficient measurement evidence. Where the cause
is unknown, say so. Missing judgments or an unverified failure trigger do not
prove that another task is needed. Include material fixture and verifier limits,
such as synthetic data or checks that exercise only part of the intended behavior.

Without usable traces, report measurement as unmeasured and explain the needed
input or conversion. With partial measurement, make clear that counts apply only
to the successful subset and list the exclusions. Do not fabricate a JSON report.
If inputs change, identify affected findings as stale until refreshed; old
measurements must not appear current. Retain still-applicable findings and user
corrections when updating the report.

Close with a recommended next action grounded in the findings. For an existing
suite with useful gaps to address, recommend proposing new or improved eval tasks
and name the behaviors those proposals should target. Explain when gathering
evidence, fixing the agent, or creating the first evals should come first.
Detailed ranked proposals belong to `eval-author-task-create`; the report
identifies the direction and evidence for that handoff without claiming tasks
were created or gaps closed. Use the [guided handoff](guided-audit.md#after-review-offer-the-next-step)
to offer this next step during review and continue when the user accepts it.
Present that offer in the turn-ending response and wait before starting proposal
work. A recommendation in the report or acceptance of its findings is not an
answer to the handoff question.

## Review and updates

At the final guided milestone, explicitly say the report was created or updated,
link it, and walk through the main findings, evidence limits, and next actions.
Ask for corrections or agreement and wait. Keep **Generate and review coverage
report** incomplete while that review is pending; resolve requested corrections
before completing it. Reuse an answered review. Record conversational review
state in `audit-progress.md`.

Every turn-ending response after creating or updating the report must call out
that write and link it, including earlier specification and measurement
check-ins. At the specification check-in, follow the [guided review](guided-audit.md#3-define-what-the-evals-should-cover):
explain both artifacts, preview the intended checks, and point directly to the
complete **Intended coverage** section before requesting agreement. Keep that
scope decision separate from the later review of findings and next actions.
For read-only requests, discuss the available findings without writing
or claiming an update. A focused operation does not require unrelated test
inspection or the full guided review just to fill this template.
