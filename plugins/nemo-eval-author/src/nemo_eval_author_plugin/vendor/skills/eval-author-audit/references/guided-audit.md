<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guided audit milestones

Use the five visible labels and literal checkboxes from the [audit skill](../SKILL.md) for a full
audit. The numbered script steps in that file are technical operations inside
these milestones, not additional user check-ins. A focused generate, validate,
measure, or aggregate request keeps its scope; the trace-source rule below still
applies whenever traces would be searched or read. Focused operations do not
require an eval inventory or suite confirmation unrelated to their supplied inputs.

## Opening before the first milestone

For a fresh full audit, follow the audit skill's opening: show all five unchecked
steps with their descriptions using literal `- [ ]` syntax, ask **“Ready to get
started?”**, and end the turn in the final response. The completed opening is the
user's opportunity to understand the workflow before its execution starts.
Wait for acceptance before any repository inspection or Ethos validation or
generation, including checks of a previously accepted Ethos. No progress file is
needed to ask this question. After acceptance, begin **Understand your agent**
and retain the response when saving progress. Reuse an answered opening on return
to an audit already started; do not replay it before every milestone.

## Check in and preserve progress

Finish the authorized work in the current milestone, explain its result and
limits, link its artifacts, and refresh the checklist. Mark completed milestones
`[x]`, leave partial or deferred work `[ ]`, and mark the current milestone
**We're here**. Every deferred or unavailable milestone must include a short,
specific reason on its checklist line, including on resume and in the final
review: for example, `- [ ] **Measure coverage** — Deferred: selected trace directory is missing.`
Do not shorten this to “deferred” even when the reason appears elsewhere.
State the next action and ask one focused question about the
result or the next milestone's missing input. Deliver that checkpoint in the
turn-ending response (`final` when available) and wait for the user's answer.
A commentary question or an asynchronous input widget followed by more work
does not complete the handoff. Do not poll for an answer, keep inspecting files,
or leave audit subagents working in the background. Finish or pause any delegated
work before presenting the checkpoint. At the final milestone, invite questions or corrections and
stop. Reuse answered check-ins; an answer accepting the result and named next
milestone settles that transition. A generic request to resume does not approve
a pending content review. A full-audit request alone does not waive check-ins.
Known inputs settle input questions, not milestone transitions. Even with accepted
Ethos and confirmed absence of evals and traces, explain the first milestone and
wait at its check-in before completing the next milestone. Do not infer answered
check-ins from the ability to do later work. This also applies when the next
milestone needs no tool calls.

Before presenting a checkpoint, keep reads, writes, and delegation within the
current milestone. Once its question has been presented, stop all audit work
until the user replies, including otherwise independent source inspection.
An input question can also serve as the transition check-in when the current
milestone's result has been explained; do not follow a source-selection answer
with a duplicate readiness question. The Ethos check-in offers edits before
eval discovery; confirm the eval inventory before asking for trace selection.
Show the checklist at the opening, milestone check-ins, source-selection replies, and
resumption, rather than after every tool call. Explain what an input is needed
for and what remains possible without it.

Save candidate eval suites separately from the user-confirmed suite set, including
exclusions and any pending selection question. Record eval selection and trace
selection independently; an answer to one does not settle the other.
Save the accepted opening, confirmed paths and source scope, completed work and
supporting evidence, limitations, pending questions or answered transitions, and the next action in
`.eval-author/audit-progress.md`. Record whether the result is a specification,
partial measurement, or measured audit. Do not store conversation progress in
Ethos or edit generated coverage JSON. For read-only requests or before a file
can be saved, retain this state in the conversation. Progress notes are a memory
of decisions, not evidence that a check passed or that an unanswered question
was accepted. Keep the human-readable findings in `audit-coverage-report.md`;
whenever it is created or updated, explicitly call out that write and link the
report in the turn-ending response, even at an intermediate checkpoint.

On return, reuse applicable answers and artifacts and resume the earliest
unfinished, non-deferred milestone or its pending check-in. Preserve an explicit
deferral and the agreed independent next step; revisit it when its missing input
arrives or the user asks. A selected source stays selected until the user changes
it; do not repeat its confirmation. Changed Ethos, audit items, or trace inputs
reopen affected work and invalidate dependent measurements or judgments as
appropriate. Recheck applicability without restarting the entire audit or
searching beyond the selected source. Refresh the Markdown report to identify
affected findings as stale or unmeasured until applicable evidence is available;
do not present old counts or mappings as current after their inputs change.

## 1. Understand your agent

Input: the selected agent's intended purpose, boundaries, and success criteria.
After the user accepts the opening, introduce Ethos before validation or authoring,
including when an existing document can be reused. Explain what it is and why the
audit needs it, with a link to the
[bundled Ethos skill](../../ethos/SKILL.md).
For example: “`ETHOS.md` describes what your agent should do, what it should
avoid, and what a good result looks like. For this audit, we'll turn that intent
into the behaviors, tools, and failure cases your evaluations should cover,
then compare them with the existing tests and any selected run records.” Tie
this to the selected agent; do not assume the user knows the term.

Include this brief definition and explanation of its use in the first
turn-ending Ethos check-in, before validation results or repair requests. Keep
it in that reply even if an introduction appeared earlier in commentary, where
it may be collapsed. A link or a summary of this agent's behavior alone does
not explain what the document is for. Later check-ins can reuse this explanation
without repeating it.

Follow the audit skill's **Ethos Pre-flight**, reusing an applicable existing
Ethos and its review state. Look for that document first using selected paths
and directly named agent documentation, before inspecting implementation details.
When found, promptly show a clickable link and a brief summary of its purpose,
main behaviors, and boundaries. Limit initial checks to the document itself;
do not delay this handoff for an exhaustive code comparison or begin rewriting
it to match implementation. Include any document issues that need resolving in
the same handoff, keep this milestone incomplete, and ask about the needed fixes
before editing; an incomplete existing file is not a reason to create a replacement.
For a format mismatch, first explain the document's purpose and audit use, then
summarize the existing intent and the specific formatting issue. Explain the
proposed repair in ordinary language before asking to make it; parser success
or a list of missing headings should not lead this first handoff.
Offer the user a chance to edit the existing intent before
moving on, even if no edits are needed. This is the milestone check-in, not a
fresh intent interview or a demand to rewrite an unchanged document.

If no document exists, explain that one will capture the intended
behavior, then follow the shared Local Ethos procedure to create and review it.
This milestone does not require an eval inventory, traces, Harbor setup, or an
evaluation run.

Complete when the applicable Ethos has been checked and any new or revised
content reviewed under the shared Local Ethos procedure. For an unchanged,
usable document, end with the linked summary, refreshed checklist, and a question
such as: “Would you like to edit anything in this Ethos, or use it as-is and look
for existing evaluations next?” Wait for the reply. Reuse an answered edit
opportunity and transition; do not add another readiness question. Existing Ethos
does not by itself answer that check-in. Keep trace-location questions in the
next milestone, after reporting and confirming the eval inventory.

## 2. Confirm evals and traces

Inputs: the evaluations to audit, or the user's statement that none exist, and
the source and location of any traces to use. Reuse paths and answers from the
current conversation, discovery, or first-eval handoff when the user has selected
them for this audit. A trace records what the agent actually did during a run;
an eval definition or score alone is not that record.

Start by identifying existing evaluations from the named agent documentation,
selected eval source, or a prior inventory. Before asking about traces, report
what was found with a clickable path to each suite's directory, runner, or README
and a short description of what it tests. Link shared case definitions when useful,
but do not substitute only a case-matrix link or runner names for the suite
locations. Treat discovered suites as candidates until the user selects them.
Keep this a concise inventory; read bounded suite documentation and case
definitions to identify candidates, leaving detailed grading analysis for the
later specification and findings milestones.

For a full guided audit, unless the user already explicitly selected these
suites, end the turn with the linked inventory and a question such as: “Are these
the evaluation suites you want to audit, or should we add or exclude any?” Keep **Confirm evals
and traces** current and wait for the answer before asking about traces,
inspecting even previously selected traces, or drafting coverage. Showing the
inventory is not confirmation. A reply supplying only a trace location or saying
there are no traces leaves suite selection pending; retain that trace answer and
ask only the unanswered suite question.

An explicit request such as “audit suites A and B at these paths” already
selects that set; summarize it without asking again. A path supplied only as a
place to look, a prior discovery inventory, or a saved candidate path does not.
Apply additions and exclusions before proceeding. On resume, reuse confirmed
choices and present any still-pending suite selection before continuing.

If the user confirms no evaluations exist, say so without searching to prove it.
If the location is unknown or inaccessible, report that limit and ask for the
eval location first; do not claim absence or imply that an inventory succeeded.
Explain traces in ordinary language before the source question: recorded agent
runs show which tools and behaviors were actually exercised; test definitions
describe what was planned, and pass/fail scores alone cannot establish that
coverage. State that the specification can still be prepared without run records,
while measurement would remain unmeasured.

Before searching for, listing, opening, converting, or measuring traces, confirm
with the user both their source and location. Ask a concrete question, such as:
“Which run records should this audit use, and where are they saved?” A local file
or bounded directory plus the identified run source is sufficient; do not demand
an exact filename before searching a user-selected directory. An explicit request
such as “use the Harbor traces in `runs/release-check/`” already settles this
selection. Reuse that answer without another approval prompt. “Use available
traces,” an automatically discovered path, or a path mentioned by documentation
does not select a source. If an earlier handoff merely lists candidate paths,
ask which to use before reading them.

This rule also applies to tracing runtime tool names while drafting audit items,
opening traces referenced by a report, and implicit trace scans inside discovery
or conversion tools. Before presenting a missing-source question, inspect named
agent docs, selected eval definitions, or a tool registry within the current
milestone as useful, excluding run logs, trajectories, and trace directories
until source selection is settled. After asking the source question, end the
turn; do not continue that inspection while waiting. Independent specification
work can resume if the user explicitly defers the missing source and selects
that next step. Whole-repository `rg --files`,
`find`, or recursive globs that expose trace filenames count as trace searches,
even without opening their contents. Read directly named Ethos and agent files
first; do not batch a broad inventory alongside loading these instructions.
If the location is unknown, ask the next focused source question rather than searching the checkout, home
directory, platform accounts, or unrelated jobs for candidate traces.

Once selected, inspect only that source to establish which records are accessible
and usable. Audit measurement accepts local ATIF files or Harbor trial directories
containing ATIF; selecting a remote source does not grant access or make it ATIF.
For other formats or inaccessible services, explain the needed local export or
appropriate supported conversion. Do not query Intake or start platform discovery
from this skill. Preserve the source selection and describe the limitation.

When the selected run folder is missing, explain that measurement is unavailable.
If the confirmed evaluation definitions are accessible, offer the remaining
work in plain language: “Should I outline what your evaluations should
check and compare that with your selected evaluation tasks, or would you like
to use another folder of recorded runs?” Adapt the task description to the
confirmed suite. Use **Audit specification** when naming the saved artifact;
the choice of next action should not require understanding “coverage
specification” or “test mappings.” Wait for the answer before proceeding.

Complete this milestone when the user has confirmed the eval suite set (or that
none exist) and trace selection or confirmed unavailability is understood.
Explain the selected scope, what its evidence can establish, and any missing
inputs; check in before defining coverage.
An inaccessible eval directory or a missing report does not establish no evals.
Keep an unresolved eval location open and ask for an accessible source; offer
independent specification work if useful, carrying that limitation forward.
If the user explicitly defers suite selection and requests independent
specification work, keep this milestone incomplete and use the accepted Ethos
and registry. Do not present candidate suites as selected or their findings as
findings about the user's confirmed evaluation scope.

## 3. Define what the evals should cover

Inputs: checked Ethos and authoritative tool names from the registry, agent
configuration, or confirmed traces. Group the audit skill's **Steps 1–3** here:
draft items, generate or reconcile `audit.md`, and validate it. These items
describe intended coverage; they are not runnable tests or proof that the
existing tests cover it.

Create or update `.eval-author/audit-coverage-report.md` using the
[report guidance](coverage-report.md) and the available findings. Its **Intended
coverage** section is the readable review surface: include every proposed item,
its intended check, and its required evidence from `audit.md`. Map audit items to
inspected test definitions where possible, with measurement explicitly unmeasured
until applicable results exist. Record unresolved mappings without inventing
tests or inspecting unselected traces.
Direct the user to this section of the Audit coverage report for the current
review. The Audit specification is its structured source; do not require a
second review of the YAML. Agreement with the complete, current Intended
coverage section approves the matching specification's checks and evidence
requirements, while review of findings remains a later decision.

In the turn-ending specification check-in:

- Open with the actual creation or update of the **Audit coverage report** and
  validation of the draft **Audit specification**. Attach a brief purpose to
  each linked name in that opening, before the scope preview or findings: the
  report explains the proposed checks, what the selected evaluations test, and
  the findings and evidence limits; the specification defines what the
  evaluations should check and the evidence required for each check. Include
  this explanation in the turn-ending reply even if it appeared earlier or in
  commentary. Names, filenames, and status words alone do not explain purpose.
  Use the [opening example](coverage-report.md#artifact-names), adapting it to
  the actions actually performed. **Draft** means awaiting agreement on the
  specification; the report is preliminary here.
- Show a short, concrete preview grouped into tools, capabilities, and failure
  cases. Describe intended checks, not just totals, item names, or gap findings.
  For a large scope, label the preview as a summary and link directly to the
  report's complete **Intended coverage** section using a supported section or
  line link verified against the saved file, labeled **Audit coverage report —
  Intended coverage**. Name the section in the reply so the user can find it
  even if the client opens only the file.
- Report the validation result, then explicitly say what to read and decide:
  “Please review the Intended coverage section of the Audit coverage report.
  Are these the right checks and evidence requirements, or should anything
  change?” Link that section directly. A generic “Does this scope look right?”
  leaves the review target unclear even when both artifacts are linked above.
  Name the next milestone that agreement would allow: measurement, or report
  review when measurement is deferred.
  Approval of intended coverage does not approve the final report or establish
  that any item is covered.

Validation proves structure and references, not completeness or agreement with
intent. Keep review pending until the user confirms the proposed coverage;
apply requested changes and revalidate. Do not mark it approved just because the
schema passed. An answer accepting the specification and the named next step
settles this milestone's check-in without a second prompt.

When no evaluations exist, explicitly explain that Ethos and this specification
can establish what future tests should cover. Existing agent runs can still be
examined when selected, but their coverage is evidence about those runs, not an
evaluation suite that does not exist. Without run evidence, proceed with the
specification alone. Unresolved tool names remain open questions; do not invent
names, silently omit material behaviors, or call a partial specification complete.

## 4. Measure coverage

Inputs: the reviewed, validated specification and usable records from the
user-confirmed source. Explain which trace set and measurement methods will be
used. Execute the audit skill's **Steps 4–5**, including supported judgments when
required, then aggregate the measurements. Report tools, capabilities, and failure
cases separately; methods not run remain unmeasured. A tool-only run cannot
establish capability or failure-case coverage. Keep an explicit list of coverage
files produced for the selected trace set and pass those with repeated `--coverage`
arguments to aggregation, or use a fresh output directory containing only that
set. Do not recursively aggregate a shared measurements directory containing
other audits' results. Reused coverage must belong to the current selected scope
as well as match the current specification and trace inputs.

Complete when the agreed measurement scope has been processed and aggregated
successfully, even when coverage is low. If some inputs fail, report only the
successful subset with explicit exclusions and keep the agreed milestone partial.
Refresh `audit-coverage-report.md` with the measured subset, methods, and any
excluded inputs. Explain the result and limits, link the updated report, refresh
the checklist, and check in before **Generate and review coverage report**.
Reuse existing measurements only when their inputs still apply.

If traces are absent, inaccessible, invalid, or unavailable in a supported format,
name the specific limitation and needed evidence. Keep **Measure coverage**
unchecked with the reason on its checklist line, such as “Deferred: selected
trace directory is missing” or “Unavailable: run records need an ATIF export.”
Preserve that reason in progress notes and every later checklist until resolved.
Do not report 0% coverage, fabricate
measurement files, or treat the lack of traces as uncovered behavior. The
specification and its review can still finish. Record the limitation in the
Markdown report. At that check-in, offer **Generate and review coverage report**
as the next independent milestone and wait for the answer. Do not run evals or
install dependencies just to fill the gap without the
authorization required by the core.

## 5. Generate and review coverage report

Generate or update the **Audit coverage report**
(`.eval-author/audit-coverage-report.md`) following the
[report guidance](coverage-report.md), even for a specification-only outcome.
Then go over it with the user. Lead with what was established: a specification,
partial measurement, or coverage measured over a named set of traces. Explain
the findings, their evidence and limits, link the Audit coverage report and
supporting artifacts that actually exist, and leave deferred milestones visible.
A specification-only outcome is useful planning work, not a completed measurement.
Distinguish items not measured from measured items lacking evidence; neither automatically means
the dataset needs another test. Keep the [artifact names](coverage-report.md#artifact-names)
consistent: approving the Audit specification and reviewing the Audit coverage
report's findings are separate decisions. Both user reviews use the Audit
coverage report: first its Intended coverage section, now its findings, evidence
limits, and next actions. Link the relevant saved sections and explicitly name
those as the content to review; do not ask the user to reapprove intended checks.

Keep this milestone unchecked and current while awaiting the user's review.
Ask whether the Audit coverage report's findings need correction and offer the
recommended next step below, then end the turn. Mark it complete only after the
user has reviewed the report and any requested corrections are resolved. A reply accepting the findings and
selecting a next step satisfies the review; do not add another approval prompt.
Record the answered review in progress and refresh the report if findings change.

### After review: offer the next step

After the user accepts the findings, mark the report review complete and guide
them forward. For an existing suite, lead with proposing new or improved eval
tasks to improve coverage of the report's main gaps. Name the relevant behaviors
and explain that the next deliverable is a ranked set of candidate scenarios,
expected behavior, and verification plans. Avoid ending only with a generic
invitation for questions or a bare link to another skill.

For example: “The audit identified gaps around unverified account recovery.
I recommend proposing tasks to exercise that behavior, with checks for the
expected safe response. Would you like me to prepare those proposals?” This
offer can be included in the report-review question. Make the handoff explicit:
ask whether to start proposing tasks, end the turn, and wait for the user's
answer. Do not start proposal analysis, write proposals, or leave proposal
subagents running while the answer is pending. A broad request to audit and
suggest improvements does not skip this checkpoint. Accepting the findings
alone does not accept the proposal offer; a reply accepting both settles the
handoff without a second permission question.

Once accepted, enter `eval-author-task-create` at **Step 1: propose dataset
improvements**. Carry the Audit coverage report, reviewed Audit specification,
available Coverage measurements (JSON) and evidence, selected source scope, user
corrections, and agreed priorities.
Do not restart the audit or repeat settled questions. Explain that proposals
will be saved in `.eval-author/proposals/dataset-recommendations.md`; creating
and running tasks is a later step with its own scope. If the user declines,
finish with the reviewed report and completed checklist.

Adapt the recommendation to the findings. Without usable run evidence, explain
what measurement needs; any task ideas based on inspected definitions remain
unmeasured candidates. If missing evidence or an agent fix is the useful next
action, recommend it instead of inventing tasks. With no existing evals, offer
the first-eval workflow to turn the reviewed specification into initial cases,
grading, and a runnable suite, reusing Ethos and prior answers. Keep deferred
measurement visible and do not imply that setup or execution already passed.
