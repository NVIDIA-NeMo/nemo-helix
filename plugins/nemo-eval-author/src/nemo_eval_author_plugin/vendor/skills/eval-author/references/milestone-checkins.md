<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Check in after each milestone

The core selects the entry route before its authoring welcome and canonical
checklist. Read this reference for first-eval after the user starts
that experience; it owns stage transitions, progress, and resumption. Discovery
inventory, audits, and proposals keep their scoped entry procedures. Sub-flows
own the work within authoring stages. Throughout this procedure, a stage means
one of the ten internal stages. The core's five visible checklist steps group
their progress without combining stages or changing their check-ins.

## Complete, explain, check in, wait

1. Finish the authorized work for the current stage, or identify the specific
   requirement preventing completion. Make its result concrete and reviewable.
2. Explain what was found or produced, why it matters, and the limits of the
   evidence. Link artifacts beside their explanation and refresh the checklist.
3. Name the next stage and ask one focused question about the result or that next
   step. End the reply and wait for the answer before starting the next stage.
   Address questions or corrections within the current stage first. At the final
   results stage, ask whether anything needs explanation or revisiting and stop.

A question requesting the next stage's first needed input can serve as this
check-in. Review of a saved Ethos can settle the Ethos stage; a source-selection
answer can settle the evaluation starting point and allow scope planning. Reuse
answered transitions without another readiness prompt. A broad request to get
evals working does not waive stage conversations. Preserve existing installation
and execution authorization under the core's boundaries.

Keep tool work and instruction loading focused on the current stage. Read-only
inspection needed to answer its question is allowed, but do not begin later-stage
scans, probes, interviews, artifact creation, integration, or runs while its
check-in is unanswered. Within the current stage, complete independent authorized
work. If a prerequisite prevents completion, explain the limitation and ask whether
to defer it and move to a named independent stage. Wait for that choice, keep the
deferred stage incomplete and its visible step unchecked, and return when its
prerequisite is available.

## Progress and resumption

Use the core's five visible checklist labels, stage grouping, and checkbox syntax
throughout the sub-flows; do not render the ten internal stages as another checklist.
Render it in source-selection replies, stage check-ins, and return visits, rather
than after every tool call. A saved report, link, findings list, or promise to show
it later does not replace the visible checklist. Put findings and the checklist
before the closing question. Mark the visible step containing the current internal
stage **We're here**, including when an earlier prerequisite was explicitly
deferred. Use `[x]` only when every internal stage in that step is complete; an
answered check-in is separate from technical completion. Leave future and partial
steps unchecked, naming their remaining work. Continue the same check-ins between
internal stages even when the visible step does not change. Ordinary intent or
source questions do not need a blocked status.

After a side question or setup detour, answer or resolve that issue, then return
to this same checklist: briefly recap completed work, mark the current stage,
and name the next planned action. Preserve the pending scope question and prior
answers; a detour does not accept a plan or reopen a settled corpus choice.
Continue authorized work when no decision remains pending. At final handoff,
show every completed step checked and explain any remaining partial or blocked
step instead of silently omitting it.

Track case/grader preparation, environment availability, and agent configuration
separately. A first case is partial progress when the agreed scope is a larger
suite. Configuration is not validation; a successful control run is not evaluation
of the actual agent. Keep validation open when required checks fail or remain
unrun. A completed evaluation can have low scores without being incomplete.
The owning sub-flow defines the evidence needed to complete each stage.

Save the internal stage, evidence, pending question or answered transition, and
next action in the existing findings or task README. Before those exist, retain early-stage
state in the conversation or existing intent notes; do not run discovery just to
obtain a report. Keep selected-runtime probe evidence, local interpreter paths, and milestone
or approval records in that workflow state, not in `ETHOS.md`. This does not exclude
substantive evaluation requirements from Ethos's Evaluation Setup section.
For first-eval, use `.eval-author/first-eval.md` once the source decision is
established; keep the selected corpus, user priorities, agreed per-behavior scope,
and delivered-example inventory there alongside progress. Update it at each
completed stage and when feedback changes the plan, so resumption reads the
current agreement rather than reconstructing it from scattered messages.
On return, reuse applicable answers and completed work, rechecking
readiness when inputs changed. Do not infer an answer from an installed tool or
saved file, or restart a completed opening or Ethos interview. Inventory before
authoring may already have settled the evaluation starting point, but does not
complete Ethos or runtime setup. Narrow inventory, readiness, audit, proposal,
trace, and internal validation requests keep their scoped flow.

## Early stages: Ethos, runtime, then evals

### 1. Establish the agent’s Ethos

After the opening answer, follow [Local Ethos](local-ethos.md). Identify the agent,
understand its intended purpose and limits, and reuse or create its document.
This stage uses agent context, not an eval inventory. Do not load authoring or
audit sub-flows, probe runtimes, or scan reports and traces to choose an eval source.
A missing Ethos calls for the next focused intent question. An applicable existing
Ethos calls for a brief explanation of its intent, not a fresh interview.

New or revised saved content is reviewed through the local procedure. That review
is the check-in before runtime setup; name runtime setup as the next stage. For an applicable
unchanged document, summarize what it establishes and check in before runtime setup
without demanding another content approval. Neither mentioning Ethos in the plan
nor answering an intent question completes the document procedure.

### 2. Get the evaluation runtime ready

After the Ethos check-in, follow the core's [provider-selection rule](../SKILL.md#select-the-evaluation-provider)
and introduce the selected setup before probing its installation. Both Gym and
Harbor are supported. Explain how the selected tools run cases with their
environment and grading, and retain results for reruns. Verify the runtimes the
selected setup uses; apply both checks below when it uses both.

For **Gym**, link [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym). Reuse the
repository's documented installation and verify `gym --version` and `gym --help`
with its actual CLI. Authoring requires v0.6.0+ in its separate Python 3.13.14+
environment; verify that interpreter can import `nemo_gym.environment.scaffold`.
Record the command, interpreter, and version. When using both tools, keep their
Python environments separate as described in discovery's runtime guidance. If
setup is missing or broken, report the specific need and follow the core's
installation authorization boundary. Respect the user's selected setup.

For **Harbor**, link [Harbor's documentation](https://www.harborframework.com/docs)
and use discovery's [runtime prerequisite checks](../../eval-author-discover/references/runtime-prerequisites.md#runtime-prerequisite-checks),
including its optional-skill advisory, without a full evaluation scan. Use its
[setup guide](../../eval-author-discover/references/harbor-setup.md) for a missing
or broken installation. Preserve the verified command, interpreter, and version.

The checkpoint reply explains the selected framework and observed setup result,
including when reusing an installation. Setup does not prove task-specific
environment readiness, agent access, or successful runs. Keep technical paths
in the findings unless they help the user act, and show the shared checklist.

Check in before the next unfinished stage. If the evaluation starting point and
the trace source or no-trace choice are already settled, carry them forward and
proceed to **Define the evaluation scope** after this check-in. Otherwise,
asking about the remaining source choice can serve as the
transition to **Understand the evaluation starting point**. If a source is already
supplied, name the planned inspection without asking them to select it again.
Missing selected-runtime setup can be deferred using the rule above; it does not prevent finding
the user's eval material.

### 3. Understand the evaluation starting point

Reuse an eval starting point already settled by the user or discovery inventory;
no repeat eval scan or eval-source question is needed. Otherwise, after the runtime
check-in use `eval-author-discover` for inventory and source selection, reusing
verified runtime evidence. Determine whether there is an existing Gym or Harbor suite,
other eval material, or confirmed absence of evals. Ask about actual candidates
or a missing location; keep this stage unchecked and current until the source is
settled. The report's summary and examples supply findings and question wording,
not a complete onboarding reply in place of the checklist.

For first-eval, an established absence of evals settles the eval starting point,
but not which traces, if any, to use. Follow
[Corpus selection](coverage-planning.md#select-trace-evidence-before-using-it)
before trace searches or reads; reuse an explicit source or no-trace choice.
Use the source decision as this stage's check-in and save it for scope planning.

A source answer identifies the material, not its scoring quality or coverage.
Name **Define the evaluation scope** as the next stage and use the core's
**Route after the evaluation starting point** handoff. Carry forward Ethos,
setup evidence, and answered transitions; do not restart them when authoring
continues. Any conflict between discovered criteria and Ethos is
resolved during scope planning, not by silently changing either one.

## Scope checkpoint

At **Define the evaluation scope**, explain what the selected cases will measure
before moving to **Prepare cases and grading**. For first-eval and expansion,
follow [Corpus and coverage planning](coverage-planning.md) to record proposed
and agreed scope in the existing plan before generation. Make it concrete:

- **Behavior and purpose:** what the agent should accomplish and why these cases
  matter, using established workflow priorities when available.
- **Success and evidence:** the expected outcome, what the grader must inspect,
  and whether the rule comes from existing criteria or is a proposed decision.
- **Breadth and difficulty:** distinct examples per behavior, meaningful
  variations, positive and negative scenarios, relevant difficult negatives,
  and why the proposed breadth fits the purpose. If recommending a pilot,
  explain in this reply what that limited first set will establish and what
  remains untested even if it passes. Separate example counts from native tasks
  and attempts; name exclusions and remaining gaps.
- **Priorities and cost:** explicitly ask what the user wants to change or
  prioritize, with relevant tradeoffs such as lower ongoing cost, broader
  coverage, harder cases, or another concern. Explain execution work and grounded
  estimates or uncertainty; a generic approval question alone is insufficient.
- **Execution requirements:** the capabilities, software, starting state, and
  access the cases need; distinguish known requirements from verified readiness.
- **Limits and open decisions:** missing criteria or evidence, which checks or
  runs they affect, and what is needed to resolve them.

Explain this through a representative case, grouping others with the same needs
and calling out material differences. Distinguish checking a tool sequence from
checking its resulting outcome. Explain the role of operational metrics such as
cost or latency when present; they establish success only where the agreed
criteria use them. Preserve existing scoring and explain its limits rather than
silently replacing it.

Use the existing stage check-in to settle the next affected decision or confirm
the scope, following the [feedback invitation](coverage-planning.md#explain-ongoing-cost-and-settle-the-plan).
Reuse known priorities when framing that question, without adding another
approval loop for an already agreed plan. Incorporate feedback in the saved plan
before generating affected fixtures, instructions, or graders. Reuse established answers; missing inputs
for one check need not hold up preparation of independently agreed cases.
This checkpoint explains the selected scope without requiring a coverage audit
or a new intake document.
