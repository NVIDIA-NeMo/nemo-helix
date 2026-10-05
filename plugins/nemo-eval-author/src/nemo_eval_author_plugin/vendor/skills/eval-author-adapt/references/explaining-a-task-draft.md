<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Explaining a task draft

Use this at the first task-creation milestone, especially for a user new to
Harbor. Apply [Milestone check-ins](../../eval-author/references/milestone-checkins.md)
for stage transitions and the core's evidence standard for readiness claims.
This reference owns the case and grading explanation, not another checklist.

## Make the created files understandable

Lead with the concrete artifact, such as a draft of the first eval case. Explain
the result in the reply itself; a README supports inspection but should not carry
the whole explanation. Scale later updates to what changed.

Show the actual request and expected behavior together in a short list or table.
A count such as “nine criteria” does not explain what is tested. Distinguish source
criteria, confirmed Ethos intent, and newly proposed rules, including any scoring
changes the user chose. Preserve full requests and criteria in the task files.

State whether this is one conversation with several steps or several independent
cases. For a multi-step task, explain whether conversation history is implemented
or still required; ordered files alone do not demonstrate retained history.

Connect useful file links to their state and purpose:

- Instructions contain the messages the agent will receive.
- Verifier-only criteria record the rules; identify which have working checks
  and which remain written requirements.
- Reference data or solutions need a stated basis for correctness; identify
  generated placeholders and unverified historical responses as such.
- The task README records source case IDs, decisions, and per-part status.

For application, session, and reset requirements, use
[Execution dependencies](../../eval-author/references/execution-dependencies.md).
Keep the applicable technical details in the README rather than reciting a
directory tree.

## Explain grading before asking the user to finish it

Walk through an actual source criterion: what a good result means, what evidence
would show it, and how a check reaches a verdict. When no criterion exists, ground
a proposed rule in the request and established Ethos and identify the unresolved
behavior decision before presenting it as agreed.

Separate the parts that can be missing:

| Part | Explain to the user |
|---|---|
| Success rule | What counts as a good result; whether the rule comes from the source or is proposed |
| Evidence | What the check must inspect: a response, tool action, created file, or external state |
| Checking method | Whether code applies the rule, code must be written, or a human/model must assess meaning |
| Score | How check results combine, including weights or partial credit where the source scheme uses them |

Translate “semantic review” into the actual judgment, such as whether an answer
explains requested information correctly. Explain who or what makes that judgment
and the inputs and access it needs. Explain weights as each check's contribution
and partial credit as credit for some of a rule's conditions; neither is a mandatory
new decision when the source already defines scoring.

For “grading incomplete,” name the affected rule and the missing part: behavior
decision, implementation, evidence, or score aggregation. Do not label code that
can be implemented from available evidence as something the user must supply.

### Explain a decision through its effect on agent behavior

If tool ordering is ambiguous, show whether a correct result reached through a
different sequence would pass or fail. Ask which behavior the check should accept,
not whether the user approves a report's entire intended behavior. Keep uncertainty
local to the affected check, and implement the resolved rule in the task.

## Turn remaining gaps into an actionable handoff

Explain what the draft contains, what its completed checks establish, and what
remains necessary for live evaluation. Open checklist items and a README link
alone do not tell the user how to finish their evals.

For a replay, explain that saved responses pass through the implemented checks.
Say whether Harbor executed a job or only accepted the task format, and label
commands by their actual purpose, such as **Replay saved conversations**.
Describe grading coverage by complete and partial source criteria when measured;
subchecks can cover parts of criteria, so their counts are not interchangeable.
A mock does not supply missing grading rules or establish historical equivalence.

For the main unresolved requirements, give the gap, its effect, and the smallest
useful input or implementation needed. Keep the full per-case inventory in the
findings. For example:

| Remaining work | Effect | How to finish it |
|---|---|---|
| Requests have no grading rules | A passing result is undefined | Draft criteria from the request and Ethos; resolve the missing behavior decision and implement the checks |
| Source rules conflict with each other or Ethos | Interpretations produce different scores | Show the conflicting expectations and their scoring consequence, then resolve the affected rule |
| The agent connection is unknown | Tasks cannot reach the actual agent | Inspect a setup guide, runner/config, or request-and-response example and implement the supported interface |
| Starting data or reset instructions are absent | Repeated runs may start in different states | Identify and obtain the referenced fixture or state requirements, then implement the setup |

These are examples, not an inventory to impose on every user. For unavailable
software or licenses, explain which operation needs access and use the dependency
reference's setup guidance. A guide or pointer to its owner can be enough to start.

If only reports survive, identify the historical scoring details that cannot be
recovered and explain any new proposed grading decisions as such. If the user
requested a mock because no real endpoint exists, describe its contribution to
the drafts and the remaining real integration work. Save the concrete remaining
actions with their evidence in the adaptation findings or task README.

## Example: after the user settles a sequence rule

Suppose reports and recordings prescribe a tool sequence, and the user has settled
whether additional calls are allowed. Implement that decision and the other
supported checks. If the actual artifacts support it, the explanation can be:

> I created a draft of your first eval case. It keeps your original conversation
> in order as one task and preserves the report's checks.
>
> The implemented grader checks tool counts, ordering, and the rule you confirmed
> for additional calls. The remaining criteria need trace fields absent from the
> recordings, so they are retained as unresolved requirements.
>
> Harbor can read the task files. I also checked the implemented rules with example
> traces that satisfy and violate them. Those checks exercise the scoring code;
> a fresh agent run will measure the agent's performance.
>
> You can inspect the conversation and checks in the linked task draft. The missing
> grading inputs keep cases and grading incomplete. The next stage prepares
> dependencies, fixtures, and starting and reset conditions.
> Would you like to defer the missing grading inputs and prepare that environment?

Adapt the example to the actual criteria and evidence. Link the real artifacts,
and use the shared milestone procedure for the accompanying checklist and reply.
