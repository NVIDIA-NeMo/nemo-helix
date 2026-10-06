---
name: ethos
description: Explore an agent's code and documentation, confirm its intended behavior with the user, and create or update a local ETHOS.md for evaluation and trace analysis. Use when capturing an agent's purpose, boundaries, success criteria, or change permissions.
triggers:
  - create an ETHOS.md for my agent
  - capture my agent's intended behavior
  - update my agent's Ethos
not-for:
  - eval-author (use for evaluation workflow routing)
  - eval-author-audit (use to measure coverage against an existing Ethos)
compatibility: Local Markdown authoring with file reading and writing tools.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: Read Glob Grep Write
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Explore and write an agent's Ethos

An `ETHOS.md` records what an agent is supposed to do, what it must avoid, and
how to judge success. Implementation and traces show what it currently does;
they do not establish what its owner intends. Explore the implementation, ask
for missing intent, then write and review the document in this one workflow.

This skill works locally with an existing or proposed agent and saves the result
as `ETHOS.md`. Keep changes scoped to that document, preserving the agent, model,
runtime, and evaluation scoring. Recording change permissions does not authorize
exercising them. Do not commit automatically.

When another workflow calls this skill, honor its selected agent, explicit
paths, write boundaries, and review checkpoints. Return to that workflow's
next unfinished step; do not start an optimization or build workflow yourself.

## Locate and explore

1. Use an explicit Ethos path when supplied; otherwise use root `ETHOS.md`.
   Resolve ambiguous agent identity with the user. Reuse a substantive existing
   document and its review state; a request to update one section does not require
   restarting the interview. Preserve
   custom sections, unknown frontmatter keys, and unrelated user edits.
2. For a missing or incomplete document, read the root README, directly named
   product docs, prompts, entrypoints, tool registrations, and runtime config
   relevant to this agent. Keep the scan bounded. For a proposed agent, work
   from the user's description, design documents, and prior interview notes.
3. Separate implementation facts from intended behavior. Infer Tools, Harness,
   and existing Evaluation Setup from source, and label their provenance. A
   framework import is not evidence of how the harness executes. Existing
   tests describe current coverage, not the full set of intended behaviors.
   Do not execute evals or collect traces just to establish identity and intent.
4. Summarize the agent and the gaps before asking questions. Explain that Ethos
   will give evaluations or trace analysis a target. Reuse an introduction
   already given by the caller.

## Confirm intent

Ask focused questions grounded in the workflows you found. Prefer one question
at a time, with concrete choices when useful, and wait for the answer. Reuse
answers already provided; do not impose a question count. Prior confirmed
intent remains valid unless the user changes it. Explicit requests to draft
with incomplete information can proceed with clearly recorded gaps.

For a new Ethos, cover these topics rather than silently filling them from code:

- **Purpose & Outcomes:** why this agent exists, the result it should achieve,
  any agreed target, and who owns that target. Internal tooling need not have a
  business metric. Confirm a mission inferred from a README.
- **Principles:** concrete judgment calls when fixed behavior rules run out.
  Generic virtues do not distinguish this agent from any other.
- **Vision:** durable future direction, or an explicit absence. A backlog is
  not confirmed intent; distinguish out of scope for now from out of scope on
  principle.
- **Constraints:** actual provider/model/region restrictions, data boundaries,
  production ceilings, and required approvals. Ask whether inferred gateway
  pins or middleware reflect standing policy. Do not invent constraints.
- **Trade-offs:** hard gates, priorities among remaining goals, and unacceptable
  regressions. Ask once for a usable ranking if the answer is only “balance
  quality and cost”; otherwise record the uncertainty.
- **Change Scope:** levers that exist on this agent, each `yes`, `no`, or
  `with-approval`. Unknown permission is not permission. Name any approver in
  Constraints or Notes. Never infer authorization from the current code.
- **Success Criteria:** desired behavior independent of the current eval suite,
  ranked when priorities differ. Surface conflicts with current scoring for
  the user to resolve rather than adjusting intent to match old tests.

A concrete Role and substantive Purpose & Outcomes are needed before treating
Ethos as a usable evaluation baseline. Ask for them when vague. For other
unknowns, accept “I don't know,” use `_(none)_`, and record the unresolved issue
in Open Questions. Do not interpret an unknown constraint as unlimited freedom.
Preserve the agent's broader mission even when the caller plans only a small
first evaluation set.

Keep durable policy separate from run configuration: a production cost ceiling
belongs in Constraints; the budget or experiment count for a single run belongs
in the tool running that experiment. Keep the current model and runtime unless
the user separately requests changes.

## Document contract

Use schema version 1. Frontmatter is a mapping with nonempty `name` and `author`
and an ISO 8601 `created_timestamp`. Prefer a canonical name matching
`[a-z][a-z0-9-]*`. Preserve creation metadata on edits; add `updated_timestamp`
for the edit time. Add `owner` when known and relevant to approvals.

All fifteen `##` headings below are required and must occur once. Additional
headings and keys are allowed. Replace every placeholder with actual content or
honest `_(none)_`; record unresolved intent in Open Questions. Do not add a Model
section just to duplicate runtime configuration. Permitted models belong in
Constraints.

- Scope uses labeled bullets for Audience, Categories, In scope, and Out of
  scope. Use semicolons for list-valued labels.
- Change Scope uses labeled bullets for actual levers with `yes`, `no`, or
  `with-approval`; Notes can explain exceptions or unresolved permissions.
- Open Questions uses bullets, or `_(none)_` when settled.
- Other sections accept prose, lists, or tables. Tools may be `Prompt-only.`
  only when the agent actually has no tools. Group tools by capability or
  source when credentials, side effects, freshness, and failure modes align.
- Metric Semantics explains ambiguous or decision-critical signals and the
  claims they cannot support. Omit invented measurements and unrun commands.

Use this inline template; no second skill or template download is needed:

```markdown
---
schema_version: 1
name: <agent-name>
created_timestamp: <ISO-8601-creation-timestamp>
author: <actual-author>
---

# Ethos: <agent-name>

## Role

<One concrete sentence describing this agent's role for its users.>

## Purpose & Outcomes

<Mission: why it exists, its user value, and product/workflow context.
Outcome: the external result it is accountable for, agreed targets and their
owner when known. Say when there is no business metric.>

## Scope

- Audience: <intended users>
- Categories: <task categories separated by semicolons>
- In scope: <supported workflows>
- Out of scope: <excluded workflows>

## Tools

<Actual tools, APIs, and knowledge sources; purpose, credentials/scopes, side
effects, freshness, and expected failures. Use Prompt-only. if there are none.>

## Harness

<How this agent runs: model/tool loop, tool execution, context and memory,
stop conditions, action permissions, observability, and runtime where known.>

## Behavior

<Expected behavior, refusals, escalation, tone, accepted limitations, and non-goals.>

## Principles

<Confirmed judgment calls when Behavior does not settle a choice, or _(none)_.>

## Success Criteria

<Observable outcomes and quality standards independent of current eval coverage;
ranked priorities and representative successful behavior where known.>

## Trade-offs

<Hard gates, priority order, and unacceptable regressions, or _(none)_.>

## Constraints

<Confirmed providers/models/regions, data handling, compliance, production
ceilings and measured baselines if known, and approvals; otherwise _(none)_.>

## Evaluation Setup

<Current checks, datasets, commands, metrics, thresholds, and coverage gaps.
Explicitly say when no evals exist; do not claim proposed commands were run.>

## Metric Semantics

<Meaning and source of ambiguous signals and the claims they do not support,
or _(none)_.>

## Change Scope

- <actual change lever>: <yes | no | with-approval>
- Notes: <exceptions, approvers, or unresolved permissions; otherwise _(none)_>

## Vision

<Confirmed durable direction and future use cases, or _(none)_.>

## Open Questions

- <Unresolved issue affecting safe use, evaluation, or modification; use _(none)_ if settled.>
```

## Save, check, and review

Default to `<repo-root>/ETHOS.md`, or the caller's chosen local path. Make
focused edits to existing documents. Do not replace custom content with the
outline. If the requested path is not writable, keep the answers and provide
the complete proposed content plus the local error.

Read back the exact saved file and verify:

1. It exists and is nonempty at the selected path.
2. Frontmatter has version 1, nonempty name and author, and valid creation and
   optional update timestamps; creation metadata is preserved on edits.
3. All fifteen required headings occur once. Role and Purpose & Outcomes are
   concrete. Other sections contain answers or honest `_(none)_`, with no
   unresolved template placeholders. Custom content is preserved.
4. Intended behavior, boundaries, and change permissions agree with the user's
   answers. Inferences and unknowns remain visible.

Use an existing safe YAML parser if available, otherwise perform structural
inspection and report that limitation. These checks establish structure
and consistency, not evaluation coverage.

Present a short summary of role, mission, scope, principles, and direction, link
or show the complete saved document, and call out unresolved questions. Ask for
review of newly authored or revised content before treating it as the baseline.
Interview answers alone are not review of the saved document. Reuse approval of
unchanged content. Apply requested corrections, recheck, and show the changes.
Return the path, substantive intent, gaps, validation method, and review state to
the caller. An explicitly requested unreviewed draft must be labeled as such.
