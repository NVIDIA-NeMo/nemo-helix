<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Starting data

Read this during Step 4 of [the environment sub-flow](../SKILL.md). Starting data
is the most common source of easy tasks: a world that contains only the records
a task needs lets the agent succeed without searching, choosing, or checking.

## Base population

- Generate structured data in code from the repository's schema, using
  migrations or ORM models as the source of truth, with a fixed seed. Write the
  generated data into `data/` and record the generator, seed, and per-table
  digests ([Digests for preservation checks](#digests-for-preservation-checks)),
  so every build and every trial starts from identical records.
- Size and shape the population like the data the agent meets in practice:
  entity counts, status distributions, typical field values, and history. Use
  the selected traces when they exist; otherwise choose plausible volumes and
  say so in the plan. Enough records that the agent must search is usually
  enough; bulk that only slows builds adds nothing.
- Include the variety real data has: similar names, shared attributes, mixed
  statuses, historical and archived records, and edge values the schema allows.
- Write prose fields such as notes and descriptions from the structured facts
  they describe, so text never contradicts the data it sits beside.
- Use no real personal data. Use a sanitized production snapshot only with the
  user's explicit approval, keep it and everything derived from it private and
  out of version control, and record its provenance in the plan.

## Per-task records

The calling flow's case plan names the records each case needs and the outcome
it expects. Add them on top of the base population rather than editing base
records. Keep them with their task: in the task's own `environment/` for Harbor,
or in the dataset row or session seed for Gym. The shared base then stays
reusable, and each task's world stays reviewable on its own.

## Decoys

When a case names a condition that needs confusable records, such as a similar
customer, an ineligible order, or a stale duplicate, build them so a careful
agent can still tell them apart:

- **Similar** to the target in the ways the request describes it, and
  **different in exactly one way** the agent can discover through permitted
  reads, such as status, owner, date, or eligibility.
- **Added, never edited.** Decoys must not change the records the expected
  outcome depends on.
- **Real.** Every identifier the request or the decoy mentions exists; nothing
  refers to records that do not exist, which would only send the agent
  searching forever.
- **Hidden.** Which record is the decoy is verifier-only knowledge. Avoid telling
  names, comments, identifiers, or insertion order.
- **Purposeful.** Add a decoy only for a condition a case names. Unrelated
  padding makes tasks slower, not more informative.

A decoy the agent might wrongly change belongs in the verifier's checks: an
unexpected change to it should fail the case.

## Integrity checks

Before any task relies on the data, check and record:

- every reference resolves: foreign keys, ownership lists, nested identifiers;
- each record's status permits the actions the cases expect, and forbids the
  actions the cases expect the agent to refuse;
- totals and balances recompute from their parts under the domain's rules;
- identifiers the system assigns at runtime are not pre-created;
- required fields hold schema-valid values, and enumerations use exact values.

A script or query that asserts these properties is more reliable than
inspection; keep it in `data/` and rerun it whenever the generator changes.

## Digests for preservation checks

Each time the generator runs, record a content digest for every table or file in
the starting data in `data/digests.json`. Compute each digest independently of
row order and storage format: serialize every row canonically, sort the
serialized rows, and hash them with SHA-256. Row counts are no substitute, since
a count still matches after values are edited. When a task layers its own
records onto a table, record that table's digest for the task's starting state.

The digests are verifier-only: copy the entries a task needs into its verifier
files, never into the agent's environment. The verifier recomputes the digest of
each table the case must leave unchanged from the exported end state and fails
the case on a mismatch. Tables the case is meant to change get field-level
assertions instead, and a table missing from the export is an infrastructure
error.

## Expected end state

The verifier needs the end state each case should produce. Derive it from the
case's Ethos-backed outcome applied to this starting data: apply the intended
actions to a copy of the seeded state, or state the expected changes as
field-level assertions. Keep it verifier-only.

When Ethos does not settle what the correct end state is, for example whether a
partial refund is acceptable, do not choose. Mark the item unresolved and raise
it in the calling flow's check-in; a guessed rule becomes an unapproved policy
inside the grader.

## Reset

Every trial must start from the same seeded state. Prefer seeding at image build
or service start inside containers that are recreated for each trial. When state
lives outside the containers, document the reset command, run it before each
trial, and keep trials serial until isolation is proven.
