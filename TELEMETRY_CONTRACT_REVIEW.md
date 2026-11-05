<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Usage telemetry contract follow-up — review draft

**No expanded collection is enabled by this change.** Privacy/legal approval requirements and schema/NVDP promotion remain pending. This document is a proposal, not an approval or a deployed contract.

## Implemented now

The jobs telemetry version resolver prefers installed `nemo-helix` distribution metadata, falling back to `nhx-jobs` for standalone source installs. Bundled release wheels may not have standalone service metadata, which previously resulted in `client.version: undefined`. This changes the populated value of an already-collected package-version field, not its scope. Tests cover bundled, standalone, and unknown versions.

Current platform telemetry stays on schema 1.10 with the same required fields, opt-outs, and rotating session identifiers. No new event types, fields, identity scopes, or attribution detection are wired into emission. Tests explicitly reject proposed fields on the current runtime models until a reviewed contract is promoted.

## Companion schema proposal

The `nemo-telemetry` review branch `helix-telemetry-contract-review/mkornfield` contains:

- `schemas/proposals/helix_usage_events.json`: isolated named-event validation candidate, not the production registry upload format.
- `HELIX_CONTRACT_REVIEW.md`: exact field inventory, privacy/legal decision checklist, NVDP mapping requirements, and promotion gates.
- `helix_contract_test.py`: offline compatibility and privacy-boundary tests; no network emission.

The active registry there is already 1.11; do not assume 1.11 is available for this change or independently bump Helix to it. Registry owners allocate the approved version.

Proposed optional fields: random per-event `eventId`; setup-only random `onboardingAttemptId`; explicit `invocationOrigin` (`human`/`agent`/`unknown`); bounded `toolCategory`; and job `tokenCountsSource` (`job-summary`/`unknown`). The existing onboarding event would gain a `setup_started` step value after opt-in checks. These additions must not be sent by current emitters.

No durable install identity, actual model name, endpoint URL, account/workspace/job name, command arguments, process inspection, or agent-harness fingerprinting is proposed. Existing session rotation and retention are not extended. New correlation and behavioral/tool attribution can still require legal review despite being bounded and content-free.

## Follow-up implementation, after review and promotion

1. Get privacy/legal and telemetry-owner decisions for each new field/purpose, including correlation scope, retention, permitted joins, and notification/opt-out text. Do not label this approved because it is called anonymous.
2. Allocate/promote the approved schema and NVDP mappings, with old-client compatibility. `durationSec` already accepts fractional JSON numbers; investigate why indexed values truncate rather than blindly changing the event type.
3. Add matching typed Helix event fields and serializer parity tests against the approved schema. Preserve older payload compatibility. Prevent mixed-version batches in the single-version envelope.
4. Add per-logical-event IDs once, before enqueue, preserving them across transport retries. Decide separately how to deduplicate repeated job terminal notifications; generating a new UUID each notification does not solve that.
5. Scope an onboarding attempt ID to one opted-in setup invocation, propagated only through its stages. Emit `setup_started` before setup work and terminal outcome on success/failure/cancellation. Never persist the attempt ID as an installation identifier or propagate it into unrelated workloads.
6. Accept explicit bounded invocation-origin declarations only. Unknown is not human. Do not infer origin from process/environment fingerprints. Legacy `agentMode=false` cannot distinguish human from uninstrumented agent use; keep dashboard treatment honest.
7. Map recognized operations to bounded functional categories. Administrative CLI activity is not tool/workload execution. Review categories for guardrails, agent optimization and other previously custom jobs before changing those collection values. Keep unknown sources in a safe bucket rather than leaking names.
8. Label existing terminal job counts as job-summary measurements. Document coverage, unknown `-1`, and counting boundaries. Do not add job-summary counts to underlying inference/library counts.
9. Confirm UAT and production indexed documents/types, fractional duration aggregations, retry/duplicate behavior, and opt-out in all paths. Only then release/enable expanded emitters under the approved rollout mechanism.

## Metrics not solved by this first proposal

- Installation counts: setup is not installation. Bootstrap ownership and lifecycle semantics need a separate decision; no permanent install ID is added.
- Lifetime activation/retention: requires a definition of qualifying workloads, cohort/window and approved identity scope. Existing rotating-session observations are not lifetime users.
- Provider share of inference or top model identity: onboarding provider connections do not measure runs. Actual workload/provider metadata requires its own bounded schema and privacy decision; raw model names remain excluded.
- Complete token totals: define one authoritative layer and coverage per workload/provider before adding broad inference instrumentation. Current reported job counts remain partial.
- Ordered conversion and tool composition: require session/attempt-level transforms and denominator definitions, not only new fields. Independent stage counts are not a true funnel.

The current Kibana dashboard intentionally uses active sessions, observed commands/jobs, independent onboarding progress, provider connection events, and reported token counts. It should not be relabeled with the mockup's install/activation/human-share claims until the relevant evidence exists.
