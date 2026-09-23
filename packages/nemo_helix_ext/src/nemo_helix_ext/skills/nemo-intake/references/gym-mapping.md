<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gym mapping contract

## Prerequisites

A completed Gym rollout JSONL artifact containing `ng_trajectory` with `schema_version: "1.0"`
is required. Enable Gym rollout observability before collecting the artifact; older exports
without `ng_trajectory` are not supported.

## Identities and hierarchy

Task and rollout IDs are scoped to the supplied run ID. Each rollout is a trace and session;
repeated attempts remain separate. Anonymous model calls use their position in the artifact for
identity, so replay the same artifact rather than reordering anonymous calls.

| Gym evidence | Intake representation |
|---|---|
| Rollout | Root chain span, `source=gym`, run/task/rollout identity attributes |
| Agent invocation | Agent span with explicit parent invocation and conversation output |
| Model call | LLM span with request, response, model name, and available token counts |
| Tool observation | Tool span with recorded result, uniquely matched conversation arguments, status and timing |
| Top-level `reward` | `gym.reward` evaluator result on the rollout root; zero is preserved |
| Turns, gaps, compaction, sandbox evidence and other fields | Complete original rollout under the root's `gym.raw` attribute |

## Ownership and validation

Model ownership requires a unique explicit model-call ID or exact model-reference/response-ID
match. Ambiguous or unavailable ownership attaches the model span to the rollout root and records
`gym.ownership=unavailable_or_ambiguous`; timestamps and list order do not establish ownership.
Missing invocation parents also attach to the rollout root while preserving the native relationship.
Unsupported schema versions, duplicate identities, cycles, and reversed observed intervals fail
before any upload. `--no-include-feedback` suppresses evaluator-result writes, retaining the raw reward.

## Evidence coverage

This version preserves Gym turns and delegation-tool references as source evidence; it does not
invent additional step spans or evaluation leaderboard records. Agent conversations retain their
Gym Responses API format.

## Next Steps

- Follow [Import Gym rollout evidence](import-gym.md) to preview, upload, and verify the artifact.
- Refer to the [Gym status and timing contract](gym-timing.md) for timestamp anchors, rollout
  status, and duration.
