<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Import Gym rollout evidence

Use `scripts/import_gym.py` for completed Gym rollout JSONL containing `ng_trajectory` with
`schema_version: "1.0"`. This importer uses the existing direct span and evaluator-result APIs;
it does not require Gym to be installed in the platform environment or a new Intake endpoint.
Older Gym exports without `ng_trajectory` are rejected. Enable Gym rollout observability before
collecting new evidence; OTEL service spans alone do not contain these trajectories.

From this skill directory, preview the mapping:

```bash
uv run python scripts/import_gym.py \
  --input /absolute/path/rollouts.jsonl \
  --run-id my-evaluation-2026-09-21 \
  --agent-name my-agent \
  --dry-run
```

Use a unique run ID for each evaluation job and reuse it when replaying the same artifact.
Task and rollout IDs are scoped to that run. Each rollout is a trace and session; repeated
attempts remain separate. Anonymous model calls use their position in the artifact for identity,
so replay the same artifact rather than reordering anonymous calls.

When a trajectory has no absolute timestamps, supply `--started-at` with a known, timezone-aware
run timestamp (for example `2026-09-21T10:00:00Z`). The importer never uses the current time as a
substitute. Untimed spans have `gym.timing=anchor_only`, no end time, and any recorded duration
in `gym.observed_duration_ms`. A fallback anchor is a placement aid, not measured execution time.
If timestamps exist, the earliest observed model/tool start or turn timestamp anchors untimed spans.

For upload, verify Intake's read path as described in `SKILL.md`, then remove `--dry-run` and
supply `--nmp-base-url` and `--workspace` if the active CLI context does not select the destination.
The shared importer runtime handles authentication, batching, and read-back verification of span IDs.
The same 90-day Intake retention limit as other direct importers applies.

## Mapping

| Gym evidence | Intake representation |
|---|---|
| Rollout | Root chain span, `source=gym`, run/task/rollout identity attributes |
| Agent invocation | Agent span with explicit parent invocation and conversation output |
| Model call | LLM span with request, response, model name, and available token counts |
| Tool observation | Tool span with recorded result, uniquely matched conversation arguments, status and timing |
| Top-level `reward` | `gym.reward` evaluator result on the rollout root; zero is preserved |
| Turns, gaps, compaction, sandbox evidence and other fields | Complete original rollout under the root's `gym.raw` attribute |

Model ownership requires a unique explicit model-call ID or exact model-reference/response-ID
match. Ambiguous or unavailable ownership attaches the model span to the rollout root and records
`gym.ownership=unavailable_or_ambiguous`; timestamps and list order do not establish ownership.
Missing invocation parents also attach to the rollout root while preserving the native relationship.
Unsupported schema versions, duplicate identities, cycles, and reversed observed intervals fail
before any upload. `--no-include-feedback` suppresses evaluator-result writes, retaining the raw reward.

This version preserves Gym turns and delegation-tool references as source evidence; it does not
invent additional step spans or evaluation leaderboard records. Agent conversations retain their
Gym Responses API format. Inspect the dry-run output for the actual harness's evidence coverage.
