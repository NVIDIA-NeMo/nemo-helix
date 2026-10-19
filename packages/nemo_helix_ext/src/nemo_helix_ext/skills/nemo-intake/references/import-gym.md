<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Import Gym rollout evidence

Use `scripts/import_gym.py` for completed Gym rollout JSONL containing `ng_trajectory` with
`schema_version: "1.0"`. This importer uses the existing direct span and evaluator-result APIs;
it does not require Gym to be installed in the platform environment or a new Intake endpoint.

## Prepare the rollout

Enable Gym rollout observability before collecting evidence. Older exports without `ng_trajectory`
are rejected; OTEL service spans alone do not contain these trajectories.

Choose a unique run ID for each evaluation job and reuse it when replaying the same artifact.
Keep the original artifact order when replaying anonymous model calls. Refer to the
[Gym mapping contract](gym-mapping.md) for identity, ownership, validation, and evidence coverage.

When a trajectory has no absolute timestamps, supply `--started-at` with a known, timezone-aware
run timestamp (for example `2026-09-21T10:00:00Z`). Refer to the
[Gym status and timing contract](gym-timing.md) for anchor selection, status rollups, and the
difference between recorded duration and an observed execution window.

## Preview the import

From this skill directory, run:

```bash
uv run python scripts/import_gym.py \
  --input /absolute/path/rollouts.jsonl \
  --run-id my-evaluation-2026-09-21 \
  --agent-name my-agent \
  --dry-run
```

Inspect the dry-run output for the actual harness's evidence coverage. Add `--no-include-feedback`
if evaluator results should not be written; the raw reward remains preserved.

## Upload and verify

Verify Intake's read path as described in [the Intake skill](../SKILL.md), then remove `--dry-run`
and supply `--nhx-base-url` and `--workspace` if the active CLI context does not select the destination.
The shared importer runtime handles authentication, batching, and read-back verification of span IDs.
The same 90-day Intake retention limit as other direct importers applies.
