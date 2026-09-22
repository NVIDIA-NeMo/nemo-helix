<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gym status and timing contract

For import commands, refer to [Import Gym rollout evidence](import-gym.md). For span relationships
and evidence preservation, refer to the [Gym mapping contract](gym-mapping.md).

## Anchor selection

The earliest available source timestamp anchors spans without their own start time. Candidates are
rollout, invocation, model-call, and tool-call `started_at` values, plus turn `timestamp` values.
An individual span's explicit start time is preserved even when another source supplies the anchor.

When no candidate exists, `--started-at` must supply a known, timezone-aware run timestamp. The
importer never substitutes the current time. A fallback anchor is a placement aid, not measured
execution time. Untimed spans have `gym.timing=anchor_only`, no end time, and any recorded
`duration_ms` in `gym.observed_duration_ms`, except for the rollout inference described below.

## Rollout status

The rollout wrapper keeps the hierarchy consistent for one or multiple top-level invocations.
Its explicit status or error takes precedence; otherwise, top-level invocation outcomes determine
status: an error wins, then cancellation, then success only when every top-level invocation
succeeded and no invocation parent is missing. Insufficient evidence remains unknown.

Reward remains independent of execution status. Nested failures do not override a successful
top-level invocation that may have recovered from them.

## Rollout timing

When the rollout supplies neither absolute boundary, at least one child span with an observed
start is required to infer an end. Every such span must have an observed end; the latest of those
ends becomes the rollout end. The rollout start remains the earliest source anchor described
above, including any earlier turn timestamp. The importer does not replace that anchor with the
earliest child start, preserving the storage key when replaying an import.

This derived window is marked `gym.timing=observed_child_window`. Overlapping intervals are not
summed. The window may exclude unobserved setup or teardown. A child with a recorded start but no
end prevents an inferred end. Existing explicit rollout boundaries are not extended by this rule.

Recorded `ng_perf.total_latency_ms` is retained separately as `gym.observed_duration_ms`, with
`gym.duration_source=ng_perf.total_latency_ms`; it never manufactures an absolute timestamp.
Studio shows this as **Recorded rollout duration**, alongside the observed window used by the
trace summary.
