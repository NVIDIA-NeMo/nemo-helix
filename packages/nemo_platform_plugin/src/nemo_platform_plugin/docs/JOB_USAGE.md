<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Job token usage reporting

## Purpose

Platform jobs can cause model inference in libraries and services that already
observe provider token counts. The Jobs service cannot reconstruct those counts
from container, process, or pod status. A workload therefore publishes
cumulative totals before it exits:

```python
ctx.usage.report_totals(input_tokens=120, output_tokens=45)
```

Reports use canonical nonnegative integer fields, may omit an unknown
dimension, and replace the previous whole-attempt totals. They are not deltas:
the job status-details API uses last-write-wins merging and does not provide an
atomic increment operation. Platform reporting is best-effort and cannot alter
the job result. Local execution retains the latest report in memory.

## Producer roadmap

Producer integrations build on the shared `JobContext` contract and can be
reviewed independently.

### Data Designer

- Subscribe to the library's per-request token-usage events around generation.
- Sum actual provider calls, including retries, and publish once before return.
- Publish known partial usage before propagating a later failure.
- Verify that event subscriptions are isolated across concurrent jobs.

### Anonymizer

- Reuse Data Designer token-usage events around anonymization.
- Publish both input and output totals without consuming or changing the
  library's separate anonymous telemetry counters.
- Keep non-model workflows explicitly unknown or zero according to a tested
  policy.

### Evaluator

- Normalize OpenAI, Responses, Anthropic, ATIF, and Harbor usage vocabularies.
- Aggregate raw executions rather than medians or presentation summaries.
- Define whether whole-job usage includes target calls, judge calls, retries,
  cache reads, and cache creation before implementation.
- Avoid presenting partial usage as complete when some calls omit counts.

### Agents

- Normalize usage from non-streaming Fabric results and raw Harbor/NAT trials.
- Count actual repeated executions rather than aggregate leaderboard records.
- Leave streaming usage unknown until the stream carries a final usage record.

### Safe Synthesizer

- Publish the measured completion-token total as `output_tokens`.
- Leave `input_tokens` unknown until upstream instrumentation exposes it.
- Do not estimate prompt usage or double-count multi-stage summaries.

## Deferred work

- Training-token throughput from customization and RL workloads has different
  semantics from inference input/output usage.
- Automatic inference/Intake aggregation needs job-attempt correlation,
  ingestion-lag handling, and retry deduplication.
- Accurate cost reporting needs per-model totals and pricing attribution;
  aggregate job counts alone are insufficient for multi-model jobs.
