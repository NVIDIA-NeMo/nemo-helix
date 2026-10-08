<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Read this file when SDK runs fail, judge outputs do not parse, provider calls
time out, or generated artifacts need recovery.

## Failure Modes

- Empty completions: record the raw provider response, prompt, params, finish
  reason, and provider token/request metadata when available. In SDK results,
  look for request metadata in row-level request details rather than aggregate
  scores. Retry only if the provider response indicates a transient failure.
- All-reasoning outputs: inspect completion token use and final-answer length;
  record token budget and reasoning settings separately from benchmark
  semantics.
- Unparsable judge rows: store raw judge text, parser error, row ID, external
  criterion ID from the expanded BYOB artifact if present, and retry count;
  count unresolved rows as failures unless the protocol defines different
  failure accounting.
- SDK row failures: by default, request or scoring failures abort the run. Use
  `ignore_request_failure=True` only when the benchmark protocol allows failed
  rows to be marked as `NaN` and carried into reporting.
- SDK provider timeouts and rate limits: configure SDK request behavior with
  fields such as `request_timeout` and `max_retries`; the SDK handles retry
  attempts internally and does not expose a persistent retry queue.
- External BYOB harness timeouts and rate limits: if building a checkpointed
  harness outside the SDK, use bounded retries with backoff, persist the retry
  queue, and keep progress artifacts valid after interruption.
- Failed-row retries: retry the smallest failed unit, usually one generation
  row or one criterion judge row, rather than rerunning completed work.

## Debugging Checklist

- Wrong schema: compare the metric constructor fields against
  `values/metrics.py`.
- Missing field: compare dataset row keys with Jinja templates.
- Bad parser: make the judge return exactly the JSON or regex format the parser
  expects.
- Judge produced reasoning but no final answer: increase final-answer token
  budget or adjust reasoning params if the provider supports them.
- Secret error: for local SDK runs, ensure the environment variable resolved
  from `api_key_secret` exists; for remote platform jobs, create the platform
  secret in the job workspace.
- Unsupported provider parameter: remove optional params first, then add them
  back one at a time.
- Tool-calling mismatch: check case sensitivity, function name normalization,
  argument JSON validity, and response shape.
- SDK mismatch: verify the request/config payload uses current SDK metric
  fields.
