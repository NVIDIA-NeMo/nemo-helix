<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Benchmark Reproduction

Read this file for BYOB protocol, long-running benchmark reproduction, artifact
contracts, and external-safe output requirements.

## Bring Your Own Benchmark

For external, custom, or bring-your-own benchmarks, keep the protocol explicit
and reproducible.

- Separate judge-quality evaluation from generation-quality evaluation. If the
  target is report generation, generate fresh model responses, score them with
  a fixed judge, and aggregate fulfilment. Do not treat human labels for
  existing baseline responses as labels for new model outputs.
- Use provider-agnostic configuration. Require users or harness config to
  provide generator and judge base URLs, model IDs, and secret references.
- Validate model IDs with a `/v1/models`-style endpoint when the provider
  supports it. Do not assume a web catalog equals runnable API model IDs.
- Run a tiny smoke prompt before the full benchmark to confirm authentication,
  response shape, parser behavior, and unsupported generation parameters.
- Record artifacts that make the run replayable: `README.md`, `manifest.yaml`,
  `scoring_protocol.md`, `results_report.md`, `generations.jsonl`,
  `judge_predictions.jsonl`, and `scores.json`.
- Store secret variable names or secret file paths in manifests, never secret
  values.
- Report calibration deltas transparently. Do not apply hidden correction
  factors; inspect endpoint/model ID, generation params, sampling plan, judge
  prompt, reasoning policy, parser, and aggregation if deltas are large.

## Long-Running Benchmark Reproduction

Use this pattern when a BYOB run is too large for a single in-memory SDK
example.

1. Freeze the dataset, rubric, sample plan, generator parameters, judge
   parameters, and any external scoring weights used by the benchmark protocol
   before starting the full run.
2. Smoke-test one or two rows end to end: generate, judge, parse, aggregate,
   and write artifacts.
3. Use one run directory per generator model or model configuration. Do not mix
   outputs from different generator settings in the same checkpoint files.
4. Checkpoint generation and judging separately so either stage can resume.
   Prefer append-only JSONL with stable row IDs, external criterion IDs from
   the expanded BYOB artifact, status, parsed values, and error fields.
5. For external harness JSONL recovery, recover by stable IDs, validate each
   line before reuse, rewrite a clean compacted file after recovery, and keep
   the corrupted fragment for audit.
6. Retry failed rows individually with bounded attempts. Fail closed on missing
   rows, missing criteria, or unparsable judge outputs; do not silently drop
   them from benchmark-protocol scoring counts.
7. Write final task, domain, and overall scores only after every required
   criterion row is present and parseable, or after failures are explicitly
   counted according to the scoring protocol.
8. Report progress periodically by stage: generated rows, judged criterion
   rows, parse failures, retry queue, rate-limit sleeps, and estimated
   remaining work.

ProfBench-style rubric-to-leaderboard shape:

```text
dataset rows
-> generated responses per model
-> criterion-level judge rows per response
-> fixed binary judge with parser-compatible yes/no output
-> weighted criterion fulfilment
-> task, domain, and overall scores
```

Keep the fixed judge, rubric text, parser, and any external benchmark weights
constant when comparing generator models. If the judge itself is being
evaluated, run a separate judge-quality experiment against labeled existing
responses.

## Artifact Contract

For an external BYOB handoff or checkpointed benchmark harness, these artifacts
make the run inspectable and rerunnable. The SDK does not generate this full
contract automatically; teams that need strict enforcement should implement a
typed artifact contract outside the skill.

- `manifest.yaml`: dataset version, rubric version, sample plan, model aliases,
  parameter summaries, scoring protocol, and artifact paths.
- `expanded_dataset.jsonl`: immutable rows after sampling and task expansion.
- `generations.jsonl`: generator outputs with row IDs, model alias, params
  summary, status, output text, and errors.
- `criterion_rows.jsonl`: one row per generated response and rubric criterion.
- `judge_predictions.jsonl`: raw and parsed judge outputs with retry metadata.
- `failure_log.jsonl`: failures, skipped rows, parser errors, and retry
  exhaustion.
- `scores.json`: task, domain, and overall aggregates with scoring counts
  defined by the benchmark protocol.
- `results_report.md`: concise explanation of setup, scores, caveats, and
  interpretation.

## External-Safe Outputs

For public or broadly shared skills and reports, avoid internal endpoint names,
internal model IDs, authentication details, and secret paths. Use placeholders
such as `<provider-base-url>`, `<generator-model-id>`, `<judge-model-id>`, and
`<secret-reference>`. Report parameter categories and aliases instead of
revealing private infrastructure.
