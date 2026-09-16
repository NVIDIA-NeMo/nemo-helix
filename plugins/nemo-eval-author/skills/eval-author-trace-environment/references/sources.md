<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace sources and normalization

One task per trace. Do not merge unrelated traces. Preserve the source's trace
and span identifiers in `extra`, order steps by recorded time, and record every
missing or lossy field under `extra.normalization.uncertainties` or
`extra.normalization.losses`. Never invent an instruction, tool result, file,
or final answer.

## Existing ATIF

Use the trajectory as the canonical input. Do not relabel its ATIF version.

## MLflow

Read and follow the `mlflow-to-atif` skill. Put its owner-private output under
the task workspace and use the emitted `.atif.json` file as the canonical input.

## Intake

Resolve the CLI exactly as `eval-author-inspect-trace` specifies. Read one exact
trace and every detailed span into owner-private JSON files:

```bash
umask 077
nemo intake traces get --output-format=json --workspace="WORKSPACE" --mode=detailed "TRACE_ID" > <task-dir>/private/intake-trace.json
nemo intake spans list --output-format=json --workspace="WORKSPACE" --filter.trace-id="TRACE_ID" --mode=detailed --sort=started_at --all-pages > <task-dir>/private/intake-spans.json
```

Require every returned span's `trace_id` to match the selected trace. Convert
those records to ATIF: root input becomes the user instruction; model output
becomes agent steps; each tool span becomes a paired tool call and observation;
errors and unmapped attributes remain cited in `extra.intake`. If there is no
complete human instruction, record no_candidate.

## OpenTelemetry

Prefer an existing Intake trace produced from the telemetry and use the Intake
path above. For a bounded JSON OTLP export, map the root and descendant spans
directly using the same rules, preserving trace IDs, span IDs, parent IDs,
timestamps, status, and semantic attributes under `extra.otel`. Do not parse
protobuf bytes, contact a collector, or ingest remote data in this flow. If the
export cannot establish parentage or a human instruction, record no_candidate
instead of guessing.

## Canonical verification and bounded normalization

Before continuing, verify the canonical file has one ATIF v1.0-v1.7 object,
one-based sequential step IDs, at least one user step, and resolvable tool-call
references. The helper accepts at most 128 MiB of exact source bytes and
produces canonical and safe files of at most 25 MiB. It may make only two
bounded normalizations: insert a missing JSON escape when the parser-implicated
quote immediately follows a provider-redaction placeholder; and convert
string-encoded ATIF image objects into image parts while omitting encoded
binary data, including image metadata embedded in text fields. Every operation,
offset, count, and loss is recorded in the canonical ATIF and summary. Exact
source bytes remain unchanged and hashed. Reject unrelated JSON damage instead
of repairing it heuristically.
