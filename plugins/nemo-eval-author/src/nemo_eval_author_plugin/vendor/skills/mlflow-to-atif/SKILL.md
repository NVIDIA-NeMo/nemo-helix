---
name: mlflow-to-atif
description: >-
  Convert bounded live MLflow traces or exported Trace.to_dict() JSON into one
  canonical ATIF trajectory per trace for Harbor or Eval Author audit coverage.
  Also load complete local exports through NeMo Compass for normalized span evidence.
  Use for MLflow-to-ATIF conversion or local loading, not Intake ingestion.
triggers:
  - load MLflow export evidence with NeMo Compass
  - convert MLflow traces to ATIF
  - prepare an MLflow trace for an ATIF consumer
  - measure Eval Author audit coverage from an MLflow run
not-for:
  - nemo-intake (use to import MLflow into Intake or query normalized spans)
  - eval-author-inspect-trace (use to explain one trace that is already in Intake)
  - eval-author-audit (use after this skill emits ATIF to measure audit coverage)
compatibility: >-
  Offline conversion uses Python 3.11+ and the standard library. Live queries
  require an existing Python environment with MLflow. Optional reference
  validation requires Harbor. Output is ATIF v1.7 for current downstream
  compatibility. The optional NeMo Compass loader requires Python 3.12 or 3.13
  and the pinned trace-ingest MLflow extra.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: Bash Read Write
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Convert MLflow to ATIF

## Purpose

Produce canonical ATIF without routing trace data through Intake. The bundled
script writes one owner-private `.atif.json` file per MLflow trace and prints
only a content-free summary.

For shared normalized span evidence, read
[NeMo Compass ingestion](references/trace-intel-ingest.md) and use the optional
`scripts/load_mlflow_trace.py` command. Its output is not ATIF and does not
replace the conversion or downstream validation contract below.

## Protect the trace

Treat the source and converted files as restricted data unless the user proves
otherwise. Do not print trace payloads, place them in Git, or write them into a
public or shared output directory. The script makes its output directory mode
`0700` and trajectory files mode `0600`.

## Choose the input

For exported data, accept one `Trace.to_dict()` value, a JSON array of those
values, or an object shaped as `{"traces": [...]}`:

```bash
python scripts/convert_mlflow_to_atif.py \
  --input <private-export.json> \
  --output-dir <private-atif-dir> \
  --agent-name <stable-agent-name> \
  --agent-version <agent-version>
```

For a live MLflow store, require an explicit experiment and bounded time range.
Use a Python environment where MLflow is already available; do not install it
or change MLflow authentication on the user's behalf:

```bash
python scripts/convert_mlflow_to_atif.py \
  --tracking-uri <mlflow-uri> \
  --experiment-id <experiment-id> \
  --since <inclusive-ISO-8601-time> \
  --until <exclusive-ISO-8601-time> \
  --output-dir <private-atif-dir> \
  --agent-name <stable-agent-name> \
  --agent-version <agent-version>
```

Live conversion requires HTTPS for remote tracking servers, permits HTTP only
for loopback hosts, rejects disabled TLS verification, and does not follow HTTP
redirects. It uses the paginated `MlflowClient.search_traces()` API so it also
works with MLflow versions that predate the fluent API's `return_type="list"`.

The script refuses to overwrite an existing trajectory unless the user asks to
replace the same conversion and `--overwrite` is supplied.

## Version and validation

Emit ATIF v1.7 and add `--validate-with-harbor` when Harbor is installed.

ATIF v1.8 is newer and adds audio content parts. This converter currently
projects MLflow text and JSON data only, so v1.8 adds no needed representation.
Move the converter and downstream consumers to v1.8 together when they support
that version; do not relabel v1.7 output as v1.8.

## Conversion contract

- Recover the human instruction from the root span or trace input. Fail instead
  of inventing an instruction when none exists.
- Emit LLM spans as agent steps and preserve model and token metrics when MLflow
  recorded them.
- Emit tool, retriever, embedding, reranker, and guardrail spans as deterministic
  agent steps with paired tool calls and observations.
- Flatten the span tree into timestamp order while retaining native IDs, parent
  IDs, attributes, events, status, and assessments under namespaced `mlflow`
  metadata. When an event's parent-span bounds prove that its nominal
  `time_unix_nano` value is actually in microseconds, normalize that value to
  nanoseconds and record the correction under
  `extra.mlflow_to_atif.normalization_codes`.
- Reject unresolved parents, cycles, and exported search pages with a non-empty
  `next_page_token`; collect the complete export before conversion.
- Distinguish missing, explicit null, and populated tool outputs in each result's
  `extra.mlflow.output_state`.
- Record every known lossy projection under
  `extra.mlflow_to_atif.loss_codes`.

After conversion, pass one emitted file to the downstream consumer as canonical
ATIF. Keep restricted originals separate from restricted converted output.

## Prerequisites

Offline conversion needs Python 3.11+ and a complete local `Trace.to_dict()`
export. Live conversion additionally needs MLflow in the existing environment,
configured access to the selected store, an experiment ID, and explicit time
bounds. Optional Harbor validation uses an already installed Harbor environment.
Choose a private output directory and a stable agent name and version.

## Available Scripts

Run from this skill's directory, or use the script's absolute installed path.
Use the Python environment already prepared for the chosen input mode.

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/load_mlflow_trace.py` | Load a complete export through pinned NeMo Compass; no ATIF | `--input --max-traces --output-dir` |
| `scripts/convert_mlflow_to_atif.py` | Export one ATIF trajectory per selected MLflow trace | `--input` or bounded live-store flags; `--output-dir --agent-name --agent-version` |

## Limitations

The converter projects text and JSON into ATIF v1.7, preserving known losses;
it cannot recover unrecorded human input, tool output, or multimedia. Conversion
does not prove task correctness or coverage. Live reads are bounded to the
supplied experiment and time range and do not change authentication.

## Troubleshooting

- Missing human instruction: obtain a complete export; never invent input text.
- Non-empty `next_page_token`: collect the remaining export pages before conversion.
- Unresolved parents or cycles: obtain a consistent trace export rather than
  deleting spans to make it parse.
- TLS or redirect rejection: use the authorized HTTPS endpoint with verification
  enabled; HTTP is supported only for loopback.
- Existing output: use a fresh directory unless replacement of this same
  conversion is explicitly requested; only then use `--overwrite`.
