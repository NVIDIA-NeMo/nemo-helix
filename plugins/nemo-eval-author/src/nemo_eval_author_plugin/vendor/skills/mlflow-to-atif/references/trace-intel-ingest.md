<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Load MLflow evidence with NeMo Compass

Read this when normalized provider evidence is useful alongside ATIF, or when
ATIF cannot represent the available capture. This path uses the shared
[`trace-ingest` package](https://github.com/NVIDIA-NeMo/labs-nemo-compass/tree/692d1bf57b6a9372f628b7c2004852aec7a9a83e/packages/trace-ingest)
at revision `692d1bf57b6a9372f628b7c2004852aec7a9a83e`. It emits NeMo Compass models,
**not ATIF**. Do not pass its JSON files to Harbor, audit measurement, or the
experimental trace-environment `prepare` command as trajectories.

## Environment and command

Use Python 3.12 or 3.13. In an environment authorized for dependency setup,
install the pinned package (the scripts never install dependencies themselves):

```bash
uv pip install \
  'trace-ingest[mlflow] @ git+https://github.com/NVIDIA-NeMo/labs-nemo-compass.git@692d1bf57b6a9372f628b7c2004852aec7a9a83e#subdirectory=packages/trace-ingest'
```

Run from this skill directory using that environment's Python:

```bash
python scripts/load_mlflow_trace.py --input <private-export.json> \
  --max-traces 100 --output-dir <new-private-evidence-dir>
```

## Supported evidence

The shared file loader accepts complete MLflow exports, including supported
JSONL and SDK export shapes. Loading uses the MLflow extra but performs no live
store query. It preserves normalized span trees, source metadata, assessments,
and the distinction between missing and explicit-null outputs.

`--max-traces` is required. A larger export, a continuation token, unresolved
parents, an empty corpus, or a loader validation error fails the command; no
truncated normalized corpus is accepted. For live ATIF conversion, continue
using the existing bounded converter and its transport checks.

Outputs are `source.mlflow.json` (exact export bytes), `traces.normalized.json`
(an array of `trace_ingest.Trace` models), and `loading.json` (revision, count,
bound, and digests). The upstream file loader determines corpus ordering;
this is not an ATIF step sequence. Keep the original for source fields that
are not projected into normalized spans.

## Evidence handling and failures

Use a fresh private directory. The wrapper creates it with mode `0700`, writes
retained files with mode `0600`, and refuses existing paths. Embedded media or
source paths are not opened. Output contains restricted trace data; do not
publish it or treat it as a sanitized report. CLI success and errors are
content-free; provider exceptions are not echoed.

On failure, check the installed pinned revision, supported source shape, and
input completeness privately. Failed loads remove only the newly created output
directory. Existing directories remain untouched. These loaders do not provide
an ATIF exporter: replacing the converters requires a separate fixture-tested
mapping for ATIF ordering, interaction evidence, and losses.
