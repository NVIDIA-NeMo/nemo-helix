<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Import MLflow traces

The importer calls `mlflow.search_traces(..., return_type="list")`. It maps every `Trace.data.spans`,
then imports `Trace.info.assessments`: LLM/code feedback becomes evaluator results, human feedback
becomes annotations, and expectations become metadata annotations. The complete native assessment
and unmodeled trace fields remain under `mlflow.signals` and `mlflow.raw`.

Set `MLFLOW_TRACKING_URI` using the same URI that the MLflow client already uses, then run the script
from the directory containing this reference:

```bash
uv run --with mlflow python ../scripts/import_mlflow.py \
  --project <experiment-id> \
  --since 2026-08-01T00:00:00Z \
  --until 2026-08-02T00:00:00Z \
  --workspace "$WORKSPACE" \
  --nhx-base-url "$NHX_BASE_URL"
```

Use `--input exported-traces.json` for an offline object shaped as `{"traces":[Trace.to_dict()]}`.
`--project` is optional offline. MLflow serializes span IDs as base64 in `Span.to_dict()` while its
assessment API uses hex span IDs; the importer canonicalizes both to MLflow's user-facing hex ID.

Credentials are owned by the MLflow client configuration. Intake auth uses the active NeMo CLI
context and its OAuth refresh flow; `NHX_ACCESS_TOKEN` remains an explicit override.
