<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix Evals SDK

## Quickstart

```python
from nhx_evals_sdk import evaluate

result = evaluate(
    dataset="data/eval.jsonl",
    metric="exact_match",
    reference="{{ expected }}",
    candidate="{{ prediction }}",
)

result.print_summary()
rows_df = result.to_pandas()
aggregate_df = result.to_pandas(view="aggregate")
```

`evaluate(...)` is the recommended product-level API for scripts and notebooks. It accepts:
- inline rows
- `DatasetRows`
- `pyarrow.Table`
- a local file path
- a local directory path plus an optional `pattern`

The returned `OfflineEvaluationResult` supports:
- `print_summary()`
- `format_summary()`
- `to_records()`
- `to_table()`
- `to_pandas()`

Pandas conversion is optional. Install `nhx-evals-sdk[pandas]` to use `to_pandas()`.

Offline evaluation uses each dataset row as both `item` and `sample`. In practice,
that means templates should usually read directly from the row, for example
`{{item.answer}}` and `{{item.model_output}}`.

## Advanced Usage

If you want direct control over metric construction or async execution, the low-level APIs remain available:

```python
from nhx_evals_sdk import ExactMatchMetric, evaluate_offline_sync

metric = ExactMatchMetric.from_template(
    reference="{{item.expected}}",
    candidate="{{item.prediction}}",
)

result = evaluate_offline_sync(metric=metric, dataset=[{"expected": "4", "prediction": "4"}])
```

## Gym captured usage in OTLP

`ng_trajectory_otlp.rollout_to_resource_spans` renders captured model calls using
Intake-compatible usage attributes for input, output, reported total, cache read,
cache write, and reasoning tokens. A captured call with `tokens_in` or `tokens_out`
uses Gym's normalized counters; missing or unknown normalized counters stay absent
even when raw provider usage is present. This avoids reinterpreting provider cache
accounting after Gym has normalized it.

Legacy captures without normalized input/output fields read `response.usage`,
including `input_tokens_details.cached_tokens`,
`input_tokens_details.cache_write_tokens`, and
`output_tokens_details.reasoning_tokens`. Explicit zero counts are preserved;
missing totals are not calculated from input and output tokens. Nonempty embedded
`ng_model_call_capture.calls` lists of mappings take precedence over the separate
`model_calls` argument. Empty or malformed embedded captures leave that argument
in effect.

## Docker Compose sandbox provider

`compose.py` is the stable public facade for the production Docker Compose
provider. Focused private modules separate contracts, lifecycle state, CLI and
subprocess behavior, inspection and readiness, file transfer and ownership,
lifecycle primitives, and provider orchestration.

## Docker Compose sandbox tests

The hermetic tests for `DockerComposeSandboxProvider` are organized by the
provider component they exercise: contracts, CLI execution, inspection,
lifecycle, and file transfer. Shared Compose fakes live in the agent-eval test
kit, while the vendored-import test remains a standalone compatibility check.
