<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Execution Architecture

The execution package exposes a single public entrypoint:

- `Evaluator`
  Result-oriented API that evaluates one metric or a sequence of metrics and
  returns finished `EvaluationResult` or `BenchmarkEvaluationResult` values.

## Public Surface

- `Evaluator.run()` / `Evaluator.run_sync()`
  Evaluate one metric or many metrics and return the completed result.
- `RunConfig`
  A Pydantic config type for execution-only settings such as parallelism and
  sample limits.
- `EvaluationBackend`
  The protocol implemented by result-returning execution backends.

## Backend Selection

```python
# Local SDK execution
evaluator = Evaluator()
result = await evaluator.run(
    metrics=[ExactMatchMetric(reference="{{item.reference}}")],
    dataset=[{"reference": "Paris", "output_text": "Paris"}],
)
```

## Design Notes

- `Evaluator()` uses `LocalBackend`.
- `Evaluator(client=...)` accepts an evaluator backend object. Platform-specific
  backend adapters are provided by
  [`nemo-evals-plugin`](../../../../../plugins/nemo-evals/README.md).
