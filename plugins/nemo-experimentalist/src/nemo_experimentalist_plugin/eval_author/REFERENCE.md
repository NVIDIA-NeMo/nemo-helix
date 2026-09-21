<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Insight suite authoring Python reference

## `EvalAuthorResult`

`run_eval_author(...)` returns an `EvalAuthorResult` with these fields:

| Field | Type | Description |
| --- | --- | --- |
| `train_dataset` | `Dataset` | Staged training dataset with the authored metrics and verifier changes. |
| `validation_dataset` | `Dataset` | Staged validation dataset with the authored metrics and verifier changes. |
| `insight_suite` | `Dataset \| None` | Finalized experiment-local task set generated from the Insight's trace references. |
| `insight_suite_identity` | `str \| None` | SHA-256 identity of the finalized Insight task and verifier content. |
| `metric_keys` | `tuple[str, ...]` | Metric keys authored across all three datasets. |
| `summary` | `str` | Analysis summary. |

When an Insight suite is materialized successfully, `insight_suite` and
`insight_suite_identity` are both populated. The authoring step does not split, merge,
or evaluate the returned suite; its caller owns that downstream integration.
