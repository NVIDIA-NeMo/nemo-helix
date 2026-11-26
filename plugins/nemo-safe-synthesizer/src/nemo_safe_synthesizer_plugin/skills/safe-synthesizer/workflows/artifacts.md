<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Safe Synthesizer Artifacts

## Prerequisites

- NeMo CLI access through `nemo` or `uv run nemo`; use the repo version from this branch or newer.
- Workspace access to the job and result filesets for platform artifacts.
- `config.evaluation.enabled` must be `true` for evaluation reports to be produced.
- A synthesis run must train and save an adapter for `adapter` artifacts to exist.

## Interpreting Outputs

- `synthetic-data.csv`: generated or processed tabular data.
- `summary.json`: run summary and timing metadata.
- `evaluation-report.html`: optional evaluation report when enabled and produced.
- `evaluation_report_data.json`: optional evaluation results as JSON, written alongside the HTML report.
- `adapter/`: optional adapter output when training produces one.
- `summary`
- `synthetic-data`
- `evaluation-report`
- `evaluation-report-data`
- `adapter`
- `summary` / `summary.json`: use this first for row counts, timing, and high-level run status.
- `synthetic-data`: final CSV data. For PII-only runs this may be processed data rather than newly synthesized records.
- `evaluation-report`: HTML report if evaluation was enabled and completed.
- `evaluation-report-data`: JSON with the scores, per-metric results (the values behind the report's charts), column statistics, and the PII replacement applied. Missing if evaluation was skipped, or if the report ran but its data could not be built or uploaded; the job logs say why.
- `adapter`: model adapter artifacts, present only for runs that train and save an adapter.

## Missing Artifacts

- If `evaluation-report` is missing, confirm `config.evaluation.enabled` and check whether the runtime produced `evaluation_report_html`.
- If `adapter` is missing, confirm the run trained a model and produced an adapter path.
- If `synthetic-data` is missing, treat it as a failed run and inspect job or local logs first.

## Next Steps

- Retrieve result files with `workflows/results.md`.
- Check evaluation and adapter settings in `workflows/config.md`.
- Start artifact inspection with `summary` or `summary.json`.
- Debug missing outputs with `workflows/diagnose.md`.
