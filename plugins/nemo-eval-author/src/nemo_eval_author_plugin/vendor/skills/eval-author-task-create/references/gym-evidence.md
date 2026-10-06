<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native Gym evidence

Use `task_evidence.py` with its maintained `gym_evidence.py` adapter. Compatibility
is exercised against the repository's pinned Gym 0.6.0 runtime; newer runtime
formats must pass the native integration checks before claiming compatibility.
This records native evidence; it grants no execution or model-spend permission.

## Prepare the selected inputs

In the shared result contract, set `provider: gym`, `native_run_id: /native_run_id`,
and a `gym` object like:

```json
{
  "dataset": "environments/ledger/data/example.jsonl",
  "dataset_row": 1,
  "config": "evidence-inputs/resolved-config.yaml",
  "runtime": "evidence-inputs/runtime.json",
  "reset": "evidence-inputs/reset.json"
}
```

These paths are relative to the task tree. Select one one-based physical dataset
row per manifest. The recorder hashes the full dataset, resolved configuration,
runtime record, and reset plan. Do not place inputs in excluded cache directories.
The runtime record should identify Gym revision/version, Python, and dependencies;
the reset plan describes the actual initial state/reset procedure. Prepare after
resolving services and configuration so ephemeral settings are known. Keep
expected inputs in the task tree and execution evidence outside it.

For each declared verifier control, include `gym_case` and `gym_kind` matching
its native `gym env test --json` case name and kind. The adapter retains
`case.observed_rewards`; numeric cases also expose the first observation as
`reward`. Malformed-input fixtures retain their native kind and empty observation
list; they do not receive an invented reward. Declare assertions accordingly.
For example, a reference uses a reward band; a malformed case can assert
`/case/kind == "malformed"`. These are completed native fixture checks, distinct
from infrastructure failure while collecting their report.

Normalized `health.verdict` is `healthy` for completed verifier cases. Rollouts
retain Gym's `healthy`, `unhealthy`, or `unobserved` classification. Native failures,
missing records, and disabled health checks cannot pass. Define behavior assertions
on `/reward` and health assertions on `/health/ignored_checks` (must equal `[]`),
plus relevant task-specific properties. The adapter enforces the native health
boundary independently of those assertions.

## Select fresh native output paths

Pass `--gym-source <descriptor.json>` to `task_evidence.py run`. For a fixture:

```json
{
  "mode": "fixture",
  "report": "/absolute/fresh-run/verifier-report.json",
  "inputs": {
    "dataset": "/absolute/fresh-run/executed-dataset.jsonl",
    "config": "/absolute/fresh-run/executed-config.yaml",
    "runtime": "/absolute/fresh-run/executed-runtime.json",
    "reset": "/absolute/fresh-run/executed-reset.json"
  }
}
```

The native command must produce the verifier report and capture the actual
execution inputs at these fresh paths. The recorder runs argv without a shell and
does not capture stdout, so record `gym env test --json` through a wrapper kept
in the task tree that writes the report and captures the inputs. The wrapper must
not manufacture results, suppress errors, or substitute a declared configuration
for the configuration actually used. Fixture cases use a stable
report-and-case identity; they do not need rollout IDs or ATIF. Retain every
native case even when the descriptor selects only one for a control receipt.

For an agent rollout, use `mode: rollout`, the same `inputs` mapping, and:

```json
{
  "rollouts": "/absolute/fresh-run/rollouts.jsonl",
  "row": 1,
  "verdicts": "/absolute/fresh-run/rollout_verdicts.jsonl",
  "summary": "/absolute/fresh-run/quality_summary.json",
  "conversion": "/absolute/fresh-run/converted/conversion.json"
}
```

Here `row` selects a physical rollout line; it is separate from `dataset_row`.
The adapter matches the selected dataset fields, rollout ID, task/repeat indexes,
health record, summary counts, and execution-input digests. Keep all native rows;
do not pre-filter rollouts without their matching health evidence. The command
writes native files; the recorder itself writes `--result` as normalized evidence.
Use a new directory outside the draft for each recorded run: Gym writes
`rollout_verdicts.jsonl` and `quality_summary.json` beside the rollout output.

## Baseline and closure are different claims

Recording a first-eval agent baseline is optional; when recorded, use
`--purpose baseline`, and one agent attempt may omit `--trace` and
`conversion`. The receipt retains rewards, legitimate behavioral failures, and
`health_status: unobserved` when model health was not observable. Successful native
execution with unobserved model health must not be described as healthy.
Baseline receipts cannot establish measured tool-gap closure.

For task-create closure, use `--purpose closure` and run the existing
`gym-to-atif` converter inside the recorded command with the same native JSONL
and selected row. Supply its fresh ATIF via `--trace` and its conversion receipt
via the descriptor. The adapter checks selected line, raw-row hash, and ATIF hash.
For preserved original ATIF, also supply `source_atif` pointing to the retained
native ATIF artifact; its hash must match the converter's source digest and the
unchanged output bytes. Preserve the native attachment provenance as well.
Closure requires healthy native agent rollouts, all declared controls, and two
distinct agent runs through `task_pipeline.py verify`.

All native source files are hashed and reread during acceptance. Changed config,
wrong dataset rows, missing or duplicated health identities, stale conversion
receipts, and changed traces fail closed. The recorder still trusts the native
invocation and input capture; it reports `deployment_identity: not_attested`.
It does not independently establish the source deployed to a remote service or
prove that an author-declared reset procedure was followed.
