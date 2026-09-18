<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Per-check result grammar

`tests/test.sh` must report one row per scored check to
`/logs/verifier/results` so proof evidence identifies *which* behavior passed,
not only an aggregate reward.

```sh
status=FAIL
if your_check_command; then status=PASS; fi
printf 'stable-check-id\t%s\n' "$status" >> /logs/verifier/results
```

Rules enforced by `validate-task` (static) and `record-validation` (retained
job evidence):

- Row shape is exactly `<check-id>\tPASS` or `<check-id>\tFAIL`. IDs are
  lowercase kebab-case, unique within the task, and stable across runs and
  edits; renaming an ID invalidates comparison with earlier evidence.
- First write may use `>`; later writes use `>>`. Write rows even when the
  script exits early: a check that never emits a row reads as missing
  evidence, not as a failure.
- The final reward stays binary: reward 1 exactly when every scored check is
  PASS. How the script derives process exit status from the rows is up to the
  task; Harbor turns that exit status into the reward.
- `solution/solve.sh` and agent-writable files never write
  `/logs/verifier/results`; only verifier code under `tests/` does.
- Never compare two solver-writable files as the oracle, and never assert a
  long expected literal that also appears verbatim in `instruction.md` or
  `environment/` — that is satisfiable by copying and `validate-task` reports
  it as `copyable_literal`. Derive expectations, or keep them verifier-private.
- A syntax-only parse (`sh -n`, `bash -n`, `node --check`, `php -l`) is never
  a scored check's sole condition; it proves nothing about behavior
  (`syntax_only_check`).
- When the instruction names a runnable command, test that exact command with
  the ordinary inherited task environment. Do not inject verifier-only `PATH`,
  import paths, `HOME`, or tool-specific variables to make it pass.

## Per-check proof evidence

Harbor copies the verifier's `/logs/verifier/` tree back into each retained
job as `task__*/verifier/`. `record-validation` parses every proof job's
`task__*/verifier/results` and requires:

- every NOP and Oracle and negative-control job carries the rows, with an
  identical check-ID set across all jobs;
- NOP jobs report FAIL for every scored check (a PASS is
  `nop_contamination` evidence of a trivially satisfied or leaked check);
- an Oracle job with reward 1 reports PASS for every check;
- a reward-0 negative control reports at least one FAIL.

Jobs without rows fail with `missing_check_evidence`. Historical proofs
recorded before this contract can be re-recorded once with
`record-validation --allow-aggregate-only`; the report and the exported
product then carry `check_evidence: "aggregate_only"` instead of
`"per_check"`. Mixed evidence (some jobs with rows, some without) is always
rejected: rerun the missing arms so the proof set is uniform.
