<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Per-check result grammar

**Experimental:** This guide is part of the
[trace-to-environment workflow](../SKILL.md).

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

## Review what each check proves

Before proof, map each literal check ID to its requested outcome and the complete
assertion path, including invoked helper scripts. For runtime outcomes, exercise
the supported interface and observe its result. Compilation, source text, file
existence, successful exit, or a submission's claim that an event happened do not
alone establish runtime behavior. File content or configuration state may be
checked directly when that state is itself requested. Group prerequisites and
preservation assertions with the outcome they constrain.

Try a short, ordinary incomplete implementation within the requested domain.
If it can pass a check while omitting that check's outcome, record the ID,
requirement, assertion path, and concrete counterexample in private construction
notes, then fix the verifier before proof. Distinguish source reasoning from
executed evidence. A passing unrelated check is not false credit; additional
coverage ideas without a concrete counterexample are advisory. Do not invent
requirements, require a particular implementation, or treat every copy operation
as cheating. The aggregate reward remains binary.

Use [task fidelity review](task-fidelity.md#review-the-public-contract-and-every-assertion)
for boundary/default/option-interaction cases and valid-alternative diagnostics.
An untouched failure caused only by unrelated setup errors is not evidence that
the check distinguishes the requested behavior. Keep authoritative result rows
fresh and verifier-owned; missing or unparseable child output is not a PASS.

## Per-check proof evidence

Harbor copies the verifier's `/logs/verifier/` tree back into each retained
job as `task__*/verifier/`. `record-validation` parses every proof job's
`task__*/verifier/results` and requires:

- every NOP and Oracle and negative-control job carries the rows, with an
  identical check-ID set across all jobs;
- NOP jobs report FAIL for every scored check (a PASS is
  `nop_contamination` evidence of a trivially satisfied or leaked check);
- an Oracle job with reward 1 reports PASS for every check;
- a reward-0 negative control reports at least one FAIL;
- supplied copy probes have the same check IDs as Oracle and a binary reward
  consistent with their rows. Aggregate-only proof cannot include copy probes.

Jobs without rows fail with `missing_check_evidence`. Historical proofs
recorded before this contract can be re-recorded once with
`record-validation --allow-aggregate-only`; the report and the exported
product then carry `check_evidence: "aggregate_only"` instead of
`"per_check"`. Mixed evidence (some jobs with rows, some without) is always
rejected: rerun the missing arms so the proof set is uniform.

## Executed copy probes

Static literal detection does not execute a candidate and can miss assertions in
helper files. When visible instructions or fixtures could satisfy a behavioral
check by transcription, run a custom copy control through Harbor in a fresh
container. Use only solver-visible material and solver-writable destinations;
never read hidden tests or the reference solution. Keep instruction-literal and
fixture-copy attempts separate when both apply. Declare what requested behavior
each attempt omits; if copying is the requested outcome, it is not a suitable
negative control. Record non-applicability in the private construction notes.

Retain the custom control source under `private/` and check its installed Harbor
agent interface as for the ordinary negative control. Before running it:

```bash
python <skill_dir>/scripts/trace_environment.py record-run-inputs \
  --task-dir <task-dir> --arm copy --job-dir private/jobs/copy-1 \
  --control-agent copy_agent:CopyOnly \
  --control-source private/copy_agent.py \
  --control-rationale "<visible material copied and requested behavior omitted>"
PYTHONPATH=<task-dir>/private harbor run -p <task-dir>/task \
  -a copy_agent:CopyOnly \
  --jobs-dir <task-dir>/private/jobs --job-name copy-1
```

Add `--copy-job-dir private/jobs/copy-1` to the normal `record-validation`
command, repeating the option for other attempts. These runs supplement the
required NOP, Oracle and negative-control jobs; they cannot replace them.
The helper binds the retained source and agent identity, but does not prove the
control's semantics or that Python imported that exact file. Review that wiring.

A copy run with reward 1 or an exception makes technical validation fail. A
reward-0 run may still pass individual checks: review each reported
`passed_check_ids` against its requirement and full assertion path using the
review above. Partial passes alone do not prove false credit. Repair demonstrated
defects and rerun proof; record legitimate passes in private construction notes.
Missing rows and inconsistent check IDs or rewards are contract errors, never
successful probes. Absent copy runs are reported as `not_run`, not as resistance
to copying. `passed: true` establishes evidence consistency, not completion of
this source-level review.
