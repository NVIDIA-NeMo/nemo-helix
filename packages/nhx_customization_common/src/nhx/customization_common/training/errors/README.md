<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Training failure classification

Automodel, NeMo-RL, and Unsloth training tasks classify a failure with the rules in `rules.yaml` and report it
to the Jobs service:

- `error_details`: a user-facing `message`, the error `type`, the original error as `detail`, and the `cause`,
  which is the id of the rule that matched, or `null` when none did. The Jobs service keeps these details when
  the controller later reports the same failure.
- `status_details.resumable`: whether a rerun of the same job could succeed. When it is `false`,
  `status_details.non_resumable_reason` says why, and a rerun of the job returns that reason as `rerun_warning`.
  A task killed before it could report sets neither, so its rerun has no warning.

## How a failure is classified

- Automodel and NeMo-RL train in a subprocess, so the classifier reads its output (`parser.py`). It takes every
  distinct exception line (`TypeName: message`), plus log lines that name a failure without an exception, such as
  Ray's memory monitor, a raylet that fate-shares with a dead agent, or NCCL. NeMo-RL prints the Ray session's
  agent logs before it returns a failure, and appends one line naming a port conflict, a grpcio crash, or the
  agent being killed when the logs show one.
- Unsloth trains in-process, so the classifier reads the exception and the exceptions it was raised from.

Each candidate is matched against the rules, in order; the first matching rule wins. The failure is classified by
the latest candidate whose rule is a cause. If none is, it is classified by the latest symptom, and failing that by
the latest candidate, as an `InternalError` that can be rerun. Symptoms are failures that usually follow another one,
such as a peer's NCCL error after a rank ran out of memory, so the cause wins even though it was printed first.

## Adding a rule

1. Add the rule to `rules.yaml` with a new, stable `id`. Never rename or reuse an id: it is reported to users and
   may be counted.
2. Set `rerunnable: false` only when a rerun of the same job with the same configuration would fail the same way.
   A rerun continues from the newest checkpoint, so a transient failure partway through training costs little.
3. Set `symptom: true` when the failure usually follows another one. That includes wrappers that report another
   error, such as `DatasetGenerationError`, and notices that a worker died, such as `WorkerCrashedError`.
4. Put the rule before any broader rule that would also match.
5. Add lines the rule must match to `EXAMPLES` in
   `packages/nhx_customization_common/tests/training/test_error_rules.py`. The test fails for a rule without
   examples.
