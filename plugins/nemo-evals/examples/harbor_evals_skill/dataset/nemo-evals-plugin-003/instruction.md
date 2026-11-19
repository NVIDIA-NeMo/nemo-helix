<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Create and publish a Harbor benchmark using the NeMo Helix instance at `http://127.0.0.1:8080`, workspace `default`.

1. Create a runnable Harbor task in `/workspace/pi-digit` with native ID `pi-digit`. Its `instruction.md` must ask an agent to compute the 100th digit after the decimal point in π, counting the first fractional digit as position 1. Require one digit in `/workspace/pi_digit.txt` and do not give the digit away in the instruction. Give the task a no-network environment and a deterministic verifier for the answer.
2. Publish the Harbor task as a single-member Taskset. Read it back from the running Helix instance by revision to verify it was saved. Write `/workspace/taskset.json` with `taskset_ref` and `tasks`, using fully qualified digest-pinned references and exactly one member in `tasks`.
3. Write `/workspace/pi_eval.py` so an operator can run the saved Taskset in closed-book mode on a Kubernetes Helix deployment with Builder and OpenSandbox. Accept `--taskset`, `--model-ref`, `--context`, and `--sandbox-profile`. The model reference selects an existing closed-book gateway route. Wait for the run, report its run ID and result, and exit nonzero when the run fails or the score is unavailable. Accept an inference secret reference when authentication is required.

This trial ends after the publication readback and script are ready. The operator runs the script on Kubernetes. State the Taskset pin and script path in your final response.
