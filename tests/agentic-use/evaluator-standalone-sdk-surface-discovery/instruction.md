<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Discover the Standalone Evals SDK Surface

Find the local standalone Evals SDK workflow for running a tiny exact-match evaluation.

Use only `packages/nhx_evals_sdk` and SDK-level APIs in your answer. Do not propose the `nemo` CLI, plugin SDK APIs, or any `services/*` implementation path.

Your final answer must include:

- The package/path `packages/nhx_evals_sdk`.
- The import or API symbols `Evaluator`, `ExactMatchMetric`, and either `run_sync` or `run`.
- A minimal Python snippet, not a shell command or prose outline, showing how an agent would run a two-row exact-match evaluation locally with the standalone SDK from the repo root.
- The two-row dataset:
  - `{"question": "2+2?", "expected": "4", "prediction": "4"}`
  - `{"question": "Capital of France?", "expected": "Paris", "prediction": "Lyon"}`
- An `ExactMatchMetric` configured with `reference="{{item.expected}}"` and `candidate="{{item.prediction}}"`.
- Either a synchronous SDK call such as `Evaluator().run_sync(...)` or an async SDK flow using `await Evaluator().run(...)`.
