<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configure An Agent Target

Use the local standalone Evals SDK to configure a mock agent endpoint as the evaluated target and run a one-row exact-match evaluation.

Use only `packages/nhx_evals_sdk` and SDK-level APIs in your answer. Do not propose the `nemo` CLI, plugin SDK APIs, or any `services/*` implementation path.

Your final answer must include:

- The package/path `packages/nhx_evals_sdk`.
- The import or API symbols `Evaluator`, `run_sync`, `Agent`, `AgentFormat.GENERIC`, and `ExactMatchMetric`, including `Agent` from `nhx_evals_sdk.values.agents`.
- A directly runnable Python snippet that starts a mock HTTP endpoint returning `{"answer": "4", "trajectory": {"steps": [{"tool_calls": [{"name": "calculator"}]}]}}`.
- A generic `Agent` target configured with `body`, `response_path="$.answer"`, and `trajectory_path="$.trajectory"`.
- A dataset row asking `What is 2+2?` with expected answer `4`.
- A call to `Evaluator().run_sync(...)` using the `Agent` target.
- Output that includes a printed JSON object with keys `answer`, `exact_match`, `trajectory_tool_calls`, `candidate_agent_runtime`, and `candidate_agent_model`.
- In that printed JSON object, the extracted answer must be `4`, the exact-match score must be `1.0`, the trajectory must include the `calculator` tool call, and the metadata must identify the candidate agent runtime and model.
