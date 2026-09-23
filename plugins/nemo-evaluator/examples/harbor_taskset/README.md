<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor taskset: upload, evaluate, inspect

## Prerequisites

Use Python 3.12+ with the evaluator Harbor extra and a Jupyter kernel using that
same environment. Follow [SETUP.md](../../../../SETUP.md) to start Files,
Entities, Evaluator, and Secrets services plus a host-subprocess evaluator
worker with Docker available. This source-checkout example requires the worker
and notebook to use the same repository checkout.

Export `OPENAI_API_KEY` before starting Jupyter, and allow task containers to
reach `api.openai.com`. Set `NMP_BASE_URL`, `NMP_WORKSPACE`, and, if required,
`NMP_API_KEY` in the notebook environment. The defaults are
`http://localhost:8080` and `default`; no machine-specific configuration file
is needed. Non-loopback Platform URLs must use HTTPS because the notebook
uploads `OPENAI_API_KEY` to Platform Secrets.

## Run the Evaluation

Open [harbor_taskset_e2e.ipynb](harbor_taskset_e2e.ipynb) and run the cells in order.
The notebook publishes the bundled dataset, submits its pinned Taskset to the
Evaluator plugin, runs the tasks with Harbor's built-in Codex agent against the
OpenAI API, and reads saved trial and score records.

- `harbor_dataset/`: three native Harbor tasks demonstrating a greeting,
  arithmetic, and an intentional agent runtime error.
- `agent/`: a small deterministic reference agent and its Harbor wrapper.

Identical publication is rerunnable; changed entity content requires an explicit
replacement choice. Every submission starts a new job. Outputs are cleared in
the checked-in notebook, and resource cleanup is opt-in.
