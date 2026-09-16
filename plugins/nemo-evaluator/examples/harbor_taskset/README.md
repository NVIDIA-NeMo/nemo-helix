<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor taskset: upload, evaluate, inspect

Open [harbor_taskset_e2e.ipynb](harbor_taskset_e2e.ipynb) and run the cells in order.
The notebook publishes the bundled dataset, submits its pinned Taskset to the
Evaluator plugin, and reads saved trial and score records. No LLM is required.

- `harbor_dataset/`: three native Harbor tasks demonstrating a greeting, an
  unsupported arithmetic request, and an intentional agent runtime error.
- `agent/`: a small deterministic agent and its Harbor wrapper. Only its three
  runtime Python files are uploaded into task containers.

Use Python 3.12+, the evaluator Harbor extra, and a Jupyter kernel using that
same environment. Follow [SETUP.md](../../../../SETUP.md) to start the platform
and a host-subprocess evaluator worker with Docker available. The worker imports
the wrapper from the editable `nemo_evaluator` package, so no custom
`PYTHONPATH` is needed. This source-checkout example is intended for a worker
running from the same repository checkout.

Set `NMP_BASE_URL`, `NMP_WORKSPACE`, and, if required, `NMP_API_KEY` in the
notebook environment. Defaults are `http://localhost:8080` and `default`.
No machine-specific configuration file is needed.

Identical publication is rerunnable; changed entity content requires an explicit
replacement choice. Every submission starts a new job. Outputs are cleared in
the checked-in notebook, and resource cleanup is opt-in.
