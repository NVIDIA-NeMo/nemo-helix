<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor taskset: upload, evaluate, inspect

## Prerequisites

Use Python 3.12+ with the evaluator Harbor extra and a Jupyter kernel using that
same environment. Follow [SETUP.md](../../../../SETUP.md) to start Files,
Entities, Evaluator, and Secrets services plus a host-subprocess evaluator
worker with Docker available. Built-in Codex requires no uploaded wrapper.

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

## Uploaded wrapper

[uploaded_agent.ipynb](uploaded_agent.ipynb) submits a native runner and automatically publishes
`agent/` with its helper and runtime assets.
The notebook also shows a separate upload to Filesets followed by job submission using the
returned source descriptor, so the same archive can be reused without the local agent directory.
The worker needs Harbor and the evaluator installed, but does not need this example checkout.
Enable `evaluator.harbor_agent_source_enabled` in the
service and `NEMO_EVALUATOR_HARBOR_AGENT_SOURCE_ENABLED=true` in the subprocess execution profile's
`env` configuration. The worker intentionally does not inherit every service environment setting.
This permits trusted host Python execution and is disabled by default. The wrapper notebook
needs no model API key; its deterministic runtime is useful for validating source transport.

## Acceptance tests

Run `RUN_HARBOR_AGENT_INTEGRATION=1 uv run pytest
plugins/nemo-evaluator/tests/integration/test_harbor_agent_submission.py -v -s`
from the repository root with Docker available. The tests start an isolated platform
on port 8094 and verify saved trials and rewards for uploaded source and built-in Codex.
Codex uses `OPENAI_API_KEY` by default. To use an existing compatible gateway, set
`HARBOR_CODEX_BASE_URL`, `HARBOR_CODEX_MODEL`, and `HARBOR_CODEX_API_KEY` explicitly.
The base URL must be reachable from the task container; on Docker Desktop use
`host.docker.internal` for a host service. An example gateway base URL is
`http://host.docker.internal:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1`.
Use a registered model name without a workspace prefix because Harbor's Codex adapter
strips prefixes. For a local gateway with authentication disabled, an explicit test
placeholder can satisfy Codex's API-key requirement; the gateway resolves provider
credentials itself. Neither test reads the host's Codex login files.
