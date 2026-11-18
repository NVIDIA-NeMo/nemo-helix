<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Studio Assistant Fabric smoke test

The assistant is a `nemo-agents-spec-v1` Fabric agent using the preinstalled
Deep Agents adapter. Its platform and Studio tools are exposed by the packaged
`nemo-studio-assistant-mcp` stdio server.

## Static and unit validation

From the repository root:

```bash
PYTHONPATH=agents/nemo-studio-assistant/src \
  uv run --frozen pytest agents/nemo-studio-assistant/tests/test_nemo_studio_assistant.py -v
uv run ruff check agents/nemo-studio-assistant/src agents/nemo-studio-assistant/tests
uv run ruff format --check agents/nemo-studio-assistant/src agents/nemo-studio-assistant/tests
uv run --frozen ty check
```

The tests validate Fabric translation, the Deep Agents adapter, all packaged
skills, the MCP tool surface, SDK serialization, session validation, and
approval gating for mutations.

## One-command local deploy

With the platform running (`nemo services run`, see `SETUP.md`), Docker up, and
`NEMO_DEFAULT_MODEL` set to a model entity id from `nemo models list`:

```bash
agents/nemo-studio-assistant/scripts/deploy-local.sh
```

The script snapshots `docs/` for `search_docs`, replaces any previous
registration and deployment, runs the steps below, and finishes with a one-line
invocation. Run the steps by hand when one fails and you need its output.

## Register, package, and deploy

These commands change platform state and should be run independently so each
result can be verified:

```bash
agents/nemo-studio-assistant/scripts/snapshot-docs.sh
```

```bash
export NHX_BASE_URL=http://127.0.0.1:8080
nemo agents register --name nemo-studio-assistant \
  --agent-config agents/nemo-studio-assistant-ethos/agent.yaml
nemo agents package \
  --agent agents/nemo-studio-assistant/agent.yaml \
  --pyproject agents/nemo-studio-assistant/pyproject.toml \
  --tag nemo-studio-assistant:fabric-local
nemo agents deploy --agent nemo-studio-assistant \
  --name nemo-studio-assistant-fabric-deployment \
  --mode docker --image nemo-studio-assistant:fabric-local
nemo agents deployments wait nemo-studio-assistant-fabric-deployment
```

The generated image must use the Fabric server entrypoint, not `nat serve`.

## Invocation

```bash
nemo agents invoke --agent-deployment nemo-studio-assistant-fabric-deployment \
  --input "List all workspaces on the platform"
nemo agents invoke --agent-deployment nemo-studio-assistant-fabric-deployment \
  --input "List the available models and inference providers using the platform API."
```

For mutation approval, invoke through Studio and ask the assistant to create a
workspace named `nemo-studio-assistant-smoke-test`. Verify that Studio renders an
approval prompt before the SDK call, and that declining leaves platform state
unchanged.

## Studio streaming

Open `http://127.0.0.1:8080/studio`, select the assistant deployment, and send
`List all workspaces`. Verify incremental, nonduplicated text and a terminal
success or error. Then exercise a selector and an approved mutation to verify
the callback MCP tools.

## Guide behavior

In Studio, verify:

- On a Virtual Models page, "how do I add a guardrail here?" names the current
  page and links to the right destination.
- "How does skill optimization work?" shows a "Searched docs for ..." step and
  cites a doc title.
- "List my workspaces" shows a "Called list workspaces" step before the answer.
- "How do I bake a cheesecake?" returns the fixed off-topic refusal without an
  agent run. The topic gate reads `STUDIO_ASSISTANT_TOPIC_GATE_MODEL`, falling
  back to `NEMO_DEFAULT_MODEL`, from the environment of `nemo services run`;
  set it to `off` to disable the gate. It needs the guardrails service and fails
  open when the check is unavailable.

## Cleanup

After verification, remove the temporary workspace, then undeploy and delete
the agent so another smoke-test run starts cleanly:

```bash
nemo workspaces delete nemo-studio-assistant-smoke-test --yes
nemo agents undeploy nemo-studio-assistant-fabric-deployment --yes
nemo agents delete nemo-studio-assistant --yes
```

## Automated evaluation coverage

The former NAT `/generate/full` evaluation YAML remains removed. Each case in
`src/nemo_studio_assistant/nemo-studio-assistant-eval-data.json` now has
deterministic unit coverage for its required `nemo_api` tool path in
`test_nemo_studio_assistant.py`. Root pytest discovery includes these unit tests,
so `make test-unit-ci` runs them in CI without a live model endpoint.
