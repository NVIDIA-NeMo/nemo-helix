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

## Caller authorization contract

Deploy the Studio service, agents gateway/Fabric server, and assistant image from
this same revision. The image's `sitecustomize.py` installs the Fabric 0.6 bridge;
updating only the prompt or only the MCP package is not sufficient.

- The gateway replaces any caller-supplied tool-workspace header with the resolved
  deployment workspace. The Fabric server issues an ephemeral capability per HTTP
  invocation. Only this capability crosses Fabric's private adapter protocol.
- The bundled stdio MCP peer redeems the capability at a fixed loopback broker.
  Redemption is refused from remote peers or gateway-forwarded requests, and the
  capability is revoked when the ASGI invocation finishes, including cancellation.
  Caller credentials never enter prompts, tool arguments, graph state, or Fabric
  invocation artifacts. The serving process and its local stdio peer are trusted.
- Every tool call uses a fresh, caller-authenticated client. Final HTTP origins,
  workspace paths and query overrides are checked; redirects are not followed.
  No CLI config, cached process identity, or ambient workload credential is used.
  Authenticated platform URLs must use HTTPS (loopback HTTP is supported for local
  development). Explicit no-auth local requests remain credential-free.
  Callback capabilities also require HTTPS/loopback. The shipped developer
  configuration opts into `NHX_STUDIO_LOCAL_HTTP=1`: only credential-free tools
  may send callbacks over HTTP to the exact `host.docker.internal` hostname.
  This trusts the local Docker-host network, is not a production/multitenant
  boundary, and never permits sending caller bearer tokens or cookies there.
- Studio independently binds callbacks to the active conversation's owner and
  workspace with a private, revocable callback capability. Only the authenticated
  session owner can resolve pending approvals/inputs. Approval does not grant API
  permissions. Direct gateway reads work; writes still require Studio approval.

The broker and stdio peer must share the serving process's network namespace and
serving worker. The shipped single-worker container meets this contract. Keep
its HTTP listener behind the authenticated agents gateway; do not expose it as a
public replacement for the gateway. Broker locality uses the ASGI peer address;
the shipped Uvicorn entrypoint disables proxy-header rewriting. Custom entrypoints
must preserve this socket-peer boundary, not allow untrusted `X-Forwarded-For`
values to claim a loopback peer. Other execution
topologies fail closed; do not enable an ambient-credential fallback.
Missing context, an old image, or a direct unauthenticated MCP invocation must not
perform platform operations.

Additional offline regressions (no model or listening sockets):

```bash
PYTHONPATH=agents/nemo-studio-assistant/src \
  uv run --frozen pytest agents/nemo-studio-assistant/tests/test_caller_authorization.py \
  plugins/nemo-agents/tests/unit/test_tool_authority.py \
  plugins/nemo-agents/tests/unit/test_gateway.py \
  plugins/nemo-agents/tests/unit/test_fabric_server.py \
  services/studio/tests/unit/test_assistant_authorization.py
```

## Register, package, and deploy

These commands change platform state and should be run independently so each
result can be verified:

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
