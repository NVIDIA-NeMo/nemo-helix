<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Package and deploy custom agent code

Use this reference when the agent contains a local MCP server or another Python
package that must run with the deployed agent.

The commands below assume the installed `nemo` CLI is on `PATH`.

## Project layout

Keep the complete deployable package under the agent directory:

```text
agents/<agent-name>/
  agent.yaml
  pyproject.toml
  uv.lock
  src/<package_name>/server.py
  skills/<skill-name>/SKILL.md
  tests/
  evals/cases.yaml
```

Omit directories the implementation does not need. Every `skills.paths` value
must be relative to `agent.yaml`, remain inside this project and point to a
directory containing `SKILL.md`.

## Inspect the build context

Before packaging:

1. Inspect `.gitignore` and `.dockerignore`.
2. Search the context for `.env` files, private keys, tokens, credential files,
   local databases, traces and customer data.
3. Remove or exclude anything that is not required at runtime.
4. Confirm `pyproject.toml` defines every MCP console script referenced by
   `agent.yaml`.
5. Confirm `uv.lock` is current and local tests pass with `uv run`.

Stop if the context contains an unresolved secret or sensitive dataset.

## Validate and build

Use project mode so the Python package and console scripts are installed into
the image:

```bash
IMAGE_TAG="$AGENT_NAME:local"
nemo agents package \
  --agent "agents/$AGENT_NAME/agent.yaml" \
  --pyproject "agents/$AGENT_NAME/pyproject.toml" \
  --tag "$IMAGE_TAG"
```

Do not use `--skip-validation`. For a source checkout, build the current
Platform wheel and point packaging at it:

```bash
uv build --package nemo-helix --wheel --out-dir dist
NEMO_AGENTS_WHEEL=LATEST nemo agents package \
  --agent "agents/$AGENT_NAME/agent.yaml" \
  --pyproject "agents/$AGENT_NAME/pyproject.toml" \
  --tag "$IMAGE_TAG"
```

`NEMO_AGENTS_WHEEL=LATEST` is the supported source checkout path. Do not treat
the checkout's `0.0.0` project metadata as the released Platform version and do
not silently weaken validation.

## Deploy the packaged image

After the user approves registration and deployment, use the workspace confirmed
earlier:

```bash
WORKSPACE="<confirmed-workspace>"

nemo agents register \
  --name "$AGENT_NAME" \
  --agent-config "agents/$AGENT_NAME/agent.yaml" \
  --workspace "$WORKSPACE"
```

If an AgentEnvironment was selected, deploy with its workspace-qualified
reference:

```bash
AGENT_ENVIRONMENT="$WORKSPACE/<confirmed-environment>"

nemo agents deploy \
  --agent "$AGENT_NAME" \
  --name "$AGENT_NAME-deployment" \
  --mode docker \
  --image "$IMAGE_TAG" \
  --workspace "$WORKSPACE" \
  --environment "$AGENT_ENVIRONMENT" \
  --timeout 300
```

If no AgentEnvironment was selected, omit both the assignment and option:

```bash
nemo agents deploy \
  --agent "$AGENT_NAME" \
  --name "$AGENT_NAME-deployment" \
  --mode docker \
  --image "$IMAGE_TAG" \
  --workspace "$WORKSPACE" \
  --timeout 300
```

Docker is the supported local container path. Kubernetes requires a published
image reachable by the configured executor and a separately verified runtime
contract.

## Deploy through OpenShell

Confirm the operator has configured `agents.deployments.openshell_executor`
with a reachable gateway and platform egress.

Package with the OpenShell image profile and publish to a registry the gateway
can pull from:

```bash
nemo agents package \
  --agent "agents/$AGENT_NAME/agent.yaml" \
  --pyproject "agents/$AGENT_NAME/pyproject.toml" \
  --sandbox-runtime openshell \
  --tag "$IMAGE_TAG"
docker push "$IMAGE_TAG"
```

Use a registry-qualified `IMAGE_TAG`; omit `--pyproject` for a config-only agent.
Register as above, then deploy with `--mode openshell` in place of `--mode docker`.
Keep any selected `--environment` and the confirmed `--workspace`. Verify the
deployment reaches `running`, invocation returns usable output, and the expected
network policy is enforced. Use the deploy-sandbox skill for that workflow.
Do not remap `k8s_executor` to OpenShell or assume that a local image is available
on remote gateway nodes.
