---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: deploy-sandbox
description: >-
  Package and deploy an existing Fabric agent through the OpenShell deployment
  mode, then verify invocation and network policy. Use for sandboxed agent
  deployments on an already configured gateway; not for building a new agent,
  installing the platform, or submitting a run-to-completion job.
compatibility: >-
  Requires NeMo Helix with --mode openshell, the deployments OpenShell extra,
  a reachable gateway, a configured openshell executor, and image access from
  the gateway's compute nodes.
maturity: experimental
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read, Write, Edit
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Deploy a Fabric agent through OpenShell

Use the operator-configured `openshell` deployment mode. The platform talks to
its gateway; Docker versus Kubernetes is a gateway setting, not an agent mode.
Do not point `k8s_executor` at OpenShell or use `--mode k8s` for this deployment.

## Capabilities and prerequisites

This workflow needs access to the platform, gateway, and image registry, plus a
container builder if the image is not already built. It does not host platform
services inside the assistant sandbox. If a required capability is unavailable,
report it and stop the dependent step.

1. Confirm the CLI context, platform workspace, existing Fabric config, deployment
   name, and image reference. Reuse choices and authorization already established.
   For a new agent, follow the repo's `nemo-build-agent` workflow first; do not
   introduce a NAT wrapper.
2. Verify platform health using the setup/status workflow. Confirm
   `agents.deployments.openshell_executor` names an `openshell` backend and the
   controller has not skipped it for missing dependencies. Do not narrow the
   platform environment with `uv sync --package` to install the extra.
3. Confirm gateway endpoint, gateway workspace, authentication, and image access
   with the operator. `openshell sandbox list` verifies the CLI connection, not
   the controller's credentials. The executor supports TLS/mTLS material but no
   OIDC bearer-token configuration. Do not weaken shared gateway authentication.
4. Confirm a working platform model and a sandbox-reachable `platform_egress`
   endpoint. On Kubernetes use the API Service FQDN. The source checkout includes
   `k8s/helm/examples/openshell/values.yaml` and the operator guide at
   `docs/set-up/helm/openshell.mdx`.

## Package and register

Use an existing, validated Fabric `agent.yaml`. Set these from the confirmed context:

```bash
WORKSPACE="<platform-workspace>"
AGENT_NAME="<agent-name>"
AGENT_CONFIG="<path-to-agent.yaml>"
IMAGE_TAG="<registry>/<team>/<image>:<version>"
DEPLOYMENT_NAME="$AGENT_NAME-sandbox"
```

Skip build and push if a compatible image is already published. Otherwise:

```bash
nemo agents package --agent "$AGENT_CONFIG" \
  --sandbox-runtime openshell --tag "$IMAGE_TAG"
docker push "$IMAGE_TAG"
```

Include `--pyproject <path>` for local Python code. Inspect the build context for
credentials before publishing. For a source build, use the current `nemo-helix`
wheel with `NEMO_AGENTS_WHEEL=LATEST`; do not bypass validation. Verify build and
push success before deployment. For a local gateway, an explicitly selected
image-loading path can replace push. Kubernetes private images need gateway-side
`sandboxImagePullSecrets`; platform image-pull credentials do not configure it.

Register if the agent does not already exist:

```bash
nemo agents create --name "$AGENT_NAME" \
  --agent-config "$AGENT_CONFIG" --workspace "$WORKSPACE"
```

## Deploy and invoke

```bash
nemo agents deploy --agent "$AGENT_NAME" --name "$DEPLOYMENT_NAME" \
  --mode openshell --image "$IMAGE_TAG" --workspace "$WORKSPACE" --timeout 600
```

Preserve a selected AgentEnvironment with its `--environment` reference. Wait for
`running`. If verification fails or times out, inspect the deployment/controller
error, report it, and ask the user before continuing with deployment changes.

```bash
nemo agents invoke --agent-deployment "$DEPLOYMENT_NAME" \
  --workspace "$WORKSPACE" --input "Reply with a short greeting."
```

Verify usable model output. Readiness, inference, and platform-to-gateway routing
are separate checks. Do not claim end-to-end success from readiness alone.

## Verify policy

Use `openshell sandbox list` on the selected gateway and gateway workspace to
identify the sandbox by deployment-name and deployment-workspace labels. Set
`SANDBOX_NAME` to that sandbox and `PLATFORM_API_URL` to the operator's platform
egress endpoint. Do not guess a sandbox from its name prefix.

```bash
openshell sandbox exec --name "$SANDBOX_NAME" -- \
  curl --fail --show-error --max-time 10 "$PLATFORM_API_URL/health/ready"
openshell sandbox exec --name "$SANDBOX_NAME" -- \
  curl --fail --show-error --max-time 10 https://example.com
```

The default policy permits the platform request and denies the external one.
Confirm a policy denial in output or gateway logs; DNS failures and timeouts
alone do not prove enforcement. For a custom policy, choose a destination outside
its allowlist. Invoke the agent again. Report failed checks and get confirmation
before changing configuration.

This path permits direct platform egress: do not describe it as zero egress or
claim it uses `inference.local`. Gateway-brokered inference is a separate explicit
configuration. Default Landlock mode is `best_effort`; only claim filesystem
confinement after verifying support, or use an operator-approved
`hard_requirement` setting. Do not expand policy merely to make a test pass.

## Cleanup and report

Keep the deployment if the user requested a running agent. For an authorized
disposable test:

```bash
nemo agents undeploy "$DEPLOYMENT_NAME" --workspace "$WORKSPACE" --yes
nemo agents delete "$AGENT_NAME" --workspace "$WORKSPACE" --yes
```

Delete the registration only if this workflow created it and it is no longer
needed. Verify the sandbox disappears. Report deployment, invocation, policy
checks, and retained or removed resources. Do not delete the gateway or platform.
