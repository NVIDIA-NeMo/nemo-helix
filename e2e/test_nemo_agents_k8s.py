# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E test for Kubernetes-mode agent deployments.

The Kubernetes counterpart to ``test_nemo_agents_docker.py``: it deploys an
agent as a real **Kubernetes Deployment + Service** through the nemo-deployments
plugin and invokes it through the agents gateway. The backend-agnostic
deploy/invoke/assert core is shared via ``e2e.agents_deploy_helpers``; this
module owns only the k8s-specific wiring.

What it proves — the container-mode chain end to end for both supported agent
config formats, on Kubernetes::

    sdk.agents.invoke (gateway proxy, container-mode endpoint resolution)
      -> k8s agent pod (NAT or Fabric/DeepAgents server), fronted by a ClusterIP Service
      -> Inference Gateway /openai (base_url injected at deploy time)
      -> mock provider short-circuit (no real upstream / no API key)
      -> response back through the gateway

How it runs, and where:

- This test only runs against an **external cluster** (``NMP_BASE_URL`` set) —
  i.e. the Kind CPU e2e CI job, where a Helm-deployed platform is configured
  with a nemo-deployments ``k8s`` executor (see ``e2e/k8s/values/kind.yaml``).
  The ``container_only`` marker skips it for the subprocess harness (local /
  plain e2e job), which is the inverse of the docker module's ``subprocess_only``.
- The agent runs from the platform's own ``nmp-api`` image, which already ships
  both the NAT runtime and Fabric/DeepAgents runtime (see the docker module
  docstring). The deployments k8s executor overrides the image entrypoint with
  the server for the selected config format. In CI the image is pre-pulled into
  the kind nodes and referenced by its commit-SHA tag, so the pod's default
  ``imagePullPolicy: IfNotPresent`` uses the node-local image (the k8s backend
  does not use image pull secrets).
- The image ref is composed from ``NMP_E2E_IMAGE_REGISTRY`` /
  ``NMP_E2E_IMAGE_TAG`` (the existing e2e image convention). The
  ``needs_nmp_api_image`` marker skips the test unless both are set; the Kind
  CPU e2e job exports them from the built image outputs.
- The agent is registered with a deterministic single-LLM config served by the
  e2e mock inference provider, so no ``NVIDIA_API_KEY`` or model egress is
  needed; we assert the exact mocked completion round-trips.
- Gateway reachability is in-cluster: the agent Deployment/Service land in the
  same namespace as the platform, and the Inference Gateway URL injected into
  the agent pod is the platform's own in-cluster ``NMP_BASE_URL``, reachable pod
  -> service. No docker-bridge / host-alias juggling is needed (that is a
  docker-in-a-local-process concern only), so this module has no Linux/OS skip.
"""

import os

import pytest
from nemo_agents_plugin.entities import NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT
from nemo_platform import NeMoPlatform

from e2e.agents_deploy_helpers import run_container_agent_deploy_and_invoke

# Platform image name to deploy the agent from (see module docstring). Registry
# and tag come from NMP_E2E_IMAGE_REGISTRY / NMP_E2E_IMAGE_TAG.
_AGENT_IMAGE_NAME = "nmp-api"

# Markers:
# - ``container_only``: runs only against an external cluster (``NMP_BASE_URL``
#   set) — the Kind CPU e2e job, whose Helm platform is configured with a k8s
#   deployments executor. Skipped on the subprocess harness (the inverse of the
#   docker module's ``subprocess_only``), where no k8s executor exists.
# - ``needs_nmp_api_image``: skips unless NMP_E2E_IMAGE_REGISTRY + NMP_E2E_IMAGE_TAG
#   are set, which is how the test learns the (node-pre-pulled) agent image ref.
pytestmark = [
    pytest.mark.container_only,
    pytest.mark.needs_nmp_api_image,
]


@pytest.fixture(scope="session")
def agent_deployment_image() -> str:
    """Return the prebuilt platform image ref to deploy the agent from.

    Composed from the e2e image convention (``NMP_E2E_IMAGE_REGISTRY`` /
    ``NMP_E2E_IMAGE_TAG``) as ``{registry}/nmp-api:{tag}``. In the Kind e2e job
    this exact ref is pre-pulled into the cluster nodes, so the pod resolves it
    node-locally under ``IfNotPresent``. The ``needs_nmp_api_image`` marker
    guarantees both env vars are set before this test runs; assert defensively.
    """
    registry = os.environ.get("NMP_E2E_IMAGE_REGISTRY")
    tag = os.environ.get("NMP_E2E_IMAGE_TAG")
    assert registry and tag, "needs_nmp_api_image marker should have skipped when registry/tag are unset"
    return f"{registry.rstrip('/')}/{_AGENT_IMAGE_NAME}:{tag}"


def test_nat_k8s_agent_deploys_and_invokes_through_gateway(
    sdk: NeMoPlatform, workspace: str, agent_deployment_image: str
) -> None:
    """Deploy a NAT agent as a k8s Deployment+Service and invoke it through the gateway."""
    run_container_agent_deploy_and_invoke(
        sdk,
        workspace=workspace,
        deployment_mode="k8s",
        image=agent_deployment_image,
        config_format=NAT_WORKFLOW_CONFIG_FORMAT,
        # Pod scheduling + (node-local) image resolution can take longer than the
        # docker path's local container start.
        running_timeout_seconds=420,
    )


def test_fabric_k8s_agent_deploys_and_invokes_through_gateway(
    sdk: NeMoPlatform, workspace: str, agent_deployment_image: str
) -> None:
    """Exercise non-streaming, streaming, and session calls against a Kubernetes Fabric agent."""
    run_container_agent_deploy_and_invoke(
        sdk,
        workspace=workspace,
        deployment_mode="k8s",
        image=agent_deployment_image,
        config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        # Pod scheduling + (node-local) image resolution can take longer than the
        # docker path's local container start.
        running_timeout_seconds=420,
        invocation_modes=("non_streaming", "streaming", "session"),
    )
