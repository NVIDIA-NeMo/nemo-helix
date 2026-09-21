# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E test for docker-mode agent deployments.

Unlike ``test_nemo_agents.py`` (backend-agnostic API/SDK surface, plus
subprocess-mode deployment coverage), this module deploys an agent as a real
**Docker container** through the nemo-deployments plugin and invokes it through
the agents gateway. The backend-agnostic deploy/invoke/assert core is shared
with the Kubernetes variant (``test_nemo_agents_k8s.py``) via
``e2e.agents_deploy_helpers``; this module owns only the docker-specific wiring.

What it proves — the container-mode chain end to end for both supported agent
config formats::

    sdk.agents.invoke (gateway proxy, container-mode endpoint resolution)
      -> docker agent container (NAT or Fabric/DeepAgents server)
      -> Inference Gateway /openai (base_url injected at deploy time)
      -> mock provider short-circuit (no real upstream / no API key)
      -> response back through the gateway

How it runs, and where:

- The deployed agent runs from the platform's own ``nmp-api`` image, which
  already ships both the NAT runtime and Fabric/DeepAgents runtime. The
  deployments docker executor overrides the image entrypoint with the server
  for the selected config format, so the image's own entrypoint is irrelevant.
  No agent-specific image is built.
- The image is supplied prebuilt via ``NMP_E2E_IMAGE_REGISTRY`` /
  ``NMP_E2E_IMAGE_TAG`` (the existing e2e convention). Its dedicated CI job
  (``python-e2e-image-test``) builds/pulls ``nmp-api`` and sets these; the
  ``needs_nmp_api_image`` marker skips the test everywhere they are unset (the
  plain subprocess e2e job, the kind cluster job, and local runs without them).
- The agent is registered with a deterministic single-LLM ``chat_completion``
  config served by the e2e mock inference provider, so no ``NVIDIA_API_KEY`` or
  model egress is needed; we assert the exact mocked completion round-trips.
- The platform runs as a normal local process (subprocess harness). The
  ``container_base_url_host`` harness option makes the harness bind the platform
  on all interfaces (``--host 0.0.0.0``) and rewrite ``platform.base_url`` to the
  docker bridge address; the runner seeds ``NMP_BASE_URL`` from that host paired
  with the actual bind port, so the Inference Gateway URL injected into the agent
  container is reachable from *inside* the container while the platform's own
  in-process clients still reach it. This requires the Linux docker bridge,
  which is reachable from both the container and the host process; Docker
  Desktop's host alias is not host-resolvable, so the module skips on non-Linux.
"""

import os
import platform

import pytest
from nemo_agents_plugin.entities import NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT
from nemo_platform import NeMoPlatform

from e2e.agents_deploy_helpers import run_container_agent_deploy_and_invoke

# The docker bridge gateway. On Linux (incl. GitHub Actions ubuntu runners) this
# address is reachable both from inside a container AND by the platform process
# itself (it is a local host interface). That dual reachability is required:
# platform.base_url is used both as the Inference Gateway URL injected into the
# agent container *and* by the platform's own in-process service clients.
_DOCKER_BRIDGE_HOST = "172.17.0.1"

# Platform image name to deploy the agent from. The nmp-api image already ships
# the NAT and Fabric/DeepAgents runtimes (see module docstring), so it doubles as
# the agent runtime image. Registry and tag come from NMP_E2E_IMAGE_REGISTRY /
# NMP_E2E_IMAGE_TAG (the existing e2e image convention).
_AGENT_IMAGE_NAME = "nmp-api"

# Runs the platform as a local process wired with a docker deployments executor
# (see the config), and deploys the agent as a real docker container.
#
# Markers:
# - ``needs_nmp_api_image``: skips unless NMP_E2E_IMAGE_REGISTRY + NMP_E2E_IMAGE_TAG
#   are set (its dedicated CI job builds/pulls nmp-api and sets them). This also
#   keeps the test out of the plain subprocess e2e job and the kind cluster job.
# - ``subprocess_only``: this test drives its own subprocess-harness platform
#   configured with a docker deployments executor. It must NOT run against an
#   external cluster (``NMP_BASE_URL`` set, e.g. the Kind CPU e2e job), where the
#   deployed Helm platform has no docker executor and the module's own
#   ``e2e_config``/harness are ignored.
pytestmark = [
    pytest.mark.needs_nmp_api_image,
    pytest.mark.subprocess_only,
    pytest.mark.skipif(
        platform.system() != "Linux",
        reason="Docker-mode agent deployment e2e requires a Linux docker bridge reachable by both the "
        "platform process and the agent container (Docker Desktop's host alias is not host-resolvable).",
    ),
    pytest.mark.e2e_config(
        "e2e/configs/local-docker-agents.yaml",
        harness={"backend": "subprocess", "container_base_url_host": _DOCKER_BRIDGE_HOST},
    ),
]


@pytest.fixture(scope="session")
def agent_deployment_image() -> str:
    """Return the prebuilt platform image ref to deploy the agent from.

    Composed from the e2e image convention (``NMP_E2E_IMAGE_REGISTRY`` /
    ``NMP_E2E_IMAGE_TAG``) as ``{registry}/nmp-api:{tag}``. The nmp-api image
    already ships the NAT and Fabric/DeepAgents runtimes, so no agent-specific
    image is built here. The ``needs_nmp_api_image`` marker guarantees both env
    vars are set before this test runs; assert defensively.
    """
    registry = os.environ.get("NMP_E2E_IMAGE_REGISTRY")
    tag = os.environ.get("NMP_E2E_IMAGE_TAG")
    assert registry and tag, "needs_nmp_api_image marker should have skipped when registry/tag are unset"
    return f"{registry.rstrip('/')}/{_AGENT_IMAGE_NAME}:{tag}"


def _remove_agent_container_if_present(deployment_name: str) -> None:
    """Best-effort removal of a leaked agent container after teardown."""
    try:
        from docker.errors import NotFound

        import docker
    except Exception:
        return
    try:
        client = docker.from_env()
    except Exception:
        return
    # The deployments docker backend names containers after the deployment; match
    # loosely so a naming-scheme change does not silently leak containers.
    for container in client.containers.list(all=True):
        if deployment_name in container.name:
            try:
                container.remove(force=True)
            except NotFound:
                pass
            except Exception:
                pass


def test_nat_docker_agent_deploys_and_invokes_through_gateway(
    sdk: NeMoPlatform, workspace: str, agent_deployment_image: str
) -> None:
    """Deploy a NAT agent as a docker container and invoke it through the gateway."""
    run_container_agent_deploy_and_invoke(
        sdk,
        workspace=workspace,
        deployment_mode="docker",
        image=agent_deployment_image,
        config_format=NAT_WORKFLOW_CONFIG_FORMAT,
        reap_backend_resources=_remove_agent_container_if_present,
    )


def test_fabric_docker_agent_deploys_and_invokes_through_gateway(
    sdk: NeMoPlatform, workspace: str, agent_deployment_image: str
) -> None:
    """Exercise non-streaming, streaming, and session calls against a Docker Fabric agent."""
    run_container_agent_deploy_and_invoke(
        sdk,
        workspace=workspace,
        deployment_mode="docker",
        image=agent_deployment_image,
        config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        reap_backend_resources=_remove_agent_container_if_present,
        invocation_modes=("non_streaming", "streaming", "session"),
    )
