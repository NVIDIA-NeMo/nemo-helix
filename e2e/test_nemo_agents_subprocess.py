# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for subprocess-mode agent deployments.

This module starts NeMo Platform through the subprocess E2E harness, deploys a
real agent as a child process through the agents plugin, and invokes it through
the agents gateway. The end-to-end chain is::

    sdk.agents.invoke (gateway proxy, subprocess endpoint resolution)
      -> NAT or Fabric agent child process
      -> Inference Gateway /openai (base_url injected at deploy time)
      -> mock provider short-circuit (no real upstream / no API key)
      -> response back through the gateway

Unlike the Docker and Kubernetes modules, this scenario needs neither a
prebuilt image nor an external cluster. The ``subprocess_only`` marker keeps it
in the standard Python E2E job and prevents it from running against an external
Platform, where this module's local harness configuration would be ignored.
"""

import pytest
from nemo_agents_plugin.entities import NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT
from nemo_platform import NeMoPlatform

from e2e.agents_deploy_helpers import run_agent_deploy_and_invoke, wait_for_agent_spans

pytestmark = [
    pytest.mark.subprocess_only,
    pytest.mark.e2e_config(
        "e2e/configs/local-subprocess.yaml",
        harness={"backend": "subprocess"},
    ),
]


def test_nat_agent_deploys_and_invokes_through_gateway(sdk: NeMoPlatform, workspace: str) -> None:
    """Deploy a NAT agent as a subprocess and invoke it through the gateway."""
    run_agent_deploy_and_invoke(
        sdk,
        workspace=workspace,
        deployment_mode="subprocess",
        config_format=NAT_WORKFLOW_CONFIG_FORMAT,
    )


def test_fabric_agent_deploys_and_invokes_through_gateway(sdk: NeMoPlatform, workspace: str) -> None:
    """Deploy a Fabric-backed agent as a subprocess and invoke it through the gateway.

    The agent config names no export destination, so reaching Intake proves the
    backend wired one: a subprocess deployment auto-wires its trajectory the
    same way execute jobs and container deployments do.

    Asserting while the deployment is still up is sound because a sessionless
    invoke runs on an ephemeral Fabric runtime that is started and stopped
    within the request, and Relay exports when that runtime stops -- not when
    the deployment process exits.
    """

    def assert_trajectory_reached_intake(agent_name: str) -> None:
        spans = wait_for_agent_spans(sdk, workspace=workspace, agent_name=agent_name)
        assert spans, "the agent ran but no trajectory reached Intake"

    run_agent_deploy_and_invoke(
        sdk,
        workspace=workspace,
        deployment_mode="subprocess",
        config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        after_invoke=assert_trajectory_reached_intake,
    )


def test_fabric_agent_streams_through_gateway(sdk: NeMoPlatform, workspace: str) -> None:
    """A default Fabric deployment supports SSE chat completions."""
    run_agent_deploy_and_invoke(
        sdk,
        workspace=workspace,
        deployment_mode="subprocess",
        config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        invocation_modes=("streaming",),
    )


def test_fabric_agent_invokes_with_persisted_session(sdk: NeMoPlatform, workspace: str) -> None:
    """A persisted session can start its streaming-enabled Fabric runtime."""
    run_agent_deploy_and_invoke(
        sdk,
        workspace=workspace,
        deployment_mode="subprocess",
        config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        invocation_modes=("session",),
    )
