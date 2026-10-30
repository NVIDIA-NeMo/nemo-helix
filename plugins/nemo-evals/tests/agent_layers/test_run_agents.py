# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent layer a run's agent installs into, from its resolved target."""

import pytest
from nemo_evals.agent_layers.models import AgentLayerRequest
from nemo_evals.agent_layers.run_agents import agent_layer_request
from nemo_evals.jobs.agent_spec import HarborBuiltinAgentSource, HarborImportedAgentSource, HarborRunnerTarget


def test_a_harbor_installed_agent_is_layered_with_its_kwargs() -> None:
    target = HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex"), agent_kwargs={"version": "0.162.1"})

    assert agent_layer_request(target) == AgentLayerRequest(name="codex", kwargs={"version": "0.162.1"})


@pytest.mark.parametrize(
    "target",
    [
        HarborRunnerTarget(source=HarborBuiltinAgentSource(name="oracle")),
        HarborRunnerTarget(source=HarborBuiltinAgentSource(name="nop")),
        HarborRunnerTarget(source=HarborImportedAgentSource(import_path="harbor_wrapper:WrappedAgent")),
        object(),
    ],
    ids=["oracle", "nop", "own-import-path", "not-harbor"],
)
def test_agents_that_dont_install_in_setup_get_no_layer(target: object) -> None:
    assert agent_layer_request(target) is None
