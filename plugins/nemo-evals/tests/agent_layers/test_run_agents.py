# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent layer a run's agent installs into, from its resolved target."""

import pytest
from nemo_evals.agent_layers.models import AgentLayerRequest
from nemo_evals.agent_layers.run_agents import agent_layer_request
from nemo_evals.agent_layers.service import agent_layer_key
from nemo_evals.jobs.agent_spec import HarborBuiltinAgentSource, HarborImportedAgentSource, HarborRunnerTarget
from pydantic import ValidationError


def test_a_harbor_installed_agent_is_layered_with_its_kwargs() -> None:
    target = HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex"), agent_kwargs={"version": "0.162.1"})

    assert agent_layer_request(target) == AgentLayerRequest(name="codex", kwargs={"version": "0.162.1"})


def test_only_the_install_env_of_the_runs_env_vars_reaches_the_layer() -> None:
    target = HarborRunnerTarget(
        source=HarborBuiltinAgentSource(name="codex"),
        env_vars={"PIP_INDEX_URL": "https://pypi.example/simple", "CODEX_MODE": "fast"},
    )

    request = agent_layer_request(target)

    assert request is not None and request.env == {"PIP_INDEX_URL": "https://pypi.example/simple"}
    assert agent_layer_key(request) != agent_layer_key(request.model_copy(update={"env": {}}))


def test_a_layer_request_refuses_other_env() -> None:
    with pytest.raises(ValidationError, match="CODEX_MODE"):
        AgentLayerRequest(name="codex", env={"CODEX_MODE": "fast"})


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
