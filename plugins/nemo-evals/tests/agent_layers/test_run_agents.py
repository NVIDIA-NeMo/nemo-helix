# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent layer a run's agent installs into, from its resolved target."""

from typing import Any

import pytest
from nemo_evals.agent_layers.models import AgentLayerRequest
from nemo_evals.agent_layers.run_agents import agent_layer_request, setup_network_is_public
from nemo_evals.agent_layers.service import agent_layer_key
from nemo_evals.jobs.agent_spec import HarborBuiltinAgentSource, HarborImportedAgentSource, HarborRunnerTarget
from pydantic import ValidationError

#: A fake package index URL with a username and password.
CREDENTIALED_INDEX = "https://user:token@pypi.example/simple"  # trufflehog:ignore


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


@pytest.mark.parametrize(
    ("env_vars", "kwargs"),
    [
        ({"PIP_INDEX_URL": CREDENTIALED_INDEX}, {}),
        ({"HTTPS_PROXY": "http://user@proxy.example:3128"}, {}),
        ({}, {"version": "0.162.1", "fabric_package": f"nemo-fabric --index-url {CREDENTIALED_INDEX}"}),
    ],
    ids=["index-url", "proxy-user", "kwarg"],
)
def test_an_install_with_url_credentials_gets_no_layer(env_vars: dict[str, str], kwargs: dict[str, str]) -> None:
    target = HarborRunnerTarget(source=HarborBuiltinAgentSource(name="codex"), env_vars=env_vars, agent_kwargs=kwargs)

    assert agent_layer_request(target) is None


def test_a_layer_request_refuses_url_credentials() -> None:
    with pytest.raises(ValidationError, match="username or password"):
        AgentLayerRequest(name="codex", env={"PIP_INDEX_URL": CREDENTIALED_INDEX})


def test_a_plain_url_is_layered() -> None:
    target = HarborRunnerTarget(
        source=HarborBuiltinAgentSource(name="codex"), env_vars={"PIP_INDEX_URL": "https://pypi.example/simple"}
    )

    assert agent_layer_request(target) is not None


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


@pytest.mark.parametrize(
    ("environment", "public"),
    [
        ({}, True),
        ({"network_mode": "public"}, True),
        ({"network_mode": "no-network"}, False),
        ({"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]}, False),
        ({"network_mode": "not-a-mode"}, False),
    ],
    ids=["default", "public", "no-network", "allowlist", "unparseable"],
)
def test_only_a_task_whose_setup_network_is_public_is_layered(environment: dict[str, Any], public: bool) -> None:
    """A layer installs with more network than a restricted task's setup gets (AALGO-1225), so only public ones."""
    assert setup_network_is_public({"environment": environment}) is public
