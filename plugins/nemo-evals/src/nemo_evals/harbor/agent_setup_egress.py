# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor agent's ``setup()`` inside the sandbox's agent setup egress window.

Harbor builds the agent from :data:`SETUP_EGRESS_AGENT` with the real agent (a built-in name or an
import path) in the ``agent_setup_egress_inner`` kwarg and the setup hosts in ``agent_setup_egress_hosts``.
What it gets back is an instance of a subclass of the real agent whose only change is ``setup()``, so
everything else (name, version, run, context population, error patterns) is the real agent's.

An environment with its own ``agent_setup_egress()`` window (our OpenSandbox environment) opens it.
Any other Harbor environment that can switch to an allowlist after start gets the setup hosts added to
the task's policy through Harbor's ``set_network_policy()`` for the duration of ``setup()``. Elsewhere,
setup runs under the task's own policy.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from functools import cache
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.acp_registry import is_acp_registry_shorthand, registry_spec_from_agent_name
from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.config import AgentConfig

SETUP_EGRESS_AGENT = "nemo_evals.harbor.agent_setup_egress:AgentSetupEgress"
INNER_AGENT_KWARG = "agent_setup_egress_inner"
SETUP_HOSTS_KWARG = "agent_setup_egress_hosts"


@cache
def _inner_agent_class(inner: str) -> type[BaseAgent]:
    return AgentFactory.get_agent_class_from_config(AgentConfig(name=inner))


@asynccontextmanager
async def _network_policy_window(environment: BaseEnvironment, hosts: tuple[str, ...]) -> AsyncGenerator[None, None]:
    """Add ``hosts`` to the environment's network policy for the block, then restore the policy.

    A public baseline is already open, and an environment that can't switch to an allowlist after start
    keeps the task's policy. Harbor's ``set_network_policy()`` rejects hosts the provider can't express.
    """
    if not hosts:
        yield
        return
    capabilities = environment.capabilities
    baseline = environment.network_policy
    if baseline.network_mode == NetworkMode.PUBLIC or not (
        capabilities.dynamic_network_policy and capabilities.network_allowlist
    ):
        yield
        return
    allowed = list(dict.fromkeys([*baseline.allowed_hosts, *hosts]))
    await environment.set_network_policy(NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=allowed))
    try:
        yield
    finally:
        await environment.set_network_policy(baseline)


@cache
def _with_setup_egress(agent_class: type[BaseAgent], hosts: tuple[str, ...]) -> type[BaseAgent]:
    class SetupEgressAgent(agent_class):  # ty: ignore[unsupported-base]  # a class known only at run time
        async def setup(self, environment: BaseEnvironment) -> None:
            window = getattr(environment, "agent_setup_egress", None)
            if window is None:
                async with _network_policy_window(environment, hosts):
                    await super().setup(environment)
                return
            async with window():
                await super().setup(environment)

    SetupEgressAgent.__name__ = agent_class.__name__
    SetupEgressAgent.__qualname__ = agent_class.__qualname__
    SetupEgressAgent.__module__ = agent_class.__module__
    return SetupEgressAgent


class AgentSetupEgress:
    """Stands in for the real agent class in Harbor's ``AgentConfig.import_path``; never instantiated itself."""

    def __new__(cls, *args: Any, **kwargs: Any) -> BaseAgent:
        inner = kwargs.pop(INNER_AGENT_KWARG)
        hosts = tuple(kwargs.pop(SETUP_HOSTS_KWARG, ()))
        if is_acp_registry_shorthand(inner):
            kwargs["registry_spec"] = registry_spec_from_agent_name(inner)
        return _with_setup_egress(_inner_agent_class(inner), hosts)(*args, **kwargs)

    @classmethod
    def preflight(cls, kwargs: dict[str, Any] | None = None, env: Mapping[str, str] | None = None) -> None:
        """The real agent's preflight, without the wrapper's kwargs."""
        options = dict(kwargs or {})
        inner = options.pop(INNER_AGENT_KWARG)
        options.pop(SETUP_HOSTS_KWARG, None)
        _inner_agent_class(inner).preflight(kwargs=options, env=env)
