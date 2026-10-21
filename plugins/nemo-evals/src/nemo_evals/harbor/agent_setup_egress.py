# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor agent's ``setup()`` inside the sandbox's agent setup egress window.

Harbor builds the agent from :data:`SETUP_EGRESS_AGENT` with the real agent (a built-in name or an
import path) in the ``agent_setup_egress_inner`` kwarg. What it gets back is an instance of a subclass
of the real agent whose only change is ``setup()``, so everything else (name, version, run, context
population, error patterns) is the real agent's.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.acp_registry import is_acp_registry_shorthand, registry_spec_from_agent_name
from harbor.environments.base import BaseEnvironment
from harbor.models.trial.config import AgentConfig

SETUP_EGRESS_AGENT = "nemo_evals.harbor.agent_setup_egress:AgentSetupEgress"
INNER_AGENT_KWARG = "agent_setup_egress_inner"


@cache
def _inner_agent_class(inner: str) -> type[BaseAgent]:
    return AgentFactory.get_agent_class_from_config(AgentConfig(name=inner))


@cache
def _with_setup_egress(agent_class: type[BaseAgent]) -> type[BaseAgent]:
    class SetupEgressAgent(agent_class):  # ty: ignore[unsupported-base]  # a class known only at run time
        async def setup(self, environment: BaseEnvironment) -> None:
            window = getattr(environment, "agent_setup_egress", None)
            if window is None:
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
        if is_acp_registry_shorthand(inner):
            kwargs["registry_spec"] = registry_spec_from_agent_name(inner)
        return _with_setup_egress(_inner_agent_class(inner))(*args, **kwargs)

    @classmethod
    def preflight(cls, kwargs: dict[str, Any] | None = None, env: Mapping[str, str] | None = None) -> None:
        """The real agent's preflight, without the kwarg that names it."""
        options = dict(kwargs or {})
        inner = options.pop(INNER_AGENT_KWARG)
        _inner_agent_class(inner).preflight(kwargs=options, env=env)
