# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent layer a run's agent installs into: what submit looks up and the CLI requests before it submits."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from harbor.models.task.config import NetworkMode, TaskConfig
from nemo_evals.agent_layers.models import AgentLayerRequest
from nemo_evals.jobs.agent_spec import HarborRunnerTarget
from nemo_evals.jobs.registered_agent_resolution import upgrade_registered_agent_config
from nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer import agent_install, has_url_credentials, install_env

logger = logging.getLogger(__name__)


def agent_layer_request(target: object) -> AgentLayerRequest | None:
    """The layer request for a resolved run target's agent, or None when its agent can't be layered.

    Only a Harbor target's agent installs in setup. A registered agent must already be resolved: its Fabric package
    comes from the agent's harness. An install that needs a URL with credentials, such as an authenticated package
    index, gets no layer: a layer's request is stored and uploaded for its build, so its trials install in setup.
    """
    if not isinstance(target, HarborRunnerTarget):
        return None
    target = upgrade_registered_agent_config(target)
    name, import_path, kwargs = target.agent_name, target.agent_import_path, dict(target.agent_kwargs)
    env = install_env(target.env_vars)
    if has_url_credentials(kwargs) or has_url_credentials(env):
        logger.info("The run's agent install uses a URL with credentials; its trials install the agent in setup")
        return None
    if agent_install(agent_name=name, agent_import_path=import_path, agent_dir=None, agent_kwargs=kwargs) is None:
        return None
    if import_path is not None:
        return AgentLayerRequest(import_path=import_path, kwargs=kwargs, env=env)
    return AgentLayerRequest(name=name, kwargs=kwargs, env=env)


def setup_network_is_public(config: Mapping[str, Any]) -> bool:
    """Whether a task's agent setup runs with an open network: its ``[environment]`` baseline, which setup runs under.

    Only such a task gets an agent layer for now. A layer installs in a Builder build, whose network isn't limited to
    what a restricted task's setup may reach, so a trial could start from an install it couldn't have made itself
    (AALGO-1225). A config that doesn't parse counts as restricted.
    """
    try:
        task = TaskConfig.model_validate(dict(config))
    except ValueError:
        return False
    return task.environment.resolve_baseline().network_mode == NetworkMode.PUBLIC
