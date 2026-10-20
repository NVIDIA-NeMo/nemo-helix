# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox environment task Jobs run trials in: nhx-sandbox's environment plus an agent channel.

A no-network task (``allow_internet = false``) normally gets a deny-all sandbox. An agent that runs
inside the sandbox (codex, hermes, ...) then can't reach its model. ``agent_channel_hosts`` keeps
those hosts (the inference gateway) reachable for such tasks; the internet stays closed.

It also sizes each sandbox as submit checked it: ``agent_resource_overrides`` in the agent sandbox only,
then task.toml, then ``sandbox_resource_defaults``.
"""

from __future__ import annotations

import logging
from typing import Any, override

from harbor.models.task.config import NetworkMode
from nemo_evals.harbor.remote.resources import effective_resources
from nhx_sandbox.egress import EgressAllowlist, EgressPolicy, build_egress_policy
from nhx_sandbox.harbor_opensandbox_environment import NemoOpenSandboxEnvironment, sandbox_role

logger = logging.getLogger(__name__)


class AgentChannelOpenSandboxEnvironment(NemoOpenSandboxEnvironment):
    """``NemoOpenSandboxEnvironment`` whose no-network sandboxes may still reach ``agent_channel_hosts``."""

    def __init__(
        self,
        *args: Any,
        agent_channel_hosts: list[str] | None = None,
        agent_resource_overrides: dict[str, int] | None = None,
        sandbox_resource_defaults: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        # BaseEnvironment.__init__ may validate against the policy, so set the channel first.
        self._agent_channel_hosts = tuple(agent_channel_hosts or ())
        if agent_resource_overrides or sandbox_resource_defaults:
            # Harbor's factory passes these as keywords. A copy: the task's own config must keep task.toml's
            # values, since an inherited separate verifier's config is a copy of it, made after this one.
            role = sandbox_role(kwargs["session_id"], kwargs["trial_paths"].trial_dir.name)
            overrides = (agent_resource_overrides or {}) if role == "agent" else {}
            kwargs["task_env_config"], defaulted = effective_resources(
                kwargs["task_env_config"], overrides=overrides, defaults=sandbox_resource_defaults or {}
            )
            if defaulted:
                config = kwargs["task_env_config"]
                logger.warning(
                    "The %s sandbox of %s relies on default resources: %s. Set them in task.toml or the run's "
                    "execution.sandbox.",
                    role,
                    kwargs["session_id"],
                    ", ".join(f"{field}={getattr(config, field)}" for field in defaulted),
                )
        super().__init__(*args, **kwargs)

    @override
    def egress_policy(self) -> EgressPolicy:
        """No-network tasks get the agent channel only (empty channel: deny all); other modes are unchanged."""
        if self._network_policy.network_mode != NetworkMode.NO_NETWORK or not self._agent_channel_hosts:
            return super().egress_policy()
        return build_egress_policy(
            EgressAllowlist(targets=self._agent_channel_hosts, resolver_addresses=self._resolver_addresses)
        )
