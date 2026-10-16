# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox environment task Jobs run trials in: scaled-evals' environment plus an agent channel.

A no-network task (``allow_internet = false``) normally gets a deny-all sandbox. An agent that runs
inside the sandbox (codex, hermes, ...) then can't reach its model. ``agent_channel_hosts`` keeps
those hosts (the inference gateway) reachable for such tasks; the internet stays closed. Imported
only by the dispatcher image, which carries the ``scaled-evals`` group.
"""

from __future__ import annotations

from typing import Any, override

from harbor.models.task.config import NetworkMode
from nhx_sandbox.egress import EgressAllowlist, EgressPolicy, build_egress_policy
from scaled_evals.harbor_opensandbox_environment import NemoOpenSandboxEnvironment


class AgentChannelOpenSandboxEnvironment(NemoOpenSandboxEnvironment):
    """``NemoOpenSandboxEnvironment`` whose no-network sandboxes may still reach ``agent_channel_hosts``."""

    def __init__(self, *args: Any, agent_channel_hosts: list[str] | None = None, **kwargs: Any) -> None:
        # BaseEnvironment.__init__ may validate against the policy, so set the channel first.
        self._agent_channel_hosts = tuple(agent_channel_hosts or ())
        super().__init__(*args, **kwargs)

    @override
    def egress_policy(self) -> EgressPolicy:
        """No-network tasks get the agent channel only (empty channel: deny all); other modes are unchanged."""
        if self._network_policy.network_mode != NetworkMode.NO_NETWORK or not self._agent_channel_hosts:
            return super().egress_policy()
        return build_egress_policy(
            EgressAllowlist(targets=self._agent_channel_hosts, resolver_addresses=self._resolver_addresses)
        )
