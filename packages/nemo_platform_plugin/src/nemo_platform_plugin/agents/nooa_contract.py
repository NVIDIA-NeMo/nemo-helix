# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The calling convention for NOOA agents run by NeMo Platform.

NOOA has no standardized entrypoint, so Fabric ships no generic NOOA adapter.
This module is the *platform's* convention over NOOA: export one async callable,
name it in ``agent.yaml``, and the ``nvidia.nemo-platform.nooa`` adapter will
import and invoke it.

    # my_pkg/agent.py
    from nemo_platform_plugin.agents.nooa_contract import NooaInvocation
    from nemo_platform_plugin.nooa_model_client import platform_model_clients

    async def run(invocation: NooaInvocation) -> AgentRunResult:
        async with platform_model_clients(sdk, refs):
            ...

    # agent.yaml
    harnesses:
      main:
        kind: nvidia.nemo-platform.nooa
        settings:
          entrypoint: my_pkg.agent:run

The contract lives in this package, not in ``nemo_agents_plugin``, so writing an
agent against it does not pull in the agents plugin and its bundled harness
binaries.

Stability: adding a field to :class:`NooaInvocation` is backward-compatible and
may happen; removing one or changing its meaning is a breaking change to the
convention and will not be done silently. The underlying Fabric adapter contract
is ``fabric.adapter/v1alpha2`` and is pre-1.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nemo_fabric_adapter_contract import models as contract


@dataclass(frozen=True, kw_only=True)
class NooaInvocation:
    """Everything a platform-run NOOA callable is handed for one invocation.

    A single parameter object rather than positional arguments so the convention
    can grow: a new field breaks no existing callable, whereas a fourth
    positional parameter would break every one of them. ``kw_only`` for the same
    reason one step out — callables construct this in their own tests, and field
    order should not be part of the contract.
    """

    request: contract.AgentRunRequest
    """The invocation's input, as Fabric received it."""

    context: contract.RuntimeContext
    """Fabric's per-invocation runtime context (correlation ids, cancellation)."""

    settings: Mapping[str, Any]
    """The agent config's ``harness.settings`` bag, verbatim.

    Carries ``entrypoint`` itself plus whatever else the agent declared, so a
    callable can take configuration without the adapter knowing its shape.
    """

    models: Mapping[str, contract.AgentModelConfig]
    """The agent config's ``models:`` mapping.

    Present because the adapter deliberately does not resolve platform models on
    a callable's behalf — the default/fast pair is a two-slot shape and not every
    agent fits it. A callable that wants platform models reads its refs from
    here (or from ``settings``) and opens them with
    :func:`nemo_platform_plugin.nooa_model_client.platform_model_clients`.
    """
