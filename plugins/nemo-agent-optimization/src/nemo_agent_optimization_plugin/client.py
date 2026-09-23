# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed client for this plugin's own API.

Lives here rather than under ``nemo_helix_plugin/<service>/`` with the core-service
clients: this plugin serves the route, so it owns the client for it. Only the client
*infrastructure* — the endpoint decorators, ``NemoClient``, ``send()`` — comes from the
framework package, which every plugin already depends on.

The path and the response model are the same objects the service mounts its route from,
so the two cannot drift.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_agent_optimization_plugin.schemas.strategies import STRATEGIES_PATH, OptimizationStrategyList
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.endpoint import get
from nemo_helix_plugin.client.method import method


@get(STRATEGIES_PATH)
@abstractmethod
def list_strategies() -> OptimizationStrategyList: ...


class AgentOptimizationClient(NemoClient):
    """Sync client for this plugin's API.

    Sync only: the one caller is the CLI. Add the async twin when something async needs it.
    """

    list_strategies = method(list_strategies)
