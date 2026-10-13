# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire types for the installed-strategies listing.

Answering from the *client's* environment would be the wrong answer whenever the
client and the platform have different plugins installed — and no answer at all for
Studio.  These models back the server-side listing, so a caller asks the platform
that will actually run the job which ``--strategy`` values it accepts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Route path, relative to the service's ``/v2`` router prefix.
STRATEGIES_ROUTE = "/strategies"

#: The same route as an absolute platform path, for clients that build URLs by
#: hand.  Derived from :data:`STRATEGIES_ROUTE` rather than written twice, so the
#: CLI cannot drift from what the service actually mounts.
STRATEGIES_PATH = f"/apis/agent-optimization/v2{STRATEGIES_ROUTE}"


class OptimizationStrategy(BaseModel):
    """One installed strategy, as ``--strategy`` accepts it.

    Also the declaration a strategy job makes: it is the type of the
    ``nemo_agent_optimization_strategy`` class variable, so the name users pass
    and the sentence explaining it are written once, by the plugin that owns the
    strategy, and reach this listing unchanged.
    """

    name: str = Field(description="Value to pass as 'strategy' when submitting a run.")
    description: str = Field(
        default="",
        description="What this strategy optimizes, in the words of the plugin that ships it.",
    )


class OptimizationStrategyList(BaseModel):
    """Every strategy installed on this platform.

    Unpaginated on purpose: the list is whatever plugins the deployment installed,
    so it is bounded by the install and does not grow with usage.
    """

    data: list[OptimizationStrategy] = Field(
        default_factory=list,
        description="Installed strategies, sorted by name.",
    )
