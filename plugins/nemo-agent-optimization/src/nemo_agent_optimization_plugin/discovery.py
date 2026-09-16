# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find the installed agent optimization strategies.

Strategies are ordinary ``nemo.jobs`` entries; what makes one a strategy is
subclassing :class:`AgentOptimizeJob`, not its name.  That keeps naming free
for plugins and makes membership type-checked rather than string-matched.
"""

from __future__ import annotations

from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_platform_plugin.discovery import discover_jobs


class AgentOptimizeDiscoveryError(RuntimeError):
    """Raised when an agent-optimize job is registered incorrectly."""


def discover_agent_optimize_jobs() -> dict[str, type[AgentOptimizeJob]]:
    """Every registered agent-optimize job, keyed by strategy name."""
    found: dict[str, type[AgentOptimizeJob]] = {}
    for cls in discover_jobs().values():
        if not (isinstance(cls, type) and issubclass(cls, AgentOptimizeJob)):
            continue
        if cls is AgentOptimizeJob:
            continue
        strategy = getattr(cls, "strategy", None)
        if not strategy:
            raise AgentOptimizeDiscoveryError(
                f"{cls.__qualname__} subclasses AgentOptimizeJob but declares no 'strategy'."
            )
        found[strategy] = cls
    return found
