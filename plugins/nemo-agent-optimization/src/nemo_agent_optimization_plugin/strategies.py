# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery contract for ``nemo agents optimize`` strategy plugins."""

from __future__ import annotations

import importlib.metadata
from functools import cache
from typing import Any, Protocol, runtime_checkable

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext

OPTIMIZATION_STRATEGY_GROUP = "nemo.optimization-strategy"

#: Optional key in a strategy's result dict naming the single file that *is* the optimization's
#: output (e.g. the optimized agent config), so ``--output foo.yaml`` can publish it directly
#: instead of copying the whole results tree.  ``OptimizeJob`` strips the key before returning.
#:
#: The value **must be an absolute path**: the job resolves it after ``_bundle_workdir`` has
#: restored the original working directory, so a relative path would resolve against the wrong
#: root.  Strategies build theirs from ``ctx.storage.persistent``, which is already absolute.
PRIMARY_ARTIFACT_KEY = "_primary_artifact"


class OptimizationStrategyDiscoveryError(RuntimeError):
    """Raised when an optimization strategy plugin cannot be loaded."""


@runtime_checkable
class OptimizationStrategy(Protocol):
    """Plugin contract for a named ``nemo agents optimize`` strategy."""

    name: str

    def validate_config(self, config: dict[str, Any], *, agent: str | None) -> None:
        """Validate strategy configuration before a run."""

    def run(
        self,
        *,
        agent_config: dict[str, Any] | None,
        source_agent_config: dict[str, Any] | None,
        config: dict[str, Any],
        ctx: JobContext,
        workspace: str,
        sdk: NeMoPlatform | None = None,
    ) -> dict[str, Any]:
        """Execute the strategy against a resolved agent and return a result dict.

        ``agent_config`` is ``None`` when the run named no agent.  ``OptimizeSpec`` permits that
        for every strategy — some need no agent at all, and ``nat`` can carry its agent package
        inline in ``config`` — so a strategy that does need a resolved agent rejects the omission
        from its own ``validate_config``, which sees the original ``agent`` value.
        """


@cache
def discover_optimization_strategies() -> dict[str, OptimizationStrategy]:
    """Load installed strategy plugins by entry-point name."""
    strategies: dict[str, OptimizationStrategy] = {}
    for entry in importlib.metadata.entry_points(group=OPTIMIZATION_STRATEGY_GROUP):
        try:
            loaded = entry.load()
            strategy = loaded() if isinstance(loaded, type) else loaded
        except Exception as exc:
            raise OptimizationStrategyDiscoveryError(f"Failed to load optimization strategy {entry.name!r}") from exc
        if not isinstance(strategy, OptimizationStrategy):
            raise OptimizationStrategyDiscoveryError(
                f"Optimization strategy {entry.name!r} must implement OptimizationStrategy"
            )
        if strategy.name != entry.name:
            raise OptimizationStrategyDiscoveryError(
                f"Optimization strategy entry point {entry.name!r} loaded a strategy named {strategy.name!r}"
            )
        strategies[entry.name] = strategy
    return strategies
