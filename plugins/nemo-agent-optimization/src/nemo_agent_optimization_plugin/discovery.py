# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find the installed agent optimization strategies.

A strategy is an ordinary ``nemo.jobs`` entry that declares one extra class
variable::

    class OptimizeJob(NemoJob):
        nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(
            name="nat",
            description="Numeric hyperparameter search over a Fabric agent workflow.",
        )

That is the whole contract.  There is no base class to inherit and no
strategy-specific entry-point group to register under, so a strategy job stays
a plain :class:`~nemo_helix_plugin.job.NemoJob` that its own plugin owns
end to end — this package only needs to be able to *find* it.

The declared value is the same :class:`OptimizationStrategy` the listing route
returns, so what a plugin says about its strategy is what callers read: the
name and its description are authored together rather than the description
being borrowed from the job's own, which describes the job to CLI users.

Keep this module free of imports from sibling modules that jobs import: it is
reached through :func:`~nemo_helix_plugin.discovery.discover_jobs`, which
imports every installed job module.  ``schemas.strategies`` is safe — it imports
nothing but pydantic, which is why the declaration type lives there.
"""

from __future__ import annotations

import logging

from nemo_agent_optimization_plugin.schemas.strategies import OptimizationStrategy
from nemo_helix_plugin.discovery import discover_jobs
from nemo_helix_plugin.job import NemoJob

logger = logging.getLogger(__name__)

#: Class variable a job sets to advertise itself as an optimization strategy.
STRATEGY_ATTR = "nemo_agent_optimization_strategy"


def declared_strategy(job_cls: type[NemoJob]) -> OptimizationStrategy | None:
    """The strategy *job_cls* advertises, or ``None`` when it advertises none.

    A job that sets :data:`STRATEGY_ATTR` to anything but a named
    :class:`OptimizationStrategy` is treated as not a strategy at all: the
    listing and the dispatcher must agree on what counts, and an unnamed
    strategy is one ``--strategy`` could never select.
    """
    strategy = getattr(job_cls, STRATEGY_ATTR, None)
    if not isinstance(strategy, OptimizationStrategy) or not strategy.name:
        return None
    return strategy


def discover_strategy_jobs() -> dict[str, type[NemoJob]]:
    """Every installed optimization strategy job, keyed by strategy name.

    Entry-point keys are irrelevant here — a strategy is identified by the
    :attr:`~OptimizationStrategy.name` it declares in its :data:`STRATEGY_ATTR`
    class variable, so plugins keep naming their jobs however their own routing
    needs.
    """
    found: dict[str, type[NemoJob]] = {}
    for key, job_cls in discover_jobs().items():
        strategy = declared_strategy(job_cls)
        if strategy is None:
            continue
        if strategy.name in found and found[strategy.name] is not job_cls:
            logger.warning(
                "Optimization strategy %r is claimed by both %s and %s (via %r); keeping the first",
                strategy.name,
                found[strategy.name].__qualname__,
                job_cls.__qualname__,
                key,
            )
            continue
        found[strategy.name] = job_cls
    return found


def discover_strategies() -> list[OptimizationStrategy]:
    """Every installed strategy as its own plugin declares it, sorted by name.

    Derived from :func:`discover_strategy_jobs` rather than from a second scan,
    so a strategy dropped there for being a duplicate is not listed as
    selectable here.
    """
    declared = (declared_strategy(job_cls) for job_cls in discover_strategy_jobs().values())
    return sorted((strategy for strategy in declared if strategy is not None), key=lambda strategy: strategy.name)
