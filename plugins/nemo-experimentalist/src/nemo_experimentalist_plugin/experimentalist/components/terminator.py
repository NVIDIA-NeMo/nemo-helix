# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminator step — decides when the evolutionary optimization loop should stop.

The loop consults :meth:`Terminator.run` once per round; it stops when the round
budget is exhausted or the optimization has converged (a deterministic Pareto-front
check, with a qualitative LLM check as a tie-breaker). Publishing the winner is the
loop's job (``backend.publish_candidate``), not the Terminator's.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_experimentalist_plugin.entities import Candidate, MetricTarget
from nemo_experimentalist_plugin.experimentalist import roles
from nemo_experimentalist_plugin.experimentalist.components.models import (
    has_metric_dimensions,
    pareto_front,
    pareto_objectives,
)
from nemo_experimentalist_plugin.skills import skills_dir
from nemo_helix_plugin.nooa_model_client import get_fast_model
from nooa import Agent, CodeActStrategy, TextSkill, hidden, strategy
from nooa.agentdoc import doc
from nooa.config import CodeActConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TerminatorConfig(BaseModel):
    """Tuning parameters for early stopping.

    ``min_rounds_before_stopping`` is deliberately absent: it sits on the run config
    beside ``max_rounds``, and reaches this component through the constructor, the same
    way the objective and regression targets do. Two homes for one value is what made it
    silently ignorable.
    """


class TerminationDecision(BaseModel):
    """Structured result of a termination assessment.

    Attributes:
        stop: True when the optimization loop should stop.
        reason: Human-readable explanation; empty string when ``stop`` is False.
    """

    stop: bool
    reason: str = ""


class ConvergenceTerminator(Agent, roles.Terminator):
    """Decides when the evolutionary optimization loop should stop."""

    config_type = TerminatorConfig

    name = "convergence"

    def __init__(
        self,
        *,
        config: TerminatorConfig | None = None,
        min_rounds_before_stopping: int = 3,
        objective_metrics: list[MetricTarget] | None = None,
        regression_metrics: list[MetricTarget] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=kwargs.pop("llm", None) or get_fast_model(), **kwargs)
        self._config = config or TerminatorConfig()
        self._min_rounds_before_stopping = min_rounds_before_stopping
        self._objective_metrics = objective_metrics or [MetricTarget(name="reward", direction="maximize")]
        self._regression_metrics = regression_metrics or []
        self.terminator_skill = TextSkill(path=skills_dir() / "terminator")

    @hidden
    async def run(
        self,
        *,
        round_num: int,
        candidates: list[Candidate],
        prior_analysis: str | None = None,
    ) -> TerminationDecision:
        """Canonical termination decision, consulted once per round at the top of the loop.

        Early stopping only: the round budget belongs to the loop, so a terminator that
        never stops still cannot produce an unbounded run.

        Args:
            round_num: Rounds completed so far.
            candidates: The scored population; convergence is judged from their rewards.
            prior_analysis: Markdown analysis from the previous round, if any.

        Returns:
            A :class:`TerminationDecision`.
        """
        reached = self.assess_objective_reached(candidates)
        if reached.stop:
            return reached
        return await self.assess_convergence(candidates=candidates, prior_analysis=prior_analysis)

    @hidden
    def assess_objective_reached(self, candidates: list[Candidate]) -> TerminationDecision:
        """Stop when a live candidate already satisfies every targeted objective.

        Convergence asks whether progress has stalled; this asks whether any is left to
        make. It is not part of that judgement, so selecting no terminator does not
        disable it -- a target is a threshold the caller stated, not an opinion about
        stagnation. To keep going, set no target.

        Only what finalization would consider counts: a live candidate, or the
        generation-0 baseline even if survivor selection killed it — the same
        re-admission the selector makes when deciding whether anything beat doing
        nothing. Otherwise the run could end in favour of a winner that never reaches
        the target, or fail to end when shipping nothing already satisfies it.
        """
        targets = [target for target in self._objective_metrics if target.target is not None]
        if not targets:
            return TerminationDecision(stop=False)
        for candidate in candidates:
            if candidate.killed_generation is not None and candidate.generation != 0:
                continue
            metrics = candidate.rewards["validation"].metrics or {}
            if all(target.is_satisfied_by(metrics.get(target.name)) for target in targets):
                summary = ", ".join(f"{target.name}={metrics.get(target.name)}" for target in targets)
                return TerminationDecision(stop=True, reason=f"objective reached by {candidate.label} ({summary})")
        return TerminationDecision(stop=False)

    @hidden
    async def assess_convergence(
        self,
        *,
        candidates: list[Candidate],
        prior_analysis: str | None,
    ) -> TerminationDecision:
        """Decide whether to early-stop before running another round.

        Returns ``stop=False`` without consulting a model when there is no prior round
        to compare against. Otherwise consults :meth:`_has_converged`.

        Args:
            candidates: The scored population.
            prior_analysis: Markdown analysis from the previous round, if any.

        Returns:
            A :class:`TerminationDecision`.
        """
        if prior_analysis is None:
            return TerminationDecision(stop=False)
        converged = await self._has_converged(
            candidates=candidates,
            prior_analysis=prior_analysis,
            min_rounds_before_stopping=self._min_rounds_before_stopping,
            objective_metrics=self._objective_metrics,
            regression_metrics=self._regression_metrics,
        )
        if converged:
            return TerminationDecision(stop=True, reason="optimization converged (Pareto front stagnated)")
        return TerminationDecision(stop=False)

    @hidden
    async def _has_converged(
        self,
        *,
        candidates: list[Candidate],
        prior_analysis: str,
        min_rounds_before_stopping: int,
        objective_metrics: list[MetricTarget] | None = None,
        regression_metrics: list[MetricTarget],
    ) -> bool:
        """Return True if the optimization has converged and should stop early."""
        # Truthy check, not ``is not None``: an unscored candidate has an empty metrics
        # dict, and counting those would skew both the round cutoff and the front.
        scored = [c for c in candidates if c.rewards["validation"].metrics]
        rounds = sorted({n.generation for n in scored})
        if len(rounds) < min_rounds_before_stopping:
            return False
        cutoff_round = rounds[-min_rounds_before_stopping]
        old = [n for n in scored if n.generation <= cutoff_round]
        if not old:
            return False
        objectives = objective_metrics or []
        scored = [c for c in scored if has_metric_dimensions(c.rewards["validation"].metrics or {}, objectives)]
        if not scored:
            return False
        old = [c for c in old if c in scored]
        if not old:
            return False
        ranked = lambda c: pareto_objectives(c.rewards["validation"].metrics or {}, objectives)  # noqa: E731
        old_front_ids = {c.id for c in pareto_front(old, ranked)}
        full_front_ids = {c.id for c in pareto_front(scored, ranked)}
        if full_front_ids.issubset(old_front_ids):
            return True
        return await self.qualitative_stop_check(
            prior_analysis,
            objectives,
            regression_metrics,
        )

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def qualitative_stop_check(
        self,
        analysis: str,
        objective_metrics: list[MetricTarget],
        regression_metrics: list[MetricTarget],
    ) -> bool:  # ty: ignore[invalid-return-type]  # pyright: ignore[reportReturnType]
        """Decide whether the optimization has qualitatively plateaued; return True to stop.

        Judge the round ``analysis`` text against the terminator skill's stop
        heuristics (prefilled below). Return True only with concrete textual
        evidence of stagnation in ``analysis``; when in doubt, return False.

        ``objective_metrics`` are the evaluator metrics being improved;
        ``regression_metrics`` are those that must not worsen. Judge stagnation
        only in that context; do not invent a scalar score, weights, or a new
        selection rule.
        """
        print(doc(self.terminator_skill))
        ...
