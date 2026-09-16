# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which candidates breed, and which one wins."""

from typing import Any

from nemo_experimentalist_plugin.entities import Candidate, MetricTarget
from nemo_experimentalist_plugin.experimentalist import roles
from nemo_experimentalist_plugin.experimentalist.components.models import (
    has_metric_dimensions,
    pareto_front,
    pareto_objectives,
    pareto_sort,
)
from nemo_helix_plugin.nooa_model_client import get_default_model, get_fast_model
from nooa import Agent, CodeActStrategy, strategy
from nooa.agents import TokenBudgetSummarizer
from nooa.config import CodeActConfig
from nooa.config.summarizer_config import TokenBudgetConfig
from pydantic import BaseModel, Field


class SelectorConfig(BaseModel):
    """Tuning parameters for survivor selection."""

    max_summary_tokens: int = Field(default=80_000, description="Max tokens the token-budget summarizer may use.")
    objectives: list[str] = Field(
        default_factory=lambda: ["validation"],
        description=(
            "Reward channels ranked on. Selection policy lives here rather than on the "
            "entity: 'insight rewards must never feed Pareto selection' is a selector "
            "concern, and a flag on the channel would be wrong for anyone ranking on train."
        ),
    )


# Selection is two questions, and only the first needs a model: Pareto ranking is
# arithmetic over reward channels, while choosing among incomparable candidates inside one
# front is a judgement about architectural diversity. Splitting them is what lets a
# numeric strategy reuse the ranking and skip the model entirely.
class ParetoDiversitySelector(Agent, roles.Selector):
    """Pareto-rank on the configured objectives, then pick diverse survivors with a model."""

    config_type = SelectorConfig

    name = "pareto-diversity"

    def __init__(
        self,
        config: SelectorConfig | None = None,
        objective_metrics: list[MetricTarget] | None = None,
        regression_metrics: list[MetricTarget] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the selector."""
        super().__init__(llm=kwargs.pop("llm", None) or get_default_model(), **kwargs)
        self._config = config or SelectorConfig()
        self._objective_metrics = objective_metrics or [MetricTarget(name="reward", direction="maximize")]
        self._regression_metrics = regression_metrics or []
        TokenBudgetSummarizer.install(
            self, llm=get_fast_model(), config=TokenBudgetConfig(max_tokens=self._config.max_summary_tokens)
        )

    def rank(self, candidates: list[Candidate]) -> list[Candidate]:
        """Sort by non-domination rank over the configured objectives."""
        return pareto_sort(candidates, self._objectives_of)

    async def survivors(self, candidates: list[Candidate], *, k: int) -> list[Candidate]:
        """Up to *k* candidates to carry into the next round as parents."""
        return await self.select_diverse_survivors(self.rank(candidates), k)

    def winner(self, candidates: list[Candidate]) -> Candidate | None:
        """The run's winner, or None when nothing carries a comparable measurement.

        Finalization asks a different question from survivor selection. `survivors` is
        told to keep a candidate created this round, and the generation-0 baseline never
        is one, so a baseline can be killed mid-loop. But "is any of this better than
        shipping nothing?" has to be able to answer *no*, so the baseline is re-admitted
        here regardless, and anchors the regression comparison even when it does not win.

        A candidate that worsens a `regression_metrics` target against that baseline is
        dropped before ranking, so a gain on the objectives cannot pay for a regression.
        Ties go to the oldest candidate: equal score means the diff bought nothing.
        """
        measured = [c for c in candidates if has_metric_dimensions(self._ranked_metrics(c), self._objective_metrics)]
        baseline = next((c for c in measured if c.generation == 0), None)
        eligible = [c for c in measured if c.killed_generation is None]
        if baseline is not None and all(c is not baseline for c in eligible):
            eligible.insert(0, baseline)
        if not eligible:
            return None
        reference = baseline if baseline is not None else min(eligible, key=self._creation_order)
        kept = [c for c in eligible if not self._regresses(c, reference)]
        front = pareto_front(kept or eligible, self._objectives_of)
        return min(front, key=self._creation_order) if front else None

    def _objectives_of(self, candidate: Candidate) -> dict[str, float]:
        """The candidate's ranked metrics projected onto the objectives.

        Ranking is uniformly "higher is better", so a minimized target is negated and
        anything outside the objectives is dropped — a protected metric must not sway
        the front, only disqualify through :meth:`_regresses`.
        """
        return pareto_objectives(self._ranked_metrics(candidate), self._objective_metrics)

    def _regresses(self, candidate: Candidate, baseline: Candidate) -> bool:
        """Whether *candidate* is worse than *baseline* on a metric that must not regress.

        A target missing from either side is skipped: absent evidence is not a regression.
        """
        if candidate is baseline:
            return False
        before, after = self._ranked_metrics(baseline), self._ranked_metrics(candidate)
        for target in self._regression_metrics:
            was, now = before.get(target.name), after.get(target.name)
            if was is None or now is None:
                continue
            if now < was if target.direction == "maximize" else now > was:
                return True
        return False

    @staticmethod
    def _creation_order(candidate: Candidate) -> tuple[int, int, str]:
        """Order by creation, independently of how the caller ordered the list.

        Labels are run-scoped and handed out in sequence (``agent-0``, ``agent-1``, ...),
        so the numeric suffix recovers creation order within a generation. The label
        itself is the final component, keeping the key total if a label ever stops
        matching that shape.
        """
        _, _, suffix = candidate.label.rpartition("-")
        return (candidate.generation, int(suffix) if suffix.isdigit() else 0, candidate.label)

    def _ranked_metrics(self, candidate: Candidate) -> dict[str, float]:
        """The metrics this selector ranks on, merged across its objective channels."""
        merged: dict[str, float] = {}
        for channel in self._config.objectives:
            merged.update(candidate.rewards[channel].metrics or {})
        return merged

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=100, cell_timeout=3600.0)))
    async def select_diverse_survivors(self, ranked: list[Candidate], k: int) -> list[Candidate]:  # pyright: ignore[reportReturnType]
        """Choose up to k survivors from Pareto-ranked candidates.

        ``ranked`` is already Pareto-sorted using outcome and trajectory scores:
        front 0 (non-dominated) first, then front 1, etc. Inside a front,
        candidates are incomparable, so prefer agents with distinct architecture
        changes, complementary task coverage, and different trajectory strengths.

        To avoid getting stuck on the same agents round after round:
        - Always include at least one candidate created this round — the ones whose
          `generation` equals the highest across `ranked`. If every new candidate sits on
          a worse Pareto front, still take the best of them.
        - Prefer candidates with different `optimization_type` values, or that address
          different root causes.
        - Prefer a mix of recent and older lineages over k agents from one branch.
        - Prefer an agent whose failures differ from the others already chosen.
        - Read each candidate's `description` and `generated_from` to tell what
          actually changed; two agents with near-identical changes are one choice.

        Return exactly the candidates you choose, in preference order, at most k.
        """
        ...
