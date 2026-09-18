# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt GA individual state."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from nemo_evaluator_sdk.agent_eval.scores import AgentEvalTaskScore


@dataclass
class GaIndividual:
    """One prompt candidate in a GA population."""

    prompts: dict[str, str]
    generation: int
    individual_index: int
    parent_ids: tuple[str, ...] = ()
    carried_from: str | None = None
    phase_trial_number: int | None = None
    global_trial_number: int | None = None
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    normalized_metrics: dict[str, float] = field(default_factory=dict)
    raw_scores: tuple[AgentEvalTaskScore, ...] = ()
    fitness: float | None = None
    status: str = "pending"
    failure_reason: str | None = None
    transform_failures: list[str] = field(default_factory=list)
    transform_successes: list[str] = field(default_factory=list)

    @property
    def individual_id(self) -> str:
        """Stable generation-local individual identifier."""

        return f"g{self.generation:03d}-i{self.individual_index:03d}"

    @property
    def is_valid(self) -> bool:
        """Whether this individual can participate in selection."""

        return self.status == "completed" and self.fitness is not None

    @property
    def prompt_signature(self) -> tuple[tuple[str, str], ...]:
        """Deterministic prompt tuple for duplicate/diversity checks."""

        return tuple(sorted(self.prompts.items()))

    def clone_as_elite(self, *, generation: int, individual_index: int) -> GaIndividual:
        """Copy an already evaluated elite into a later generation without reevaluation."""

        return GaIndividual(
            prompts=copy.deepcopy(self.prompts),
            generation=generation,
            individual_index=individual_index,
            parent_ids=(self.individual_id,),
            carried_from=self.individual_id,
            phase_trial_number=self.phase_trial_number,
            global_trial_number=self.global_trial_number,
            aggregate_metrics=dict(self.aggregate_metrics),
            normalized_metrics=dict(self.normalized_metrics),
            raw_scores=self.raw_scores,
            fitness=self.fitness,
            status=self.status,
            failure_reason=self.failure_reason,
            transform_failures=list(self.transform_failures),
            transform_successes=list(self.transform_successes),
        )

    def to_history_row(self) -> dict[str, Any]:
        """Return a JSON/CSV-safe summary row for this individual."""

        return {
            "individual_id": self.individual_id,
            "phase": "prompt",
            "generation": self.generation,
            "individual_index": self.individual_index,
            "phase_trial_number": self.phase_trial_number,
            "global_trial_number": self.global_trial_number,
            "status": self.status,
            "fitness": self.fitness,
            "failure_reason": self.failure_reason,
            "parent_ids": ",".join(self.parent_ids),
            "carried_from": self.carried_from,
            "transform_failures": " | ".join(self.transform_failures),
            "transform_successes": " | ".join(self.transform_successes),
            **{f"metric.{name}": value for name, value in sorted(self.aggregate_metrics.items())},
            **{f"normalized.{name}": value for name, value in sorted(self.normalized_metrics.items())},
            **{f"prompt.{name}": value for name, value in sorted(self.prompts.items())},
        }


__all__ = ["GaIndividual"]
