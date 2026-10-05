# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface and read models for Evaluation rollups."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoreRollup:
    sum: float | None
    mean: float | None
    median: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    count: int


@dataclass
class EvaluationRollup:
    evaluation_name: str
    run_count: int = 0
    test_case_count: int = 0
    model_names: list[str] = field(default_factory=list)
    agent_names: list[str] = field(default_factory=list)
    agent_versions: list[str] = field(default_factory=list)
    evaluator_scores: dict[str, ScoreRollup] = field(default_factory=dict)
    cost_usd: ScoreRollup | None = None
    latency_ms: ScoreRollup | None = None
    tokens: ScoreRollup | None = None

    @property
    def evaluator_names(self) -> list[str]:
        return sorted(self.evaluator_scores)


class EvaluationRollupRepository(ABC):
    """Domain-facing interface for Evaluation rollup reads."""

    @abstractmethod
    async def get_rollups(
        self,
        *,
        workspace: str,
        evaluation_names: list[str],
    ) -> dict[str, EvaluationRollup]:
        """Return rollups keyed by Evaluation name."""
        pass
