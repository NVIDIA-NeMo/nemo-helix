# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for the paired Evaluation comparison."""

from abc import ABC, abstractmethod

from nhx.intake.experiments.compare import CompareMetric, EvaluationMetricSession



class EvaluationCompareRepository(ABC):
    """Domain-facing interface for the per-session inputs of a comparison."""

    @abstractmethod
    async def get_metric_sessions(
        self,
        *,
        workspace: str,
        evaluation_names: list[str],
        metric: CompareMetric,
    ) -> list[EvaluationMetricSession]:
        """Return every session's value for ``metric`` across both evaluations.

        One query covers all requested evaluations and reads every session —
        no page cap, because the pairing pipeline needs each attempt.
        """
        pass
