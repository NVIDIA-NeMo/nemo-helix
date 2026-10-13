# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value types used during pipeline execution."""

from enum import Enum


class EvaluationPhase(str, Enum):
    """Phase where a row failure is raised instead of converted to NaN."""

    SAMPLE_GENERATION = "sample_generation"
    METRIC_SCORING = "metric_scoring"


class EvaluationError(Exception):
    """Raised when evaluation fails with ``fail_fast=True``.

    ``metric_key`` is the public metric identifier. For single-metric
    pipelines it is the metric type name; for multi-metric benchmark
    pipelines it is the fully-qualified metric ref.
    """

    def __init__(
        self,
        index: int,
        message: str,
        *,
        phase: EvaluationPhase = EvaluationPhase.METRIC_SCORING,
        metric_key: str | None = None,
    ) -> None:
        """Create an evaluation error that keeps sample index context.

        Strict mode means ``fail_fast=True``: row failures abort evaluation
        instead of being converted into NaN row results.

        Args:
            index: Position of the failing sample in the input list.
            message: Original exception message.
            phase: Execution phase where the row failed.
            metric_key: Public metric identifier for the failing metric.
        """
        self.index = index
        self.message = message
        self.phase = phase
        self.metric_key = metric_key
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Return a concise user-facing error message."""
        phase = self.phase.value.replace("_", " ")
        metric = f" for metric {self.metric_key!r}" if self.metric_key else ""
        return f"Evaluation failed during {phase}{metric} on row {self.index}: {self.message}"
