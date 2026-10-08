# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-oriented run handle types for evaluator backends."""

from __future__ import annotations

from typing import Any, Protocol

from nhx_evals_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nhx_evals_sdk.values.results import EvaluationResult


class JobEvaluationRun(Protocol):
    """Structural contract for job-backed single-metric run handles."""

    async def status(self) -> str:
        """Return the current lifecycle status for the run.

        Returns:
            A backend-defined status string such as `created`, `active`, or
            `completed`.
        """
        ...

    async def result(
        self,
        *,
        timeout_s: float | None = None,
        poll_interval_s: float = 1.0,
    ) -> EvaluationResult:
        """Wait for the run to finish and return its metric result.

        Args:
            timeout_s: Optional maximum wait time before raising `TimeoutError`.
            poll_interval_s: Poll interval used by backends that require
                repeated status checks.

        Returns:
            The completed single-metric evaluation result.
        """
        ...

    def result_sync(
        self,
        *,
        timeout_s: float | None = None,
        poll_interval_s: float = 1.0,
    ) -> EvaluationResult:
        """Synchronously wait for the run to finish and return its result.

        Args:
            timeout_s: Optional maximum wait time before raising `TimeoutError`.
            poll_interval_s: Poll interval used by backends that require
                repeated status checks.

        Returns:
            The completed single-metric evaluation result.
        """
        ...

    def job(self) -> Any | None:
        """Return the primary backend job object, if one exists.

        Returns:
            A backend-specific job object for single-job runs, or `None` for
            purely local completed runs.
        """
        ...

    def jobs(self) -> list[Any]:
        """Return all backend job objects associated with the run.

        Returns:
            One or more backend-specific job objects.
        """
        ...


class JobBenchmarkEvaluationRun(Protocol):
    """Structural contract for job-backed multi-metric run handles."""

    async def status(self) -> str:
        """Return the current lifecycle status for the combined run.

        Returns:
            A backend-defined status string aggregated across all underlying
            metric runs.
        """
        ...

    async def result(
        self,
        *,
        timeout_s: float | None = None,
        poll_interval_s: float = 1.0,
    ) -> BenchmarkEvaluationResult:
        """Wait for all metrics to finish and return the combined result.

        Args:
            timeout_s: Optional maximum wait time before raising `TimeoutError`.
            poll_interval_s: Poll interval used by backends that require
                repeated status checks.

        Returns:
            The completed multi-metric evaluation result.
        """
        ...

    def result_sync(
        self,
        *,
        timeout_s: float | None = None,
        poll_interval_s: float = 1.0,
    ) -> BenchmarkEvaluationResult:
        """Synchronously wait for all metrics to finish and return the result.

        Args:
            timeout_s: Optional maximum wait time before raising `TimeoutError`.
            poll_interval_s: Poll interval used by backends that require
                repeated status checks.

        Returns:
            The completed multi-metric evaluation result.
        """
        ...

    def job(self) -> Any | None:
        """Return a primary backend job object when one exists.

        Returns:
            A backend-specific job object, or `None` when the run is backed by
            multiple jobs or no remote jobs at all.
        """
        ...

    def jobs(self) -> list[Any]:
        """Return all backend job objects associated with the run.

        Returns:
            One or more backend-specific job objects.
        """
        ...
