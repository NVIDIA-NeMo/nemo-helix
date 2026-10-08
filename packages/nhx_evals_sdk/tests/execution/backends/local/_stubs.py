# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test stubs for local evaluator backend tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from nhx_evals_sdk.inference import PostprocessResponse, PreprocessRequest
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.values import RunConfig
from pydantic import BaseModel


class IdentityPreprocessHook(PreprocessRequest):
    """Named preprocess hook used to verify hook ordering in backend tests."""

    def preprocess(self, request: dict, id: str | None = None) -> dict:  # noqa: A002
        """Return the request unchanged."""
        del id
        return request


class IdentityPostprocessHook(PostprocessResponse):
    """Named postprocess hook used to verify hook ordering in backend tests."""

    def postprocess(self, response: dict, id: str | None = None) -> dict:  # noqa: A002
        """Return the response unchanged."""
        del id
        return response


class DuplicateMetric:
    """Minimal metric stub used to exercise unique metric key handling."""

    @property
    def type(self) -> str:
        """Return the metric key used by the backend when namespacing results."""
        return "duplicate"

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return the outputs exposed by this metric."""
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Return one structured score result for protocol conformance in tests."""
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


class PreparedBenchmarkMetric(BaseModel):
    """Metric stub that exposes all local preparation hooks."""

    type: str = "prepared"
    applied_parallelism: int | None = None
    secrets_resolved: bool = False
    preflight_ran: bool = False

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return the outputs exposed by this metric."""
        return [MetricOutputSpec.continuous_score("score")]

    def secrets(self) -> dict[str, object]:
        """Return no concrete secret refs while satisfying the secrets protocol."""
        return {}

    def apply_evaluation_job_params(self, params: RunConfig) -> None:
        """Capture runtime job params applied during local metric preparation."""
        self.applied_parallelism = params.parallelism

    async def resolve_secrets(self, secret_resolver: Callable[[str], Awaitable[str | None]]) -> None:
        """Record that local metric preparation attempted secret resolution."""
        del secret_resolver
        self.secrets_resolved = True

    async def preflight(self) -> None:
        """Record that local metric preparation ran preflight."""
        self.preflight_ran = True

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Return one structured score result for protocol conformance in tests."""
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])
