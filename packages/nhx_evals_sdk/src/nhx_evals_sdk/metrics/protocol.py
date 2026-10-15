# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime protocol for implementing Evaluator metrics."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nhx_evals_sdk.resolver_protocols import ModelResolver, SecretResolver
from nhx_evals_sdk.values.common import SecretRef
from nhx_evals_sdk.values.models import ModelRef
from nhx_evals_sdk.values.protocol import (
    BooleanValue,
    CandidateOutput,
    ContinuousScore,
    DatasetRow,
    DiscreteScore,
    Label,
    MetricDescriptor,
    MetricDiagnostic,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
    MetricTypeName,
    validate_metric_result,
)

__all__ = [
    "BooleanValue",
    "CandidateOutput",
    "ContinuousScore",
    "CorpusMetric",
    "DatasetRow",
    "DiscreteScore",
    "Label",
    "Metric",
    "MetricDescriptor",
    "MetricDiagnostic",
    "MetricInput",
    "MetricOutput",
    "MetricOutputSpec",
    "MetricResult",
    "MetricTypeName",
    "MetricWithModels",
    "MetricWithPreflight",
    "MetricWithSecrets",
    "validate_metric_result",
]


@runtime_checkable
class Metric(Protocol):
    """Shared row-scoring primitive for SDK runtime metrics."""

    @property
    def type(self) -> MetricTypeName:
        """Return the public metric key/type identifier."""
        ...

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return declared row-level outputs emitted by this metric."""
        ...

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute structured output for one row/candidate pair."""
        ...


@runtime_checkable
class CorpusMetric(Protocol):
    """Protocol for metrics that also emit corpus-level scores."""

    async def compute_corpus_scores(self, inputs: list[MetricInput]) -> MetricResult | None:
        """Compute corpus-level scores across all evaluated rows."""
        ...


@runtime_checkable
class MetricWithSecrets(Protocol):
    """Protocol for metrics that require secrets."""

    def secrets(self) -> dict[str, SecretRef]:
        """Return environment variables mapped to secret references."""
        ...

    async def resolve_secrets(self, secret_resolver: SecretResolver) -> None:
        """Resolve secrets before the metric is used for evaluation."""
        ...


@runtime_checkable
class MetricWithModels(Protocol):
    """Protocol for metrics that require model resolution."""

    def model_refs(self) -> dict[str, ModelRef]:
        """Return metric field names mapped to model references.

        Example: ``{"model": ModelRef("workspace/model")}``.
        """
        ...

    async def resolve_models(self, model_resolver: ModelResolver) -> None:
        """Resolve model references before the metric is used for evaluation."""
        ...


@runtime_checkable
class MetricWithPreflight(Protocol):
    """Protocol for metrics that need one-time setup before parallel evaluation starts."""

    async def preflight(self) -> None:
        """Run one-time preflight before processing rows."""
        ...
