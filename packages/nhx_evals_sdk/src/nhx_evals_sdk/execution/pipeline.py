# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline infrastructure types for online execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from nhx_evals_sdk.metrics.protocol import MetricResult
from nhx_evals_sdk.values.results import RowScore


@dataclass
class GeneratedSampleEvent:
    """Carries one generated sample row from producer workers to scorer workers."""

    row_index: int
    item: MappingProxyType
    sample: MappingProxyType
    requests_log: list[dict[str, Any]]


class GeneratedSampleScoringPipeline(Protocol):
    """Internal contract for pipelines that generate a sample before scoring."""

    rows: list[dict[str, Any]]
    parallelism: int

    async def generate_sample(self, index: int, row: dict[str, Any]) -> dict[str, Any]:
        """Prepare the generated sample payload for one input row."""
        ...

    def handle_generation_error(
        self,
        index: int,
        row: dict[str, Any],
        error: Exception,
        generation_requests: list[dict[str, Any]],
    ) -> tuple[int, MetricResult | None, RowScore]:
        """Convert a generation failure into a completed row result or raise."""
        ...

    async def score_row(
        self,
        index: int,
        row: dict[str, Any],
        sample: dict[str, Any],
        generation_requests: list[dict[str, Any]],
    ) -> tuple[int, MetricResult | None, RowScore]:
        """Score one prepared sample and return the completed row result."""
        ...


@dataclass
class PipelineRuntime:
    """Mutable queue-worker state shared by the generic pipeline helpers."""

    pipeline: GeneratedSampleScoringPipeline
    sample_queue: asyncio.Queue[GeneratedSampleEvent | object]
    results: list[tuple[int, MetricResult | None, RowScore] | None]
